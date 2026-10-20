# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 transformer (80B MoE language model + diffusion heads) for Neuron.

HunyuanImage-3.0 generates an image with its language model: the latent tokens are
part of one joint sequence

    <bos> prompt <boi> <img_size_1024> <img_ratio_k> <timestep> <img> x N <eoi>

and every denoising step runs the 32 MoE decoder layers over the timestep token and
the N image tokens. The attention is causal, except that the image tokens attend to
each other bidirectionally. Two consequences shape this port:

- The text (everything before ``<timestep>``) is causal and timestep-independent, so
  its keys and values are computed once per request (:meth:`forward_prefix`) and
  reused by every step, as the reference's static KV cache does.
- A step (:meth:`forward_step`) sees only the timestep token and the image tokens:
  the image tokens attend to the image, the timestep token and the text; the timestep
  token to itself and the text.

Classifier-free guidance runs the conditional and unconditional sequences (the latter
with each prompt token replaced by ``<cfg>``) as two rows of one batch, or with CFG
parallelism one row on each of two TP groups; the guided velocity and the
flow-matching Euler update are computed in-graph.

Layer structure (all 32 layers):
  - RMSNorm -> GQA attention (32 query / 8 KV heads, RoPE *then* per-head RMS QK-norm)
  - RMSNorm -> MoE: softmax top-8 of 64 SwiGLU experts (renormalized) + a shared
    SwiGLU expert

Diffusion heads (their large convolutions sharded over TP):
  - ``patch_embed``: conv 32->1024 and a timestep-conditioned ResBlock 1024->4096
  - ``final_layer``: ResBlock 4096->1024, GroupNorm, SiLU, conv 1024->32
  - three timestep embedders (the ``<timestep>`` token, and each head's ResBlock)

Parallelism (one TP group):
  - Attention heads sharded over TP; with TP > 8 each KV head is replicated on
    TP / 8 ranks.
  - Experts sharded over the same ranks (expert parallelism, 64 / TP experts each,
    blockwise NKI MoE kernel); the shared expert runs sequence-parallel (or, to save
    memory, TP-sharded).
  - Megatron sequence parallelism: norms and residuals run on a 1/TP slice of the
    tokens; all-gather before attention/MoE, reduce-scatter after.
"""

import json
import logging
import math
import os

import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    QKVOutputLayout,
)
from safetensors import safe_open
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.functional.moe.moe_cte import MoECTEImplementation
from vllm_omni.diffusion.distributed.parallel_state import get_cfg_group, get_world_group

logger = logging.getLogger(__name__)

# The attention kernel reads per-query bounds in tiles of this many rows.
_BOUNDS_TILE = 128
# Tokens per block of the blockwise MoE kernel.
_MOE_BLOCK_SIZE = 512


def expert_placement(load, ranks: int) -> list[int]:
    """The order of a layer's experts such that rank ``r`` holds the ``r``-th slice.

    ``load`` gives each expert's expected share of the routed tokens. Every rank gets
    the same number of experts and, greedily (heaviest expert first, to the least
    loaded rank with a free slot), about the same total load: the MoE layers' time is
    set by the busiest rank, which all others wait for at the next collective.
    """
    per_rank = len(load) // ranks
    slots: list[list[int]] = [[] for _ in range(ranks)]
    totals = [0.0] * ranks
    for expert in sorted(range(len(load)), key=lambda e: -load[e]):
        rank = min((r for r in range(ranks) if len(slots[r]) < per_rank), key=totals.__getitem__)
        slots[rank].append(expert)
        totals[rank] += load[expert]
    return [expert for slot in slots for expert in slot]


def _moe_token_count(tokens: int) -> int:
    """Tokens the MoE kernels run on: a multiple of 128 (the kernel's row tile), and
    above 2048 of 2048, so that the blockwise mapping runs as NKI kernels (its
    ``indexed_flatten`` needs ``T / 128`` divisible by 16) instead of a slow fallback."""
    multiple = 128 if tokens <= 2048 else 2048
    return -(-tokens // multiple) * multiple


# ----------------------------------------------------------------------
# Host-side helpers
# ----------------------------------------------------------------------


def rope_cos_sin(pos_y: torch.Tensor, pos_x: torch.Tensor, head_dim: int, theta: float):
    """HunyuanImage-3.0 2-D RoPE tables ``[S, head_dim]`` (float32, rotate-half).

    As ``build_2d_rope``: frequency ``k`` of the ``head_dim / 2`` takes its position from
    the y axis when ``k`` is even and from the x axis when odd. Text tokens have
    ``y = x = position``, which is plain 1-D RoPE.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.stack([pos_y, pos_x], dim=-1).float()  # [S, 2]
    angles = (pos[:, None, :] * inv_freq.view(1, head_dim // 4, 2)).reshape(-1, head_dim // 2)
    angles = angles.repeat(1, 2)
    return angles.cos(), angles.sin()


def text_rope(length: int, head_dim: int, theta: float, start: int = 0):
    pos = torch.arange(start, start + length)
    return rope_cos_sin(pos, pos, head_dim, theta)


def image_rope(image_start: int, height: int, width: int, head_dim: int, theta: float):
    """RoPE of an ``height x width`` token grid whose first token sits at ``image_start``.

    The grid is centered on the diagonal: ``y = L + (h*w - h)/2 + i``,
    ``x = L + (h*w - w)/2 + j`` with ``L = image_start``.
    """
    beta_y = image_start + (width * height - height) // 2
    beta_x = image_start + (width * height - width) // 2
    ys = torch.arange(height).repeat_interleave(width) + beta_y
    xs = torch.arange(width).repeat(height) + beta_x
    return rope_cos_sin(ys, xs, head_dim, theta)


def timestep_frequencies(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0):
    """Sinusoidal ``[cos, sin]`` embedding of ``t`` ``[B]`` (float32)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Checkpoint:
    """Lazy reader of a sharded safetensors checkpoint (``model.safetensors.index.json``)."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._files: dict[str, object] = {}

    def _file(self, key: str):
        name = self.weight_map[key]
        if name not in self._files:
            self._files[name] = safe_open(os.path.join(self.path, name), framework="pt")
        return self._files[name]

    def get(self, key: str) -> torch.Tensor:
        return self._file(key).get_tensor(key)

    def rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        return self._file(key).get_slice(key)[start:stop]

    def cols(self, key: str, start: int, stop: int) -> torch.Tensor:
        return self._file(key).get_slice(key)[:, start:stop]


# ----------------------------------------------------------------------
# Collectives
# ----------------------------------------------------------------------


def _all_gather(x: torch.Tensor, group, size: int) -> torch.Tensor:
    if size == 1:
        return x
    out = torch.empty((x.shape[0] * size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _all_reduce(x: torch.Tensor, group, size: int) -> torch.Tensor:
    if size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


def _reduce_scatter(x: torch.Tensor, group, size: int) -> torch.Tensor:
    if size == 1:
        return x
    out = torch.empty((x.shape[0] // size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), op=dist.ReduceOp.SUM, group=group)
    return out


# ----------------------------------------------------------------------
# Language-model layers
# ----------------------------------------------------------------------


class HunyuanRMSNorm(nn.Module):
    """RMSNorm normalized in float32; the weight multiplies the input-dtype result."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class HunyuanImage3Attention(nn.Module):
    """GQA attention with RoPE followed by per-head RMS QK-norm, TP-sharded by heads."""

    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.head_dim = config.attention_head_dim
        num_heads, num_kv = config.num_attention_heads, config.num_key_value_heads
        if num_heads % self.tp_size:
            raise ValueError(f"{num_heads} attention heads do not shard over TP={self.tp_size}")
        if self.tp_size <= num_kv and num_kv % self.tp_size:
            raise ValueError(f"{num_kv} KV heads do not shard over TP={self.tp_size}")
        if self.tp_size > num_kv and self.tp_size % num_kv:
            raise ValueError(f"TP={self.tp_size} is not a multiple of {num_kv} KV heads")
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = max(1, num_kv // self.tp_size)
        self.groups = num_heads // num_kv  # query heads per KV head
        self.scale = self.head_dim**-0.5
        q_size, kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.split = [q_size, q_size + kv_size]
        # [in, out] layout: local [Q | K | V] columns.
        self.qkv_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=dtype)
        )
        self.o_weight = nn.Parameter(torch.empty(q_size, self.hidden_size, dtype=dtype))
        self.eps = config.rms_norm_eps
        self.query_layernorm = HunyuanRMSNorm(self.head_dim, config.rms_norm_eps, dtype)
        self.key_layernorm = HunyuanRMSNorm(self.head_dim, config.rms_norm_eps, dtype)

    def load(self, ckpt: Checkpoint, prefix: str, num_kv: int) -> dict[str, torch.Tensor]:
        """Local weights from the checkpoint's fused ``qkv_proj``.

        Its rows are grouped per KV head as ``[q x groups, k, v]`` (``head_dim`` rows each).
        """
        d, groups = self.head_dim, self.groups
        w = ckpt.get(f"{prefix}qkv_proj.weight").view(num_kv, groups + 2, d, self.hidden_size)
        if self.tp_size <= num_kv:
            kv = slice(self.tp_rank * self.num_kv_heads, (self.tp_rank + 1) * self.num_kv_heads)
            q, k, v = w[kv, :groups], w[kv, groups], w[kv, groups + 1]
        else:
            replicas = self.tp_size // num_kv
            kv = self.tp_rank // replicas
            first = (self.tp_rank % replicas) * self.num_heads
            q = w[kv : kv + 1, first : first + self.num_heads]
            k, v = w[kv : kv + 1, groups], w[kv : kv + 1, groups + 1]
        qkv = torch.cat([t.reshape(-1, self.hidden_size) for t in (q, k, v)])
        # Local query heads are global heads [rank * num_heads, ...), contiguous columns
        # of o_proj in either case.
        cols = self.num_heads * d
        o = ckpt.cols(f"{prefix}o_proj.weight", self.tp_rank * cols, (self.tp_rank + 1) * cols)
        return {
            "qkv_weight": qkv.t(),
            "o_weight": o.t(),
            "query_layernorm.weight": ckpt.get(f"{prefix}query_layernorm.weight"),
            "key_layernorm.weight": ckpt.get(f"{prefix}key_layernorm.weight"),
        }

    def project(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, fused: bool = True):
        """``x`` ``[T, H]`` -> ``q`` ``[heads, T, D]``, ``k``/``v`` ``[kv_heads, T, D]``.

        As the reference: RoPE in float32, then the per-head RMS QK-norm, then a cast to
        the activation dtype. On device one fused NKI kernel does the projection, RoPE
        and the (post-RoPE) norm and writes the head-major layout attention consumes.
        """
        tokens = x.shape[0]
        if fused and x.device.type != "cpu":
            qkv = NF.qkv_proj(
                x[None],
                self.qkv_weight,
                d_head=self.head_dim,
                output_layout=QKVOutputLayout.NBSd,
                cos_cache=cos[None],
                sin_cache=sin[None],
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                qk_norm_post_rope_q_norm=NormType.RMS_NORM,
                qk_norm_post_rope_k_norm=NormType.RMS_NORM,
                qk_norm_post_rope_eps=self.eps,
                qk_norm_post_rope_q_gamma=self.query_layernorm.weight[None],
                qk_norm_post_rope_k_gamma=self.key_layernorm.weight[None],
            ).squeeze(1)
            kv = self.num_heads + self.num_kv_heads
            return qkv[: self.num_heads], qkv[self.num_heads : kv], qkv[kv:]
        qkv = torch.matmul(x, self.qkv_weight)
        q, k, v = torch.tensor_split(qkv, self.split, dim=-1)
        q = q.view(tokens, self.num_heads, self.head_dim).float()
        k = k.view(tokens, self.num_kv_heads, self.head_dim).float()
        cos, sin = cos[:, None], sin[:, None]
        q = self.query_layernorm(q * cos + _rotate_half(q) * sin).to(x.dtype)
        k = self.key_layernorm(k * cos + _rotate_half(k) * sin).to(x.dtype)
        v = v.view(tokens, self.num_kv_heads, self.head_dim)
        return q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

    def expand_kv(self, t: torch.Tensor) -> torch.Tensor:
        """KV heads ``[kv_heads, S, D]`` repeated to the local query heads."""
        repeat = self.num_heads // self.num_kv_heads
        return t.repeat_interleave(repeat, dim=0) if repeat > 1 else t

    def attend_causal(self, q, k, v) -> torch.Tensor:
        """Causal attention of one row ``[heads, S, D]`` -> ``[S, heads * D]``."""
        out = NF.flash_attention(
            q.contiguous(),
            self.expand_kv(k).contiguous(),
            self.expand_kv(v).contiguous(),
            scale=self.scale,
            causal_mask=True,
            tp_q=True,
            tp_k=True,
        )
        return out.transpose(0, 1).reshape(q.shape[1], -1)

    def attend_bounded(self, q, k, v, bound_min, bound_max) -> torch.Tensor:
        """Query ``i`` of ``q`` ``[heads, S, D]`` attends keys ``[bound_min[i], bound_max[i])``.

        ``bound_min``/``bound_max`` are ``[S_pad, 1]`` int32 with ``S_pad`` the query
        count rounded up to the kernel's bounds tile.
        """
        heads, seq_q, _ = q.shape
        pad = bound_min.shape[0] - seq_q
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
        keys = k.shape[1]
        kpad = -keys % _BOUNDS_TILE
        k, v = self.expand_kv(k), self.expand_kv(v)
        if kpad:
            k, v = F.pad(k, (0, 0, 0, kpad)), F.pad(v, (0, 0, 0, kpad))
        out = NF.flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=bound_min[None].expand(heads, -1, -1).contiguous(),
            bound_max=bound_max[None].expand(heads, -1, -1).contiguous(),
        )
        return out[:, :seq_q].transpose(0, 1).reshape(seq_q, -1)


class HunyuanImage3MoE(nn.Module):
    """Softmax top-k routing over expert-parallel SwiGLU experts plus a shared expert."""

    def __init__(self, config, dtype: torch.dtype, shared_sequence_parallel: bool = True):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_coordinator = get_tp_group()
        hidden = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk[0] if isinstance(config.moe_topk, list) else config.moe_topk
        inter = config.moe_intermediate_size
        self.intermediate = inter[0] if isinstance(inter, list) else inter
        shared = config.num_shared_expert
        shared = shared[0] if isinstance(shared, list) else shared
        if self.num_experts % self.tp_size:
            raise ValueError(f"{self.num_experts} experts do not shard over TP={self.tp_size}")
        self.num_local_experts = self.num_experts // self.tp_size
        self.expert_start = self.tp_rank * self.num_local_experts
        # The shared expert either runs sequence-parallel (full weights on each rank's
        # tokens) or, to save 2.25 GB per rank at TP=16, with its intermediate dim
        # sharded over TP on all tokens (summed with the routed experts).
        self.shared_sp = shared_sequence_parallel
        self.shared_full = self.intermediate * shared
        if not self.shared_sp and self.shared_full % self.tp_size:
            raise ValueError(f"the shared expert does not shard over TP={self.tp_size}")
        self.shared_inter = self.shared_full // (1 if self.shared_sp else self.tp_size)

        # The reference keeps the router in float32.
        self.router_weight = nn.Parameter(
            torch.empty(hidden, self.num_experts, dtype=torch.float32)
        )
        self.gate_up_weight = nn.Parameter(
            torch.empty(self.num_local_experts, hidden, 2, self.intermediate, dtype=dtype)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.num_local_experts, self.intermediate, hidden, dtype=dtype)
        )
        self.shared_gate_weight = nn.Parameter(torch.empty(hidden, self.shared_inter, dtype=dtype))
        self.shared_up_weight = nn.Parameter(torch.empty(hidden, self.shared_inter, dtype=dtype))
        self.shared_down_weight = nn.Parameter(torch.empty(self.shared_inter, hidden, dtype=dtype))

    def load(self, ckpt: Checkpoint, prefix: str, order=None) -> dict[str, torch.Tensor]:
        """Local experts and the shared expert.

        ``order`` (see :func:`expert_placement`) permutes the experts: this rank holds
        ``order[expert_start : expert_start + num_local_experts]`` and the router's
        columns follow the same order, which leaves the routing unchanged (top-k and
        softmax do not depend on the experts' order). The checkpoint's
        ``gate_and_up_proj`` computes ``x1 * silu(x2)`` for ``x1, x2 = chunk(2)``: its
        first half is the up projection, the second the gate.
        """
        inter = self.intermediate
        order = list(order) if order is not None else list(range(self.num_experts))
        local = order[self.expert_start : self.expert_start + self.num_local_experts]
        gate_up, down = [], []
        for e in local:
            w = ckpt.get(f"{prefix}experts.{e}.gate_and_up_proj.weight")  # [2I, H]
            gate_up.append(torch.stack([w[inter:].t(), w[:inter].t()], dim=1))  # [H, 2, I]
            down.append(ckpt.get(f"{prefix}experts.{e}.down_proj.weight").t())  # [I, H]
        shared = ckpt.get(f"{prefix}shared_mlp.gate_and_up_proj.weight")
        shared_down = ckpt.get(f"{prefix}shared_mlp.down_proj.weight")
        lo = 0 if self.shared_sp else self.tp_rank * self.shared_inter
        hi = lo + self.shared_inter
        return {
            "router_weight": ckpt.get(f"{prefix}gate.wg.weight").float().t()[:, order],
            "gate_up_weight": torch.stack(gate_up),
            "down_weight": torch.stack(down),
            "shared_gate_weight": shared[self.shared_full + lo : self.shared_full + hi].t(),
            "shared_up_weight": shared[lo:hi].t(),
            "shared_down_weight": shared_down[:, lo:hi].t(),
        }

    def shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        """The shared expert (or its TP shard) on ``x``."""
        tokens = x.shape[0]
        # The NKI MLP kernel needs an even token count.
        h = F.pad(x, (0, 0, 0, tokens % 2)) if tokens % 2 else x
        out = NF.mlp(h, self.shared_gate_weight, self.shared_up_weight, self.shared_down_weight)
        return out[:tokens] if tokens % 2 else out

    def decode(self, x: torch.Tensor, part: int = 0, parts: int = 1) -> torch.Tensor:
        """Routed experts for one token ``x`` ``[1, H]`` -> this rank's partial sum.

        At one token the experts are bound by reading their weights: every local expert
        is computed densely and weighted by its affinity (zero unless selected), which
        beats the blockwise kernel's routing and per-block overhead. With ``parts`` > 1
        only the ``part``-th slice of the local experts is computed (the ranks holding
        the same experts share the work).
        """
        affinities = NF.router(
            hidden_states=x,
            router_weights=self.router_weight,
            top_k=self.top_k,
            activation="softmax",
            computation_dtype=torch.float32,
        )
        local = affinities[0, self.expert_start : self.expert_start + self.num_local_experts]
        share = self.num_local_experts // parts
        experts = slice(part * share, (part + 1) * share)
        gate_up, down, local = (
            self.gate_up_weight[experts],
            self.down_weight[experts],
            local[experts],
        )
        gate = torch.einsum("h,ehi->ei", x[0], gate_up[:, :, 0])
        up = torch.einsum("h,ehi->ei", x[0], gate_up[:, :, 1])
        out = torch.einsum("ei,eih->eh", F.silu(gate) * up, down)
        return (out.float() * local[:, None]).sum(0, keepdim=True).to(x.dtype)

    def shared_expert_part(self, x: torch.Tensor, part: int, parts: int) -> torch.Tensor:
        """The ``part``-th slice of the (TP-sharded) shared expert's intermediate dim."""
        share = self.shared_inter // parts
        cols = slice(part * share, (part + 1) * share)
        h = F.pad(x, (0, 0, 0, 1))  # the NKI MLP kernel needs an even token count
        out = NF.mlp(
            h,
            self.shared_gate_weight[:, cols],
            self.shared_up_weight[:, cols],
            self.shared_down_weight[cols],
        )
        return out[:1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Routed experts: ``x`` ``[T, H]`` (all tokens) -> this rank's partial sum ``[T, H]``."""
        affinities = NF.router(
            hidden_states=x,
            router_weights=self.router_weight,
            top_k=self.top_k,
            activation="softmax",
            computation_dtype=torch.float32,
        )
        local = affinities[:, self.expert_start : self.expert_start + self.num_local_experts]
        # Padding tokens have no expert (zero affinity), so they add no blocks.
        tokens = x.shape[0]
        padded = _moe_token_count(tokens)
        if padded != tokens:
            local = F.pad(local, (0, 0, 0, padded - tokens))
        masked, position_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=local.contiguous(),
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=_MOE_BLOCK_SIZE,
            moe_group=self.tp_coordinator,
            tp_degree=1,
        )
        # The block count is static (every token could pick every local expert) but
        # only a few blocks hold tokens: the hybrid kernel's dynamic loop runs just
        # those (``num_static_block=0``; ``shard_on_block`` would run them all). Its
        # inputs use the [T + 1] convention: padding slots point at an extra zero row.
        routed = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_i_hybrid,
            num_static_block=0,
            conditions=F.pad(conditions, (0, 1)),
            hidden_states=F.pad(x, (0, 0, 0, padded + 1 - tokens)),
            expert_affinities_masked=F.pad(masked, (0, 0, 0, self.num_local_experts)),
            gate_up_proj_weight=self.gate_up_weight,
            down_proj_weight=self.down_weight,
            activation_function=ActFnType.SiLU,
            block_size=_MOE_BLOCK_SIZE,
            token_position_to_id=torch.where(
                position_to_id < 0, torch.full_like(position_to_id, padded), position_to_id
            ),
            block_to_expert=block_to_expert.view(-1, 1),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )
        return routed[:tokens].to(x.dtype)


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config, dtype: torch.dtype, shared_sequence_parallel: bool = True):
        super().__init__()
        self.input_layernorm = HunyuanRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.self_attn = HunyuanImage3Attention(config, dtype)
        self.post_attention_layernorm = HunyuanRMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype
        )
        self.mlp = HunyuanImage3MoE(config, dtype, shared_sequence_parallel)


# ----------------------------------------------------------------------
# Diffusion heads (TP-sharded convolutions)
# ----------------------------------------------------------------------


def _group_norm(x: torch.Tensor, groups: int, weight, bias, eps: float = 1e-5) -> torch.Tensor:
    """GroupNorm computed in float32 (``weight``/``bias`` cover ``x``'s channels)."""
    return F.group_norm(x.float(), groups, weight.float(), bias.float(), eps).to(x.dtype)


def _conv(x: torch.Tensor, weight, bias=None) -> torch.Tensor:
    return F.conv2d(x, weight, bias, padding=weight.shape[-1] // 2)


def _params(module: nn.Module, shapes: dict[str, tuple], dtype: torch.dtype) -> None:
    for name, shape in shapes.items():
        module.register_parameter(name, nn.Parameter(torch.empty(*shape, dtype=dtype)))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=dtype),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_frequencies(t, self.freq_dim).to(self.mlp[0].weight.dtype))


class PatchEmbed(nn.Module):
    """``UNetDown`` (patch size 1): latents -> this rank's partial sum of the image tokens.

    conv 32->1024 (replicated), then the ResBlock 1024->4096 with its convolutions
    sharded over TP: the first conv by output channels (the following GroupNorm's
    groups of 128 channels stay whole on a rank), the second by input channels. The
    ranks' outputs sum to the tokens; bias and skip connection are added on rank 0.
    """

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, dtype: torch.dtype):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.local = out_ch // self.tp_size
        if self.local % (out_ch // 32):
            raise ValueError(f"patch_embed's GroupNorm does not shard over TP={self.tp_size}")
        self.groups = self.local // (out_ch // 32)
        local = self.local
        _params(
            self,
            {
                "conv_in_weight": (hidden_ch, in_ch, 3, 3),
                "conv_in_bias": (hidden_ch,),
                "in_norm_weight": (hidden_ch,),
                "in_norm_bias": (hidden_ch,),
                "in_conv_weight": (local, hidden_ch, 3, 3),
                "in_conv_bias": (local,),
                "emb_weight": (2 * local, out_ch),
                "emb_bias": (2 * local,),
                "out_norm_weight": (local,),
                "out_norm_bias": (local,),
                "out_conv_weight": (out_ch, local, 3, 3),
                "out_conv_bias": (out_ch,),
                "skip_weight": (out_ch, hidden_ch, 1, 1),
                "skip_bias": (out_ch,),
            },
            dtype,
        )

    def load(self, ckpt: Checkpoint, prefix: str) -> dict[str, torch.Tensor]:
        lo, hi = self.tp_rank * self.local, (self.tp_rank + 1) * self.local
        full = self.out_conv_weight.shape[0]
        r = f"{prefix}model.1."
        emb_w, emb_b = ckpt.get(f"{r}emb_layers.1.weight"), ckpt.get(f"{r}emb_layers.1.bias")
        return {
            "conv_in_weight": ckpt.get(f"{prefix}model.0.weight"),
            "conv_in_bias": ckpt.get(f"{prefix}model.0.bias"),
            "in_norm_weight": ckpt.get(f"{r}in_layers.0.weight"),
            "in_norm_bias": ckpt.get(f"{r}in_layers.0.bias"),
            "in_conv_weight": ckpt.rows(f"{r}in_layers.2.weight", lo, hi),
            "in_conv_bias": ckpt.rows(f"{r}in_layers.2.bias", lo, hi),
            # [scale | shift] rows of the local channels.
            "emb_weight": torch.cat([emb_w[lo:hi], emb_w[full + lo : full + hi]]),
            "emb_bias": torch.cat([emb_b[lo:hi], emb_b[full + lo : full + hi]]),
            "out_norm_weight": ckpt.rows(f"{r}out_layers.0.weight", lo, hi),
            "out_norm_bias": ckpt.rows(f"{r}out_layers.0.bias", lo, hi),
            "out_conv_weight": ckpt.cols(f"{r}out_layers.3.weight", lo, hi),
            "out_conv_bias": ckpt.get(f"{r}out_layers.3.bias"),
            "skip_weight": ckpt.get(f"{r}skip_connection.weight"),
            "skip_bias": ckpt.get(f"{r}skip_connection.bias"),
        }

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """``x`` ``[B, C, h, w]``, ``emb`` ``[B, E]`` -> partial tokens ``[B, h*w, H]``."""
        x = _conv(x, self.conv_in_weight, self.conv_in_bias)
        h = _group_norm(x, 32, self.in_norm_weight, self.in_norm_bias)
        h = _conv(F.silu(h), self.in_conv_weight, self.in_conv_bias)
        emb = F.linear(F.silu(emb), self.emb_weight, self.emb_bias)[..., None, None]
        scale, shift = emb.chunk(2, dim=1)
        h = _group_norm(h, self.groups, self.out_norm_weight, self.out_norm_bias)
        h = _conv(F.silu(h * (1 + scale) + shift), self.out_conv_weight)
        if self.tp_rank == 0:
            h = h + self.out_conv_bias[:, None, None] + _conv(x, self.skip_weight, self.skip_bias)
        return h.flatten(2).transpose(1, 2)


class FinalLayer(nn.Module):
    """``UNetUp`` (patch size 1, output norm): tokens ``[B, h*w, H]`` -> velocity ``[B, C, h, w]``.

    ResBlock 4096->1024 then GroupNorm, SiLU, conv 1024->32, sharded over TP: the
    first and last convs by input channels (all-reduced), the ResBlock's second conv
    and its skip connection by output channels (so the output GroupNorm's groups of 32
    channels stay whole on a rank).
    """

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, dtype: torch.dtype):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group().device_group
        self.in_local = in_ch // self.tp_size
        self.local = hidden_ch // self.tp_size
        if self.local % (hidden_ch // 32):
            raise ValueError(f"final_layer's GroupNorm does not shard over TP={self.tp_size}")
        self.groups = self.local // (hidden_ch // 32)
        _params(
            self,
            {
                "in_norm_weight": (in_ch,),
                "in_norm_bias": (in_ch,),
                "in_conv_weight": (hidden_ch, self.in_local, 3, 3),
                "in_conv_bias": (hidden_ch,),
                "emb_weight": (2 * hidden_ch, in_ch),
                "emb_bias": (2 * hidden_ch,),
                "mid_norm_weight": (hidden_ch,),
                "mid_norm_bias": (hidden_ch,),
                "mid_conv_weight": (self.local, hidden_ch, 3, 3),
                "mid_conv_bias": (self.local,),
                "skip_weight": (self.local, in_ch, 1, 1),
                "skip_bias": (self.local,),
                "out_norm_weight": (self.local,),
                "out_norm_bias": (self.local,),
                "out_conv_weight": (out_ch, self.local, 3, 3),
                "out_conv_bias": (out_ch,),
            },
            dtype,
        )

    def load(self, ckpt: Checkpoint, prefix: str) -> dict[str, torch.Tensor]:
        ilo, ihi = self.tp_rank * self.in_local, (self.tp_rank + 1) * self.in_local
        lo, hi = self.tp_rank * self.local, (self.tp_rank + 1) * self.local
        r = f"{prefix}model.0."
        return {
            "in_norm_weight": ckpt.get(f"{r}in_layers.0.weight"),
            "in_norm_bias": ckpt.get(f"{r}in_layers.0.bias"),
            "in_conv_weight": ckpt.cols(f"{r}in_layers.2.weight", ilo, ihi),
            "in_conv_bias": ckpt.get(f"{r}in_layers.2.bias"),
            "emb_weight": ckpt.get(f"{r}emb_layers.1.weight"),
            "emb_bias": ckpt.get(f"{r}emb_layers.1.bias"),
            "mid_norm_weight": ckpt.get(f"{r}out_layers.0.weight"),
            "mid_norm_bias": ckpt.get(f"{r}out_layers.0.bias"),
            "mid_conv_weight": ckpt.rows(f"{r}out_layers.3.weight", lo, hi),
            "mid_conv_bias": ckpt.rows(f"{r}out_layers.3.bias", lo, hi),
            "skip_weight": ckpt.rows(f"{r}skip_connection.weight", lo, hi),
            "skip_bias": ckpt.rows(f"{r}skip_connection.bias", lo, hi),
            "out_norm_weight": ckpt.rows(f"{prefix}model.1.0.weight", lo, hi),
            "out_norm_bias": ckpt.rows(f"{prefix}model.1.0.bias", lo, hi),
            "out_conv_weight": ckpt.cols(f"{prefix}model.1.2.weight", lo, hi),
            "out_conv_bias": ckpt.get(f"{prefix}model.1.2.bias"),
        }

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tp_group)
        return x

    def forward(self, x: torch.Tensor, emb: torch.Tensor, height: int, width: int):
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
        h = F.silu(_group_norm(x, 32, self.in_norm_weight, self.in_norm_bias))
        h = h[:, self.tp_rank * self.in_local : (self.tp_rank + 1) * self.in_local]
        h = self._all_reduce(_conv(h, self.in_conv_weight)) + self.in_conv_bias[:, None, None]
        emb = F.linear(F.silu(emb), self.emb_weight, self.emb_bias)[..., None, None]
        scale, shift = emb.chunk(2, dim=1)
        h = _group_norm(h, 32, self.mid_norm_weight, self.mid_norm_bias) * (1 + scale) + shift
        h = _conv(F.silu(h), self.mid_conv_weight, self.mid_conv_bias)
        h = h + _conv(x, self.skip_weight, self.skip_bias)
        h = F.silu(_group_norm(h, self.groups, self.out_norm_weight, self.out_norm_bias))
        return self._all_reduce(_conv(h, self.out_conv_weight)) + self.out_conv_bias[:, None, None]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------


class NeuronHunyuanImage3Model(nn.Module):
    """Language model and diffusion heads of HunyuanImage-3.0 (see the module docstring).

    The token embedding table stays on the host; the caller embeds the prefix tokens.
    """

    def __init__(
        self,
        config,
        dtype: torch.dtype = torch.bfloat16,
        num_layers: int | None = None,
        shared_sequence_parallel: bool = True,
        with_lm_head: bool = False,
    ):
        super().__init__()
        self.config = config
        self._dtype = dtype
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group().device_group
        self.hidden_size = config.hidden_size
        self.head_dim = config.attention_head_dim
        self.rope_theta = config.rope_theta
        self.num_layers = num_layers or config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                HunyuanImage3DecoderLayer(config, dtype, shared_sequence_parallel)
                for _ in range(self.num_layers)
            ]
        )
        latent = config.vae["latent_channels"]
        self.latent_channels = latent
        self.patch_embed = PatchEmbed(latent, 1024, config.hidden_size, dtype)
        self.final_layer = FinalLayer(config.hidden_size, 1024, latent, dtype)
        self.timestep_emb = TimestepEmbedder(config.hidden_size, dtype)
        self.time_embed = TimestepEmbedder(config.hidden_size, dtype)
        self.time_embed_2 = TimestepEmbedder(config.hidden_size, dtype)
        self._embed_tokens: torch.Tensor | None = None
        # The LM head for autoregressive text (think / recaption), vocab-sharded.
        self.with_lm_head = with_lm_head
        if with_lm_head:
            if config.vocab_size % self.tp_size:
                raise ValueError(f"the vocabulary does not shard over TP={self.tp_size}")
            self.ln_f = HunyuanRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
            self.lm_head_weight = nn.Parameter(
                torch.empty(config.vocab_size // self.tp_size, config.hidden_size, dtype=dtype)
            )
            # The vocab shard of the token embedding, for decoding without the host.
            self.wte_local = nn.Parameter(
                torch.empty(config.vocab_size // self.tp_size, config.hidden_size, dtype=dtype)
            )
        self._compiled_prefix = None
        self._compiled_step = None
        # CFG parallelism: replicas (one TP group each) run the conditional and the
        # unconditional row.
        self.cfg_size, self.cfg_rank, self.cfg_group = 1, 0, None
        try:
            cfg = get_cfg_group()
        except AssertionError:
            cfg = None
        if cfg is not None and cfg.world_size > 1:
            if cfg.world_size != 2:
                raise ValueError("CFG parallelism supports two replicas (cond, uncond)")
            self.cfg_size, self.cfg_rank, self.cfg_group = 2, cfg.rank_in_group, cfg.device_group
            self.world_group = get_world_group().device_group

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # -- weights -------------------------------------------------------

    def load_weights(self, model_path: str, expert_load=None) -> None:
        """Loads this rank's shards; ``expert_load`` (``[layer][expert]`` token shares)
        balances the experts over the ranks."""
        ckpt = Checkpoint(model_path)
        state: dict[str, torch.Tensor] = {}
        num_kv = self.config.num_key_value_heads
        for i, layer in enumerate(self.layers):
            src, dst = f"model.layers.{i}.", f"layers.{i}."
            for name in ("input_layernorm", "post_attention_layernorm"):
                state[f"{dst}{name}.weight"] = ckpt.get(f"{src}{name}.weight")
            for k, v in layer.self_attn.load(ckpt, f"{src}self_attn.", num_kv).items():
                state[f"{dst}self_attn.{k}"] = v
            order = None
            if expert_load is not None:
                order = expert_placement(
                    expert_load[i], layer.mlp.num_experts // layer.mlp.num_local_experts
                )
            for k, v in layer.mlp.load(ckpt, f"{src}mlp.", order).items():
                state[f"{dst}mlp.{k}"] = v
        for head in ("patch_embed", "final_layer"):
            for k, v in getattr(self, head).load(ckpt, f"{head}.").items():
                state[f"{head}.{k}"] = v
        for name, _ in self.named_parameters():
            if name.startswith("time"):
                state[name] = ckpt.get(name)
        if self.with_lm_head:
            state["ln_f.weight"] = ckpt.get("model.ln_f.weight")
            rows = self.lm_head_weight.shape[0]
            state["lm_head_weight"] = ckpt.rows(
                "lm_head.weight", self.tp_rank * rows, (self.tp_rank + 1) * rows
            )
            state["wte_local"] = ckpt.rows(
                "model.wte.weight", self.tp_rank * rows, (self.tp_rank + 1) * rows
            )
        state = {
            k: v.to(torch.float32 if k.endswith("router_weight") else self._dtype).contiguous()
            for k, v in state.items()
        }
        self.load_state_dict(state, strict=True, assign=True)
        self._embed_tokens = ckpt.get("model.wte.weight").to(self._dtype)
        logger.info(
            "Loaded HunyuanImage-3.0 transformer: %d layers (rank=%d, tp=%d)",
            self.num_layers,
            self.tp_rank,
            self.tp_size,
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Host token embedding ``[R, P]`` -> ``[R, P, H]``."""
        return F.embedding(input_ids.cpu(), self._embed_tokens)

    # -- graphs --------------------------------------------------------

    def _shard(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's sequence-parallel slice of ``x`` ``[T, ...]``."""
        if self.tp_size == 1:
            return x
        local = x.shape[0] // self.tp_size
        return x[self.tp_rank * local : (self.tp_rank + 1) * local]

    def _mlp(self, layer, x: torch.Tensor) -> torch.Tensor:
        h = layer.post_attention_layernorm(x)
        routed = layer.mlp(_all_gather(h, self.tp_group, self.tp_size))
        if not layer.mlp.shared_sp:
            routed = routed + layer.mlp.shared_expert(_all_gather(h, self.tp_group, self.tp_size))
            return x + _reduce_scatter(routed, self.tp_group, self.tp_size)
        return x + _reduce_scatter(routed, self.tp_group, self.tp_size) + layer.mlp.shared_expert(h)

    def forward_prefix(self, embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Text K/V of every layer for ``R`` right-padded rows.

        ``embeds`` ``[R, P, H]``, ``cos``/``sin`` ``[P, D]`` -> ``k``, ``v``
        ``[layers, R, kv_heads, P, D]``. The attention is causal, so the padding after a
        row's valid tokens never reaches them.
        """
        rows, length, hidden = embeds.shape
        x = self._shard(embeds.reshape(rows * length, hidden))
        return self._prefix_layers(x, rows, length, cos, sin)

    def forward_prefix_edit(self, embeds, cos, sin, bound_max, t0, latents, vit_tokens, layout):
        """Prefix K/V of an edit (text and condition images), for ``R`` rows.

        ``embeds`` ``[R, P, H]`` holds the text embeddings (zeros at the image slots);
        each condition image ``k`` contributes its latents ``[1, C, h_k, w_k]`` (patch
        embedded at timestep ``t0`` = 0), its SigLIP2 tokens ``[1024, H]`` and a
        ``<timestep>`` token, at the static positions ``layout[k] = (timestep, vae, vit)``.
        ``bound_max`` ``[P_pad, 1]`` gives each query's key range (causal, full within a
        condition image's joint span).
        """
        x = self._edit_embeddings(embeds, t0, latents, vit_tokens, layout)
        rows, length, _ = embeds.shape
        return self._prefix_layers(x, rows, length, cos, sin, bound_max)

    def _edit_embeddings(self, embeds, t0, latents, vit_tokens, layout):
        """The SP-sharded prefix embeddings: text, with the condition images' slots
        filled (see :meth:`forward_prefix_edit`)."""
        rows = embeds.shape[0]
        t0 = t0.expand(1)
        temb = self.time_embed(t0)
        timestep = self.timestep_emb(t0)  # [1, H]
        # Rank 0 holds the text, SigLIP2 and <timestep> embeddings; every rank its
        # partial sum of the patch embedding; the reduce-scatter adds them up. The image
        # slots are written with two index copies.
        lead = self.tp_rank == 0
        lead_index, lead_values, image_index, image_values = [], [], [], []
        for (ts_pos, vae_pos, vit_pos), lat, vit in zip(layout, latents, vit_tokens):
            image = self.patch_embed(lat.to(self._dtype), temb)[0]  # [h*w, H]
            image_index.append(torch.arange(vae_pos, vae_pos + image.shape[0], device=image.device))
            image_values.append(image)
            lead_index += [
                torch.arange(ts_pos, ts_pos + 1, device=image.device),
                torch.arange(vit_pos, vit_pos + vit.shape[0], device=image.device),
            ]
            lead_values += [timestep.to(self._dtype), vit.to(self._dtype)]
        out_rows = []
        for r in range(rows):
            row = embeds[r] if lead else torch.zeros_like(embeds[r])
            if image_values:
                row = row.index_copy(0, torch.cat(image_index), torch.cat(image_values))
                if lead:
                    row = row.index_copy(0, torch.cat(lead_index), torch.cat(lead_values))
            out_rows.append(row)
        return _reduce_scatter(torch.cat(out_rows), self.tp_group, self.tp_size)

    def _prefix_layers(self, x, rows, length, cos, sin, bound_max=None, keep_output=False):
        """Every layer's K/V ``[layers, R, kv_heads, P, D]`` of the SP-sharded prefix ``x``.

        Causal attention, or with ``bound_max`` each query's keys ``[0, bound_max)``.
        With ``keep_output`` the last layer runs too and its (SP-sharded) output is
        also returned.
        """
        cos, sin = cos.repeat(rows, 1), sin.repeat(rows, 1)
        if bound_max is not None:
            bound_min = torch.zeros_like(bound_max)
        keys, values = [], []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = _all_gather(layer.input_layernorm(x), self.tp_group, self.tp_size)
            q, k, v = attn.project(h, cos, sin)
            keys.append(k.view(attn.num_kv_heads, rows, length, -1).transpose(0, 1))
            values.append(v.view(attn.num_kv_heads, rows, length, -1).transpose(0, 1))
            if i == len(self.layers) - 1 and not keep_output:
                break  # the last layer's output is not needed
            out = []
            for r in range(rows):
                row = slice(r * length, (r + 1) * length)
                if bound_max is None:
                    out.append(attn.attend_causal(q[:, row], k[:, row], v[:, row]))
                else:
                    out.append(
                        attn.attend_bounded(q[:, row], k[:, row], v[:, row], bound_min, bound_max)
                    )
            out = torch.cat(out)
            x = x + _reduce_scatter(torch.matmul(out, attn.o_weight), self.tp_group, self.tp_size)
            x = self._mlp(layer, x)
        if keep_output:
            return torch.stack(keys), torch.stack(values), x
        return torch.stack(keys), torch.stack(values)

    # -- autoregressive text (think / recaption) -------------------------

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        """Final norm and the vocab-sharded LM head: ``h`` ``[1, H]`` -> logits ``[V]`` (fp32)."""
        h = self.ln_f(h)
        local = torch.matmul(h, self.lm_head_weight.t()).float()  # [1, V / TP]
        logits = _all_gather(local.t().contiguous(), self.tp_group, self.tp_size).view(-1)
        if self.cfg_size > 1:
            # The CFG replicas run the same text; averaging their logits makes them
            # bit-identical on every rank, so they sample the same tokens.
            logits = _all_reduce(logits, self.world_group, self.cfg_size * self.tp_size)
            logits = logits / (self.cfg_size * self.tp_size)
        return logits

    def forward_ar_prefill(
        self, embeds, cos, sin, bound_max, t0, latents, vit_tokens, layout, last, cache_len
    ):
        """One row's prefix K/V padded to ``cache_len`` (``[layers, kv_heads, S, D]``) and
        the logits of the token at ``last`` (``[1]`` int64)."""
        x = self._edit_embeddings(embeds, t0, latents, vit_tokens, layout)
        length = embeds.shape[1]
        k, v, x = self._prefix_layers(x, 1, length, cos, sin, bound_max, keep_output=True)
        pad = (0, 0, 0, cache_len - length)
        k, v = F.pad(k[:, 0], pad), F.pad(v[:, 0], pad)
        h = _all_gather(x, self.tp_group, self.tp_size).index_select(0, last)
        return k, v, self._logits(h)

    def forward_decode(self, embed, cos_table, sin_table, k_cache, v_cache, position):
        """One token: ``embed`` ``[1, H]`` at ``position`` (``[1]`` int64, device); RoPE from
        the ``[S, D]`` tables. Returns its logits ``[V]``, the caches with its K/V written
        and ``position + 1``, so that consecutive steps need only the embedding from the host."""
        x, new_k, new_v = self._decode_layers(
            embed, cos_table, sin_table, k_cache, v_cache, position
        )
        return self._logits(x), new_k, new_v, position + 1

    def forward_decode_sample(
        self, token, position, step, k_cache, v_cache, cos_table, sin_table, tokens, ranks, sampling
    ):
        """One token fully on device: embed ``token`` (``[1]`` int32), run the layers and
        sample the next token (``NF.sample``: top-k from the vocab shards, temperature,
        top-p). ``sampling`` = (temperature, top_k, top_p, forced_after, forced) is
        static; when ``token`` is ``forced_after`` the next token is ``forced``. ``ranks``
        = (TP rank int32, 1.0 on global rank 0 else 0.0) as tensors, so that every rank
        compiles the same graph. The next token is written to ``tokens`` at ``step``; all
        outputs feed the next call, so steps queue up without a host round trip."""
        temperature, top_k, top_p, forced_after, forced = sampling
        rank, lead = ranks
        embed = self._embed_tokens_device(token, rank)
        x, new_k, new_v = self._decode_layers(
            embed, cos_table, sin_table, k_cache, v_cache, position
        )
        local = torch.matmul(self.ln_f(x), self.lm_head_weight.t()).float()  # [1, V / TP]
        nxt = NF.sample(
            local,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tp_group=self.tp_group,
            max_top_k=top_k,
            tp_rank=rank,
        ).view(1)
        # Every rank continues with global rank 0's token (each rank draws its own random
        # number). Token ids are exact in float32.
        group = self.world_group if self.cfg_size > 1 else self.tp_group
        own = _all_reduce(nxt.float() * lead, group, self.cfg_size * self.tp_size)
        nxt = own.round().to(nxt.dtype)
        if forced_after >= 0:
            nxt = torch.where(token == forced_after, torch.full_like(nxt, forced), nxt)
        tokens = tokens.index_copy(0, step, nxt)
        return nxt, position + 1, step + 1, new_k, new_v, tokens

    def _embed_tokens_device(self, token: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        """Token embedding from the vocab-sharded device table: ``[1]`` -> ``[1, H]``."""
        rows = self.wte_local.shape[0]
        local = token.long() - rank.long() * rows
        inside = (local >= 0) & (local < rows)
        embed = self.wte_local.index_select(0, local.clamp(0, rows - 1)) * inside[:, None]
        return _all_reduce(embed, self.tp_group, self.tp_size)

    def _decode_layers(self, embed, cos_table, sin_table, k_cache, v_cache, position):
        x = embed
        cos, sin = cos_table.index_select(0, position), sin_table.index_select(0, position)
        length = k_cache.shape[2]
        visible = (torch.arange(length, device=x.device) <= position)[None, None, :]
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            q, k, v = attn.project(layer.input_layernorm(x), cos, sin, fused=False)
            keys = k_cache[i].index_copy(1, position, k)
            vals = v_cache[i].index_copy(1, position, v)
            new_k.append(keys)
            new_v.append(vals)
            scores = torch.matmul(q, attn.expand_kv(keys).transpose(1, 2)).float() * attn.scale
            probs = scores.masked_fill(~visible, float("-inf")).softmax(dim=-1)
            out = torch.matmul(probs.to(x.dtype), attn.expand_kv(vals))  # [heads, 1, D]
            out = torch.matmul(out.transpose(0, 1).reshape(1, -1), attn.o_weight)
            x = x + _all_reduce(out, self.tp_group, self.tp_size)
            h = layer.post_attention_layernorm(x)
            if self.cfg_size > 1 and not layer.mlp.shared_sp:
                # The CFG replicas hold the same weights: each computes half of the
                # experts (and of the shared expert), summed over all ranks, so a token
                # reads half of the expert weights per rank (as TP = 2 x 16 would).
                part = (self.cfg_rank, self.cfg_size)
                moe = layer.mlp.decode(h, *part) + layer.mlp.shared_expert_part(h, *part)
                x = x + _all_reduce(moe, self.world_group, self.cfg_size * self.tp_size)
                continue
            routed = layer.mlp.decode(h)
            if layer.mlp.shared_sp:
                x = x + _all_reduce(routed, self.tp_group, self.tp_size)
                x = x + layer.mlp.shared_expert(h)
            else:
                x = x + _all_reduce(
                    routed + layer.mlp.shared_expert(h), self.tp_group, self.tp_size
                )
        return x, torch.stack(new_k), torch.stack(new_v)

    def _velocity(self, latents, t, text_k, text_v, cos, sin, bound_min, bound_max):
        """Velocity ``[R, C, h, w]`` of ``latents`` ``[1, C, h, w]`` for each of the ``R`` rows.

        A row is ``[N image tokens, <timestep>, padding]`` (see :func:`step_row_len`); its keys are
        ``[image, <timestep>, text]`` (the padding's keys are dropped).
        """
        rows = text_k.shape[1]
        _, _, height, width = latents.shape
        num_image = height * width
        row_len = step_row_len(num_image, rows, self.tp_size)
        dtype = self._dtype
        image = latents.to(dtype).expand(rows, -1, -1, -1)
        t_rows = t.expand(rows)
        # Each TP rank holds a partial sum of the image tokens (rank 0 also the
        # <timestep> token); the reduce-scatter sums it into the SP slices.
        tokens = self.patch_embed(image, self.time_embed(t_rows))  # [R, N, H]
        timestep = self.timestep_emb(t_rows)[:, None]  # [R, 1, H]
        if self.tp_rank != 0:
            timestep = torch.zeros_like(timestep)
        pad = tokens.new_zeros(rows, row_len - num_image - 1, self.hidden_size)
        x = torch.cat([tokens, timestep, pad], dim=1).reshape(rows * row_len, self.hidden_size)
        x = _reduce_scatter(x, self.tp_group, self.tp_size)
        cos, sin = cos.repeat(rows, 1), sin.repeat(rows, 1)

        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = _all_gather(layer.input_layernorm(x), self.tp_group, self.tp_size)
            q, k, v = attn.project(h, cos, sin)
            out = []
            for r in range(rows):
                row = slice(r * row_len, r * row_len + row_len)
                keep = slice(r * row_len, r * row_len + num_image + 1)
                keys = torch.cat([k[:, keep], text_k[i, r]], dim=1)
                vals = torch.cat([v[:, keep], text_v[i, r]], dim=1)
                out.append(attn.attend_bounded(q[:, row], keys, vals, bound_min, bound_max))
            out = torch.cat(out)
            x = x + _reduce_scatter(torch.matmul(out, attn.o_weight), self.tp_group, self.tp_size)
            x = self._mlp(layer, x)

        x = _all_gather(x, self.tp_group, self.tp_size).view(rows, row_len, self.hidden_size)
        return self.final_layer(x[:, :num_image], self.time_embed_2(t_rows), height, width)

    def forward_step(
        self, latents, t, dt, text_k, text_v, cos, sin, bound_min, bound_max, cfg_scale, guide
    ):
        """One Euler step of ``latents`` ``[1, C, h, w]`` (float32).

        With ``guide`` the velocity is ``uncond + cfg_scale * (cond - uncond)`` over the
        conditional and unconditional rows: both rows of this graph, or under CFG
        parallelism this replica's row and the other replica's (all-gathered).
        """
        velocity = self._velocity(latents, t, text_k, text_v, cos, sin, bound_min, bound_max)
        velocity = velocity.float()
        if self.cfg_size > 1:
            rows = torch.empty(
                (velocity.shape[0] * self.cfg_size, *velocity.shape[1:]),
                dtype=velocity.dtype,
                device=velocity.device,
            )
            dist.all_gather_into_tensor(rows, velocity.contiguous(), group=self.cfg_group)
            velocity = rows
        if guide:
            cond, uncond = velocity.chunk(2)
            velocity = uncond + cfg_scale * (cond - uncond)
        else:
            velocity = velocity[:1]
        return latents + velocity * dt

    def compile(self, *args, prefix_options=None, step_options=None, **kwargs):
        self._compiled_prefix = torch.compile(
            self.forward_prefix, *args, options=prefix_options, **kwargs
        )
        edit_options = {**(prefix_options or {})}
        edit_options["model_name"] = f"{edit_options.get('model_name', 'prefix')}_edit"
        self._compiled_prefix_edit = torch.compile(
            self.forward_prefix_edit, *args, options=edit_options, **kwargs
        )
        if self.with_lm_head:
            ar_options = {**(prefix_options or {})}
            name = ar_options.get("model_name", "prefix")
            self._compiled_ar_prefill = torch.compile(
                self.forward_ar_prefill,
                *args,
                options={**ar_options, "model_name": f"{name}_ar"},
                **kwargs,
            )
            self._compiled_decode = torch.compile(
                self.forward_decode,
                *args,
                options={**ar_options, "model_name": f"{name}_decode"},
                **kwargs,
            )
            self._compiled_decode_sample = torch.compile(
                self.forward_decode_sample,
                *args,
                options={**ar_options, "model_name": f"{name}_decode_sample"},
                **kwargs,
            )
        self._compiled_step = torch.compile(
            self.forward_step, *args, options=step_options, **kwargs
        )

    def prefix(self, *args):
        fn = self._compiled_prefix if self._compiled_prefix is not None else self.forward_prefix
        return fn(*args)

    def ar_prefill(self, *args):
        fn = getattr(self, "_compiled_ar_prefill", None) or self.forward_ar_prefill
        return fn(*args)

    def decode_sample(self, *args):
        fn = getattr(self, "_compiled_decode_sample", None) or self.forward_decode_sample
        return fn(*args)

    def decode(self, *args):
        fn = getattr(self, "_compiled_decode", None) or self.forward_decode
        return fn(*args)

    def prefix_edit(self, *args):
        fn = getattr(self, "_compiled_prefix_edit", None) or self.forward_prefix_edit
        return fn(*args)

    def step(self, *args):
        fn = self._compiled_step if self._compiled_step is not None else self.forward_step
        return fn(*args)


def step_row_len(num_image: int, rows: int, tp_size: int) -> int:
    """Tokens per step row: the image tokens, ``<timestep>`` and the fewest padding
    tokens for the rows to split over the TP ranks into even slices (the NKI MLP
    kernel of the sequence-parallel shared expert needs an even token count)."""
    length = num_image + 1
    while (rows * length) % (2 * tp_size):
        length += 1
    return length


def step_layout(
    prefix_len: int,
    height: int,
    width: int,
    head_dim: int,
    theta: float,
    rows: int = 2,
    tp_size: int = 1,
):
    """Host inputs of a step of ``rows`` rows for a prefix of ``prefix_len`` valid tokens.

    Returns ``cos``/``sin`` ``[step_row_len, D]`` for one row (image tokens, then the
    timestep token and its padding) and the query bounds ``[S_pad, 1]`` over the
    keys ``[image (N), <timestep>, text]``: image queries see the image,
    the timestep token and the valid text; the timestep token (and the padding) only
    the timestep token and the text.
    """
    num_image = height * width
    # Token ``prefix_len`` is <timestep>; the image starts right after it.
    img_cos, img_sin = image_rope(prefix_len + 1, height, width, head_dim, theta)
    ts_cos, ts_sin = text_rope(1, head_dim, theta, start=prefix_len)
    row_len = step_row_len(num_image, rows, tp_size)
    cos = torch.cat([img_cos, ts_cos.expand(row_len - num_image, -1)])
    sin = torch.cat([img_sin, ts_sin.expand(row_len - num_image, -1)])
    padded = -(-row_len // _BOUNDS_TILE) * _BOUNDS_TILE
    bound_min = torch.zeros(padded, 1, dtype=torch.int32)
    bound_min[num_image:] = num_image
    # The text keys beyond prefix_len are padding.
    bound_max = torch.full((padded, 1), num_image + 1 + prefix_len, dtype=torch.int32)
    return cos, sin, bound_min, bound_max
