# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image DiT for Neuron.

Neuron re-implementation of diffusers' ``MiniMaxH3ImageTransformer2DModel`` (diffusers
0.42, ``transformer_minimax_h3_image.py``) following the vLLM-Neuron LLaMA pattern used by
the Wan2.2 DiT: raw ``nn.Parameter`` tensors with TP-sharded weight loaders.

Neuron-specific restructuring (numerically equivalent to the reference):

* **Packed layout.** The reference scatters text and image rows into one packed sequence
  ordered ``[text | references | generated image]``. Unmasked self-attention is
  permutation-equivariant (every row carries its own rotary position, modality tag and
  timestep), so the physical order is free. This model packs ``[image rows | text rows |
  text padding]``: the image rows sit at fixed positions (no scatter / gather), and the
  text is padded to a bucket so one compiled graph serves many prompt lengths. Padded keys
  are excluded with the attention kernel's runtime ``kv_used_len`` bound.
* **AdaLN tables on the host.** Every block's ``adaln_proj`` (2688 -> 96768, ~40% of the
  checkpoint's parameters) is a function of the timestep only. The checkpoint samples a
  fixed six-step grid, so the per-(timestep, modality) modulation rows are computed once on
  the CPU (exactly as the reference does, in bfloat16) and passed to the compiled graph;
  the projection weights never reach the device. Likewise the timestep embedder, the
  output-norm modulation and the per-step PDD output heads.
* **Text stream once per request.** ``context_embedder`` + ``token_refiner`` depend on the
  prompt only, so they run in their own graph once per request instead of once per step.

Parallelism: TP over heads / FFN columns (all-reduce after the output and down
projections) and CP over the packed sequence (K/V all-gathered across the CP group).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from functools import cache

import nki
import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    get_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group as get_cp_group

from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups
from vllm_omni_neuron.lite_compat import nki_op, wrap_nki_kernel

logger = logging.getLogger(__name__)

# Per-row modality tags of the checkpoint's AdaLN table: 0 image (video), 1 text, 2 audio.
MODALITY_NUM = 3
IMAGE_TAG = 0
TEXT_TAG = 1
# The `t` the reference rows of an edit are held at for every step.
REFERENCE_NOISE_AUG = 0.999

# Rows of the per-block modulation table passed to the step graph.
TAB_TARGET = 0  # generated image rows at the step's timestep
TAB_TEXT = 1  # text rows at the step's timestep
TAB_REFERENCE = 2  # reference image rows at REFERENCE_NOISE_AUG
NUM_TABS = 3


@dataclass
class MiniMaxH3ImageConfig:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    num_pdd_heads: int = 6

    @property
    def patch_dim(self) -> int:
        return self.in_channels * math.prod(self.patch_size)

    @property
    def rotary_dim(self) -> int:
        return 2 * 3 * self.rope_freq_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> MiniMaxH3ImageConfig:
        names = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in cfg.items() if k in names}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return cls(**kwargs)


# ===================================================================
# Kernel gates and attention
# ===================================================================


def _resolve_can_run_kernel():
    try:
        from vllm_neuron.utils.neuron_utils import can_run_kernel
    except ModuleNotFoundError as error:
        if error.name != "vllm_neuron.utils.neuron_utils":
            raise
        from vllm_neuron.nki.nki_hop import can_run_kernel
    return can_run_kernel


@cache
def _can_run_kernel_impl():
    return _resolve_can_run_kernel()


def can_run_kernel(tensor: torch.Tensor) -> bool:
    return _can_run_kernel_impl()(tensor)


# The checkpoint's head dim is fixed at 128; the softmax scale is baked into the kernel so
# that q is not pre-rounded to bf16 after scaling.
_ATTN_SCALE = 1.0 / math.sqrt(128)


@nki.jit
def _h3_attention_kernel(q, k, v, kv_len):
    """Non-causal flash attention over ``[BN, S, D]`` q/k/v; keys ``>= kv_len`` are masked."""
    from nkilib.core.attention.attention_cte import _attention_cte

    return _attention_cte(
        q=q,
        k=k,
        v=v,
        scale=_ATTN_SCALE,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
        kv_used_len=kv_len,
    )


@nki_op("minimax_h3_image::attention_kvlen")
def _h3_nki_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_len: torch.Tensor
) -> torch.Tensor:
    return wrap_nki_kernel(_h3_attention_kernel, supported_lnc=(2,))(q, k, v, kv_len)


def _torch_attention(q, k, v, kv_len):
    """Fallback: ``[BN, Sq, D]`` x ``[BN, Sk, D]`` with keys ``>= kv_len`` masked."""
    scores = torch.matmul(q.float() * _ATTN_SCALE, k.float().transpose(-2, -1))
    key_pos = torch.arange(k.shape[1], device=k.device)
    mask = key_pos[None, None, :] < kv_len.reshape(1, 1, 1).to(key_pos.dtype)
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def h3_attention(q, k, v, kv_len):
    """``q`` ``[1, Sq, N, D]``, ``k``/``v`` ``[1, Sk, N, D]`` -> ``[1, Sq, N*D]``."""
    _, sq, n, d = q.shape
    if d != 128:
        raise ValueError(f"MiniMax-H3-Image attention expects head_dim 128, got {d}")
    sk = k.shape[1]
    q3 = q.transpose(1, 2).reshape(n, sq, d).contiguous()
    k3 = k.transpose(1, 2).reshape(n, sk, d).contiguous()
    v3 = v.transpose(1, 2).reshape(n, sk, d).contiguous()
    if can_run_kernel(v3):
        out = _h3_nki_attention(q3, k3, v3, kv_len)
    else:
        out = _torch_attention(q3, k3, v3, kv_len)
    return out.reshape(1, n, sq, d).transpose(1, 2).reshape(1, sq, n * d)


# ===================================================================
# Small math helpers (reference numerics)
# ===================================================================


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``torch.nn.RMSNorm`` numerics: fp32 compute including the weight, cast back."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``rotary_dim`` channels of every head (rotate-half convention).

    ``x`` ``[1, S, N, D]``; ``cos``/``sin`` ``[S, rotary_dim]`` already in ``x.dtype``.
    """
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rot = x_rot * cos + rotated * sin
    return torch.cat((x_rot, x_pass), dim=-1)


# ===================================================================
# Weight loaders
# ===================================================================


def _col_weight_loader(shard_size, num_shards):
    return sharding_weight_loader(
        shard_dim=1, shard_size=shard_size, num_shards=num_shards, is_storage_transposed=True
    )


def _row_weight_loader(shard_size, num_shards):
    return sharding_weight_loader(
        shard_dim=0, shard_size=shard_size, num_shards=num_shards, is_storage_transposed=True
    )


def _transposed_loader():
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].T)


def _swiglu_up_gate_loader(ffn_dim: int, shard_size: int):
    """``ff.net.0.proj.weight`` is ``[value; gate]`` stacked on dim 0 (diffusers SwiGLU).

    Each rank takes its value and gate column shards and stores ``[H, 2 * shard]``
    (value columns first, then gate columns).
    """

    def transform(slices, rank):
        sl = slices[0]
        start = rank * shard_size
        value = sl[start : start + shard_size, :]
        gate = sl[ffn_dim + start : ffn_dim + start + shard_size, :]
        return torch.cat([value, gate], dim=0).T

    return SafetensorsWeightLoader(transform=transform)


def _cast_weight_loader(loader, dtype):
    def transform(slices, rank):
        return loader.load(slices, rank).to(dtype)

    return SafetensorsWeightLoader(transform=transform)


# ===================================================================
# Modules
# ===================================================================


class H3Attention(nn.Module):
    """TP self-attention: fused column-parallel QKV, per-head QK RMSNorm, row-parallel out."""

    def __init__(self, cfg: MiniMaxH3ImageConfig, use_cp: bool):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if cfg.num_attention_heads % tp_size:
            raise ValueError(f"{cfg.num_attention_heads} heads are not divisible by TP={tp_size}")
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.head_dim = cfg.attention_head_dim
        self.num_heads = cfg.num_attention_heads // tp_size
        self.eps = cfg.qk_norm_eps
        local = self.num_heads * self.head_dim
        self.local_inner = local
        hidden = cfg.hidden_size

        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size if use_cp else 1
        self.cp_group = cp_group if self.cp_size > 1 else None

        self.qkv_weight = nn.Parameter(torch.empty(hidden, 3 * local, dtype=torch.bfloat16))
        set_weight_loader(
            self.qkv_weight,
            fused_qkv_weight_loader(
                q_size=local, kv_size=local, shard_dim=1, num_shards=tp_size, is_storage_transposed=True
            ),
        )
        self.norm_q_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.bfloat16))
        self.norm_k_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.bfloat16))
        self.o_weight = nn.Parameter(torch.empty(local, hidden, dtype=torch.bfloat16))
        set_weight_loader(self.o_weight, _row_weight_loader(local, tp_size))

    def forward(self, x, kv_len, cos=None, sin=None):
        return self.project_out(self.attend(x, kv_len, cos, sin))

    def project_out(self, out):
        out = torch.matmul(out, self.o_weight)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        return out

    def attend(self, x, kv_len, cos=None, sin=None):
        """QKV projection, QK norm, RoPE and attention: ``[1, S, local_heads * D]`` (pre-O)."""
        qkv = torch.matmul(x, self.qkv_weight)
        q, k, v = qkv.split(self.local_inner, dim=-1)
        q = rms_norm(q.unflatten(-1, (self.num_heads, self.head_dim)), self.norm_q_weight, self.eps)
        k = rms_norm(k.unflatten(-1, (self.num_heads, self.head_dim)), self.norm_k_weight, self.eps)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        if cos is not None:
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        if self.cp_size > 1:
            k = self.cp_group.all_gather(k.contiguous(), dim=1)
            v = self.cp_group.all_gather(v.contiguous(), dim=1)
        return h3_attention(q, k, v, kv_len)


class H3FeedForward(nn.Module):
    """TP SwiGLU: ``down(value * silu(gate))``, all-reduced."""

    def __init__(self, cfg: MiniMaxH3ImageConfig):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if cfg.ffn_dim % tp_size:
            raise ValueError(f"ffn_dim {cfg.ffn_dim} is not divisible by TP={tp_size}")
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.shard = cfg.ffn_dim // tp_size
        hidden = cfg.hidden_size
        self.up_gate_weight = nn.Parameter(torch.empty(hidden, 2 * self.shard, dtype=torch.bfloat16))
        set_weight_loader(self.up_gate_weight, _swiglu_up_gate_loader(cfg.ffn_dim, self.shard))
        self.down_weight = nn.Parameter(torch.empty(self.shard, hidden, dtype=torch.bfloat16))
        set_weight_loader(self.down_weight, _row_weight_loader(self.shard, tp_size))

    def forward(self, x):
        h = torch.matmul(x, self.up_gate_weight)
        value, gate = h.split(self.shard, dim=-1)
        out = torch.matmul(value * F.silu(gate), self.down_weight)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        return out


class H3RefinerBlock(nn.Module):
    def __init__(self, cfg: MiniMaxH3ImageConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.attn = H3Attention(cfg, use_cp=False)
        self.norm2_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.ff = H3FeedForward(cfg)

    def forward(self, x, kv_len):
        x = x + self.attn(rms_norm(x, self.norm1_weight, self.eps), kv_len)
        x = x + self.ff(rms_norm(x, self.norm2_weight, self.eps))
        return x


def _segment_modulate(x, scale, shift, segments, base):
    """``x * (1 + scale[tab]) + shift[tab]`` over static row segments.

    ``segments`` is a list of ``(start, end, tab)`` in *global* row coordinates; ``base`` is
    the global row of ``x[:, 0]``. ``scale``/``shift`` are ``[NUM_TABS, H]``.
    """
    parts = []
    for start, end, tab in segments:
        xs = x[:, start - base : end - base]
        parts.append(xs * (1.0 + scale[tab]) + shift[tab])
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def _segment_gate(x, gate, segments, base):
    parts = []
    for start, end, tab in segments:
        parts.append(gate[tab] * x[:, start - base : end - base])
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


AUTO_OVERLAP_MAX_TOKENS = 8704


def _chunk_bounds(n: int, chunks: int, align: int = 128) -> list[tuple[int, int]]:
    step = max(align, -(-n // chunks // align) * align)
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _clip_segments(segments, lo, hi):
    out = [(max(s, lo), min(e, hi), t) for s, e, t in segments if max(s, lo) < min(e, hi)]
    return out


class H3Block(nn.Module):
    """Pre-norm self-attention + SwiGLU, each AdaLN-modulated per row segment."""

    def __init__(self, cfg: MiniMaxH3ImageConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.attn = H3Attention(cfg, use_cp=True)
        self.norm2_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.ff = H3FeedForward(cfg)
        # >1: split the post-attention tail into token chunks to overlap TP all-reduces;
        # 0: automatic (see AUTO_OVERLAP_MAX_TOKENS).
        self.overlap_chunks = 0

    def forward(self, x, mod, cos, sin, kv_len, segments, base):
        # mod: [NUM_TABS, 6, H] in diffusers order shift/scale/gate (msa), shift/scale/gate (mlp)
        shift_msa, scale_msa, gate_msa = mod[:, 0], mod[:, 1], mod[:, 2]
        shift_mlp, scale_mlp, gate_mlp = mod[:, 3], mod[:, 4], mod[:, 5]
        h = _segment_modulate(rms_norm(x, self.norm1_weight, self.eps), scale_msa, shift_msa, segments, base)
        chunks = self.overlap_chunks
        if chunks == 0:  # auto: overlap pays off up to ~8.7k tokens per core (measured on trn2)
            chunks = 2 if x.shape[1] <= AUTO_OVERLAP_MAX_TOKENS else 1
        if chunks <= 1:
            x = x + _segment_gate(self.attn(h, kv_len, cos, sin), gate_msa, segments, base)
            h = _segment_modulate(rms_norm(x, self.norm2_weight, self.eps), scale_mlp, shift_mlp, segments, base)
            return x + _segment_gate(self.ff(h), gate_mlp, segments, base)
        # Token-chunked tail: every chunk's O-projection / FFN all-reduces are independent of
        # the other chunks' matmuls, so the compiler can overlap the collectives with compute.
        a = self.attn.attend(h, kv_len, cos, sin)
        outs = []
        for c0, c1 in _chunk_bounds(x.shape[1], chunks):
            seg = _clip_segments(segments, base + c0, base + c1)
            xc = x[:, c0:c1] + _segment_gate(self.attn.project_out(a[:, c0:c1]), gate_msa, seg, base + c0)
            hc = _segment_modulate(rms_norm(xc, self.norm2_weight, self.eps), scale_mlp, shift_mlp, seg, base + c0)
            outs.append(xc + _segment_gate(self.ff(hc), gate_mlp, seg, base + c0))
        return torch.cat(outs, dim=1)


class _BlockGroup(nn.Module):
    """``K`` consecutive blocks; the code of the per-group graph (weights come in as inputs)."""

    def __init__(self, blocks: list[H3Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.segments: list[tuple[int, int, int]] = []
        self.base = 0

    def set_segments(self, segments, base) -> None:
        self.segments, self.base = list(segments), int(base)

    def forward(self, x, mods, cos, sin, kv_len):
        for j, block in enumerate(self.blocks):
            x = block(x, mods[j], cos, sin, kv_len, self.segments, self.base)
        return x


class MiniMaxH3ImageTransformer(nn.Module):
    """The MiniMax-H3-Image DiT, split into a per-request text graph and a per-step graph.

    * :meth:`prepare_text` — ``context_embedder`` + ``token_refiner`` over the padded text.
    * :meth:`forward` — one denoising step over the packed ``[image | text]`` sequence,
      returning the velocity of the generated rows.

    Host-side tables (modulation, output heads) come from :class:`MiniMaxH3ImageTimeTables`.
    """

    def __init__(
        self,
        config: MiniMaxH3ImageConfig | dict | None = None,
        num_layers: int | None = None,
        layers_per_graph: int | None = 5,
    ):
        super().__init__()
        if config is None:
            config = MiniMaxH3ImageConfig()
        elif isinstance(config, dict):
            config = MiniMaxH3ImageConfig.from_dict(config)
        if num_layers is not None:
            config.num_layers = int(num_layers)
        self.config = cfg = config

        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_rank = cp_group.rank_in_group
        self.cp_group = cp_group if self.cp_size > 1 else None
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        register_replica_groups(tp_size=self.tp_size, cp_size=self.cp_size)

        H = cfg.hidden_size
        self.proj_in_weight = nn.Parameter(torch.empty(cfg.patch_dim, H, dtype=torch.float32))
        set_weight_loader(self.proj_in_weight, _transposed_loader())
        self.proj_in_bias = nn.Parameter(torch.empty(H, dtype=torch.float32))
        self.context_embedder_weight = nn.Parameter(torch.empty(cfg.text_dim, H, dtype=torch.bfloat16))
        set_weight_loader(self.context_embedder_weight, _transposed_loader())
        self.context_embedder_bias = nn.Parameter(torch.empty(H, dtype=torch.bfloat16))

        self.refiner_blocks = nn.ModuleList([H3RefinerBlock(cfg) for _ in range(cfg.num_refiner_layers)])
        self.refiner_final_norm_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.blocks = nn.ModuleList([H3Block(cfg) for _ in range(cfg.num_layers)])
        self.norm_out_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))

        # Layout of the packed sequence, set per request by the pipeline (see set_layout).
        self.num_reference_rows = 0
        self.configure_layer_groups(layers_per_graph)

    # ------------------------------------------------------------------ layout

    def set_layout(self, num_reference_rows: int) -> None:
        self.num_reference_rows = int(num_reference_rows)

    def _segments(self, num_image_rows: int, total: int) -> tuple[list[tuple[int, int, int]], int]:
        """Static ``(start, end, tab)`` segments of this CP rank's rows, and its first row."""
        local = total // self.cp_size
        lo, hi = self.cp_rank * local, (self.cp_rank + 1) * local
        bounds = [
            (0, self.num_reference_rows, TAB_REFERENCE),
            (self.num_reference_rows, num_image_rows, TAB_TARGET),
            (num_image_rows, total, TAB_TEXT),
        ]
        segments = []
        for start, end, tab in bounds:
            s, e = max(start, lo), min(end, hi)
            if s < e:
                segments.append((s, e, tab))
        return segments, lo

    # ------------------------------------------------------------------ graphs

    def prepare_text(self, text_embeds: torch.Tensor, text_kv_len: torch.Tensor) -> torch.Tensor:
        """``[1, T_pad, text_dim]`` encoder states -> ``[1, T_pad, H]`` refined text rows."""
        cfg = self.config
        x = torch.matmul(text_embeds.to(torch.bfloat16), self.context_embedder_weight) + self.context_embedder_bias
        for block in self.refiner_blocks:
            x = block(x, text_kv_len)
        return rms_norm(x, self.refiner_final_norm_weight, cfg.final_norm_eps)

    # Layer groups: the 50 blocks run as ``num_layers / layers_per_graph`` calls of ONE
    # compiled graph over a group of identical blocks, with each group's weights passed in as
    # graph inputs. This bounds the compiler's live activations (a 50-layer graph at 4096^2
    # on 8 cores needs ~30 GB of HBM), and every group reuses the same NEFF.

    def configure_layer_groups(self, layers_per_graph: int | None) -> None:
        n = len(self.blocks)
        k = math.gcd(n, int(layers_per_graph or n))
        self.layers_per_graph = k
        template = _BlockGroup(list(self.blocks[:k]))
        # Not registered as a submodule: it aliases blocks[:k] only to provide the graph's code.
        object.__setattr__(self, "_group_template", template)
        template_names = [name for name, _ in template.named_parameters()]
        self._group_param_names = template_names

    def group_params(self, g: int) -> dict[str, torch.Tensor]:
        k = self.layers_per_graph
        params = {}
        for j in range(k):
            for name, p in self.blocks[g * k + j].named_parameters():
                params[f"blocks.{j}.{name}"] = p
        return params

    def split_mods(self, block_mods: torch.Tensor) -> list[torch.Tensor]:
        """``[L, NUM_TABS, 6, H]`` (host) -> one contiguous ``[K, ...]`` tensor per group."""
        k = self.layers_per_graph
        return [block_mods[g * k : (g + 1) * k].contiguous() for g in range(len(self.blocks) // k)]

    def embed(self, image_rows: torch.Tensor, text_hidden: torch.Tensor) -> torch.Tensor:
        """Packed ``[image | text]`` sequence, this CP rank's ``[1, S_local, H]`` bf16 chunk."""
        image_embeds = torch.matmul(image_rows.float(), self.proj_in_weight) + self.proj_in_bias
        x = torch.cat([image_embeds.to(torch.bfloat16), text_hidden.to(torch.bfloat16)], dim=1)
        total = x.shape[1]
        if total % self.cp_size:
            raise ValueError(f"packed length {total} is not divisible by CP={self.cp_size}")
        if self.cp_size > 1:
            local = total // self.cp_size
            x = x[:, self.cp_rank * local : (self.cp_rank + 1) * local]
        return x

    def run_group(self, x, mods, cos, sin, kv_len, params):
        """One group of ``layers_per_graph`` blocks, weights given by ``params``."""
        return torch.func.functional_call(self._group_template, params, (x, mods, cos, sin, kv_len))

    def finalize(self, x, out_mod, head_weight, head_bias):
        """Output norm + this step's PDD head; ``[1, total, patch_dim]`` fp32 (CP-gathered)."""
        h = rms_norm(x, self.norm_out_weight, self.config.final_norm_eps)
        h = h * (1.0 + out_mod[1]) + out_mod[0]
        out = torch.matmul(h.float(), head_weight) + head_bias
        if self.cp_size > 1:
            out = self.cp_group.all_gather(out.contiguous(), dim=1)
        return out

    def compile_step(self, compile_fn) -> None:
        """Replace the step pieces with ``compile_fn(fn, name)`` versions."""
        self._embed_fn = compile_fn(self.embed, "embed")
        self._group_fn = compile_fn(self.run_group, f"blocks_x{self.layers_per_graph}")
        self._final_fn = compile_fn(self.finalize, "final")

    def forward(
        self,
        image_rows: torch.Tensor,
        text_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_len: torch.Tensor,
        group_mods: list[torch.Tensor],
        out_mod: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor,
    ) -> torch.Tensor:
        """One denoising step (host orchestration of the compiled pieces).

        Args:
            image_rows: ``[1, N_img, patch_dim]`` fp32 patchified latents, reference rows first.
            text_hidden: ``[1, T_pad, H]`` bf16 output of :meth:`prepare_text`.
            cos, sin: ``[S_local, rotary_dim]`` bf16 rotary tables of this CP rank's rows.
            kv_len: ``[1]`` int32, number of real (non-padding) rows ``N_img + T_real``.
            group_mods: per layer group, ``[K, NUM_TABS, 6, H]`` bf16 AdaLN rows (see split_mods).
            out_mod: ``[2, H]`` bf16 output-norm ``shift`` and ``scale`` for this step.
            head_weight, head_bias: ``[H, patch_dim]`` / ``[patch_dim]`` fp32 PDD head of this step.

        Returns:
            ``[1, total, patch_dim]`` fp32; rows ``[:N_img]`` are the image velocities.
        """
        n_img = image_rows.shape[1]
        total = n_img + text_hidden.shape[1]
        segments, base = self._segments(n_img, total)
        self._group_template.set_segments(segments, base)
        if not hasattr(self, "_group_params"):
            self._group_params = [self.group_params(g) for g in range(len(group_mods))]
        embed = getattr(self, "_embed_fn", self.embed)
        run = getattr(self, "_group_fn", self.run_group)
        final = getattr(self, "_final_fn", self.finalize)
        x = embed(image_rows, text_hidden)
        for g, mods in enumerate(group_mods):
            x = run(x, mods, cos, sin, kv_len, self._group_params[g])
        return final(x, out_mod, head_weight, head_bias)

    # ------------------------------------------------------------------ weights

    def load_weights(self, model_path: str, device: torch.device = torch.device("cpu")) -> None:
        """Load the diffusers-format checkpoint (``transformer_diffusers/``) TP-sharded."""
        tp_rank, tp_size = self.tp_rank, self.tp_size
        mappings: dict[str, str | list[str]] = {
            "proj_in_weight": "proj_in.weight",
            "proj_in_bias": "proj_in.bias",
            "context_embedder_weight": "context_embedder.weight",
            "context_embedder_bias": "context_embedder.bias",
            "refiner_final_norm_weight": "token_refiner.final_norm.weight",
            "norm_out_weight": "norm_out.norm.weight",
        }

        def attn_ff(prefix_model, prefix_ckpt):
            mappings[f"{prefix_model}.norm1_weight"] = f"{prefix_ckpt}.norm1.weight"
            mappings[f"{prefix_model}.norm2_weight"] = f"{prefix_ckpt}.norm2.weight"
            mappings[f"{prefix_model}.attn.qkv_weight"] = [
                f"{prefix_ckpt}.attn.to_q.weight",
                f"{prefix_ckpt}.attn.to_k.weight",
                f"{prefix_ckpt}.attn.to_v.weight",
            ]
            mappings[f"{prefix_model}.attn.norm_q_weight"] = f"{prefix_ckpt}.attn.norm_q.weight"
            mappings[f"{prefix_model}.attn.norm_k_weight"] = f"{prefix_ckpt}.attn.norm_k.weight"
            mappings[f"{prefix_model}.attn.o_weight"] = f"{prefix_ckpt}.attn.to_out.0.weight"
            mappings[f"{prefix_model}.ff.up_gate_weight"] = f"{prefix_ckpt}.ff.net.0.proj.weight"
            mappings[f"{prefix_model}.ff.down_weight"] = f"{prefix_ckpt}.ff.net.2.weight"

        for i in range(len(self.refiner_blocks)):
            attn_ff(f"refiner_blocks.{i}", f"token_refiner.refiner_blocks.{i}")
        for i in range(len(self.blocks)):
            attn_ff(f"blocks.{i}", f"transformer_blocks.{i}")

        for name, param in self.named_parameters():
            set_weight_loader(param, _cast_weight_loader(get_weight_loader(param), param.dtype))

        checkpoint = SafetensorsCheckpoint(model_path, None)
        result = checkpoint.load_sharded_pipelined(tp_rank, tp_size, self, mappings, device)
        missing = sorted(set(dict(self.named_parameters())) - set(result.state_dict))
        if missing:
            raise RuntimeError(f"MiniMax-H3-Image DiT weights not loaded: {missing[:8]}")
        self.load_state_dict(result.state_dict, strict=False, assign=True)


# ===================================================================
# Host-side timestep tables
# ===================================================================


def _timestep_sinusoid(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """diffusers ``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)``."""
    from diffusers.models.embeddings import get_timestep_embedding

    return get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift=0)


class MiniMaxH3ImageTimeTables:
    """Host-side, timestep-only parts of the DiT, evaluated exactly as the reference does.

    For a set of timesteps it produces, per block, the six AdaLN rows of every modality
    (``adaln_proj``), the output-norm ``shift``/``scale`` (``norm_out.linear``), and keeps
    the per-step PDD output heads. Weights are streamed block by block from the checkpoint
    and dropped; only the small tables are kept.
    """

    def __init__(self, model_path: str, config: MiniMaxH3ImageConfig):
        self.model_path = model_path
        self.config = config
        index_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json")
        with open(index_path) as f:
            self._weight_map = json.load(f)["weight_map"]
        self._handles: dict[str, object] = {}
        self._temb_cache: dict[float, torch.Tensor] = {}

    def _get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        fname = self._weight_map[key]
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.model_path, fname), framework="pt")
        return self._handles[fname].get_tensor(key)

    def temb(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``time_embedder(time_proj(t))`` in fp32: ``[n, time_embed_dim]``."""
        cfg = self.config
        x = _timestep_sinusoid(timesteps, cfg.freq_dim)
        x = F.linear(x, self._get("time_embedder.linear_1.weight").float(), self._get("time_embedder.linear_1.bias").float())
        x = F.silu(x)
        return F.linear(x, self._get("time_embedder.linear_2.weight").float(), self._get("time_embedder.linear_2.bias").float())

    @torch.no_grad()
    def block_tables(self, timesteps: torch.Tensor, blocks: list[int]) -> dict[int, torch.Tensor]:
        """Per block: ``[n_t, MODALITY_NUM, 6, H]`` bf16 AdaLN rows (reference rounding)."""
        cfg = self.config
        act = F.silu(self.temb(timesteps)).to(torch.bfloat16)
        out = {}
        for i in blocks:
            w = self._get(f"transformer_blocks.{i}.adaln_proj.linear.weight")
            b = self._get(f"transformer_blocks.{i}.adaln_proj.linear.bias")
            rows = F.linear(act, w.to(torch.bfloat16), b.to(torch.bfloat16))  # [n_t, 6*H*3]
            out[i] = rows.view(-1, MODALITY_NUM, 6, cfg.hidden_size).contiguous()
        return out

    @torch.no_grad()
    def out_table(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``[n_t, 2, H]`` bf16 output-norm ``(shift, scale)``."""
        act = F.silu(self.temb(timesteps)).to(torch.bfloat16)
        w = self._get("norm_out.linear.weight").to(torch.bfloat16)
        b = self._get("norm_out.linear.bias").to(torch.bfloat16)
        shift, scale = F.linear(act, w, b).chunk(2, dim=-1)
        return torch.stack([shift, scale], dim=1).contiguous()

    def head(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """PDD output head of ``step``: ``([H, patch_dim], [patch_dim])`` fp32."""
        w = self._get(f"pdd_proj_out.{step}.weight").float()
        b = self._get(f"pdd_proj_out.{step}.bias").float()
        return w.T.contiguous(), b.contiguous()


def build_rope_tables(position_ids: torch.Tensor, cfg: MiniMaxH3ImageConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``MiniMaxH3ImageRotaryPosEmbed``: ``[S, 3]`` positions -> fp32 cos/sin ``[S, 96]``."""
    d = cfg.rope_freq_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, 2 * d, 2, dtype=torch.float32) / (2 * d)))
    pos = position_ids.to(torch.float32)
    freqs = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    ft, fh, fw = freqs.unbind(dim=1)
    freqs = torch.cat((ft, fh, fw), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()
