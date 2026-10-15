# SPDX-License-Identifier: Apache-2.0
"""QwenImage21Transformer2DModel for Neuron — text-to-image, tensor parallel.

Follows the vLLM-Neuron raw ``nn.Parameter`` + weight-loader pattern used by the Wan2.2
DiT. The math mirrors diffusers' ``transformer_qwenimage21.py`` (single-stream blocks,
one modulation shared by every block, ``causal_condition``).

Qwen-Image 2.1 attends over one joint ``[text, image]`` sequence with a block-causal
mask: text is causal, the target image block is bidirectional and sees all of the text.
Under ``causal_condition`` the text tokens modulate from ``t = 0``, so their activations
— and therefore every layer's text K/V — do not depend on the denoising step or on the
image. Text-to-image therefore splits into two static graphs:

* :meth:`forward_prefix` runs the text tokens through the blocks once per request and
  returns every layer's text K/V (the prefix KV cache of the reference pipeline).
* :meth:`forward_step` runs one denoising step over the image tokens only, attending to
  ``[image K/V, cached text K/V]``, and applies the flow-matching Euler update in-graph.

This is exactly the reference's KV-cached path: its first step attends with the same
mask, and target rows of a block-causal mask see the whole prefix plus their own block.

Text is right-padded to a static length. Padded text keys are excluded from the image
attention through the attention kernel's per-query KV bounds (``[0, N + text_len)`` with
image keys first); in the causal text pass they never precede a valid query.

Supported parallelism: TP (heads and MLP intermediate dim sharded). The step graph also
runs sequence parallel over the TP group by default: token-wise work (fused AdaLN norms,
residuals, embedders) runs on ``N / TP`` tokens per rank, with an all-gather before and a
reduce-scatter after each sharded projection. The text pass all-reduces instead.
Context parallelism (``ring_degree``) splits the step graph's image queries over the CP
group and all-gathers the image K/V; the text pass and its K/V are replicated per CP
rank.
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import NormType, QKVOutputLayout
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
from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import (
    sp_all_gather_seq,
    sp_reduce_scatter_seq,
)
from vllm_omni_neuron.diffusion.quantization.adaln_kernels import adaln_modulate

logger = logging.getLogger(__name__)


# The attention kernel reads per-query KV bounds in tiles of this many query rows.
_BOUNDS_TILE = 128

# Heads per fused QKV kernel call (see QwenImage21Attention.__init__).
_MAX_QKV_KERNEL_HEADS = 4

# QwenImage21Rope position tables: [0, 8192) followed by [-1024, -1].
_ROPE_MAX_POS = 8192
_ROPE_MAX_NEG = 1024


@dataclass
class QwenImage21Config:
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True


# ===================================================================
# RoPE (host side)
# ===================================================================


def _axis_angles(index: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    inv_freq = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
    return torch.outer(index, inv_freq)


def qwen_image21_rope(
    text_len: int,
    text_buffer_len: int,
    latent_height: int,
    latent_width: int,
    axes_dim=(16, 56, 56),
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotary tables of the joint T2I sequence, float32, in rotate-half layout.

    Mirrors ``QwenImage21Rope`` for one prompt followed by the target image: text tokens
    advance a shared position on all three axes; the image block freezes the frame axis
    at ``text_len`` and lays its tokens on a height/width grid centred on zero. Padded
    text positions continue the text positions (they are masked out as keys).

    The reference rotates adjacent channel pairs ``(2i, 2i + 1)``. The fused QKV kernel
    rotates channel ``i`` with ``i + D/2`` instead, so the attention stores Q and K with
    each head's channels permuted to ``[even..., odd...]`` (see :data:`_ROPE_PERM`),
    which turns one rotation into the other; ``cos``/``sin`` repeat the per-pair values
    over both halves.

    Returns:
        ``text_cos, text_sin`` ``[text_buffer_len, D]`` and ``img_cos, img_sin``
        ``[latent_height * latent_width, D]``.
    """
    if text_len + max(latent_height, latent_width) > _ROPE_MAX_POS:
        raise ValueError("sequence exceeds the RoPE position table")
    if max(latent_height, latent_width) > 2 * _ROPE_MAX_NEG:
        raise ValueError("latent grid exceeds the negative RoPE position table")

    text_pos = torch.arange(text_buffer_len)
    text = torch.cat([_axis_angles(text_pos, d, theta) for d in axes_dim], dim=-1)

    h_idx = torch.arange(-(latent_height - latent_height // 2), latent_height // 2)
    w_idx = torch.arange(-(latent_width - latent_width // 2), latent_width // 2)
    grid_h = h_idx[:, None].expand(latent_height, latent_width).reshape(-1)
    grid_w = w_idx[None, :].expand(latent_height, latent_width).reshape(-1)
    frame = torch.full_like(grid_h, text_len)
    image = torch.cat(
        [
            _axis_angles(frame, axes_dim[0], theta),
            _axis_angles(grid_h, axes_dim[1], theta),
            _axis_angles(grid_w, axes_dim[2], theta),
        ],
        dim=-1,
    )
    text, image = torch.cat([text, text], dim=-1), torch.cat([image, image], dim=-1)
    return text.cos(), text.sin(), image.cos(), image.sin()


@dataclass
class EditLayout:
    """Host-side layout of an edit prefix (see :func:`qwen_image21_edit_layout`)."""

    gather_index: torch.Tensor  # [prefix_len] int64 into cat([text rows, condition latents])
    bound_max: torch.Tensor  # [prefix_len] int32, exclusive key bound per prefix query
    prefix_cos: torch.Tensor  # [prefix_len, D]
    prefix_sin: torch.Tensor
    target_cos: torch.Tensor  # [target tokens, D]
    target_sin: torch.Tensor
    valid_len: int  # unpadded prefix length


def qwen_image21_edit_layout(
    slot_mask: torch.Tensor,
    text_rows: int,
    condition_hw: list[tuple[int, int]],
    target_hw: tuple[int, int],
    axes_dim=(16, 56, 56),
    theta: float = 10000.0,
    multiple_of: int = _BOUNDS_TILE,
) -> EditLayout:
    """Joint text/condition-image prefix, following the reference.

    ``slot_mask`` ``[L]`` marks the vision-language image slots among the ``L`` valid text
    rows (after the system prompt); the rows are padded to ``text_rows``. Each slot stands
    for a 2x2 group of condition latents, so the slots of condition image ``i`` (taken in
    order, ``h_i * w_i / 4`` of them, as ``build_token_metadata`` counts them) become its
    ``h_i * w_i`` latent tokens in raster order. The condition latents follow the text
    rows in the gather source, concatenated in image order.

    The mask is causal over text and bidirectional inside each image block; RoPE follows
    ``QwenImage21Rope`` (text advances one position per token on all axes; an image block
    freezes the frame axis, centres its h/w grid, and advances the position by
    ``max(h, w)``); the target image comes after the prefix. Padded rows are appended
    after the valid prefix (masked out as keys), up to a multiple of ``multiple_of``
    positions (the attention kernel reads per-query bounds in 128-row tiles).
    """
    num_valid = int(slot_mask.shape[0])
    slots = slot_mask.nonzero().flatten().tolist()
    counts = [h * w // 4 for h, w in condition_hw]
    if sum(counts) != len(slots) or any(h * w % 4 for h, w in condition_hw):
        raise ValueError("image slots do not match the condition latent grids")

    # Segments in sequence order: ("text", rows) or ("image", index).
    segments, row, start = [], 0, 0
    for index, count in enumerate(counts):
        first = slots[start]
        if slots[start : start + count] != list(range(first, first + count)):
            raise ValueError(f"condition image {index}'s slots are not contiguous")
        segments.append(("text", list(range(row, first))))
        segments.append(("image", index))
        row, start = first + count, start + count
    segments.append(("text", list(range(row, num_valid))))

    def grid(h, w):
        hs = torch.arange(-(h - h // 2), h // 2)[:, None].expand(h, w).reshape(-1)
        ws = torch.arange(-(w - w // 2), w // 2)[None, :].expand(h, w).reshape(-1)
        return hs, ws

    latent_offsets = [0]
    for h, w in condition_hw:
        latent_offsets.append(latent_offsets[-1] + h * w)

    gather, bound, frame, hh, ww = [], [], [], [], []
    pos = 0
    for kind, value in segments:
        if kind == "text":
            gather += value
            bound += [len(bound) + p + 1 for p in range(len(value))]
            idx = torch.arange(pos, pos + len(value))
            frame.append(idx), hh.append(idx), ww.append(idx)
            pos += len(value)
        else:
            h, w = condition_hw[value]
            gather += [text_rows + latent_offsets[value] + i for i in range(h * w)]
            bound += [len(bound) + h * w] * (h * w)
            hs, ws = grid(h, w)
            frame.append(torch.full_like(hs, pos)), hh.append(hs), ww.append(ws)
            pos += max(h, w)
    valid_len = len(gather)
    target_pos = pos

    pad_rows = list(range(num_valid, text_rows))
    pad_rows += [text_rows - 1] * (-(valid_len + len(pad_rows)) % multiple_of)
    gather += pad_rows
    bound += [len(bound) + p + 1 for p in range(len(pad_rows))]
    idx = torch.arange(pos, pos + len(pad_rows))  # padding: any positions (masked as keys)
    frame.append(idx), hh.append(idx), ww.append(idx)

    def tables(f, h, w):
        angles = torch.cat(
            [
                _axis_angles(f, axes_dim[0], theta),
                _axis_angles(h, axes_dim[1], theta),
                _axis_angles(w, axes_dim[2], theta),
            ],
            dim=-1,
        )
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos(), angles.sin()

    prefix_cos, prefix_sin = tables(torch.cat(frame), torch.cat(hh), torch.cat(ww))
    th, tw = target_hw
    hs, ws = grid(th, tw)
    target_cos, target_sin = tables(torch.full_like(hs, target_pos), hs, ws)
    return EditLayout(
        gather_index=torch.tensor(gather, dtype=torch.int64),
        bound_max=torch.tensor(bound, dtype=torch.int32),
        prefix_cos=prefix_cos,
        prefix_sin=prefix_sin,
        target_cos=target_cos,
        target_sin=target_sin,
        valid_len=valid_len,
    )


def _rope_perm(head_dim: int) -> torch.Tensor:
    """Per-head channel order ``[0, 2, ..., D-2, 1, 3, ..., D-1]``.

    Applied to Q and K alike it leaves every ``q . k`` unchanged, and it maps the
    reference's adjacent-pair rotation onto the kernel's half-split rotation.
    """
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


# ===================================================================
# Model components
# ===================================================================


def _layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Non-affine LayerNorm computed in float32, returned in ``x``'s dtype."""
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose checkpoint weight stores ``scale - 1``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        rrms = torch.rsqrt(torch.mean(h**2, dim=-1, keepdim=True) + self.eps)
        return (h * rrms * (self.weight.float() + 1)).to(x.dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(self.text_norm(x))
        return self.out_layer(F.gelu(x, approximate="tanh"))


class _TimestepEmbedder(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, dim, bias=False)
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class QwenImage21TimestepProjEmbeddings(nn.Module):
    """Sinusoidal (cos first) timestep projection followed by the MLP embedder."""

    timestep_dim = 256

    def __init__(self, dim: int, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.time_factor = time_factor
        half = self.timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.timestep_embedder = _TimestepEmbedder(self.timestep_dim, dim)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        args = (self.time_factor * timestep.float())[:, None] * self.freqs[None]
        proj = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.timestep_embedder(proj.to(dtype))


class _NormOut(nn.Module):
    """Final adaptive norm: scale only, ``LN(x) * (1 + linear(silu(temb)))``.

    Holds the projection only; :meth:`QwenImage21Transformer2DModel.forward_step` fuses
    the norm into the last block's residual.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)


def _col_loader(shard_size: int, num_shards: int):
    return sharding_weight_loader(
        shard_dim=1, shard_size=shard_size, num_shards=num_shards, is_storage_transposed=True
    )


def _row_loader(shard_size: int, num_shards: int):
    return sharding_weight_loader(
        shard_dim=0, shard_size=shard_size, num_shards=num_shards, is_storage_transposed=True
    )


def _permute_qk_loader(loader, num_heads: int, num_groups: int, perm: torch.Tensor):
    """Fused ``[H, 3 * inner]`` QKV shard → ``[groups, H, 3 * inner / groups]``.

    Each Q and K head's output channels are reordered by ``perm``, and the heads are
    split into ``num_groups`` contiguous slabs, each laid out ``[q, k, v]`` for its heads.
    """

    def transform(slices, rank):
        weight = loader.load(slices, rank)
        hidden = weight.shape[0]
        weight = weight.view(hidden, 3, num_heads, -1).clone()
        weight[:, :2] = weight[:, :2, :, perm]
        weight = weight.view(hidden, 3, num_groups, num_heads // num_groups, -1)
        return weight.permute(2, 0, 1, 3, 4).reshape(num_groups, hidden, -1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _permuted_gamma_loader(perm: torch.Tensor):
    def transform(slices, rank):
        return slices[0][:][perm].reshape(1, -1)

    return SafetensorsWeightLoader(transform=transform)


def _tp_reduce(output: torch.Tensor, group, tp_size: int, sequence_parallel: bool) -> torch.Tensor:
    """Sum a row-parallel projection over TP: reduce-scatter along the sequence under
    sequence parallelism, all-reduce otherwise."""
    if tp_size <= 1:
        return output
    if sequence_parallel:
        return sp_reduce_scatter_seq(output, group, tp_size)
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
    return output


def _adaln(hidden, mul, eps, residual=None, gate=None):
    """Fused ``LayerNorm(residual + gate * hidden) * mul`` (see ``adaln_modulate``)."""
    return adaln_modulate(
        hidden, mul, None, eps=eps, norm_type="layer_norm", residual=residual, gate=gate
    )


class QwenImage21Attention(nn.Module):
    """Fused-QKV attention with per-head RMS QK-norm and TP head sharding."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group
        if num_heads % self.tp_size != 0:
            raise ValueError(f"num_attention_heads={num_heads} not divisible by TP={self.tp_size}")
        self.num_heads = num_heads // self.tp_size
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner = self.num_heads * head_dim
        self.qkv_split = [inner, 2 * inner]

        self.eps = eps
        perm = _rope_perm(head_dim)

        # The fused QKV kernel returns garbage above a 1536-wide (4-head) output on this
        # SDK, so ranks holding more heads project them in groups of at most 4.
        self.head_groups = math.ceil(self.num_heads / _MAX_QKV_KERNEL_HEADS)
        while self.num_heads % self.head_groups:
            self.head_groups += 1
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.head_groups, dim, 3 * inner // self.head_groups)
        )
        set_weight_loader(
            self.qkv_proj_weight,
            _permute_qk_loader(
                fused_qkv_weight_loader(
                    q_size=inner,
                    kv_size=inner,
                    shard_dim=1,
                    num_shards=self.tp_size,
                    is_storage_transposed=True,
                ),
                self.num_heads,
                self.head_groups,
                perm,
            ),
        )
        self.o_proj_weight = nn.Parameter(torch.empty(inner, dim))
        set_weight_loader(self.o_proj_weight, _row_loader(inner, self.tp_size))
        # Per-head RMS QK-norm gammas, stored [1, D] in the permuted channel order.
        self.norm_q_weight = nn.Parameter(torch.ones(1, head_dim))
        self.norm_k_weight = nn.Parameter(torch.ones(1, head_dim))
        for param in (self.norm_q_weight, self.norm_k_weight):
            set_weight_loader(param, _permuted_gamma_loader(perm))

    def project_qkv(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """``[1, S, H]`` → ``q, k, v`` head-major ``[heads, S, D]``, Q/K normed and RoPE'd.

        The fused NKI QKV kernel (one call per head group) does the projection, the per-head RMS QK-norm and the
        rotary embedding (``cos``/``sin`` ``[S, D]``), and writes the head-major layout
        attention consumes directly.
        """
        n = self.num_heads // self.head_groups
        q, k, v = [], [], []
        for weight in self.qkv_proj_weight.unbind(0):
            qkv = NF.qkv_proj(
                hidden_states,
                weight,
                d_head=self.head_dim,
                output_layout=QKVOutputLayout.NBSd,
                cos_cache=cos.unsqueeze(0),
                sin_cache=sin.unsqueeze(0),
                num_q_heads=n,
                num_kv_heads=n,
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=self.eps,
                qk_norm_pre_rope_q_gamma=self.norm_q_weight,
                qk_norm_pre_rope_k_gamma=self.norm_k_weight,
            ).squeeze(1)
            q.append(qkv[:n])
            k.append(qkv[n : 2 * n])
            v.append(qkv[2 * n :])
        if self.head_groups == 1:
            return q[0], k[0], v[0]
        return torch.cat(q), torch.cat(k), torch.cat(v)

    def project_out(self, attn: torch.Tensor, sequence_parallel: bool = False) -> torch.Tensor:
        """d-major ``[heads, D, S]`` attention output → ``[1, S, H]`` summed over TP.

        With ``sequence_parallel`` the sum is reduce-scattered along the sequence, so each
        rank gets its ``S / TP`` slice; otherwise it is all-reduced.
        """
        output = NF.o_proj(attn.unsqueeze(0), self.o_proj_weight)
        return _tp_reduce(output, self.tp_group, self.tp_size, sequence_parallel)

    def attend_causal(self, q, k, v) -> torch.Tensor:
        return NF.flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=self.scale,
            causal_mask=True,
            tp_q=True,
            tp_k=True,
            tp_out=True,
        )

    def attend_prefix(self, q, k, v, bound_max: torch.Tensor) -> torch.Tensor:
        """Query ``i`` attends to keys ``[0, bound_max[i])`` (block-causal prefix).

        The prefix length must be a multiple of 128 (see :meth:`attend_bounded`).
        """
        heads, seq_q, _ = q.shape
        bound_max = bound_max.to(torch.int32).view(1, seq_q, 1).expand(heads, seq_q, 1).contiguous()
        return NF.flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=True,
            bound_min=torch.zeros_like(bound_max),
            bound_max=bound_max,
        )

    def attend_bounded(self, q, k, v, kv_len: torch.Tensor) -> torch.Tensor:
        """Full attention of ``q`` over the first ``kv_len`` keys of ``k``/``v``.

        The kernel reads the per-query bounds in 128-row tiles, so the queries are padded
        to a multiple of 128 and the padded output rows dropped.
        """
        heads, seq_q, _ = q.shape
        pad = -seq_q % _BOUNDS_TILE
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
        bound_max = kv_len.to(torch.int32).view(1, 1, 1).expand(heads, seq_q + pad, 1).contiguous()
        bound_min = torch.zeros_like(bound_max)
        out = NF.flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=True,
            bound_min=bound_min,
            bound_max=bound_max,
        )
        return out[..., :seq_q] if pad else out


class QwenImage21FeedForward(nn.Module):
    """SwiGLU ``out(silu(gate_layer(x)) * proj(x))`` with TP intermediate sharding."""

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group
        inner = inner_dim // self.tp_size
        self.gate_proj_weight = nn.Parameter(torch.empty(dim, inner))
        self.up_proj_weight = nn.Parameter(torch.empty(dim, inner))
        self.down_proj_weight = nn.Parameter(torch.empty(inner, dim))
        set_weight_loader(self.gate_proj_weight, _col_loader(inner, self.tp_size))
        set_weight_loader(self.up_proj_weight, _col_loader(inner, self.tp_size))
        set_weight_loader(self.down_proj_weight, _row_loader(inner, self.tp_size))

    def forward(self, hidden_states: torch.Tensor, sequence_parallel: bool = False):
        """``[1, S, H]`` → ``[1, S, H]``; with ``sequence_parallel`` the input and output
        are this rank's ``S / TP`` slice (all-gathered before, reduce-scattered after)."""
        if sequence_parallel:
            hidden_states = sp_all_gather_seq(hidden_states, self.tp_group, self.tp_size)
        shape = hidden_states.shape
        output = NF.mlp(
            hidden_states.reshape(-1, shape[-1]),
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        ).reshape(shape)
        return _tp_reduce(output, self.tp_group, self.tp_size, sequence_parallel)


class QwenImage21TransformerBlock(nn.Module):
    """Single-stream block; its scales and gates come from the model-wide modulation."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: int, eps: float):
        super().__init__()
        self.eps = eps
        self.attn = QwenImage21Attention(dim, num_heads, head_dim, eps)
        self.img_mlp = QwenImage21FeedForward(dim, dim * mlp_ratio)

    def _residual_mlp(self, hidden_states, scale2, gate2):
        modulated = _layer_norm(hidden_states, self.eps) * (1 + scale2)
        return hidden_states + gate2.tanh() * self.img_mlp(modulated)

    def forward_prefix(self, hidden_states, modulation, cos, sin, last: bool, bound_max=None):
        """Prefix pass: return this layer's prefix ``k, v`` and (unless ``last``) its output.

        Text-only prefixes are causal; with ``bound_max`` (edit prefixes holding a
        condition-image block) query ``i`` attends to keys ``[0, bound_max[i])``.
        """
        scale1, gate1, scale2, gate2 = modulation
        modulated = _layer_norm(hidden_states, self.eps) * (1 + scale1)
        q, k, v = self.attn.project_qkv(modulated, cos, sin)
        if last:
            return None, k, v
        if bound_max is None:
            attn = self.attn.attend_causal(q, k, v)
        else:
            attn = self.attn.attend_prefix(q, k, v, bound_max)
        attn = self.attn.project_out(attn)
        hidden_states = hidden_states + gate1.tanh() * attn
        return self._residual_mlp(hidden_states, scale2, gate2), k, v

    def attention_step(
        self,
        modulated,
        cos,
        sin,
        text_k,
        text_v,
        kv_len,
        sequence_parallel,
        cp_group=None,
        num_image_keys=None,
    ):
        """Image self-attention over ``[image K/V, cached text K/V]``.

        ``modulated`` (and the result) is this rank's sequence slice under
        ``sequence_parallel``, else its whole (CP-local) image sequence. With a
        ``cp_group`` the queries stay local and the image K/V are all-gathered over it;
        the attention is bidirectional and RoPE is already applied, so the gathered
        order does not matter. ``num_image_keys`` (padded sequences) is the number of
        real image tokens; the keys of the padding rows after them are moved behind the
        text keys, outside the ``kv_len`` attended range.
        """
        if sequence_parallel:
            modulated = sp_all_gather_seq(modulated, self.attn.tp_group, self.attn.tp_size)
        q, k, v = self.attn.project_qkv(modulated, cos, sin)
        if cp_group is not None:
            k = cp_group.all_gather(k.contiguous(), dim=1)
            v = cp_group.all_gather(v.contiguous(), dim=1)
        if num_image_keys is None:
            k = torch.cat([k, text_k], dim=1)
            v = torch.cat([v, text_v], dim=1)
        else:
            # Padded image tokens: their keys go last, after the (valid) text keys.
            n = num_image_keys
            k = torch.cat([k[:, :n], text_k, k[:, n:]], dim=1)
            v = torch.cat([v[:, :n], text_v, v[:, n:]], dim=1)
        return self.attn.project_out(self.attn.attend_bounded(q, k, v, kv_len), sequence_parallel)


class QwenImage21Transformer2DModel(nn.Module):
    """Qwen-Image 2.1 DiT for Neuron (text-to-image).

    Args mirror the diffusers config. ``text_offset`` is the number of leading
    text-encoder positions (the system prompt) dropped before the DiT, and
    ``text_buffer_len`` the static number of DiT text positions after it.
    """

    def __init__(
        self,
        text_offset: int,
        text_buffer_len: int,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope=(16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
        sequence_parallel: bool = True,
    ):
        super().__init__()
        if patch_size != 1:
            raise ValueError("Qwen-Image 2.1 consumes latents unpatched (patch_size=1)")
        if not causal_condition:
            raise ValueError("the prefix KV split requires causal_condition=True")
        self.config = QwenImage21Config(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels or in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            context_in_dim=context_in_dim,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=tuple(axes_dims_rope),
            eps=eps,
            causal_condition=causal_condition,
        )
        self.text_offset = text_offset
        self.text_buffer_len = text_buffer_len
        inner_dim = num_attention_heads * attention_head_dim

        cp_group = get_cp_group()
        cp_size = cp_group.world_size
        self.cp_size = cp_size
        self.cp_rank = cp_group.rank_in_group
        self.cp_group = cp_group if cp_size > 1 else None
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group().device_group
        self.sequence_parallel = sequence_parallel
        register_replica_groups(tp_size=self.tp_size, cp_size=cp_size)

        self.time_text_embed = QwenImage21TimestepProjEmbeddings(inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, inner_dim, eps)
        self.img_in = nn.Linear(in_channels, inner_dim, bias=False)
        # One shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 4 * inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = _NormOut(inner_dim)
        self.proj_out = nn.Linear(inner_dim, self.config.out_channels, bias=False)

        self._compiled_prefix = None
        self._compiled_step = None

    @property
    def dtype(self) -> torch.dtype:
        return self.img_in.weight.dtype

    def _modulation(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``temb`` ``[1, H]`` → the four ``[1, 1, H]`` scale/gate tensors."""
        return tuple(t.unsqueeze(1) for t in self.modulation(temb).chunk(4, dim=-1))

    def forward_prefix(
        self,
        encoder_hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Text-encoder output ``[1, offset + buffer_len, C]`` → stacked text K and V.

        Returns ``k, v`` of shape ``[num_layers, heads_per_rank, text_buffer_len, D]``.
        """
        dtype = self.dtype
        text = encoder_hidden_states[:, self.text_offset : self.text_offset + self.text_buffer_len]
        return self._prefix_blocks(self.txt_in(text.to(dtype)), cos, sin)

    def forward_prefix_edit(
        self,
        encoder_hidden_states: torch.Tensor,
        condition_latents: torch.Tensor,
        gather_index: torch.Tensor,
        bound_max: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Edit prefix: text with the condition image's latent block → stacked K and V.

        The joint sequence is ``cat([txt_in(text rows), img_in(condition latents)])``
        gathered by ``gather_index`` (see :func:`qwen_image21_edit_layout`), so every
        vision-language image slot becomes its 2x2 latent tokens in place.
        """
        dtype = self.dtype
        text = encoder_hidden_states[:, self.text_offset :]
        rows = torch.cat([self.txt_in(text.to(dtype)), self.img_in(condition_latents.to(dtype))], 1)
        hidden_states = torch.index_select(rows, 1, gather_index)
        return self._prefix_blocks(hidden_states, cos, sin, bound_max)

    def _prefix_blocks(self, hidden_states, cos, sin, bound_max=None):
        """Run the prefix through the blocks, modulated from ``t = 0``."""
        dtype = self.dtype
        zero = torch.zeros(1, dtype=dtype, device=hidden_states.device)
        modulation = self._modulation(self.time_text_embed(zero, dtype))
        keys, values = [], []
        last = len(self.transformer_blocks) - 1
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, k, v = block.forward_prefix(
                hidden_states, modulation, cos, sin, last=i == last, bound_max=bound_max
            )
            keys.append(k)
            values.append(v)
        return torch.stack(keys), torch.stack(values)

    def forward_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        dt: torch.Tensor,
        text_k: torch.Tensor,
        text_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_len: torch.Tensor,
    ) -> torch.Tensor:
        """One denoising step: predict the flow and take the Euler step.

        Args:
            latents: Packed latents ``[1, N, C_in]``.
            timestep: ``[1]``, the scheduler timestep scaled to ``[0, 1]``.
            dt: ``[1]`` float32, ``sigma_next - sigma``.
            text_k, text_v: Output of :meth:`forward_prefix`.
            cos, sin: Image RoPE ``[N, D/2]``.
            kv_len: ``[1]`` int, ``N`` plus the number of valid text tokens.

        Returns:
            The updated latents, in ``latents.dtype``.
        """
        dtype = self.dtype
        eps = self.config.eps
        seq_len = latents.shape[1]
        # Context parallelism splits the image sequence over the CP group: each CP rank
        # owns a contiguous block of queries (and their RoPE) and gathers K/V.
        latents_local = latents
        if self.cp_size > 1:
            if seq_len % self.cp_size != 0:
                raise ValueError(
                    f"{seq_len} image tokens do not split over context parallelism "
                    f"{self.cp_size}; choose a resolution whose latent grid does"
                )
            seq_len //= self.cp_size
            start = self.cp_rank * seq_len
            latents_local = latents[:, start : start + seq_len]
            cos, sin = cos[start : start + seq_len], sin[start : start + seq_len]
        # Sequence parallelism shards the token-wise work (norms, modulation, residuals,
        # embedders) across TP ranks; it needs the (CP-local) sequence to split evenly.
        # Without CP, a sequence that does not split evenly is padded with dummy image
        # tokens; attention_step keeps their keys outside the attended range and the
        # padded rows are dropped from the output.
        sp = self.sequence_parallel and self.tp_size > 1
        pad = -seq_len % self.tp_size if sp and self.cp_size == 1 else 0
        sp = sp and (seq_len + pad) % self.tp_size == 0
        if pad:
            latents_local = F.pad(latents_local, (0, 0, 0, pad))
            cos, sin = F.pad(cos, (0, 0, 0, pad)), F.pad(sin, (0, 0, 0, pad))
        num_image_keys = seq_len if pad else None
        if sp:
            local = (seq_len + pad) // self.tp_size
            latents_local = latents_local[:, self.tp_rank * local : (self.tp_rank + 1) * local]
        hidden_states = self.img_in(latents_local.to(dtype))

        temb = self.time_text_embed(timestep, dtype)
        scale1, gate1, scale2, gate2 = self._modulation(temb)
        mul1, mul2 = 1 + scale1, 1 + scale2
        gate1, gate2 = gate1.tanh(), gate2.tanh()
        mul_out = 1 + self.norm_out.linear(F.silu(temb).to(dtype)).unsqueeze(1)

        # Every block shares one modulation, so each block's MLP residual fuses with the
        # next block's attention norm, and the last one with the output norm:
        #   h, m = LN(h + gate * sublayer_out) * mul
        modulated = _adaln(hidden_states, mul1, eps)
        last = len(self.transformer_blocks) - 1
        for i, block in enumerate(self.transformer_blocks):
            attn = block.attention_step(
                modulated, cos, sin, text_k[i], text_v[i], kv_len, sp, self.cp_group, num_image_keys
            )
            hidden_states, modulated = _adaln(attn, mul2, eps, residual=hidden_states, gate=gate1)
            mlp = block.img_mlp(modulated, sp)
            hidden_states, modulated = _adaln(
                mlp, mul_out if i == last else mul1, eps, residual=hidden_states, gate=gate2
            )
        noise_pred = self.proj_out(modulated)
        if sp:
            noise_pred = sp_all_gather_seq(noise_pred, self.tp_group, self.tp_size)
        if pad:
            noise_pred = noise_pred[:, :seq_len]
        if self.cp_group is not None:
            noise_pred = self.cp_group.all_gather(noise_pred.contiguous(), dim=1)
        # FlowMatchEulerDiscreteScheduler.step: float32 update, cast back to the
        # model output dtype.
        updated = latents.float() + dt.float() * noise_pred.float()
        return updated.to(noise_pred.dtype)

    def compile(self, *args, prefix_options=None, step_options=None, **kwargs):
        self._compiled_prefix = torch.compile(
            self.forward_prefix, *args, options=prefix_options, **kwargs
        )
        edit_options = {**(prefix_options or {})}
        edit_options["model_name"] = f"{edit_options.get('model_name', 'prefix')}_edit"
        self._compiled_prefix_edit = torch.compile(
            self.forward_prefix_edit, *args, options=edit_options, **kwargs
        )
        self._compiled_step = torch.compile(
            self.forward_step, *args, options=step_options, **kwargs
        )

    def prefix_edit(self, *args):
        fn = getattr(self, "_compiled_prefix_edit", None) or self.forward_prefix_edit
        return fn(*args)

    def prefix(self, *args):
        fn = self._compiled_prefix if self._compiled_prefix is not None else self.forward_prefix
        return fn(*args)

    def step(self, *args):
        fn = self._compiled_step if self._compiled_step is not None else self.forward_step
        return fn(*args)

    def load_weights(self, model_path: str) -> None:
        """Load the TP-sharded blocks and the replicated embedders/heads."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        mappings: dict[str, str | list[str]] = {}
        for i in range(self.config.num_layers):
            src = f"transformer_blocks.{i}."
            mappings[f"{src}attn.qkv_proj_weight"] = [
                f"{src}attn.to_q.weight",
                f"{src}attn.to_k.weight",
                f"{src}attn.to_v.weight",
            ]
            mappings[f"{src}attn.o_proj_weight"] = f"{src}attn.to_out.0.weight"
            mappings[f"{src}attn.norm_q_weight"] = f"{src}attn.norm_q.weight"
            mappings[f"{src}attn.norm_k_weight"] = f"{src}attn.norm_k.weight"
            mappings[f"{src}img_mlp.gate_proj_weight"] = f"{src}img_mlp.gate_layer.weight"
            mappings[f"{src}img_mlp.up_proj_weight"] = f"{src}img_mlp.proj.weight"
            mappings[f"{src}img_mlp.down_proj_weight"] = f"{src}img_mlp.out.weight"

        dtype = self.dtype
        for _, param in self.named_parameters():
            set_weight_loader(param, _cast_loader(get_weight_loader(param), dtype))

        checkpoint = SafetensorsCheckpoint(model_path)
        result = checkpoint.load_sharded(
            rank=tp_rank,
            world_size=tp_size,
            model=self,
            mappings=mappings,
            device=torch.device("cpu"),
        )
        self.load_state_dict(result.state_dict, strict=True, assign=True)
        logger.info(
            "Loaded Qwen-Image 2.1 DiT: %d tensors (rank=%d, tp=%d)",
            len(result.state_dict),
            tp_rank,
            tp_size,
        )


def _cast_loader(loader, dtype: torch.dtype):
    def transform(slices, rank):
        return loader.load(slices, rank).to(dtype)

    return SafetensorsWeightLoader(transform=transform)
