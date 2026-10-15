# SPDX-License-Identifier: Apache-2.0
"""Standalone Neuron Qwen3-VL vision tower with TP sharding (single image).

Mirrors ``transformers`` ``Qwen3VLVisionModel`` for one image:

  - Patch embedding (the stride == kernel Conv3d is a linear map of each flattened patch)
  - Bilinearly interpolated learned position embedding (gathered on the host)
  - N ViT blocks: LayerNorm → bidirectional MHA with 2-D RoPE → residual,
    LayerNorm → GELU(tanh) MLP → residual
  - DeepStack: after selected blocks a patch merger (2x2 merge + MLP) emits features
    that the language model adds to its first layers' hidden states
  - Final patch merger: tokens for the language model's image placeholders

Everything index-shaped — the position-embedding interpolation, the 2-D RoPE tables —
is computed on the host with the ``transformers`` helpers, so it matches the reference
exactly; the compiled graph is pure tensor math at a static patch count (one graph per
input image grid).

TP strategy:
  - Attention heads and the MLP intermediate dim sharded across ranks
  - All-reduce after the attention output and MLP down projections (biases added once)
  - Patch embedding and mergers replicated
"""

import json
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import vllm_neuron.functional as NF
from safetensors import safe_open
from transformers import AutoConfig
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    get_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

logger = logging.getLogger(__name__)

_CKPT_PREFIX = "model.visual."


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Affine LayerNorm computed in float32, returned in ``x``'s dtype."""
    return F.layer_norm(
        x.float(), norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps
    ).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _col_loader(shard_size: int, num_shards: int, shard_dim: int):
    return sharding_weight_loader(shard_dim=shard_dim, shard_size=shard_size, num_shards=num_shards)


def _fused_qkv_loader(dim: int, shard: int, num_shards: int):
    """Shard each of Q, K, V (rows of the fused ``[3 * dim, ...]`` tensor) by heads."""

    def transform(slices, rank):
        tensor = slices[0][:]
        parts = tensor.view(3, dim, *tensor.shape[1:])
        start = (rank % num_shards) * shard
        return parts[:, start : start + shard].reshape(3 * shard, *tensor.shape[1:])

    return SafetensorsWeightLoader(transform=transform)


class NeuronQwen3VLVisionBlock(nn.Module):
    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group
        dim = config.hidden_size
        if config.num_heads % self.tp_size or config.intermediate_size % self.tp_size:
            raise ValueError(f"vision tower does not shard over TP={self.tp_size}")
        self.num_heads = config.num_heads // self.tp_size
        self.head_dim = dim // config.num_heads
        inner = self.num_heads * self.head_dim
        mlp = config.intermediate_size // self.tp_size

        self.norm1 = nn.LayerNorm(dim, eps=1e-6, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, dtype=dtype)
        # nn.Linear layout ([out, in]); the TP shards are rows of qkv/fc1, columns of proj/fc2.
        self.qkv_weight = nn.Parameter(torch.empty(3 * inner, dim, dtype=dtype))
        self.qkv_bias = nn.Parameter(torch.empty(3 * inner, dtype=dtype))
        self.proj_weight = nn.Parameter(torch.empty(dim, inner, dtype=dtype))
        self.proj_bias = nn.Parameter(torch.empty(dim, dtype=dtype))
        self.fc1_weight = nn.Parameter(torch.empty(mlp, dim, dtype=dtype))
        self.fc1_bias = nn.Parameter(torch.empty(mlp, dtype=dtype))
        self.fc2_weight = nn.Parameter(torch.empty(dim, mlp, dtype=dtype))
        self.fc2_bias = nn.Parameter(torch.empty(dim, dtype=dtype))
        for param in (self.qkv_weight, self.qkv_bias):
            set_weight_loader(param, _fused_qkv_loader(dim, inner, self.tp_size))
        set_weight_loader(self.proj_weight, _col_loader(inner, self.tp_size, 1))
        set_weight_loader(self.fc1_weight, _col_loader(mlp, self.tp_size, 0))
        set_weight_loader(self.fc1_bias, _col_loader(mlp, self.tp_size, 0))
        set_weight_loader(self.fc2_weight, _col_loader(mlp, self.tp_size, 1))
        self.scale = self.head_dim**-0.5

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tp_group)
        return x

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``x`` ``[N, C]``, ``cos``/``sin`` ``[N, head_dim]`` float32."""
        seq_len = x.shape[0]
        qkv = F.linear(_layer_norm(x, self.norm1), self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.view(seq_len, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3).unbind(0)
        # transformers apply_rotary_pos_emb_vision: rotate in float32, cast back.
        q = (q.float() * cos + _rotate_half(q.float()) * sin).to(x.dtype)
        k = (k.float() * cos + _rotate_half(k.float()) * sin).to(x.dtype)
        attn = NF.flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
        )  # [heads, N, D]
        attn = attn.transpose(0, 1).reshape(seq_len, -1)
        x = x + (self._all_reduce(F.linear(attn, self.proj_weight)) + self.proj_bias)
        h = F.gelu(
            F.linear(_layer_norm(x, self.norm2), self.fc1_weight, self.fc1_bias), approximate="tanh"
        )
        return x + (self._all_reduce(F.linear(h, self.fc2_weight)) + self.fc2_bias)


class NeuronQwen3VLPatchMerger(nn.Module):
    """2x2 spatial merge followed by an MLP (replicated)."""

    def __init__(self, config, use_postshuffle_norm: bool, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6, dtype=dtype
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        x = _layer_norm(x, self.norm).view(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class NeuronQwen3VLVisionEncoder(nn.Module):
    """Qwen3-VL vision tower returning image tokens and DeepStack features.

    The patch-embedding position table stays on the host (like the language model's
    token embedding): it is gathered with the interpolation indices there and the
    result is a graph input.
    """

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        local_files_only = os.path.isdir(model_path)
        config = AutoConfig.from_pretrained(
            model_path, subfolder="text_encoder", local_files_only=local_files_only
        ).vision_config
        self.config = config
        self._dtype = dtype
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.merge = config.spatial_merge_size
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patch_proj = nn.Linear(patch_dim, config.hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList(
            [NeuronQwen3VLVisionBlock(config, dtype) for _ in range(config.depth)]
        )
        self.merger = NeuronQwen3VLPatchMerger(config, use_postshuffle_norm=False, dtype=dtype)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                NeuronQwen3VLPatchMerger(config, use_postshuffle_norm=True, dtype=dtype)
                for _ in self.deepstack_visual_indexes
            ]
        )
        head_dim = config.hidden_size // config.num_heads
        spatial_dim = head_dim // 2
        rope = getattr(config, "rope_parameters", None) or {}
        theta = rope.get("rope_theta", 10000.0)
        self._inv_freq = 1.0 / (
            theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self._pos_embed: torch.Tensor | None = None
        self._compiled = None

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def load_weights(self, text_encoder_path: str) -> None:
        mappings: dict[str, str] = {
            "patch_proj.weight": f"{_CKPT_PREFIX}patch_embed.proj.weight",
            "patch_proj.bias": f"{_CKPT_PREFIX}patch_embed.proj.bias",
        }
        for i in range(len(self.blocks)):
            src, dst = f"{_CKPT_PREFIX}blocks.{i}.", f"blocks.{i}."
            for name, key in (
                ("qkv", "attn.qkv"),
                ("proj", "attn.proj"),
                ("fc1", "mlp.linear_fc1"),
                ("fc2", "mlp.linear_fc2"),
            ):
                mappings[f"{dst}{name}_weight"] = f"{src}{key}.weight"
                mappings[f"{dst}{name}_bias"] = f"{src}{key}.bias"
        for name, _ in self.named_parameters():
            if name.startswith(("merger.", "deepstack_merger_list.")) or ".norm" in name:
                mappings.setdefault(name, f"{_CKPT_PREFIX}{name}")
        # The patch embedding is a Conv3d in the checkpoint: flatten it to [out, C*T*P*P].
        conv = get_weight_loader(self.patch_proj.weight)
        set_weight_loader(
            self.patch_proj.weight,
            SafetensorsWeightLoader(
                transform=lambda s, r: conv.load(s, r).reshape(self.config.hidden_size, -1)
            ),
        )
        checkpoint = SafetensorsCheckpoint(text_encoder_path)
        result = checkpoint.load_sharded(
            rank=self.tp_rank,
            world_size=self.tp_size,
            model=self,
            mappings=mappings,
            device=torch.device("cpu"),
        )
        state_dict = {k: v.to(self._dtype) for k, v in result.state_dict.items()}
        self.load_state_dict(state_dict, strict=True, assign=True)
        index = os.path.join(text_encoder_path, "model.safetensors.index.json")
        key = f"{_CKPT_PREFIX}pos_embed.weight"
        with open(index) as f:
            file_name = json.load(f)["weight_map"][key]
        with safe_open(os.path.join(text_encoder_path, file_name), framework="pt") as f:
            self._pos_embed = f.get_tensor(key).to(self._dtype)
        logger.info("Loaded Qwen3-VL vision tower (rank=%d, tp=%d)", self.tp_rank, self.tp_size)

    def host_inputs(self, grid_thw: torch.Tensor):
        """Position embedding ``[N, C]`` and 2-D RoPE ``cos``/``sin`` ``[N, head_dim]``."""
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self.merge,
        )
        # As in transformers: the bf16 table times float32 weights, summed in float32.
        pos = (F.embedding(indices, self._pos_embed) * weights[:, :, None]).sum(1).to(self._dtype)
        position_ids = get_vision_position_ids(grid_thw, self.merge)
        freqs = position_ids[..., None].float() * self._inv_freq
        freqs = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return pos, freqs.cos(), freqs.sin()

    def _encode(self, pixel_values, pos_embeds, cos, sin):
        x = F.linear(pixel_values.to(self._dtype), self.patch_proj.weight, self.patch_proj.bias)
        x = x + pos_embeds
        deepstack = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if i in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)]
                deepstack.append(merger(x))
        return self.merger(x), torch.stack(deepstack)

    def compile(self, *args, **kwargs):
        self._compiled = torch.compile(self._encode, *args, **kwargs)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, device):
        """Host ``pixel_values`` ``[N, C*T*P*P]`` → device ``(tokens [N/4, D], deepstack [3, N/4, D])``."""
        pos, cos, sin = self.host_inputs(grid_thw)
        fn = self._compiled if self._compiled is not None else self._encode
        return fn(pixel_values.to(device), pos.to(device), cos.to(device), sin.to(device))
