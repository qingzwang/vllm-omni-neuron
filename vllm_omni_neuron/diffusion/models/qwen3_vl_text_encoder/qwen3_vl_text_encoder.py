# SPDX-License-Identifier: Apache-2.0
"""Standalone Neuron Qwen3-VL text encoder (language model only) with TP sharding.

Qwen-Image 2.1 conditions its DiT on the hidden states of a Qwen3-VL language model.
For text-to-image there are no vision tokens, so only the text decoder runs:

  - Token embedding (looked up on the host; see ``NeuronQwen3VLTextEncoder.forward``)
  - N decoder layers, each with:
    - RMSNorm → GQA self-attention (per-head QK-norm, RoPE, causal) → residual
    - RMSNorm → SwiGLU MLP → residual
  - No final norm: the DiT was trained on the last decoder layer's output *before*
    the language model's final RMSNorm (the diffusers pipeline strips it with a hook).

With text-only input Qwen3-VL's interleaved MRoPE assigns the same position to its
temporal, height and width sections, so it reduces to standard 1-D rotate-half RoPE.

The graph runs at one static sequence length. Prompts are right-padded to it: the
attention is causal, so padded positions never influence the valid ones, and the
caller ignores the padded rows of the output.

TP strategy:
  - Q heads sharded across ranks; KV heads sharded (or replicated when TP > num KV heads)
  - MLP intermediate dim sharded across ranks
  - All-reduce after the output projection and after the MLP down projection
"""

import json
import logging
import os
from collections import namedtuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import vllm_neuron.functional as NF
from safetensors import safe_open
from transformers import AutoConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

logger = logging.getLogger(__name__)

_TextEncoderOutput = namedtuple("_TextEncoderOutput", ["last_hidden_state"])

# Checkpoint key prefix of the language model inside Qwen3VLForConditionalGeneration.
_CKPT_PREFIX = "model.language_model."

# Largest fused per-rank QKV width the NKI QKV kernel fits in SBUF (see WanSelfAttention).
_MAX_NKI_QKV_SIZE = 2048


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def text_rope_cos_sin(
    seq_len: int, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half RoPE tables ``[seq_len, head_dim]`` for positions ``0..seq_len-1``.

    Matches ``Qwen3VLTextRotaryEmbedding`` for text-only input: computed in float32,
    then cast to the activation dtype.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.arange(seq_len, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class Qwen3VLRMSNorm(nn.Module):
    """RMSNorm normalized in float32; the weight multiplies the input-dtype result."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class NeuronQwen3VLTextAttention(nn.Module):
    """Causal GQA self-attention with per-head QK-norm and TP head sharding."""

    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        if num_heads % self.tp_size != 0:
            raise ValueError(f"num_attention_heads={num_heads} not divisible by TP={self.tp_size}")
        self.num_heads = num_heads // self.tp_size
        if self.tp_size >= num_kv_heads:
            if self.tp_size % num_kv_heads != 0:
                raise ValueError(f"TP={self.tp_size} not a multiple of num_kv_heads={num_kv_heads}")
            self.num_kv_heads = 1
            num_kv_replicas = self.tp_size // num_kv_heads
        else:
            if num_kv_heads % self.tp_size != 0:
                raise ValueError(f"num_kv_heads={num_kv_heads} not divisible by TP={self.tp_size}")
            self.num_kv_heads = num_kv_heads // self.tp_size
            num_kv_replicas = 1
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        qkv_size = q_size + 2 * kv_size
        self.qkv_split = [q_size, q_size + kv_size]
        self._use_nf_qkv_proj = qkv_size <= _MAX_NKI_QKV_SIZE

        self.qkv_proj_weight = nn.Parameter(torch.empty(config.hidden_size, qkv_size, dtype=dtype))
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=q_size,
                kv_size=kv_size,
                shard_dim=1,
                num_shards=self.tp_size,
                is_storage_transposed=True,
                num_kv_replicas=num_kv_replicas,
            ),
        )
        self.o_proj_weight = nn.Parameter(torch.empty(q_size, config.hidden_size, dtype=dtype))
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=q_size,
                num_shards=self.tp_size,
                is_storage_transposed=True,
            ),
        )
        self.q_norm = Qwen3VLRMSNorm(self.head_dim, config.rms_norm_eps, dtype)
        self.k_norm = Qwen3VLRMSNorm(self.head_dim, config.rms_norm_eps, dtype)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """``hidden_states`` ``[1, L, H]``, ``cos``/``sin`` ``[L, head_dim]`` → ``[1, L, H]``."""
        seq_len = hidden_states.shape[1]
        if self._use_nf_qkv_proj:
            qkv = NF.qkv_proj(hidden_states, self.qkv_proj_weight, bias=None)
        else:
            qkv = torch.matmul(hidden_states, self.qkv_proj_weight)
        q, k, v = torch.tensor_split(qkv[0], self.qkv_split, dim=-1)

        # [L, heads * D] -> [heads, L, D]; QK-norm is per head, before RoPE.
        q = self.q_norm(q.view(seq_len, self.num_heads, self.head_dim)).transpose(0, 1)
        k = self.k_norm(k.view(seq_len, self.num_kv_heads, self.head_dim)).transpose(0, 1)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)

        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=0)
            v = v.repeat_interleave(self.num_kv_groups, dim=0)

        # [heads, D, L] (d-major) is the layout NF.o_proj consumes.
        attn = NF.flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=self.scale,
            causal_mask=True,
            tp_q=True,
            tp_k=True,
            tp_out=True,
        )
        output = NF.o_proj(attn.unsqueeze(0), self.o_proj_weight)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output


class NeuronQwen3VLTextMLP(nn.Module):
    """SwiGLU MLP ``down(silu(gate(x)) * up(x))`` with TP intermediate sharding."""

    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group
        inner = config.intermediate_size // self.tp_size

        self.gate_proj_weight = nn.Parameter(torch.empty(config.hidden_size, inner, dtype=dtype))
        self.up_proj_weight = nn.Parameter(torch.empty(config.hidden_size, inner, dtype=dtype))
        self.down_proj_weight = nn.Parameter(torch.empty(inner, config.hidden_size, dtype=dtype))
        col = sharding_weight_loader(
            shard_dim=1, shard_size=inner, num_shards=self.tp_size, is_storage_transposed=True
        )
        row = sharding_weight_loader(
            shard_dim=0, shard_size=inner, num_shards=self.tp_size, is_storage_transposed=True
        )
        set_weight_loader(self.gate_proj_weight, col)
        set_weight_loader(self.up_proj_weight, col)
        set_weight_loader(self.down_proj_weight, row)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        output = NF.mlp(
            hidden_states.reshape(-1, shape[-1]),
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        ).reshape(shape)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output


class NeuronQwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.input_layernorm = Qwen3VLRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.self_attn = NeuronQwen3VLTextAttention(config, dtype)
        self.post_attention_layernorm = Qwen3VLRMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype
        )
        self.mlp = NeuronQwen3VLTextMLP(config, dtype)

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cos, sin
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class NeuronQwen3VLTextEncoder(nn.Module):
    """Qwen3-VL language model returning the last decoder layer's (pre-norm) hidden states.

    Constructed from the ``text_encoder`` config; weights are loaded by
    :meth:`load_weights`. The token embedding table stays on the host (it is not a
    registered parameter, so ``.to(device)`` leaves it there): the lookup is a gather
    of ``seq_len`` rows, and keeping the 1.2 GB table off-device saves that HBM on
    every rank.

    Args:
        model_path: Pipeline root (contains the ``text_encoder/`` subfolder).
        seq_len: Static sequence length of the compiled graph.
        dtype: Activation and weight dtype.
    """

    def __init__(self, model_path: str, seq_len: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        local_files_only = os.path.isdir(model_path)
        full_config = AutoConfig.from_pretrained(
            model_path, subfolder="text_encoder", local_files_only=local_files_only
        )
        config = getattr(full_config, "text_config", full_config)
        self.config = config
        self.seq_len = seq_len
        self._dtype = dtype
        self._compiled = None

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.layers = nn.ModuleList(
            [NeuronQwen3VLTextDecoderLayer(config, dtype) for _ in range(config.num_hidden_layers)]
        )
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = config.rope_parameters["rope_theta"]
        self._rope_theta = rope_theta
        cos, sin = text_rope_cos_sin(seq_len, config.head_dim, rope_theta, dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._embed_tokens: torch.Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def load_weights(self, text_encoder_path: str) -> None:
        """Load the TP-sharded decoder layers and the (host-resident) embedding table."""
        mappings: dict[str, str | list[str]] = {}
        for i in range(len(self.layers)):
            src = f"{_CKPT_PREFIX}layers.{i}."
            dst = f"layers.{i}."
            mappings[f"{dst}self_attn.qkv_proj_weight"] = [
                f"{src}self_attn.q_proj.weight",
                f"{src}self_attn.k_proj.weight",
                f"{src}self_attn.v_proj.weight",
            ]
            for name in ("o_proj", "q_norm", "k_norm"):
                param = "o_proj_weight" if name == "o_proj" else f"{name}.weight"
                mappings[f"{dst}self_attn.{param}"] = f"{src}self_attn.{name}.weight"
            for name in ("gate_proj", "up_proj", "down_proj"):
                mappings[f"{dst}mlp.{name}_weight"] = f"{src}mlp.{name}.weight"
            for name in ("input_layernorm", "post_attention_layernorm"):
                mappings[f"{dst}{name}.weight"] = f"{src}{name}.weight"

        checkpoint = SafetensorsCheckpoint(text_encoder_path)
        result = checkpoint.load_sharded(
            rank=self.tp_rank,
            world_size=self.tp_size,
            model=self,
            mappings=mappings,
            device=torch.device("cpu"),
        )
        state_dict = {k: v.to(self._dtype) for k, v in result.state_dict.items()}
        self.load_state_dict(state_dict, strict=False, assign=True)
        self._embed_tokens = self._load_embedding(text_encoder_path)
        logger.info(
            "Loaded Qwen3-VL text encoder: %d tensors (rank=%d, tp=%d)",
            len(state_dict),
            self.tp_rank,
            self.tp_size,
        )

    def _load_embedding(self, text_encoder_path: str) -> torch.Tensor:
        key = f"{_CKPT_PREFIX}embed_tokens.weight"
        index_path = os.path.join(text_encoder_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                file_name = json.load(f)["weight_map"][key]
        else:
            file_name = "model.safetensors"
        with safe_open(os.path.join(text_encoder_path, file_name), framework="pt") as f:
            return f.get_tensor(key).to(self._dtype)

    def _encode(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.rope_cos, self.rope_sin)
        return hidden_states

    def _encode_multimodal(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        deepstack: torch.Tensor,
    ) -> torch.Tensor:
        """Text+image pass: explicit (MRoPE) tables and DeepStack residuals.

        ``deepstack`` ``[k, 1, L, H]`` holds each DeepStack feature at the image
        positions and zeros elsewhere; layer ``i < k`` adds row ``i`` to its output, as
        ``Qwen3VLTextModel._deepstack_process`` does at the visual positions.
        """
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cos, sin)
            if i < deepstack.shape[0]:
                hidden_states = hidden_states + deepstack[i]
        return hidden_states

    def compile(self, *args, **kwargs):
        self._compiled = torch.compile(self._encode, *args, **kwargs)
        self._compiled_multimodal = torch.compile(self._encode_multimodal, *args, **kwargs)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Host token embedding ``[1, L]`` → ``[1, L, H]``."""
        return F.embedding(input_ids.cpu(), self._embed_tokens)

    def mrope_cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleaved-MRoPE tables ``[L, D]`` for ``(t, h, w)`` positions ``[3, L]``.

        As ``Qwen3VLTextRotaryEmbedding``: frequency ``j`` takes its position from the
        height axis when ``j % 3 == 1``, the width axis when ``j % 3 == 2`` (within the
        sections), and the temporal axis otherwise.
        """
        head_dim = self.config.head_dim
        rope = getattr(self.config, "rope_parameters", None) or self.config.rope_scaling
        theta = self._rope_theta
        section = rope.get("mrope_section", [24, 20, 20])
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = position_ids.float()[..., None] * inv_freq  # [3, L, D/2]
        mixed = freqs[0].clone()
        for axis in (1, 2):
            idx = slice(axis, section[axis] * 3, 3)
            mixed[..., idx] = freqs[axis][..., idx]
        emb = torch.cat([mixed, mixed], dim=-1)
        return emb.cos().to(self._dtype), emb.sin().to(self._dtype)

    def forward(self, input_ids: torch.Tensor, device: torch.device) -> _TextEncoderOutput:
        """Encode ``input_ids`` ``[1, seq_len]`` (host tensor) into ``[1, seq_len, H]`` on ``device``."""
        if input_ids.shape[-1] != self.seq_len:
            raise ValueError(f"expected {self.seq_len} tokens, got {input_ids.shape[-1]}")
        inputs_embeds = self.embed(input_ids).to(device)
        fn = self._compiled if self._compiled is not None else self._encode
        return _TextEncoderOutput(fn(inputs_embeds))

    def forward_multimodal(self, inputs_embeds, cos, sin, deepstack) -> _TextEncoderOutput:
        """Device tensors from the caller (see :meth:`_encode_multimodal`), any length."""
        fn = getattr(self, "_compiled_multimodal", None) or self._encode_multimodal
        return _TextEncoderOutput(fn(inputs_embeds, cos, sin, deepstack))
