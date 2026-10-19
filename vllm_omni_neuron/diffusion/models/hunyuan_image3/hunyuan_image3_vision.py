# SPDX-License-Identifier: Apache-2.0
"""SigLIP2 (so400m, patch 16, NaFlex) vision tower and aligner of HunyuanImage-3.0-Instruct.

Mirrors ``siglip2.py`` / ``LightProjector`` of the checkpoint for one condition image:

  - patch embedding (Linear 768 -> 1152) of the processor's 1024 zero-padded patches
  - plus the 16x16 learned position table resized (bilinear, antialiased) to the
    image's patch grid; padding slots get the first row (resized on the host)
  - 27 pre-LN encoder layers: bidirectional 16-head attention whose keys are the
    valid patches, GELU(tanh) MLP 4304
  - post LayerNorm, then the aligner Linear 1152 -> 4096, GELU, Linear 4096 -> 4096

All 1024 outputs (padding slots included) are the image's SigLIP2 tokens in the
language model. The tower is small (0.4B parameters, 1024 tokens), so it is replicated
on every rank; one compiled graph serves every image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm_neuron.functional as NF

_PREFIX = "vision_model."


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    return F.layer_norm(
        x.float(), norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps
    ).to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dtype: torch.dtype):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, dtype=dtype)
        self.v_proj = nn.Linear(dim, dim, dtype=dtype)
        self.out_proj = nn.Linear(dim, dim, dtype=dtype)

    def forward(self, x: torch.Tensor, bound_min, bound_max) -> torch.Tensor:
        tokens = x.shape[0]

        def heads(t):
            return t.view(tokens, self.heads, self.head_dim).transpose(0, 1).contiguous()

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        out = NF.flash_attention(
            q,
            k,
            v,
            scale=self.head_dim**-0.5,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=bound_min.expand(self.heads, -1, -1).contiguous(),
            bound_max=bound_max.expand(self.heads, -1, -1).contiguous(),
        )
        return self.out_proj(out.transpose(0, 1).reshape(tokens, -1))


class _Layer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps, dtype=dtype)
        self.self_attn = _Attention(dim, heads, dtype)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps, dtype=dtype)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, mlp, dtype=dtype)
        self.mlp.fc2 = nn.Linear(mlp, dim, dtype=dtype)

    def forward(self, x, bound_min, bound_max):
        x = x + self.self_attn(_layer_norm(x, self.layer_norm1), bound_min, bound_max)
        h = F.gelu(self.mlp.fc1(_layer_norm(x, self.layer_norm2)), approximate="tanh")
        return x + self.mlp.fc2(h)


class HunyuanImage3VisionEncoder(nn.Module):
    """SigLIP2 tower + aligner: ``[1024, 768]`` patches -> ``[1024, 4096]`` tokens."""

    def __init__(self, vit_config: dict, aligner_config: dict, dtype: torch.dtype):
        super().__init__()
        dim = vit_config["hidden_size"]
        self._dtype = dtype
        self.patch_dim = vit_config["num_channels"] * vit_config["patch_size"] ** 2
        self.grid = int(vit_config["num_patches"] ** 0.5)
        self.patch_embedding = nn.Linear(self.patch_dim, dim, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                _Layer(
                    dim,
                    vit_config["num_attention_heads"],
                    vit_config["intermediate_size"],
                    vit_config["layer_norm_eps"],
                    dtype,
                )
                for _ in range(vit_config["num_hidden_layers"])
            ]
        )
        self.post_layernorm = nn.LayerNorm(dim, eps=vit_config["layer_norm_eps"], dtype=dtype)
        n_embed = aligner_config["n_embed"]
        self.aligner_in = nn.Linear(dim, n_embed, dtype=dtype)
        self.aligner_out = nn.Linear(n_embed, n_embed, dtype=dtype)
        self._position_table: torch.Tensor | None = None  # host [grid, grid, dim]
        self._compiled = None

    def load_weights(self, ckpt) -> None:
        state = {}
        for name, _ in self.named_parameters():
            if name.startswith("aligner_in."):
                key = "vision_aligner.layers.0." + name.split(".", 1)[1]
            elif name.startswith("aligner_out."):
                key = "vision_aligner.layers.2." + name.split(".", 1)[1]
            elif name.startswith("patch_embedding."):
                key = f"{_PREFIX}embeddings.{name}"
            elif name.startswith("layers."):
                key = f"{_PREFIX}encoder.{name}"
            else:
                key = f"{_PREFIX}{name}"
            state[name] = ckpt.get(key).to(self._dtype)
        self.load_state_dict(state, strict=True, assign=True)
        table = ckpt.get(f"{_PREFIX}embeddings.position_embedding.weight")
        self._position_table = table.to(self._dtype).view(self.grid, self.grid, -1)

    def positions(self, grid_hw: tuple[int, int], slots: int) -> torch.Tensor:
        """Host ``[slots, dim]``: the table resized to ``grid_hw``, padding = first row."""
        h, w = grid_hw
        table = self._position_table.permute(2, 0, 1)[None].float()
        resized = F.interpolate(
            table, size=(h, w), mode="bilinear", align_corners=False, antialias=True
        )
        resized = resized.reshape(table.shape[1], h * w).t().to(self._dtype)
        out = resized[:1].expand(slots, -1).clone()
        out[: h * w] = resized
        return out

    def _encode(self, pixels, positions, bound_min, bound_max):
        x = self.patch_embedding(pixels.to(self._dtype)) + positions
        for layer in self.layers:
            x = layer(x, bound_min, bound_max)
        x = _layer_norm(x, self.post_layernorm)
        return self.aligner_out(F.gelu(self.aligner_in(x)))

    def compile(self, *args, **kwargs):
        self._compiled = torch.compile(self._encode, *args, **kwargs)

    def forward(self, pixels: torch.Tensor, grid_hw: tuple[int, int], device) -> torch.Tensor:
        """``pixels`` ``[1024, 768]`` (host) -> ``[1024, 4096]`` on ``device``."""
        slots = pixels.shape[0]
        valid = grid_hw[0] * grid_hw[1]
        # Every query attends to the valid patches (key padding mask).
        bound_min = torch.zeros(1, slots, 1, dtype=torch.int32)
        bound_max = torch.full((1, slots, 1), valid, dtype=torch.int32)
        fn = self._compiled if self._compiled is not None else self._encode
        return fn(
            pixels.to(device),
            self.positions(grid_hw, slots).to(device),
            bound_min.to(device),
            bound_max.to(device),
        )
