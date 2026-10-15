# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Single-frame port of diffusers' AutoencoderKLQwenImage21
# (huggingface/diffusers@916a5791, src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py),
# which is not in the diffusers release pinned by vllm-omni.
#
# Neuron-specific changes
# 1. Single frame only, no diffusers tiling. The encoder (for edit condition images)
#    returns the posterior mode, as the edit pipeline's ``sample_mode="argmax"`` does.
# 2. Single-frame encode/decode without the feature cache. An image is one latent frame, which
#    the reference decodes as its first chunk: every causal conv sees no cache, every
#    temporal upsample takes the "Rep" path (time_conv skipped) and every DupUp3D
#    shortcut keeps only its last frame. That path is reproduced statically, so the whole
#    decoder is one fullgraph-compilable graph.
# 3. QwenImage21RMS_norm: F.normalize replaced with x * rsqrt(sum(x^2)) (no NaN guard),
#    as in the Neuron Wan VAE.
# 4. QwenImage21Upsample: nn.Upsample(nearest-exact, 2x) replaced with repeat_interleave,
#    which is identical for integer scale factors and lowers to bulk copies.
# 5. Mid-block attention as explicit query-chunked matmuls (the compiled SDPA returns NaN
#    with the 1152-wide head), and the decoder compiled per stage.
# 6. Exact tiled decode of the up blocks above 64x64 latents, spread over ranks
#    (decoder_tiles), which bounds graph size and device memory at high resolution.

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

logger = logging.getLogger(__name__)

# Queries per block of the mid-block attention (see _query_chunked_attention).
_ATTN_QUERY_CHUNK = 512
# Tiled decode (see decoder_tiles): window side, kept rows/columns per tile, and the
# margin left on each side. The margin must cover the up blocks' 13-pixel receptive field.
_DECODE_TILE = 64
_DECODE_STRIDE = 32
_DECODE_HALO = (_DECODE_TILE - _DECODE_STRIDE) // 2


class QwenImage21DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        factor = factor_t * factor_s * factor_s
        assert out_channels * factor % in_channels == 0
        self.repeats = out_channels * factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = x.shape
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(b, self.out_channels, self.factor_t, self.factor_s, self.factor_s, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, self.out_channels, t * self.factor_t, h * self.factor_s, w * self.factor_s)
        # First chunk: keep the last of the factor_t duplicated frames.
        return x[:, :, self.factor_t - 1 :, :, :]


class QwenImage21CausalConv3d(nn.Conv2d):
    """The reference's single-frame causal conv: a 2-D conv over the folded frame."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        self.padding = (0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.squeeze(2), self._padding)
        return super().forward(x).unsqueeze(2)


class QwenImage21RMS_norm(nn.Module):
    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        # F.normalize(x) * sqrt(C) == x * rsqrt(sum(x^2)) * sqrt(C), in float32.
        inv_norm = x.float().pow(2).sum(dim, keepdim=True).clamp(min=1e-24).rsqrt()
        normalized = (x.float() * inv_norm).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class QwenImage21Upsample(nn.Module):
    """2x nearest upsampling of ``[N, C, H, W]``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


class QwenImage21AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.view(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, self.out_channels, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(dim=2)


class QwenImage21Resample(nn.Module):
    """2x spatial resample. ``time_conv`` exists for checkpoint compatibility only: a
    single frame never reaches it (the reference skips it on the first chunk)."""

    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None):
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            upsample_out_dim = upsample_out_dim or dim // 2
            self.resample = nn.Sequential(
                QwenImage21Upsample(), nn.Conv2d(dim, upsample_out_dim, 3, padding=1)
            )
            if mode == "upsample3d":
                self.time_conv = QwenImage21CausalConv3d(dim, dim * 2, (1, 1), padding=(0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=2)
            )
            if mode == "downsample3d":
                self.time_conv = QwenImage21CausalConv3d(dim, dim, (1, 1), stride=(1, 1))
        else:
            raise ValueError(f"unsupported resample mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)


class QwenImage21ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = QwenImage21RMS_norm(in_dim, images=False)
        self.conv1 = QwenImage21CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImage21CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImage21CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + h


def _query_chunked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-head ``softmax(q k^T / sqrt(C)) v`` over ``[N, S, C]``, in query chunks.

    The head is the full channel dim (1152 here), above the NKI attention kernel's
    head-dim limit, and the compiled SDPA decomposition returns NaN at 64x64 latents.
    Explicit matmuls with a float32 softmax, a bounded block of queries at a time, keep
    the score tile small at any resolution.
    """
    scale = q.shape[-1] ** -0.5
    k_t = k.transpose(1, 2)
    outputs = []
    for start in range(0, q.shape[1], _ATTN_QUERY_CHUNK):
        scores = torch.matmul(q[:, start : start + _ATTN_QUERY_CHUNK], k_t).float() * scale
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v))
    return torch.cat(outputs, dim=1)


class QwenImage21AttentionBlock(nn.Module):
    """Single-head spatial self-attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = QwenImage21RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        qkv = self.to_qkv(self.norm(x))
        qkv = qkv.reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = (t_.squeeze(1) for t_ in qkv.chunk(3, dim=-1))
        x = _query_chunked_attention(q, k, v)
        x = x.permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class QwenImage21MidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__()
        resnets = [QwenImage21ResidualBlock(dim, dim, dropout)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImage21AttentionBlock(dim))
            resnets.append(QwenImage21ResidualBlock(dim, dim, dropout))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x))
        return x


class QwenImage21ResidualDownBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False
    ):
        super().__init__()
        self.avg_shortcut = QwenImage21AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = QwenImage21Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(x_copy)


class QwenImage21Encoder3d(nn.Module):
    def __init__(
        self,
        in_channels=4,
        dim=96,
        z_dim=128,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        temperal_downsample=(False, True, True, True),
        dropout=0.0,
    ):
        super().__init__()
        dim_mult = list(dim_mult)
        dims = [dim * u for u in [1] + dim_mult]
        self.conv_in = QwenImage21CausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                QwenImage21ResidualDownBlock(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks,
                    temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                    down_flag=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )
        self.mid_block = QwenImage21MidBlock(dims[-1], dropout, num_layers=1)
        self.norm_out = QwenImage21RMS_norm(dims[-1], images=False)
        self.conv_out = QwenImage21CausalConv3d(dims[-1], z_dim, 3, padding=1)


class QwenImage21ResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ):
        super().__init__()
        self.avg_shortcut = (
            QwenImage21DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2)
            if up_flag
            else None
        )
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImage21ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = QwenImage21Resample(out_dim, mode=mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy)
        return x


class QwenImage21Decoder3d(nn.Module):
    def __init__(
        self,
        dim=144,
        z_dim=64,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        temperal_upsample=(True, True, True, False),
        dropout=0.0,
        out_channels: int = 4,
    ):
        super().__init__()
        dim_mult = list(dim_mult)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = QwenImage21CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImage21MidBlock(dims[0], dropout, num_layers=1)
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(dim_mult) - 1
            self.up_blocks.append(
                QwenImage21ResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                )
            )
        self.norm_out = QwenImage21RMS_norm(dims[-1], images=False)
        self.conv_out = QwenImage21CausalConv3d(dims[-1], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x))
        for up_block in self.up_blocks:
            x = up_block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class NeuronAutoencoderKLQwenImage21(ModelMixin, ConfigMixin):
    """Decoder of ``AutoencoderKLQwenImage21`` for single-frame (image) latents.

    Loads from the ``vae`` subfolder with ``from_pretrained``; the config signature
    matches the reference so the checkpoint config applies unchanged.
    """

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: int | None = 144,
        z_dim: int = 64,
        dim_mult: list[int] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        attn_scales: list[float] = (),
        temperal_downsample: list[bool] = (False, True, True, True),
        dropout: float = 0.0,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
        is_residual: bool = True,
        in_channels: int = 4,
        out_channels: int = 4,
        patch_size: int | None = None,
        scale_factor_temporal: int | None = 8,
        scale_factor_spatial: int | None = 16,
    ):
        super().__init__()
        if not is_residual or patch_size is not None or attn_scales:
            raise NotImplementedError("only the Qwen-Image 2.1 VAE configuration is supported")
        self.encoder = QwenImage21Encoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
        )
        self.quant_conv = QwenImage21CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImage21CausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImage21Decoder3d(
            dim=decoder_base_dim or base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_upsample=list(temperal_downsample)[::-1],
            dropout=dropout,
            out_channels=out_channels,
        )
        self._compiled_stages = None
        self._compiled_encoder_stages = None

    def _encode_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pixels ``[B, C, 1, H, W]`` in ``[-1, 1]`` → first down block output."""
        x = x.to(self.post_quant_conv.weight.dtype)
        return self.encoder.down_blocks[0](self.encoder.conv_in(x))

    def _encode_tail(self, x: torch.Tensor) -> torch.Tensor:
        """Last down block output → normalized latent mode (the posterior mean)."""
        enc = self.encoder
        x = enc.conv_out(F.silu(enc.norm_out(enc.mid_block(x))))
        mean = self.quant_conv(x)[:, : self.config.z_dim]
        latents_mean = torch.tensor(self.config.latents_mean, dtype=mean.dtype, device=mean.device)
        latents_std = torch.tensor(self.config.latents_std, dtype=mean.dtype, device=mean.device)
        return (mean - latents_mean.view(1, -1, 1, 1, 1)) / latents_std.view(1, -1, 1, 1, 1)

    def _encoder_stages(self):
        return [self._encode_head, *self.encoder.down_blocks[1:], self._encode_tail]

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Pixels ``[1, C, 1, H, W]`` → normalized latents ``[1, z_dim, 1, H/16, W/16]``.

        The reference's ``_encode_vae_image`` with ``sample_mode="argmax"``: one frame,
        the posterior mode, then ``(z - latents_mean) / latents_std``.
        """
        for stage in self._compiled_encoder_stages or self._encoder_stages():
            x = stage(x)
        return x

    def _decode_head(self, z: torch.Tensor) -> torch.Tensor:
        """Normalized latents → mid-block output."""
        z = z.to(self.post_quant_conv.weight.dtype)
        mean = torch.tensor(self.config.latents_mean, dtype=z.dtype, device=z.device)
        std = torch.tensor(self.config.latents_std, dtype=z.dtype, device=z.device)
        z = z * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        return self.decoder.mid_block(self.decoder.conv_in(self.post_quant_conv(z)))

    def _decode_tail(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder.conv_out(F.silu(self.decoder.norm_out(x)))
        return torch.clamp(x, min=-1.0, max=1.0)

    def _stages(self):
        return [self._decode_head, *self.decoder.up_blocks, self._decode_tail]

    def compile(self, *args, options=None, **kwargs):
        """Compile the decoder as one graph per stage (head, each up block, tail).

        One graph over the whole decoder fails SBUF allocation at 1024x1024 (the
        compiler fuses across up blocks); each stage compiles on its own, and the
        boundaries cost only an HBM round trip of the stage output.
        """
        options = dict(options or {})
        base = options.get("model_name", "qwen_image21_vae")
        self._compiled_stages = [
            torch.compile(stage, *args, options={**options, "model_name": f"{base}_{i}"}, **kwargs)
            for i, stage in enumerate(self._stages())
        ]
        self._compiled_encoder_stages = [
            torch.compile(
                stage, *args, options={**options, "model_name": f"{base}_enc{i}"}, **kwargs
            )
            for i, stage in enumerate(self._encoder_stages())
        ]

    @staticmethod
    def num_tiles(height: int, width: int) -> int:
        return len(decoder_tiles(height, width))

    def decode_image(self, z: torch.Tensor, rank: int = 0, world_size: int = 1, cpu_group=None):
        """Normalized latents ``[1, z_dim, 1, h, w]`` → pixels ``[1, C, 1, 16h, 16w]``.

        Latents up to one tile (64x64) decode whole on the calling rank. Larger latents
        decode tiled (see :func:`decoder_tiles`): every rank of ``cpu_group`` runs the
        head on the whole latent, decodes its share of the tiles, and rank 0 gathers and
        returns the assembled image (other ranks return ``None``). Tiles move through the
        host, so ``z`` and the result are host tensors in that case.
        """
        stages = self._compiled_stages or self._stages()
        height, width = z.shape[-2:]
        if height <= _DECODE_TILE and width <= _DECODE_TILE:
            for stage in stages:
                z = stage(z)
            return z

        device = self.post_quant_conv.weight.device
        head, up_stages = stages[0], stages[1:]
        features = head(z.to(device)).cpu()
        scale = self.config.scale_factor_spatial
        tiles = decoder_tiles(height, width)
        decoded = {}
        for index in range(rank, len(tiles), world_size):
            (h0, h1, hs), (w0, w1, ws) = tiles[index]
            x = features[..., hs : hs + _window(height), ws : ws + _window(width)].to(device)
            for stage in up_stages:
                x = stage(x)
            # Keep only the tile's own rows/columns; the rest is halo.
            x = x.cpu()[
                ..., (h0 - hs) * scale : (h1 - hs) * scale, (w0 - ws) * scale : (w1 - ws) * scale
            ]
            decoded[index] = x
        if world_size > 1:
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(
                decoded, gathered, dst=dist.get_global_rank(cpu_group, 0), group=cpu_group
            )
            if rank != 0:
                return None
            for part in gathered:
                decoded.update(part)
        out = torch.empty(
            (1, decoded[0].shape[1], 1, height * scale, width * scale), dtype=decoded[0].dtype
        )
        for index, ((h0, h1, _), (w0, w1, _)) in enumerate(tiles):
            out[..., h0 * scale : h1 * scale, w0 * scale : w1 * scale] = decoded[index]
        return out


def _window(size: int) -> int:
    return min(size, _DECODE_TILE)


def decoder_tiles(height: int, width: int):
    """Exact spatial tiling of the decoder's up blocks over the head output.

    Everything after the mid block is local (3x3 convs, nearest upsampling, per-pixel
    norms): its receptive field spans 13 latent pixels. Each tile reads a fixed
    ``_DECODE_TILE`` window (clamped inside the latent, so window edges are either image
    edges, where the reference zero-pads too, or at least ``_DECODE_HALO`` pixels from
    the rows/columns it keeps), so the assembled tiles equal the untiled decode. One
    window shape per resolution means one compiled graph per up stage.

    Returns ``[((h0, h1, h_start), (w0, w1, w_start)), ...]``: the latent rows/columns a
    tile keeps and where its window starts.
    """

    def axis(size):
        window = _window(size)
        if window == size:
            return [(0, size, 0)]
        return [
            (
                start,
                min(start + _DECODE_STRIDE, size),
                min(max(start - _DECODE_HALO, 0), size - window),
            )
            for start in range(0, size, _DECODE_STRIDE)
        ]

    return [(h, w) for h in axis(height) for w in axis(width)]
