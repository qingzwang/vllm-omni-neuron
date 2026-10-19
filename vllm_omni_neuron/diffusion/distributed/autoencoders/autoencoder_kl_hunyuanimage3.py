# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 VAE decoder (``AutoencoderKLConv3D``) for Neuron, single image.

Mirrors the decoder of ``autoencoder_kl_3d.py`` in the HunyuanImage-3.0 checkpoint:

  - ``conv_in`` (plus a channel-repeat shortcut), mid ResnetBlock / single-head
    self-attention / ResnetBlock
  - 5 up levels of 3 ResnetBlocks (GroupNorm(32), swish, 3x3x3 conv); the first four
    end in a depth-to-space ``UpsampleDCAE`` that doubles H and W, and the first two
    also double the frame count (1 -> 2 -> 4)
  - GroupNorm, swish, ``conv_out``; the image is the last frame

Activations are kept as ``[frames, C, H, W]``. A 3x3x3 convolution with zero temporal
padding is one 2-D convolution with three times the output channels (one per temporal
tap), whose outputs are summed with a one-frame shift; with one frame only the centre
tap contributes.

The up levels hold almost all of the work (about 100 TFLOP at 1024x1024, on up to
128 channels x 4 frames x 1024 x 1024), so each is split into row strips over the
ranks, exactly:

  - A level starts from its full input and takes the rank's strip plus a halo of
    ``_HALO`` rows on each side: one row per 3x3 convolution of the level, so after its
    last convolution the strip's own rows are exact. Rows outside the image are zeroed
    before every convolution (its zero padding).
  - GroupNorm statistics span the whole image (and all frames): each rank sums over its
    own rows and the sums are all-reduced.
  - The level's output (own rows) is all-gathered into the next level's full input.

The head (conv_in and the mid block, with its global attention) runs on the 64x64-ish
latent grid on every rank. Each stage is one compiled graph.
"""

import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 3x3 convolutions per up level: 3 ResnetBlocks x 2, plus the upsample conv (or, for
# the last level, conv_out).
_HALO = 7


class _Strip:
    """A rank's row strip of an up level: own rows ``[halo, halo + own)`` of the strip."""

    def __init__(self, rows_index: torch.Tensor, own: int, image_rows: int, group, world: int):
        self.own = own
        self.group, self.world = group, world
        rows = rows_index - _HALO  # image row of each strip row
        self.mask = ((rows >= 0) & (rows < image_rows))[None, None, :, None]
        self.own_mask = self.mask[:, :, _HALO : _HALO + own]
        self.image_rows = image_rows

    def zero_outside(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask.to(x.dtype)

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.group)
        return t


class _Conv3d(nn.Module):
    """``nn.Conv3d`` (stride 1, symmetric padding) on ``[frames, C, H, W]`` activations."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dtype: torch.dtype):
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel, kernel, kernel, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_ch, dtype=dtype))

    def forward(self, x: torch.Tensor, strip: _Strip | None = None) -> torch.Tensor:
        if self.kernel == 1:
            return F.conv2d(x, self.weight[:, :, 0], self.bias)
        if strip is not None:
            x = strip.zero_outside(x)
        pad = self.kernel // 2
        frames = x.shape[0]
        if frames == 1:
            return F.conv2d(x, self.weight[:, :, 1], self.bias, padding=pad)
        out_ch, in_ch = self.weight.shape[:2]
        # One 2-D conv for the three temporal taps: tap 0 reads the previous frame,
        # tap 2 the next one (zeros beyond the ends).
        taps = self.weight.permute(2, 0, 1, 3, 4).reshape(3 * out_ch, in_ch, 3, 3)
        y = F.conv2d(x, taps, padding=pad).view(frames, 3, out_ch, *x.shape[2:])
        prev = F.pad(y[:-1, 0], (0, 0, 0, 0, 0, 0, 1, 0))
        nxt = F.pad(y[1:, 2], (0, 0, 0, 0, 0, 0, 0, 1))
        return y[:, 1] + prev + nxt + self.bias[None, :, None, None]


class _GroupNorm(nn.Module):
    """GroupNorm(32, eps=1e-6) over channels-in-group, frames and space, in float32.

    On a strip the statistics are those of the whole image: sums over the rank's own
    rows, all-reduced.
    """

    def __init__(self, channels: int, dtype: torch.dtype, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.groups, self.eps = groups, eps
        self.weight = nn.Parameter(torch.ones(channels, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(channels, dtype=dtype))

    def forward(self, x: torch.Tensor, strip: _Strip | None = None) -> torch.Tensor:
        frames, channels, height, width = x.shape
        g = x.float().view(frames, self.groups, channels // self.groups, height, width)
        if strip is None:
            mean = g.mean(dim=(0, 2, 3, 4), keepdim=True)
            var = (g - mean).pow(2).mean(dim=(0, 2, 3, 4), keepdim=True)
        else:
            own = g[:, :, :, _HALO : _HALO + strip.own]
            weight = strip.own_mask.float()[:, :, None]  # [1, 1, 1, own, 1]
            count = frames * (channels // self.groups) * strip.image_rows * width
            total = strip.all_reduce((own * weight).sum(dim=(0, 2, 3, 4)))
            mean = (total / count).view(1, -1, 1, 1, 1)
            sq = strip.all_reduce(((own - mean).pow(2) * weight).sum(dim=(0, 2, 3, 4)))
            var = (sq / count).view(1, -1, 1, 1, 1)
        g = (g - mean) * torch.rsqrt(var + self.eps)
        out = g.view(frames, channels, height, width) * self.weight.float()[:, None, None]
        return (out + self.bias.float()[:, None, None]).to(x.dtype)


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dtype: torch.dtype):
        super().__init__()
        self.norm1 = _GroupNorm(in_ch, dtype)
        self.conv1 = _Conv3d(in_ch, out_ch, 3, dtype)
        self.norm2 = _GroupNorm(out_ch, dtype)
        self.conv2 = _Conv3d(out_ch, out_ch, 3, dtype)
        if in_ch != out_ch:
            self.nin_shortcut = _Conv3d(in_ch, out_ch, 1, dtype)

    def forward(self, x: torch.Tensor, strip: _Strip | None = None) -> torch.Tensor:
        h = self.conv1(_swish(self.norm1(x, strip)), strip)
        h = self.conv2(_swish(self.norm2(h, strip)), strip)
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Single-head self-attention over every (frame, y, x) position."""

    def __init__(self, channels: int, dtype: torch.dtype, query_chunk: int = 1024):
        super().__init__()
        self.norm = _GroupNorm(channels, dtype)
        self.q = _Conv3d(channels, channels, 1, dtype)
        self.k = _Conv3d(channels, channels, 1, dtype)
        self.v = _Conv3d(channels, channels, 1, dtype)
        self.proj_out = _Conv3d(channels, channels, 1, dtype)
        self.query_chunk = query_chunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames, channels, height, width = x.shape
        h = self.norm(x)

        def tokens(t):  # [f, c, h, w] -> [(f h w), c]
            return t.permute(0, 2, 3, 1).reshape(-1, channels)

        q, k, v = tokens(self.q(h)), tokens(self.k(h)), tokens(self.v(h))
        scale = channels**-0.5
        # Explicit attention in query chunks: the head dim (= channels) is too wide
        # for the fused attention kernel.
        out = []
        for start in range(0, q.shape[0], self.query_chunk):
            scores = torch.matmul(q[start : start + self.query_chunk], k.t()).float() * scale
            out.append(torch.matmul(scores.softmax(dim=-1).to(v.dtype), v))
        out = torch.cat(out).view(frames, height, width, channels).permute(0, 3, 1, 2)
        return x + self.proj_out(out.contiguous())


class UpsampleDCAE(nn.Module):
    """Conv to ``factor`` x channels, depth-to-space, plus a channel-repeat shortcut."""

    def __init__(self, in_ch: int, out_ch: int, temporal: bool, dtype: torch.dtype):
        super().__init__()
        self.temporal = temporal
        factor = 8 if temporal else 4
        self.conv = _Conv3d(in_ch, out_ch * factor, 3, dtype)
        self.repeats = factor * out_ch // in_ch

    def _depth_to_space(self, x: torch.Tensor) -> torch.Tensor:
        # b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)
        frames, channels, height, width = x.shape
        r1 = 2 if self.temporal else 1
        c = channels // (r1 * 4)
        x = x.view(frames, r1, 2, 2, c, height, width).permute(0, 1, 4, 5, 2, 6, 3)
        return x.reshape(frames * r1, c, height * 2, width * 2)

    def forward(self, x: torch.Tensor, strip: _Strip | None = None) -> torch.Tensor:
        h = self._depth_to_space(self.conv(x, strip))
        return h + self._depth_to_space(x.repeat_interleave(self.repeats, dim=1))


class _UpLevel(nn.Module):
    def __init__(self, blocks, upsample):
        super().__init__()
        self.block = nn.ModuleList(blocks)
        if upsample is not None:
            self.upsample = upsample


class _Mid(nn.Module):
    def __init__(self, channels: int, dtype: torch.dtype):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels, dtype)
        self.attn_1 = AttnBlock(channels, dtype)
        self.block_2 = ResnetBlock(channels, channels, dtype)


def _gather_rows(x: torch.Tensor, group, world: int) -> torch.Tensor:
    """All-gather ``[F, C, rows, W]`` strips along the rows (rank order)."""
    if world == 1:
        return x
    rows_first = x.permute(2, 0, 1, 3).contiguous()
    out = torch.empty(
        (rows_first.shape[0] * world, *rows_first.shape[1:]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(out, rows_first, group=group)
    return out.permute(1, 2, 0, 3)


class HunyuanImage3Decoder(nn.Module):
    """``Decoder`` of ``AutoencoderKLConv3D`` (parameter names match ``vae.decoder.*``)."""

    def __init__(self, vae_config: dict, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        # The config lists the encoder's channels; the decoder runs them in reverse.
        channels = list(reversed(vae_config["block_out_channels"]))
        self.z_channels = vae_config["latent_channels"]
        self.scaling_factor = vae_config["scaling_factor"]
        num_res = vae_config["layers_per_block"]
        spatial_levels = int(vae_config["ffactor_spatial"]).bit_length() - 1
        temporal_levels = int(vae_config["ffactor_temporal"]).bit_length() - 1
        if 2 * (num_res + 1) + 1 != _HALO:
            raise ValueError(f"the strip halo assumes {_HALO} convolutions per up level")
        self._dtype = dtype
        block_in = channels[0]
        self.conv_in = _Conv3d(self.z_channels, block_in, 3, dtype)
        self.mid = _Mid(block_in, dtype)
        self.up = nn.ModuleList()
        for level, ch in enumerate(channels):
            blocks = []
            for _ in range(num_res + 1):
                blocks.append(ResnetBlock(block_in, ch, dtype))
                block_in = ch
            upsample = None
            if level < spatial_levels:
                out = channels[level + 1]
                upsample = UpsampleDCAE(block_in, out, level < temporal_levels, dtype)
                block_in = out
            self.up.append(_UpLevel(blocks, upsample))
        self.norm_out = _GroupNorm(block_in, dtype)
        self.conv_out = _Conv3d(block_in, vae_config["out_channels"], 3, dtype)
        self._compiled: dict[str, object] = {}
        self._group, self._world, self._rank = None, 1, 0

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_parallel(self, group, world: int, rank: int) -> None:
        """Split the up levels' rows over ``world`` ranks of ``group`` (all must decode)."""
        self._group, self._world, self._rank = group, world, rank

    def load_weights(self, checkpoint, prefix: str = "vae.decoder.") -> None:
        state = {
            name: checkpoint.get(prefix + name).to(self._dtype)
            for name, _ in self.named_parameters()
        }
        self.load_state_dict(state, strict=True, assign=True)

    # -- stages ---------------------------------------------------------

    def _head(self, latents: torch.Tensor) -> torch.Tensor:
        """Model-space ``latents`` ``[1, C_z, h, w]`` -> mid-block output."""
        z = (latents / self.scaling_factor).to(self._dtype)
        repeats = self.conv_in.weight.shape[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats, dim=1)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        return self.mid.block_2(h)

    def _level(self, index: int, x: torch.Tensor, rows_index: torch.Tensor) -> torch.Tensor:
        """Up level ``index`` on this rank's strip of the full input ``x`` ``[F, C, R, W]``.

        Returns the full output (all ranks' own rows, gathered); the last level also
        runs the output norm and conv and returns the last frame.
        """
        rows = x.shape[2]
        own = rows_index.shape[0] - 2 * _HALO
        padded = F.pad(x, (0, 0, _HALO, _HALO + own * self._world - rows))
        strip = _Strip(rows_index, own, rows, self._group, self._world)
        h = padded.index_select(2, rows_index)
        level = self.up[index]
        for block in level.block:
            h = block(h, strip)
        own_rows = slice(_HALO, _HALO + own)
        if hasattr(level, "upsample"):
            h = level.upsample(h, strip)
            own_rows = slice(2 * _HALO, 2 * (_HALO + own))
            rows *= 2
        else:
            h = self.conv_out(_swish(self.norm_out(h, strip)), strip)[-1:]
        out = _gather_rows(h[:, :, own_rows].contiguous(), self._group, self._world)
        return out[:, :, :rows]

    def compile(self, *args, **kwargs):
        options = dict(kwargs.pop("options", None) or {})
        name = options.get("model_name", "vae")

        def _compile(fn, suffix):
            return torch.compile(
                fn, *args, options={**options, "model_name": f"{name}_{suffix}"}, **kwargs
            )

        self._compiled["head"] = _compile(self._head, "head")
        for i in range(len(self.up)):
            self._compiled[f"up{i}"] = _compile(
                lambda x, rows_index, i=i: self._level(i, x, rows_index), f"up{i}"
            )

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``latents`` ``[1, C_z, h, w]`` (model space) -> image ``[1, 3, 16h, 16w]`` in [-1, 1].

        With :meth:`set_parallel`, every rank of the group must call it.
        """
        head = self._compiled.get("head", self._head)
        h = head(latents)
        for i in range(len(self.up)):
            rows = h.shape[2]
            own = math.ceil(rows / self._world)
            rows_index = torch.arange(own + 2 * _HALO) + self._rank * own
            level = self._compiled.get(f"up{i}", lambda x, r, i=i: self._level(i, x, r))
            h = level(h, rows_index.to(h.device))
        return h
