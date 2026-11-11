# SPDX-License-Identifier: Apache-2.0
"""Packed-sequence layout of MiniMax-H3-Image on Neuron.

The reference (diffusers ``build_image_packed_sequence``) orders the packed sequence
``[text | reference images | generated image]``. The Neuron DiT uses the physical order
``[reference images | generated image | text | padding]`` instead; attention is unmasked
and every row keeps the rotary coordinate, modality tag and timestep it has in the
reference, so the outputs are the same rows, permuted. The rotary grid is reproduced
exactly (float64 ``numpy.linspace``, as released).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# The spatial rotary axes are normalized by the square root of the latent area and scaled by 32.
_ROPE_SPATIAL_SCALE = 32
# Every text bucket is a multiple of this.
TEXT_ALIGN = 128


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int) -> torch.Tensor:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)


def get_image_canvas_size(height: int, width: int, min_generation_area: int, multiple: int = 32) -> tuple[int, int]:
    """Reference canvas: below ``min_generation_area`` the image is generated larger."""
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def patchify(latents: torch.Tensor, patch_h: int = 2, patch_w: int = 2) -> torch.Tensor:
    """``[1, C, h, w]`` latents -> ``[(h/ph)*(w/pw), C*ph*pw]`` rows (row-major patch grid)."""
    _, c, h, w = latents.shape
    x = latents.reshape(1, c, 1, 1, h // patch_h, patch_h, w // patch_w, patch_w)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * patch_h * patch_w).contiguous()


def unpatchify(rows: torch.Tensor, latent_height: int, latent_width: int, channels: int = 24, patch: int = 2):
    rows = rows.reshape(-1, latent_height // patch, latent_width // patch, channels, patch, patch)
    rows = rows.permute(0, 3, 1, 4, 2, 5)
    return rows.reshape(-1, channels, latent_height, latent_width).contiguous()


@dataclass
class PackedLayout:
    num_text: int  # real text rows
    text_pad: int  # padded text rows in the graph
    num_reference_rows: int
    num_image_rows: int  # reference + generated rows
    position_ids: torch.Tensor  # [total, 3] float64, physical order
    text_token_tags: torch.Tensor  # [num_text] reference modality tags of the text rows

    @property
    def total(self) -> int:
        return self.num_image_rows + self.text_pad

    @property
    def kv_len(self) -> int:
        return self.num_image_rows + self.num_text


def choose_text_pad(num_text: int, num_image_rows: int, cp_size: int, buckets: list[int]) -> int:
    """Smallest bucket ``>= num_text`` such that the packed length splits into CP chunks of
    ``TEXT_ALIGN`` rows."""
    align = TEXT_ALIGN * cp_size
    for bucket in sorted(buckets):
        if bucket < num_text:
            continue
        pad = bucket
        while (num_image_rows + pad) % align:
            pad += TEXT_ALIGN
        return pad
    raise ValueError(f"prompt of {num_text} tokens exceeds the largest text bucket {max(buckets)}")


def build_layout(
    num_text: int,
    text_pad: int,
    reference_grids: list[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    text_token_tags: torch.Tensor | None = None,
    patch_h: int = 2,
    patch_w: int = 2,
) -> PackedLayout:
    """Physical ``[references | generated | text | pad]`` layout with reference coordinates."""
    num_reference_rows = sum((h // patch_h) * (w // patch_w) for h, w in reference_grids)
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    num_image_rows = num_reference_rows + num_target_rows
    total = num_image_rows + text_pad
    position_ids = torch.zeros(total, 3, dtype=torch.float64)

    # Reference clock: text rows take times 0..T-1, every image the next integer slot.
    rotary_time = float(num_text)
    cursor = 0
    for h, w in reference_grids:
        n = (h // patch_h) * (w // patch_w)
        position_ids[cursor : cursor + n, 0] = rotary_time
        position_ids[cursor : cursor + n, 1:] = _frame_position_grid(h, w, patch_h, patch_w)
        cursor += n
        rotary_time += 1.0
    position_ids[cursor:num_image_rows, 0] = rotary_time
    position_ids[cursor:num_image_rows, 1:] = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    position_ids[num_image_rows : num_image_rows + num_text, 0] = torch.arange(num_text, dtype=torch.float64)

    if text_token_tags is None:
        text_token_tags = torch.ones(num_text, dtype=torch.long)
    return PackedLayout(
        num_text=num_text,
        text_pad=text_pad,
        num_reference_rows=num_reference_rows,
        num_image_rows=num_image_rows,
        position_ids=position_ids,
        text_token_tags=text_token_tags,
    )
