# SPDX-License-Identifier: Apache-2.0
"""Masks and region marks for Qwen-Image 2.1 local editing (host-side, PIL).

Qwen-Image 2.1 has no mask input: local edits are specified in the images themselves,
either as marks drawn on the image to edit (the model card's "circle-guided multi-region
editing": a colored outline around each region, referred to by color in the prompt, and
gone from the output) or as a separate mask image passed as another reference. This
module turns ``multi_modal_data["mask_image"]`` (the key vllm-omni's ``/v1/images/edits``
endpoint fills) into those inputs:

- ``"outline"`` mode draws each region's outline, in its own color, around the region on
  the image to edit.
- ``"image"`` mode renders the regions (white for a single one, else filled in their
  colors) on black, as one extra reference image after the input images.

Optionally the edited output is composited back onto the original outside the regions,
so pixels there are kept exactly.

A region is a mask image (same aspect as the image to edit; white, or for images with
an alpha channel transparent, marks the region, as in diffusers and the OpenAI images
API respectively) or a box ``(x0, y0, x1, y1)`` in the image's pixel coordinates.
"""

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFilter

MASK_MODES = ("outline", "image")
DEFAULT_COLORS = ("red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple")


def region_mask(region, size: tuple[int, int]) -> Image.Image:
    """Binary ``L`` mask (255 inside the region) of ``size`` (width, height)."""
    if isinstance(region, str):
        region = Image.open(region)
    if isinstance(region, (tuple, list)) and len(region) == 4 and np.isscalar(region[0]):
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle([round(float(c)) for c in region], fill=255)
        return mask
    if not isinstance(region, Image.Image):
        region = Image.fromarray(np.asarray(region))
    if "A" in region.getbands() and region.getchannel("A").getextrema()[0] < 255:
        # OpenAI convention: the transparent area is the one to edit.
        inside = np.asarray(region.getchannel("A")) < 128
    else:
        inside = np.asarray(region.convert("L")) >= 128
    mask = Image.fromarray(inside.astype(np.uint8) * 255)
    return mask if mask.size == size else mask.resize(size, Image.NEAREST)


def _odd(n: float) -> int:
    n = max(3, round(n))
    return n if n % 2 else n + 1


def _rgba(color) -> tuple[int, int, int, int]:
    rgb = ImageColor.getrgb(color) if isinstance(color, str) else tuple(color)
    return (*rgb[:3], 255)


def draw_outlines(image: Image.Image, masks: list[Image.Image], colors) -> Image.Image:
    """``image`` (RGBA) with a band just outside each mask drawn in its color."""
    out = image.copy()
    width = _odd(max(image.size) / 160)  # ~6 px at 1024, like a marker stroke
    for mask, color in zip(masks, colors):
        # Outside the region, so the content to edit stays fully visible.
        band = np.asarray(mask.filter(ImageFilter.MaxFilter(width))) > np.asarray(mask)
        layer = Image.new("RGBA", image.size, _rgba(color))
        out.paste(layer, mask=Image.fromarray(band.astype(np.uint8) * 255))
    return out


def render_mask_image(masks: list[Image.Image], colors) -> Image.Image:
    """The regions on black: white for a single region, else in their colors."""
    out = Image.new("RGBA", masks[0].size, (0, 0, 0, 255))
    for mask, color in zip(masks, colors):
        fill = (255, 255, 255, 255) if len(masks) == 1 else _rgba(color)
        out.paste(Image.new("RGBA", mask.size, fill), mask=mask)
    return out


def apply_masks(images: list[Image.Image], regions, options: dict):
    """Condition images with the masks applied, and the union mask for compositing.

    ``regions`` (one region or a list) refer to the last image. Returns
    ``(images, union_mask)``; ``union_mask`` is ``None`` unless ``mask_composite``.
    """
    if not images:
        raise ValueError("mask_image needs an image to edit in multi_modal_data['image']")
    is_box = isinstance(regions, (tuple, list)) and len(regions) == 4 and np.isscalar(regions[0])
    regions = [regions] if is_box or not isinstance(regions, (tuple, list)) else list(regions)
    if not regions:
        return images, None
    mode = options.get("mask_mode", "outline")
    if mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mode!r}")
    colors = list(options.get("mask_colors") or DEFAULT_COLORS)
    if len(colors) < len(regions):
        raise ValueError(f"{len(regions)} regions but only {len(colors)} mask_colors")
    target = images[-1] if images[-1].mode == "RGBA" else images[-1].convert("RGBA")
    masks = [region_mask(r, target.size) for r in regions]
    images = list(images)
    if mode == "outline":
        images[-1] = draw_outlines(target, masks, colors)
    else:
        images.append(render_mask_image(masks, colors))
    union = None
    if options.get("mask_composite", False):
        union = masks[0]
        for mask in masks[1:]:
            union = Image.fromarray(np.maximum(np.asarray(union), np.asarray(mask)))
    return images, union


def composite(generated: torch.Tensor, original: torch.Tensor, union: Image.Image) -> torch.Tensor:
    """Keep ``original`` outside the (slightly grown, feathered) ``union`` mask.

    ``generated`` and ``original`` are ``[1, C, H, W]`` in the same value range;
    ``union`` is an ``L`` mask at any size.
    """
    height, width = generated.shape[-2:]
    mask = union.resize((width, height), Image.BILINEAR)
    grow = _odd(max(width, height) / 128)
    mask = mask.filter(ImageFilter.MaxFilter(grow)).filter(ImageFilter.GaussianBlur(grow / 2))
    alpha = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)[None, None]
    return original * (1 - alpha) + generated * alpha
