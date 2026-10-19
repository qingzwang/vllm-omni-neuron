# SPDX-License-Identifier: Apache-2.0
"""Host-side inputs of HunyuanImage-3.0-Instruct image editing.

Mirrors the Instruct checkpoint's tokenizer and image processor for an edit request
with ``K`` condition images and direct generation (``bot_task="image"``, no
think/recaption pass). The conditional row is

    <bos> system "\\n\\n" "User: "
      per image: <boi> <img_size_1024> <img_ratio_k> <timestep>
                 <img> x (h_k * w_k)       VAE latent tokens (t = 0)
                 <joint_img_sep>
                 <img> x 1024              SigLIP2 tokens (padded to 1024)
                 <eoi>
      prompt "\\n\\n" "Assistant: " <answer> <boi> <img_size_1024> <img_ratio>
    <timestep> <img> x N ...               the generated image (the step graph)

and the unconditional row replaces the prompt tokens with ``<cfg>`` (system prompt,
role prefixes and condition images are kept). Attention is causal except within each
condition image's joint span (its VAE tokens, separator and SigLIP2 slots), which is
bidirectional. Everything before the generated ``<timestep>`` is timestep-independent
(condition images use t = 0) and forms the cached prefix.

2-D RoPE: each image span (VAE grid, SigLIP2 grid, generated grid) is centred on the
diagonal at its own start index; every other token (text, special tokens, SigLIP2
padding slots) has 1-D position = its index.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageOps

from .hunyuan_image3_transformer import image_rope, rope_cos_sin

_BASE_SIZE = 1024
_VAE_FACTOR = 16
_VIT_SLOTS = 1024
_BOUNDS_TILE = 128
# Extra (height, width) buckets of the Instruct image processor, after the 33 ratio
# buckets (indices 33-36).
_EXTRA_BUCKETS = ((1024, 768), (1280, 720), (768, 1024), (720, 1280))


def instruct_buckets(base_buckets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The Instruct ``vae_reso_group``: the base buckets plus extra ratios not yet present."""
    buckets = list(base_buckets)
    ratios = {h / w for h, w in buckets}
    for h, w in _EXTRA_BUCKETS:
        if h / w not in ratios:
            buckets.append((h, w))
            ratios.add(h / w)
    return buckets


def nearest_bucket(height: int, width: int, buckets) -> int:
    ratios = np.array([h / w for h, w in buckets])
    return int(np.argmin(np.abs(ratios - height / width)))


def load_rgb(image) -> Image.Image:
    """As ``transformers.image_utils.load_image``: EXIF orientation, then RGB."""
    if isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return ImageOps.exif_transpose(image).convert("RGB")


def resize_and_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    """Cover-resize (LANCZOS) and centre-crop to ``height x width``, as the official processor."""
    w, h = image.size
    if h / w < height / width:
        rh, rw = height, int(round(height / h * w))
    else:
        rw, rh = width, int(round(width / w * h))
    top, left = int(round((rh - height) / 2.0)), int(round((rw - width) / 2.0))
    image = image.resize((rw, rh), resample=Image.Resampling.LANCZOS)
    return image.crop((left, top, left + width, top + height))


def vae_pixels(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """``[1, 3, H, W]`` in [-1, 1]."""
    array = np.asarray(resize_and_crop(image, height, width), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1)[None] - 0.5) / 0.5


@dataclass
class ConditionImage:
    """One preprocessed condition image."""

    vae_pixels: torch.Tensor  # [1, 3, H, W] in [-1, 1]
    latent_hw: tuple[int, int]
    ratio_index: int
    vit_pixels: torch.Tensor  # [1024, 768] patches (zero padded)
    vit_hw: tuple[int, int]  # SigLIP2 patch grid (valid patches = h * w)


def prepare_condition(image, buckets, vit_processor) -> ConditionImage:
    image = load_rgb(image)
    w, h = image.size
    index = nearest_bucket(h, w, buckets)
    bh, bw = buckets[index]
    vit = vit_processor(image, return_tensors="pt")
    nh, nw = (int(v) for v in vit["spatial_shapes"][0])
    return ConditionImage(
        vae_pixels=vae_pixels(image, bh, bw),
        latent_hw=(bh // _VAE_FACTOR, bw // _VAE_FACTOR),
        ratio_index=index,
        vit_pixels=vit["pixel_values"][0],
        vit_hw=(nh, nw),
    )


@dataclass
class EditPrefix:
    """Token rows and layout of an edit prefix (everything before the generated <timestep>)."""

    ids: torch.Tensor  # [R, P] (P = valid length)
    # Per condition image: start of its <timestep> token, VAE tokens and SigLIP2 slots.
    timestep_pos: list[int] = field(default_factory=list)
    vae_pos: list[int] = field(default_factory=list)
    vit_pos: list[int] = field(default_factory=list)
    joint_spans: list[tuple[int, int]] = field(default_factory=list)  # [start, end)


class InstructTokens:
    """Token ids of the Instruct template."""

    def __init__(self, tokenizer, bos_id: int):
        ids = tokenizer.convert_tokens_to_ids
        self.tokenizer = tokenizer
        self.bos = bos_id
        self.boi, self.eoi = ids("<boi>"), ids("<eoi>")
        self.img, self.timestep, self.cfg = ids("<img>"), ids("<timestep>"), ids("<cfg>")
        self.sep, self.answer = ids("<joint_img_sep>"), ids("<answer>")
        self.size = ids(f"<img_size_{_BASE_SIZE}>")
        self.ratio = lambda i: ids(f"<img_ratio_{i}>")
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
        self.encode = enc
        self.newlines = enc("\n\n")
        self.user, self.assistant = enc("User: "), enc("Assistant: ")


def edit_prefix(
    tokens: InstructTokens,
    system_prompt: str,
    prompt: str,
    conditions: list[ConditionImage],
    gen_ratio: int,
    cfg: bool,
) -> EditPrefix:
    """The prefix rows (conditional, and unconditional if ``cfg``) and their layout."""
    t = tokens
    head = [t.bos]
    if system_prompt:
        head += t.encode(system_prompt) + t.newlines
    head += t.user
    out = EditPrefix(ids=torch.empty(0))
    images: list[int] = []
    for cond in conditions:
        base = len(head) + len(images)
        images += [t.boi, t.size, t.ratio(cond.ratio_index), t.timestep]
        out.timestep_pos.append(base + 3)
        vae = cond.latent_hw[0] * cond.latent_hw[1]
        out.vae_pos.append(base + 4)
        out.vit_pos.append(base + 4 + vae + 1)
        out.joint_spans.append((base + 4, base + 4 + vae + 1 + _VIT_SLOTS))
        images += [t.img] * vae + [t.sep] + [t.img] * _VIT_SLOTS + [t.eoi]
    text = t.encode(prompt)
    tail = t.newlines + t.assistant + [t.answer, t.boi, t.size, t.ratio(gen_ratio)]
    rows = [head + images + text + tail]
    if cfg:
        rows.append(head + images + [t.cfg] * len(text) + tail)
    out.ids = torch.tensor(rows, dtype=torch.long)
    return out


def prefix_rope(prefix: EditPrefix, conditions, head_dim: int, theta: float, length: int):
    """RoPE ``cos``/``sin`` ``[length, D]`` of the prefix (padded to ``length``)."""
    pos_y = torch.arange(length)
    pos_x = torch.arange(length)
    for cond, vae, vit in zip(conditions, prefix.vae_pos, prefix.vit_pos):
        for start, (h, w) in ((vae, cond.latent_hw), (vit, cond.vit_hw)):
            beta_y = start + (w * h - h) // 2
            beta_x = start + (w * h - w) // 2
            pos_y[start : start + h * w] = torch.arange(h).repeat_interleave(w) + beta_y
            pos_x[start : start + h * w] = torch.arange(w).repeat(h) + beta_x
    return rope_cos_sin(pos_y, pos_x, head_dim, theta)


def prefix_bounds(prefix: EditPrefix, length: int) -> torch.Tensor:
    """Per-query key bound ``[length, 1]`` int32: causal, extended to the end of the
    query's joint (condition image) span."""
    bound = torch.arange(1, length + 1, dtype=torch.int32)
    for start, end in prefix.joint_spans:
        bound[start:end] = end
    return bound[:, None]


__all__ = [
    "ConditionImage",
    "EditPrefix",
    "InstructTokens",
    "edit_prefix",
    "image_rope",
    "instruct_buckets",
    "nearest_bucket",
    "prefix_bounds",
    "prefix_rope",
    "prepare_condition",
]
