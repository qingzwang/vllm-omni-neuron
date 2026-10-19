# SPDX-License-Identifier: Apache-2.0
"""NeuronQwenImage21Pipeline — Qwen-Image 2.1 text-to-image and edit pipeline for Neuron.

vllm-omni 0.24 (and the diffusers release it pins) has no Qwen-Image 2.1 pipeline, so
this is a standalone pipeline that mirrors diffusers' ``QwenImage21Pipeline`` for
text-to-image and image editing (up to 10 reference images), with optional true
classifier-free guidance (off by default, as the model is meant to be sampled without
it). Per text-to-image request:

1. Tokenize the prompt with the T2I chat template; the Qwen3-VL text encoder (TP,
   compiled) returns its pre-norm hidden states at a static length.
2. The DiT's prefix graph turns the text (minus the system prompt) into every layer's
   text K/V, once.
3. The DiT's step graph runs each denoising step over the image tokens only and applies
   the Euler update in-graph; the latents stay on device for the whole loop.
4. The VAE decoder (compiled per stage) decodes the latents: on rank 0 up to a 64x64
   latent, above that as exact tiles spread over the TP ranks.

An edit request (one condition image in ``multi_modal_data["image"]``) differs in steps
1-2: the Qwen3-VL vision tower encodes the image for the language model (image tokens
and DeepStack features, MRoPE positions), the VAE encoder turns it into a latent block,
and the prefix graph runs text plus that block with a block-causal mask.

Local edits use masks the way the model was trained for them, as image content (see
``region_marks``): ``multi_modal_data["mask_image"]`` regions are drawn as colored
outlines on the image to edit, or passed as an extra mask image.

Timesteps, sigmas, RoPE tables and the initial noise are computed on the host exactly
as the reference does, so a seed gives the reference's starting point.
"""

import logging
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image as PILImage
from torch import nn
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage21 import (
    NeuronAutoencoderKLQwenImage21,
)
from vllm_omni_neuron.diffusion.models.qwen3_vl_text_encoder.qwen3_vl_text_encoder import (
    NeuronQwen3VLTextEncoder,
)
from vllm_omni_neuron.diffusion.models.qwen3_vl_text_encoder.qwen3_vl_vision_encoder import (
    NeuronQwen3VLVisionEncoder,
)
from vllm_omni_neuron.diffusion.models.qwen_image_2_1.qwen_image_2_1_transformer import (
    QwenImage21Transformer2DModel,
    qwen_image21_edit_layout,
    qwen_image21_rope,
)
from vllm_omni_neuron.diffusion.models.qwen_image_2_1.region_marks import apply_masks, composite
from vllm_omni_neuron.lite_compat import is_lite_runtime

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        "model_arch": "QwenImage21Pipeline",
        "class_name": "NeuronQwenImage21Pipeline",
        "post_process_func_name": "get_qwen_image21_post_process_func",
    },
]

# The VAE compresses 16x spatially and the transformer consumes latents unpatched.
_VAE_SCALE_FACTOR = 16
# Width/height must be multiples of this (the reference rounds down to it).
_MULTIPLE_OF = _VAE_SCALE_FACTOR * 2
_DEFAULT_RESOLUTION = 1024
_DEFAULT_STEPS = 40
# Static number of DiT text positions (after the system prompt) unless the stage
# config's model_config sets ``max_text_tokens``.
_DEFAULT_MAX_TEXT_TOKENS = 512

_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
_T2I_TEMPLATE = (
    f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# Image-conditioned (edit) template; the placeholder repeats once per condition image.
_IMAGE_PLACEHOLDER = "<image1><|vision_start|><|image_pad|><|vision_end|>"
_TI2I_TEMPLATE = (
    f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{_IMAGE_PLACEHOLDER}{{}}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# Reference images per edit the model was trained with.
_MAX_CONDITION_IMAGES = 10

_TRANSFORMER_CONFIG_KEYS = (
    "patch_size",
    "in_channels",
    "out_channels",
    "num_layers",
    "attention_head_dim",
    "num_attention_heads",
    "context_in_dim",
    "mlp_ratio",
    "axes_dims_rope",
    "eps",
    "causal_condition",
    # Not a diffusers config field; may be set in the stage config's model_config.
    "sequence_parallel",
)

# Compiled variants allowed per function (see NeuronQwenImage21Pipeline.compile).
_RECOMPILE_LIMIT = 64

_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]
_VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]


def get_qwen_image21_post_process_func(od_config):
    image_processor = VaeImageProcessor(vae_scale_factor=_MULTIPLE_OF)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + base_shift - m * base_seq_len


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """``(width, height)`` of ``target_area`` pixels at ``ratio``, multiples of 32."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def _wait_for(tensor: torch.Tensor) -> None:
    """Block until ``tensor`` is computed.

    Lite executes graphs asynchronously, and enqueueing every denoising step at once
    overflows its execution queue. A one-element D2D copy waits on the producing graph
    without a host transfer (as in ``NeuronWanPipeline.scheduler_step_maybe_with_cfg``).
    """
    if is_lite_runtime():
        source = tensor.view(-1)[:1]
        torch.empty_like(source).copy_(source)


def _model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


class NeuronQwenImage21Pipeline(nn.Module):
    """Qwen-Image 2.1 text-to-image and image editing on Neuron (TP, optional CP and CFG)."""

    # Requests may carry a condition image (multi_modal_data["image"]) to edit.
    support_image_input = True
    color_format = "RGBA"

    # Weights are loaded by load_weights() from the checkpoint directories.
    weights_sources: list = []

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = _model_path(od_config.model)
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model_config = dict(od_config.model_config or {})
        self.max_text_tokens = int(model_config.pop("max_text_tokens", _DEFAULT_MAX_TEXT_TOKENS))
        # Area (as a side length) the edit condition image, and by default the edited
        # output, are resized to: the reference's ``output_resolution``.
        self.edit_resolution = int(model_config.pop("edit_resolution", _DEFAULT_RESOLUTION))

        self.processor = AutoProcessor.from_pretrained(self.model_path, subfolder="processor")
        sys_message = [{"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]}]
        sys_tokens = self.processor.apply_chat_template(
            sys_message, tokenize=True, return_dict=False
        )
        # Leading system-role tokens dropped from the text-encoder output.
        self.drop_idx = len(sys_tokens[0])
        self.encoder_seq_len = self.drop_idx + self.max_text_tokens
        self.pad_token_id = self.processor.tokenizer.pad_token_id or 0

        self.text_encoder = NeuronQwen3VLTextEncoder(
            self.model_path, seq_len=self.encoder_seq_len, dtype=dtype
        )
        # Edit (image-conditioned) requests: the Qwen3-VL vision tower, and the static
        # ``get_rope_index`` of the transformers model for the image's MRoPE positions.
        self.vision_encoder = NeuronQwen3VLVisionEncoder(self.model_path, dtype=dtype)
        vl_config = AutoConfig.from_pretrained(self.model_path, subfolder="text_encoder")
        self.image_token_id = vl_config.image_token_id
        self._rope_index_owner = SimpleNamespace(
            config=vl_config,
            get_vision_position_ids=lambda *a, **k: Qwen3VLModel.get_vision_position_ids(
                None, *a, **k
            ),
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=_VAE_SCALE_FACTOR, vae_latent_channels=64
        )

        tf_config = load_transformer_config(self.model_path, "transformer", True)
        tf_config.update(model_config)
        self.transformer = QwenImage21Transformer2DModel(
            text_offset=self.drop_idx,
            text_buffer_len=self.max_text_tokens,
            **{k: tf_config[k] for k in _TRANSFORMER_CONFIG_KEYS if k in tf_config},
        )
        self.latent_channels = self.transformer.config.in_channels

        self.is_output_rank = not dist.is_initialized() or dist.get_rank() == 0
        # Every rank holds the (0.5 GB) VAE decoder: above one 64x64 latent tile the
        # decode is split over all of them (NeuronAutoencoderKLQwenImage21.decode_image).
        world = get_world_group()
        self._vae_rank, self._vae_world = world.rank_in_group, world.world_size
        self._vae_cpu_group = world.cpu_group
        self.vae = NeuronAutoencoderKLQwenImage21.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=dtype
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_path, subfolder="scheduler"
        )
        self.last_stage_seconds: dict[str, float] = {}

    def to(self, *args, **kwargs):
        for module in (self.text_encoder, self.vision_encoder, self.transformer, self.vae):
            module.to(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        self.text_encoder.load_weights(os.path.join(self.model_path, "text_encoder"))
        self.vision_encoder.load_weights(os.path.join(self.model_path, "text_encoder"))
        self.transformer.load_weights(os.path.join(self.model_path, "transformer"))
        return None

    def compile(self, *args, **kwargs):
        # Every new output size (and, for edits, condition-image size) adds a graph per
        # stage, and the VAE's up blocks share one forward: lift Dynamo's default limit
        # of 8 compiled variants per function.
        dynamo_config = torch._dynamo.config
        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, _RECOMPILE_LIMIT)
        dynamo_config.accumulated_recompile_limit = max(
            dynamo_config.accumulated_recompile_limit, 8 * _RECOMPILE_LIMIT
        )
        options = dict(kwargs.pop("options", None) or {})
        kwargs.setdefault("fullgraph", True)
        kwargs.setdefault("dynamic", False)

        def _opts(name, compiler_args=_COMPILER_ARGS):
            return {**options, "model_name": name, "compiler_args": list(compiler_args)}

        self.text_encoder.compile(*args, options=_opts("qwen_image21_text_encoder"), **kwargs)
        self.vision_encoder.compile(*args, options=_opts("qwen_image21_vision"), **kwargs)
        self.transformer.compile(
            *args,
            prefix_options=_opts("qwen_image21_dit_prefix"),
            step_options=_opts("qwen_image21_dit_step"),
            **kwargs,
        )
        self.vae.compile(*args, options=_opts("qwen_image21_vae", _VAE_COMPILER_ARGS), **kwargs)
        return self

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_of(prompt) -> str:
        return prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
        """Right-padded ``[1, encoder_seq_len]`` ids and the number of DiT text tokens."""
        # Qwen has no bos token, so an empty prompt would leave the encoder nothing to read.
        text = _T2I_TEMPLATE.format(prompt or " ")
        input_ids = self.processor(text=[text], return_tensors="pt").input_ids
        num_tokens = input_ids.shape[1]
        if num_tokens > self.encoder_seq_len:
            raise ValueError(
                f"prompt is {num_tokens - self.drop_idx} tokens; the compiled text length is "
                f"{self.max_text_tokens} (raise model_config.max_text_tokens)"
            )
        padded = torch.full((1, self.encoder_seq_len), self.pad_token_id, dtype=input_ids.dtype)
        padded[:, :num_tokens] = input_ids
        return padded, num_tokens - self.drop_idx

    def _timesteps(self, num_steps: int, image_seq_len: int, sigmas=None):
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        cfg = self.scheduler.config
        mu = calculate_shift(
            image_seq_len,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(len(sigmas), sigmas=sigmas, mu=mu)
        return self.scheduler.timesteps, self.scheduler.sigmas

    def _initial_latents(self, generator, height: int, width: int, dtype) -> torch.Tensor:
        """Reference ``prepare_latents``: ``randn`` in the model dtype, then spatial flatten."""
        shape = (1, 1, self.latent_channels, height, width)
        latents = torch.randn(shape, generator=generator, dtype=dtype)
        return latents.view(1, self.latent_channels, height * width).transpose(1, 2).contiguous()

    @staticmethod
    def _condition_images(prompt) -> list:
        """The request's condition images (empty for text-to-image)."""
        if isinstance(prompt, str):
            return []
        images = (prompt.get("multi_modal_data") or {}).get("image")
        if images is None:
            return []
        images = list(images) if isinstance(images, (list, tuple)) else [images]
        out = []
        for image in images:
            if isinstance(image, str):
                image = PILImage.open(image)
            elif not isinstance(image, PILImage.Image):
                image = PILImage.fromarray(np.asarray(image))
            out.append(image)
        return out

    def _prefix_t2i(self, prompt: str, latent_hw, device):
        """Text-to-image prefix: text K/V, target RoPE and the key count."""
        latent_h, latent_w = latent_hw
        input_ids, text_len = self._tokenize(prompt)
        text_hidden = self.text_encoder(input_ids, device).last_hidden_state
        txt_cos, txt_sin, img_cos, img_sin = qwen_image21_rope(
            text_len,
            self.max_text_tokens,
            latent_h,
            latent_w,
            axes_dim=self.transformer.config.axes_dims_rope,
        )
        text_k, text_v = self.transformer.prefix(
            text_hidden, txt_cos.to(device), txt_sin.to(device)
        )
        return text_k, text_v, img_cos, img_sin, text_len

    def _encode_condition_images(self, images, device) -> dict:
        """Prompt-independent part of an edit: each image's vision-language features
        and its VAE latents (shared by the positive and negative CFG prefixes).

        Mirrors ``QwenImage21Pipeline``: each condition image is resized to the area of
        ``edit_resolution``² at its own aspect ratio; the vision-language encoder reads it
        with alpha composited over white, and the VAE encodes all four channels.
        """
        rgba_images, sizes, vlm_images = [], [], []
        for image in images:
            rgba = image if image.mode == "RGBA" else image.convert("RGBA")
            cond_w, cond_h = calculate_dimensions(
                self.edit_resolution**2, rgba.size[0] / rgba.size[1]
            )
            resized = self.image_processor.resize(rgba, height=cond_h, width=cond_w)
            white = PILImage.new("RGB", resized.size, (255, 255, 255))
            white.paste(resized, mask=resized.getchannel("A"))
            rgba_images.append(rgba)
            sizes.append((cond_h, cond_w))
            vlm_images.append(white)
        pixels = self.processor.image_processor(images=vlm_images, return_tensors="pt")
        grid_thw = pixels.image_grid_thw
        # The vision tower runs per image: its attention never crosses images.
        image_tokens, deepstack = [], []
        patches = grid_thw.prod(-1).tolist()
        for grid, image_pixels in zip(grid_thw, pixels.pixel_values.split(patches)):
            tokens, deep = self.vision_encoder(image_pixels, grid[None], device)
            image_tokens.append(tokens.cpu())
            deepstack.append(deep.cpu())

        cond, cond_hw = [], []
        for rgba, (cond_h, cond_w) in zip(rgba_images, sizes):
            vae_input = self.image_processor.preprocess(rgba, height=cond_h, width=cond_w)
            latents = self.vae.encode_image(vae_input.unsqueeze(2).to(device)).cpu()
            cond_hw.append((latents.shape[-2], latents.shape[-1]))
            cond.append(latents.view(1, self.latent_channels, -1).transpose(1, 2))
        return {
            "vlm_images": vlm_images,
            "grid_thw": grid_thw,
            "image_tokens": torch.cat(image_tokens),
            "deepstack": torch.cat(deepstack, dim=1),
            "latents": torch.cat(cond, dim=1).contiguous().to(device),
            "latent_hw": cond_hw,
        }

    def _prefix_edit(self, prompt: str, condition: dict, latent_hw, device):
        """Edit prefix: text and condition-image K/V, target RoPE and the key count.

        The prompt goes through the TI2I template with one ``<imageN>`` placeholder per
        condition image; the image tokens and DeepStack features from
        :meth:`_encode_condition_images` fill the placeholders, and the condition
        latents replace the image slots in the DiT prefix.
        """
        placeholders = " ".join(
            f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(condition["vlm_images"]))
        )
        text = _TI2I_TEMPLATE.replace(_IMAGE_PLACEHOLDER, placeholders).format(prompt or " ")
        input_ids = self.processor(
            text=[text], images=condition["vlm_images"], return_tensors="pt"
        ).input_ids
        grid_thw = condition["grid_thw"]
        num_tokens = input_ids.shape[1]
        slots = int(grid_thw.prod(-1).sum()) // 4
        # Even token counts for the NKI MLP kernel; a multiple of 16 for good measure.
        enc_len = -(-(self.drop_idx + slots + self.max_text_tokens) // 16) * 16
        if num_tokens > enc_len:
            raise ValueError(
                f"prompt is {num_tokens - self.drop_idx - slots} tokens; the compiled text "
                f"length is {self.max_text_tokens} (raise model_config.max_text_tokens)"
            )

        # The language model with the image tokens and DeepStack features at the
        # placeholder positions (assembled on the host).
        padded = torch.full((1, enc_len), self.pad_token_id, dtype=input_ids.dtype)
        padded[:, :num_tokens] = input_ids
        is_image = padded[0] == self.image_token_id
        embeds = self.text_encoder.embed(padded)
        embeds[0, is_image] = condition["image_tokens"].to(embeds.dtype)
        deepstack = condition["deepstack"]
        deepstack_dense = embeds.new_zeros((deepstack.shape[0], 1, enc_len, embeds.shape[-1]))
        deepstack_dense[:, 0, is_image] = deepstack.to(embeds.dtype)
        position_ids = self._mrope_positions(input_ids, grid_thw, enc_len)
        cos, sin = self.text_encoder.mrope_cos_sin(position_ids)
        hidden = self.text_encoder.forward_multimodal(
            embeds.to(device), cos.to(device), sin.to(device), deepstack_dense.to(device)
        ).last_hidden_state

        layout = qwen_image21_edit_layout(
            input_ids[0, self.drop_idx :] == self.image_token_id,
            enc_len - self.drop_idx,
            condition["latent_hw"],
            latent_hw,
            axes_dim=self.transformer.config.axes_dims_rope,
        )
        text_k, text_v = self.transformer.prefix_edit(
            hidden,
            condition["latents"],
            layout.gather_index.to(device),
            layout.bound_max.to(device),
            layout.prefix_cos.to(device),
            layout.prefix_sin.to(device),
        )
        return text_k, text_v, layout.target_cos, layout.target_sin, layout.valid_len

    def _mrope_positions(self, input_ids, grid_thw, length: int) -> torch.Tensor:
        """Qwen3-VL ``(t, h, w)`` positions ``[3, length]`` (padding continues the text)."""
        mm_token_type_ids = (input_ids == self.image_token_id).to(torch.int32)
        positions, _ = Qwen3VLModel.get_rope_index(
            self._rope_index_owner, input_ids, mm_token_type_ids, image_grid_thw=grid_thw
        )
        positions = positions[:, 0]
        pad = length - positions.shape[1]
        tail = positions.max() + 1 + torch.arange(pad)
        return torch.cat([positions, tail.expand(3, pad)], dim=1)

    def forward(self, req) -> DiffusionOutput:
        if req.request_ids == ["dummy_req_id"] and self._prompt_of(req.prompts[0]) == "dummy run":
            logger.info("Skipping warmup request on Neuron pipeline")
            return DiffusionOutput(output=None)
        if len(req.prompts) != 1:
            raise NotImplementedError("Qwen-Image 2.1 on Neuron runs one prompt per request")

        sp = req.sampling_params
        # True CFG as in the reference: on when true_cfg_scale > 1 and a negative prompt
        # (possibly empty) is given; the model is meant to be sampled without it.
        cfg_scale = sp.true_cfg_scale or 1.0
        negative_prompt = (
            req.prompts[0].get("negative_prompt") if isinstance(req.prompts[0], dict) else None
        )
        do_cfg = cfg_scale > 1.0 and negative_prompt is not None
        if cfg_scale > 1.0 and negative_prompt is None:
            logger.warning("true_cfg_scale > 1 without a negative_prompt: CFG is not enabled")
        if (sp.num_outputs_per_prompt or 1) > 1:
            raise NotImplementedError("num_outputs_per_prompt > 1 is not supported")

        prompt = self._prompt_of(req.prompts[0])
        images = self._condition_images(req.prompts[0])
        regions = None
        if isinstance(req.prompts[0], dict):
            regions = (req.prompts[0].get("multi_modal_data") or {}).get("mask_image")
        mask_union = None
        target = images[-1] if images else None
        if regions is not None:
            images, mask_union = apply_masks(images, regions, sp.extra_args or {})
        if len(images) > _MAX_CONDITION_IMAGES:
            raise ValueError(f"at most {_MAX_CONDITION_IMAGES} condition images are supported")
        height, width = sp.height, sp.width
        if images:
            image = target
            # By default the reference sizes the output to the edit resolution's area at
            # the last condition image's aspect ratio.
            cond_w, cond_h = calculate_dimensions(
                self.edit_resolution**2, image.size[0] / image.size[1]
            )
            height, width = height or cond_h, width or cond_w
        height = (height or _DEFAULT_RESOLUTION) // _MULTIPLE_OF * _MULTIPLE_OF
        width = (width or _DEFAULT_RESOLUTION) // _MULTIPLE_OF * _MULTIPLE_OF
        latent_h, latent_w = height // _VAE_SCALE_FACTOR, width // _VAE_SCALE_FACTOR
        image_seq_len = latent_h * latent_w
        num_steps = sp.num_inference_steps or _DEFAULT_STEPS
        dtype = self.transformer.dtype
        device = self.device
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        latent_hw = (latent_h, latent_w)
        if not images:
            prefixes = [
                self._prefix_t2i(p, latent_hw, device)
                for p in (prompt, negative_prompt)[: 1 + do_cfg]
            ]
        else:
            condition = self._encode_condition_images(images, device)
            prefixes = [
                self._prefix_edit(p, condition, latent_hw, device)
                for p in (prompt, negative_prompt)[: 1 + do_cfg]
            ]
        text_k, text_v, img_cos, img_sin, prefix_len = prefixes[0]
        _wait_for(prefixes[-1][1])
        timings["text_encode_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        generator = sp.generator[0] if isinstance(sp.generator, list) else sp.generator
        latents = self._initial_latents(generator, latent_h, latent_w, dtype).to(device)
        timesteps, sigmas = self._timesteps(num_steps, image_seq_len, sp.sigmas)
        # The reference passes the timestep in the latents dtype, divided by 1000.
        step_inputs = [
            (
                (t.expand(1).to(dtype) / 1000).to(device),
                (sigmas[i + 1] - sigmas[i]).reshape(1).to(torch.float32).to(device),
            )
            for i, t in enumerate(timesteps)
        ]
        img_cos, img_sin = img_cos.to(device), img_sin.to(device)
        kv_len = torch.tensor([image_seq_len + prefix_len], dtype=torch.int32).to(device)
        if do_cfg:
            neg_k, neg_v, neg_cos, neg_sin, neg_len = prefixes[1]
            neg_cos, neg_sin = neg_cos.to(device), neg_sin.to(device)
            neg_kv_len = torch.tensor([image_seq_len + neg_len], dtype=torch.int32).to(device)
            scale = torch.tensor([cfg_scale], dtype=torch.float32).to(device)
        for timestep, dt in step_inputs:
            if do_cfg:
                latents = self.transformer.step_cfg(
                    latents,
                    timestep,
                    dt,
                    text_k,
                    text_v,
                    neg_k,
                    neg_v,
                    img_cos,
                    img_sin,
                    neg_cos,
                    neg_sin,
                    kv_len,
                    neg_kv_len,
                    scale,
                )
            else:
                latents = self.transformer.step(
                    latents, timestep, dt, text_k, text_v, img_cos, img_sin, kv_len
                )
            # Bounds the number of queued step executions.
            _wait_for(latents)
        timings["denoise_seconds"] = time.perf_counter() - t0

        output = None
        t0 = time.perf_counter()
        z = latents.cpu().transpose(1, 2).reshape(1, self.latent_channels, 1, latent_h, latent_w)
        if self.vae.num_tiles(latent_h, latent_w) > 1:
            image = self.vae.decode_image(z, self._vae_rank, self._vae_world, self._vae_cpu_group)
        elif self.is_output_rank:
            image = self.vae.decode_image(z.to(device))
        if self.is_output_rank:
            output = image[:, :, 0].cpu().float()
            if mask_union is not None:
                original = self.image_processor.preprocess(
                    target.convert("RGBA"), height=height, width=width
                )
                output = composite(output, original.float(), mask_union)
            timings["vae_decode_seconds"] = time.perf_counter() - t0
            timings["num_steps"] = len(step_inputs)
            self.last_stage_seconds = timings
            logger.info(
                "Qwen-Image 2.1 %dx%d, %d steps: %s",
                height,
                width,
                len(step_inputs),
                {k: round(v, 3) for k, v in timings.items()},
            )
        return DiffusionOutput(output=output)
