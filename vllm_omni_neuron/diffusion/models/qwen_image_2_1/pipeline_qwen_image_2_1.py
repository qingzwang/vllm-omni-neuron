# SPDX-License-Identifier: Apache-2.0
"""NeuronQwenImage21Pipeline — Qwen-Image 2.1 text-to-image pipeline for Neuron.

vllm-omni 0.24 (and the diffusers release it pins) has no Qwen-Image 2.1 pipeline, so
this is a standalone pipeline that mirrors diffusers' ``QwenImage21Pipeline`` for
text-to-image without classifier-free guidance (the model is meant to be sampled
without it). Per request:

1. Tokenize the prompt with the T2I chat template; the Qwen3-VL text encoder (TP,
   compiled) returns its pre-norm hidden states at a static length.
2. The DiT's prefix graph turns the text (minus the system prompt) into every layer's
   text K/V, once.
3. The DiT's step graph runs each denoising step over the image tokens only and applies
   the Euler update in-graph; the latents stay on device for the whole loop.
4. The VAE decoder (compiled per stage) decodes the latents: on rank 0 up to a 64x64
   latent, above that as exact tiles spread over the TP ranks.

Timesteps, sigmas, RoPE tables and the initial noise are computed on the host exactly
as the reference does, so a seed gives the reference's starting point.
"""

import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torch import nn
from transformers import AutoProcessor
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
from vllm_omni_neuron.diffusion.models.qwen_image_2_1.qwen_image_2_1_transformer import (
    QwenImage21Transformer2DModel,
    qwen_image21_rope,
)
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
    """Qwen-Image 2.1 text-to-image on Neuron (TP, optional CP; no CFG)."""

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
        for module in (self.text_encoder, self.transformer, self.vae):
            module.to(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        self.text_encoder.load_weights(os.path.join(self.model_path, "text_encoder"))
        self.transformer.load_weights(os.path.join(self.model_path, "transformer"))
        return None

    def compile(self, *args, **kwargs):
        options = dict(kwargs.pop("options", None) or {})
        kwargs.setdefault("fullgraph", True)
        kwargs.setdefault("dynamic", False)

        def _opts(name, compiler_args=_COMPILER_ARGS):
            return {**options, "model_name": name, "compiler_args": list(compiler_args)}

        self.text_encoder.compile(*args, options=_opts("qwen_image21_text_encoder"), **kwargs)
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

    def forward(self, req) -> DiffusionOutput:
        if req.request_ids == ["dummy_req_id"] and self._prompt_of(req.prompts[0]) == "dummy run":
            logger.info("Skipping warmup request on Neuron pipeline")
            return DiffusionOutput(output=None)
        if len(req.prompts) != 1:
            raise NotImplementedError("Qwen-Image 2.1 on Neuron runs one prompt per request")

        sp = req.sampling_params
        if (sp.true_cfg_scale or 1.0) > 1.0 and isinstance(req.prompts[0], dict):
            if req.prompts[0].get("negative_prompt") is not None:
                logger.warning("Qwen-Image 2.1 on Neuron ignores CFG (true_cfg_scale > 1)")
        if (sp.num_outputs_per_prompt or 1) > 1:
            raise NotImplementedError("num_outputs_per_prompt > 1 is not supported")

        height = (sp.height or _DEFAULT_RESOLUTION) // _MULTIPLE_OF * _MULTIPLE_OF
        width = (sp.width or _DEFAULT_RESOLUTION) // _MULTIPLE_OF * _MULTIPLE_OF
        latent_h, latent_w = height // _VAE_SCALE_FACTOR, width // _VAE_SCALE_FACTOR
        image_seq_len = latent_h * latent_w
        num_steps = sp.num_inference_steps or _DEFAULT_STEPS
        dtype = self.transformer.dtype
        device = self.device
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        input_ids, text_len = self._tokenize(self._prompt_of(req.prompts[0]))
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
        _wait_for(text_v)
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
        kv_len = torch.tensor([image_seq_len + text_len], dtype=torch.int32).to(device)
        for timestep, dt in step_inputs:
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
