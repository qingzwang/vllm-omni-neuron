# SPDX-License-Identifier: Apache-2.0
"""NeuronHunyuanImage3Pipeline — HunyuanImage-3.0 text-to-image for Neuron.

Replaces vllm-omni's ``HunyuanImage3Pipeline`` (architecture ``HunyuanImage3ForCausalMM``)
and follows the official ``generate_image`` of the checkpoint for text-to-image with an
explicit size (no think/recaption pass, no system prompt, ``pretrain`` template):

1. The prompt becomes ``<bos> prompt <boi> <img_size_1024> <img_ratio_k>``; the
   unconditional (CFG) row replaces every prompt token with ``<cfg>``. The rows are
   right-padded to the static ``max_text_tokens`` and embedded on the host.
2. The transformer's prefix graph turns them into every layer's text K/V, once.
3. The step graph runs each denoising step over the timestep token and the image
   tokens of both rows, guides the velocity and applies the Euler update in-graph.
4. The VAE decoder (compiled per stage) decodes the last frame, its up levels split
   into row strips over all ranks.

Sizes snap to the checkpoint's 33 resolution buckets around 1024x1024 (as the official
image processor does); the flow-matching schedule is the official one
(``sigma = linspace(1, 0, steps + 1)`` with shift 3), as is the guidance
``uncond + scale * (cond - uncond)`` (scale 5 by default).
"""

import json
import logging
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from diffusers.image_processor import VaeImageProcessor
from torch import nn
from transformers import PreTrainedTokenizerFast
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.distributed.utils import get_local_device

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_hunyuanimage3 import (
    HunyuanImage3Decoder,
)
from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    Checkpoint,
    NeuronHunyuanImage3Model,
    step_layout,
    text_rope,
)
from vllm_omni_neuron.lite_compat import is_lite_runtime

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        "model_arch": "HunyuanImage3ForCausalMM",
        "class_name": "NeuronHunyuanImage3Pipeline",
        "pre_process_func_name": "get_hunyuan_image3_pre_process_func",
        "post_process_func_name": "get_hunyuan_image3_post_process_func",
    },
]

_VAE_SCALE_FACTOR = 16
_BASE_SIZE = 1024
_DEFAULT_STEPS = 50
_DEFAULT_GUIDANCE = 5.0
_FLOW_SHIFT = 3.0
# Static number of prefix tokens (<bos>, prompt, <boi>, size and ratio tokens) in the
# compiled graphs unless the stage config's model_config sets ``max_text_tokens``.
_DEFAULT_MAX_TEXT_TOKENS = 1024

_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]
_VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]
_RECOMPILE_LIMIT = 64


def get_hunyuan_image3_pre_process_func(od_config):
    """Text-to-image requests need no pre-processing (replaces vllm-omni's image-conditioning one)."""

    def pre_process_func(request):
        return request

    return pre_process_func


def get_hunyuan_image3_post_process_func(od_config):
    image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


def resolution_buckets(base: int = _BASE_SIZE, step: int | None = None) -> list[tuple[int, int]]:
    """The official ``ResolutionGroup``: ``(height, width)`` from 512x2048 to 2048x512, by ratio."""
    step = step or base // 16
    sizes = [(base, base)]
    h = w = base
    while not (h >= 2 * base and w <= base // 2):
        h, w = min(h + step, 2 * base), max(w - step, base // 2)
        sizes.append((h, w))
    h = w = base
    while not (h <= base // 2 and w >= 2 * base):
        h, w = max(h - step, base // 2), min(w + step, 2 * base)
        sizes.append((h, w))
    return sorted(sizes, key=lambda s: s[0] / s[1])


def snap_size(height: int, width: int, buckets) -> tuple[int, int, int]:
    """The bucket nearest in aspect ratio: ``(height, width, ratio_index)``."""
    ratio = height / width
    index = min(range(len(buckets)), key=lambda i: abs(buckets[i][0] / buckets[i][1] - ratio))
    return (*buckets[index], index)


def flow_sigmas(num_steps: int, shift: float = _FLOW_SHIFT) -> torch.Tensor:
    """Official ``FlowMatchDiscreteScheduler`` sigmas: ``linspace(1, 0, n + 1)``, SD3 shift."""
    sigmas = torch.linspace(1, 0, num_steps + 1)
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def _wait_for(tensor: torch.Tensor) -> None:
    """Block until ``tensor`` is computed (bounds Lite's asynchronous execution queue)."""
    if is_lite_runtime():
        source = tensor.view(-1)[:1]
        torch.empty_like(source).copy_(source)


def _model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


class NeuronHunyuanImage3Pipeline(nn.Module):
    """HunyuanImage-3.0 text-to-image on Neuron (TP + expert parallelism)."""

    weights_sources: list = []

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = _model_path(od_config.model)
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model_config = dict(od_config.model_config or {})
        self.max_text_tokens = int(model_config.pop("max_text_tokens", _DEFAULT_MAX_TEXT_TOKENS))
        # Development only: run the first ``num_layers`` language-model layers.
        num_layers = model_config.pop("num_layers", None)

        # The checkpoint's config and tokenizer classes are remote code; plain JSON
        # and the fast tokenizer are all this pipeline needs.
        with open(os.path.join(self.model_path, "config.json")) as f:
            config = SimpleNamespace(**json.load(f))
        self.config = config
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(self.model_path, "tokenizer.json")
        )
        ids = self.tokenizer.convert_tokens_to_ids
        self.bos_id = config.bos_token_id
        self.boi_id, self.cfg_id = ids("<boi>"), ids("<cfg>")
        self.size_id = ids(f"<img_size_{_BASE_SIZE}>")
        self.ratio_offset = ids("<img_ratio_0>")
        self.pad_id = config.pad_token_id
        self.buckets = resolution_buckets()

        self.transformer = NeuronHunyuanImage3Model(config, dtype=dtype, num_layers=num_layers)
        self.latent_channels = self.transformer.latent_channels
        self.is_output_rank = not dist.is_initialized() or dist.get_rank() == 0
        # Every rank decodes a row strip of the image.
        self.vae = HunyuanImage3Decoder(config.vae, dtype=dtype)
        world = get_world_group()
        self.vae.set_parallel(world.device_group, world.world_size, world.rank_in_group)
        self.image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)
        self.last_stage_seconds: dict[str, float] = {}

    def to(self, *args, **kwargs):
        self.transformer.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        self.transformer.load_weights(self.model_path)
        self.vae.load_weights(Checkpoint(self.model_path))
        return None

    def compile(self, *args, **kwargs):
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

        self.transformer.compile(
            *args,
            prefix_options=_opts("hunyuan_image3_prefix"),
            step_options=_opts("hunyuan_image3_step"),
            **kwargs,
        )
        self.vae.compile(*args, options=_opts("hunyuan_image3_vae", _VAE_COMPILER_ARGS), **kwargs)
        return self

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_of(prompt) -> str:
        return prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")

    def _prefix_ids(self, prompt: str, ratio_index: int, cfg: bool) -> tuple[torch.Tensor, int]:
        """Right-padded ``[R, max_text_tokens]`` prefix ids and the valid length."""
        text = self.tokenizer.encode(prompt, add_special_tokens=False)
        tail = [self.boi_id, self.size_id, self.ratio_offset + ratio_index]
        rows = [[self.bos_id, *text, *tail]]
        if cfg:
            rows.append([self.bos_id, *([self.cfg_id] * len(text)), *tail])
        length = len(rows[0])
        if length > self.max_text_tokens:
            raise ValueError(
                f"prompt is {length} tokens with its special tokens; the compiled prefix holds "
                f"{self.max_text_tokens} (raise model_config.max_text_tokens)"
            )
        ids = torch.full((len(rows), self.max_text_tokens), self.pad_id, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, :length] = torch.tensor(row)
        return ids, length

    def forward(self, req) -> DiffusionOutput:
        if req.request_ids == ["dummy_req_id"] and self._prompt_of(req.prompts[0]) == "dummy run":
            logger.info("Skipping warmup request on Neuron pipeline")
            return DiffusionOutput(output=None)
        if len(req.prompts) != 1:
            raise NotImplementedError("HunyuanImage-3.0 on Neuron runs one prompt per request")
        sp = req.sampling_params
        if (sp.num_outputs_per_prompt or 1) > 1:
            raise NotImplementedError("num_outputs_per_prompt > 1 is not supported")
        prompt = self._prompt_of(req.prompts[0])
        if not isinstance(req.prompts[0], str) and (req.prompts[0].get("multi_modal_data") or {}):
            raise NotImplementedError("HunyuanImage-3.0 on Neuron supports text-to-image only")

        req_h, req_w = sp.height or _BASE_SIZE, sp.width or _BASE_SIZE
        height, width, ratio_index = snap_size(req_h, req_w, self.buckets)
        if (height, width) != (req_h, req_w):
            logger.info(
                "HunyuanImage-3.0: %dx%d snaps to the %dx%d bucket", req_h, req_w, height, width
            )
        latent_h, latent_w = height // _VAE_SCALE_FACTOR, width // _VAE_SCALE_FACTOR
        num_steps = sp.num_inference_steps or _DEFAULT_STEPS
        guidance = (
            sp.guidance_scale
            if getattr(sp, "guidance_scale_provided", False)
            else _DEFAULT_GUIDANCE
        )
        do_cfg = guidance > 1.0
        model = self.transformer
        device = self.device
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        ids, prefix_len = self._prefix_ids(prompt, ratio_index, do_cfg)
        if model.cfg_size > 1:
            # CFG parallelism: this replica's row (both replicas run the conditional
            # row when guidance is off).
            ids = ids[model.cfg_rank : model.cfg_rank + 1] if do_cfg else ids[:1]
        embeds = model.embed(ids).to(device)
        cos, sin = text_rope(self.max_text_tokens, model.head_dim, model.rope_theta)
        text_k, text_v = model.prefix(embeds, cos.to(device), sin.to(device))
        _wait_for(text_k)
        timings["prefix_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if sp.seed is not None:
            generator = torch.Generator("cpu").manual_seed(sp.seed)
        else:
            generator = sp.generator[0] if isinstance(sp.generator, list) else sp.generator
        # As the official pipeline: bf16 noise, float32 latents from the first update on.
        latents = torch.randn(
            (1, self.latent_channels, latent_h, latent_w), generator=generator, dtype=torch.bfloat16
        )
        latents = latents.float().to(device)
        cos, sin, bound_min, bound_max = step_layout(
            prefix_len,
            latent_h,
            latent_w,
            model.head_dim,
            model.rope_theta,
            rows=ids.shape[0],
            tp_size=model.tp_size,
        )
        cos, sin = cos.to(device), sin.to(device)
        bound_min, bound_max = bound_min.to(device), bound_max.to(device)
        scale = torch.tensor([guidance], dtype=torch.float32).to(device)
        sigmas = flow_sigmas(num_steps)
        steps = [
            (
                (sigmas[i : i + 1] * 1000).to(device),
                (sigmas[i + 1] - sigmas[i]).reshape(1).to(device),
            )
            for i in range(num_steps)
        ]
        for t, dt in steps:
            latents = model.step(
                latents, t, dt, text_k, text_v, cos, sin, bound_min, bound_max, scale, do_cfg
            )
            _wait_for(latents)
        timings["denoise_seconds"] = time.perf_counter() - t0

        output = None
        t0 = time.perf_counter()
        image = self.vae.decode(latents)
        if self.is_output_rank:
            output = image.cpu().float()
            timings["vae_decode_seconds"] = time.perf_counter() - t0
            timings["num_steps"] = num_steps
            self.last_stage_seconds = timings
            logger.info(
                "HunyuanImage-3.0 %dx%d, %d steps, guidance %.1f: %s",
                height,
                width,
                num_steps,
                guidance,
                {k: round(v, 3) for k, v in timings.items()},
            )
        return DiffusionOutput(output=output)
