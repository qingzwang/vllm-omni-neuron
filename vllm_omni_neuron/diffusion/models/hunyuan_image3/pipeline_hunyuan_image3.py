# SPDX-License-Identifier: Apache-2.0
"""NeuronHunyuanImage3Pipeline — HunyuanImage-3.0 text-to-image and editing for Neuron.

Replaces vllm-omni's ``HunyuanImage3Pipeline`` (architecture ``HunyuanImage3ForCausalMM``)
and follows the official ``generate_image`` of the checkpoints with direct generation
(no think/recaption pass) at an explicit or derived size. For the base checkpoint
(text-to-image, ``pretrain`` template, no system prompt):

1. The prompt becomes ``<bos> prompt <boi> <img_size_1024> <img_ratio_k>``; the
   unconditional (CFG) row replaces every prompt token with ``<cfg>``. The rows are
   right-padded to the static ``max_text_tokens`` and embedded on the host.
2. The transformer's prefix graph turns them into every layer's text K/V, once.
3. The step graph runs each denoising step over the timestep token and the image
   tokens of both rows, guides the velocity and applies the Euler update in-graph.
4. The VAE decoder (compiled per stage) decodes the last frame, its up levels split
   into row strips over all ranks.

The Instruct checkpoint (``instruct`` template, unified system prompt) also takes
condition images (``multi_modal_data["image"]``) to edit: each is encoded by the VAE
encoder (latent tokens at t = 0) and the SigLIP2 tower, and the prefix graph runs the
system prompt, the condition images and the instruction, with each image's tokens
attending to each other bidirectionally (see ``hunyuan_image3_edit``). Steps 2-4 are
unchanged.

Sizes snap to the checkpoint's resolution buckets around 1024x1024 (as the official
image processor does); an edit defaults to the first condition image's bucket. The
flow-matching schedule is the official one (``sigma = linspace(1, 0, steps + 1)`` with
shift 3), as is the guidance ``uncond + scale * (cond - uncond)`` (scale 5, or 2.5 for
Instruct, by default; taken from the checkpoint's generation config).
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
from transformers import PreTrainedTokenizerFast, Siglip2ImageProcessor
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.hunyuan_image3.system_prompt import get_system_prompt

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_hunyuanimage3 import (
    HunyuanImage3Decoder,
    HunyuanImage3Encoder,
)
from vllm_omni_neuron.diffusion.models.hunyuan_image3 import hunyuan_image3_edit as edit
from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_expert_load import (
    INSTRUCT_EXPERT_LOAD,
)
from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    Checkpoint,
    NeuronHunyuanImage3Model,
    step_layout,
    text_rope,
)
from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_vision import (
    HunyuanImage3VisionEncoder,
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
# Instruct prefixes (system prompt, condition images) are padded to a multiple of this,
# so a new prefix graph compiles only when the padded length changes.
_PREFIX_MULTIPLE = 1024
# Think / recaption: the official limit of generated text tokens.
_AR_MAX_NEW_TOKENS = 2048
# On-device text sampling: decode steps queued per host sync, and the top-k candidates
# (the checkpoint's top_k is 1024; at temperature 0.6 / top_p 0.95 the nucleus is far
# inside the top 256).
_DECODE_CHUNK = 16
_DECODE_IN_FLIGHT = 4
_DEVICE_TOP_K = 256
_BOT_TASKS = ("image", "recaption", "think", "think_recaption")

_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]
_VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]
_RECOMPILE_LIMIT = 64


def get_hunyuan_image3_pre_process_func(od_config):
    """No pre-processing: the pipeline prepares condition images itself (this replaces
    vllm-omni's)."""

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
    """HunyuanImage-3.0 text-to-image and (Instruct) image editing on Neuron."""

    # Requests may carry condition images (multi_modal_data["image"]) to edit.
    support_image_input = True

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
        gen_path = os.path.join(self.model_path, "generation_config.json")
        gen = json.load(open(gen_path)) if os.path.exists(gen_path) else {}
        self.default_steps = int(gen.get("diff_infer_steps", _DEFAULT_STEPS))
        self.default_guidance = float(gen.get("diff_guidance_scale", _DEFAULT_GUIDANCE))
        # Sampling of the think / recaption text (the checkpoint's generation config).
        self.temperature = float(gen.get("temperature", 0.6))
        self.top_k = int(gen.get("top_k", 1024))
        self.top_p = float(gen.get("top_p", 0.95))
        # The Instruct checkpoint: instruct template, system prompt, condition images.
        self.instruct = gen.get("sequence_template") == "instruct"
        if self.instruct:
            self.edit_buckets = edit.instruct_buckets(self.buckets)
            self.tokens = edit.InstructTokens(self.tokenizer, self.bos_id)
            prompt = get_system_prompt(gen.get("use_system_prompt", "None"), "image", None)
            self.system_prompt = (prompt or "").strip()
            self.vit_processor = Siglip2ImageProcessor.from_dict(config.vit_processor)

        # Instruct also holds the vision tower, the VAE encoder and the larger edit
        # graphs: shard its shared expert over TP to keep memory headroom.
        self.transformer = NeuronHunyuanImage3Model(
            config,
            dtype=dtype,
            num_layers=num_layers,
            shared_sequence_parallel=not self.instruct,
            with_lm_head=self.instruct,
        )
        self.latent_channels = self.transformer.latent_channels
        self.is_output_rank = not dist.is_initialized() or dist.get_rank() == 0
        # Every rank decodes a row strip of the image.
        self.vae = HunyuanImage3Decoder(config.vae, dtype=dtype)
        world = get_world_group()
        self._world_cpu_group = world.cpu_group
        self.vae.set_parallel(world.device_group, world.world_size, world.rank_in_group)
        self.vision = self.vae_encoder = None
        if self.instruct:
            # Condition images: the SigLIP2 tower (replicated) and the VAE encoder (row
            # strips over all ranks, like the decoder).
            self.vision = HunyuanImage3VisionEncoder(config.vit, config.vit_aligner, dtype)
            self.vae_encoder = HunyuanImage3Encoder(config.vae, dtype=dtype)
            self.vae_encoder.set_parallel(world.device_group, world.world_size, world.rank_in_group)
        self.image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)
        self.last_stage_seconds: dict[str, float] = {}

    def _components(self):
        return [m for m in (self.transformer, self.vae, self.vision, self.vae_encoder) if m]

    def to(self, *args, **kwargs):
        for module in self._components():
            module.to(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        # The Instruct experts are balanced over the ranks by their measured load (the
        # busiest rank sets each MoE layer's time); the base model keeps the plain order.
        load = INSTRUCT_EXPERT_LOAD if self.instruct else None
        self.transformer.load_weights(self.model_path, expert_load=load)
        ckpt = Checkpoint(self.model_path)
        self.vae.load_weights(ckpt)
        if self.instruct:
            self.vision.load_weights(ckpt)
            self.vae_encoder.load_weights(ckpt)
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
        if self.instruct:
            self.vision.compile(*args, options=_opts("hunyuan_image3_vision"), **kwargs)
            self.vae_encoder.compile(
                *args, options=_opts("hunyuan_image3_vae_encoder", _VAE_COMPILER_ARGS), **kwargs
            )
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

    @staticmethod
    def _condition_images(prompt) -> list:
        if isinstance(prompt, str):
            return []
        images = (prompt.get("multi_modal_data") or {}).get("image")
        if images is None:
            return []
        return list(images) if isinstance(images, (list, tuple)) else [images]

    def _own_rows(self, ids: torch.Tensor, cfg: bool) -> torch.Tensor:
        """Under CFG parallelism this replica's row (both run the conditional row when
        guidance is off)."""
        model = self.transformer
        if model.cfg_size == 1:
            return ids
        return ids[model.cfg_rank : model.cfg_rank + 1] if cfg else ids[:1]

    def _encode_conditions(self, conditions, generator):
        """Condition images: VAE latents (the official sampling noise is drawn first,
        image by image, from the request's generator) and SigLIP2 tokens."""
        device = self.device
        latents, vit_tokens = [], []
        for cond in conditions:
            h, w = cond.latent_hw
            eps = torch.randn((1, self.latent_channels, h, w), generator=generator)
            latents.append(self.vae_encoder.encode(cond.vae_pixels.to(device), eps.to(device)))
            vit_tokens.append(self.vision(cond.vit_pixels, cond.vit_hw, device))
            _wait_for(vit_tokens[-1])
        return latents, vit_tokens

    def _edit_inputs(self, prefix, conditions, rows: torch.Tensor):
        """Device inputs of the prefix graphs for ``rows`` padded to a multiple of 1024."""
        model, device = self.transformer, self.device
        length = prefix.ids.shape[1]
        padded = -(-length // _PREFIX_MULTIPLE) * _PREFIX_MULTIPLE
        ids = torch.full((rows.shape[0], padded), self.pad_id, dtype=torch.long)
        ids[:, :length] = rows
        cos, sin = edit.prefix_rope(prefix, conditions, model.head_dim, model.rope_theta, padded)
        bound_max = edit.prefix_bounds(prefix, padded)
        layout = tuple(zip(prefix.timestep_pos, prefix.vae_pos, prefix.vit_pos))
        return (
            model.embed(ids).to(device),
            cos.to(device),
            sin.to(device),
            bound_max.to(device),
            torch.zeros(1).to(device),
            layout,
            padded,
        )

    def _prefix_instruct(self, prompt, conditions, encoded, ratio_index, cfg, cot=None):
        """Instruct prefix K/V: system prompt, condition images, instruction (and the
        think / recaption text)."""
        model = self.transformer
        prefix = edit.edit_prefix(
            self.tokens, self.system_prompt, prompt, conditions, ratio_index, cfg, cot=cot
        )
        rows = self._own_rows(prefix.ids, cfg)
        embeds, cos, sin, bound_max, t0, layout, _ = self._edit_inputs(prefix, conditions, rows)
        latents, vit_tokens = encoded
        text_k, text_v = model.prefix_edit(
            embeds, cos, sin, bound_max, t0, latents, vit_tokens, layout
        )
        return text_k, text_v, prefix.ids.shape[1], rows.shape[0]

    def _sample(self, logits: torch.Tensor, generator) -> int:
        """Temperature, top-k, then top-p sampling (as the official ``generate``)."""
        logits = logits.float() / self.temperature
        values, index = torch.topk(logits, min(self.top_k, logits.numel()))
        probs = values.softmax(dim=-1)
        # Keep the smallest prefix of the sorted tokens whose probability reaches top_p.
        keep = (probs.cumsum(dim=-1) - probs) < self.top_p
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        choice = torch.multinomial(probs / probs.sum(), 1, generator=generator)
        return int(index[choice])

    def _generate_text(
        self, prompt, conditions, encoded, bot_task, predict_ratio, generator, max_new_tokens
    ):
        """Think / recaption text, and with ``predict_ratio`` the output's ratio bucket.

        As the official ``generate_image``: the reply starts with ``<think>`` (think
        modes) or ``<recaption>``; in think_recaption ``</think>`` is followed by
        ``<recaption>``; the text ends at ``</recaption>`` (or ``</think>`` for think).
        To predict the size, ``<answer> <boi> <img_size_1024>`` follow and the ratio
        token is the most likely of the ratio tokens. Returns the sections
        ``[(open tag, text, close tag)]``, the ratio index (or None) and the number of
        decoded tokens.
        """
        t, model, device = self.tokens, self.transformer, self.device
        trigger = t.recaption if bot_task == "recaption" else t.think
        prefix = edit.edit_prefix(
            self.tokens, self.system_prompt, prompt, conditions, None, False, ar_trigger=trigger
        )
        length = prefix.ids.shape[1]
        embeds, cos, sin, bound_max, t0, layout, padded = self._edit_inputs(
            prefix, conditions, prefix.ids
        )
        cache_len = padded + _AR_MAX_NEW_TOKENS + _PREFIX_MULTIPLE
        latents, vit_tokens = encoded
        # Every rank runs the text graphs (their collectives need all ranks); the logits
        # are averaged over the CFG replicas in-graph, so all ranks sample the same text.
        k, v, logits = model.ar_prefill(
            embeds,
            cos,
            sin,
            bound_max,
            t0,
            latents,
            vit_tokens,
            layout,
            torch.tensor([length - 1]).to(device),
            cache_len,
        )
        stops = {t.recaption_end} if bot_task != "think" else {t.think_end, t.recaption_end}
        cos_table, sin_table = text_rope(cache_len, model.head_dim, model.rope_theta)
        cos_table, sin_table = cos_table.to(device), sin_table.to(device)
        # The first token is sampled from the prefill logits on the host (every rank
        # takes rank 0's); the rest are sampled on device, _DECODE_CHUNK steps queued at
        # a time, until a stop tag or the budget.
        first = [self._sample(logits.cpu(), generator)]
        if dist.is_initialized():
            dist.broadcast_object_list(first, src=0, group=self._world_cpu_group)
        generated: list[int] = [trigger, first[0]]
        forced_after = t.think_end if bot_task == "think_recaption" else -1
        sampling = (self.temperature, _DEVICE_TOP_K, self.top_p, forced_after, t.recaption)
        lead = float(not dist.is_initialized() or dist.get_rank() == 0)
        ranks = (
            torch.tensor(model.tp_rank, dtype=torch.int32).to(device),
            torch.tensor(lead).to(device),
        )
        steps = max_new_tokens + _DECODE_CHUNK
        tokens = torch.zeros(steps, dtype=torch.int32).to(device)
        token = torch.tensor(first, dtype=torch.int32).to(device)
        position, step = torch.tensor([length]).to(device), torch.tensor([0]).to(device)
        done, submitted, in_flight = generated[-1] in stops, 0, []
        while not done and len(generated) <= max_new_tokens:
            for _ in range(min(_DECODE_CHUNK, steps - submitted)):
                token, position, step, k, v, tokens = model.decode_sample(
                    token, position, step, k, v, cos_table, sin_table, tokens, ranks, sampling
                )
                # At most _DECODE_IN_FLIGHT steps queued (Lite's execution queue is short).
                in_flight.append(token)
                if len(in_flight) > _DECODE_IN_FLIGHT:
                    _wait_for(in_flight.pop(0))
            new = tokens[submitted : submitted + _DECODE_CHUNK].cpu().tolist()
            submitted += _DECODE_CHUNK
            for value in new:
                generated.append(value)
                if value in stops or len(generated) > max_new_tokens:
                    done = True
                    break
        if generated[-1] not in stops:  # out of budget: close the open section
            close = t.recaption_end if t.recaption in generated else t.think_end
            generated.append(close)
        position = length + len(generated) - 2  # where the closing tag is decoded
        ratio_index = None
        if predict_ratio:
            # <close tag> <answer> <boi> <img_size_1024>, then the most likely ratio token.
            # The positions after the closing tag may hold the KV of tokens sampled past
            # it; they are overwritten or masked.
            position_dev = torch.tensor([position]).to(device)
            for token_id in (generated[-1], t.answer, t.boi, t.size):
                embed = model.embed(torch.tensor([[token_id]]))[0].to(device)
                logits, k, v, position_dev = model.decode(
                    embed, cos_table, sin_table, k, v, position_dev
                )
            ratio_index = int(logits.cpu()[torch.tensor(t.ratio_ids)].argmax())
        # Every rank continues with rank 0's text (a safeguard; the texts are identical).
        if dist.is_initialized():
            shared = [generated, ratio_index]
            dist.broadcast_object_list(shared, src=0, group=self._world_cpu_group)
            generated, ratio_index = shared
        sections, current, opening = [], [], None
        for token in generated:
            if token in (t.think, t.recaption):
                opening, current = token, []
            elif token in (t.think_end, t.recaption_end) and opening is not None:
                text = self.tokenizer.decode(current, skip_special_tokens=False)
                sections.append((opening, text, token))
                opening = None
            elif opening is not None:
                current.append(token)
        return sections, ratio_index, len(generated) - 1

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
        images = self._condition_images(req.prompts[0])
        bot_task = (sp.extra_args or {}).get("bot_task", "image")
        if bot_task not in _BOT_TASKS:
            raise ValueError(f"bot_task must be one of {_BOT_TASKS}, got {bot_task!r}")
        if bot_task != "image" and not self.instruct:
            raise NotImplementedError(f"bot_task={bot_task!r} needs the Instruct checkpoint")
        if images and not self.instruct:
            raise NotImplementedError(
                "image editing needs the HunyuanImage-3.0-Instruct checkpoint"
            )
        conditions = [
            edit.prepare_condition(image, self.edit_buckets, self.vit_processor) for image in images
        ]
        buckets = self.edit_buckets if self.instruct else self.buckets
        if sp.seed is not None:
            generator = torch.Generator("cpu").manual_seed(sp.seed)
            text_generator = torch.Generator("cpu").manual_seed(sp.seed)
        else:
            generator = sp.generator[0] if isinstance(sp.generator, list) else sp.generator
            text_generator = torch.Generator("cpu").manual_seed(torch.seed())
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        encoded = self._encode_conditions(conditions, generator) if self.instruct else None
        timings["condition_seconds"] = time.perf_counter() - t_start
        cot = None
        predicted = None
        if bot_task != "image":
            # Think / recaption: generate the text first (and the size, if not given).
            t0 = time.perf_counter()
            max_new = min(
                int((sp.extra_args or {}).get("max_new_tokens", _AR_MAX_NEW_TOKENS)),
                _AR_MAX_NEW_TOKENS,
            )
            cot, predicted, num_tokens = self._generate_text(
                prompt,
                conditions,
                encoded,
                bot_task,
                not (sp.height and sp.width),
                text_generator,
                max_new,
            )
            timings["text_seconds"] = time.perf_counter() - t0
            timings["text_tokens"] = num_tokens
            logger.info(
                "HunyuanImage-3.0 %s: %d tokens in %.2f s; %s",
                bot_task,
                num_tokens,
                timings["text_seconds"],
                " | ".join(text[:300] for _, text, _ in cot),
            )
        if predicted is not None:
            req_h, req_w = buckets[predicted]
        elif conditions and not (sp.height and sp.width):
            # As vllm-omni: the output takes the first condition image's bucket.
            req_h, req_w = buckets[conditions[0].ratio_index]
        else:
            req_h, req_w = sp.height or _BASE_SIZE, sp.width or _BASE_SIZE
        height, width, ratio_index = snap_size(req_h, req_w, buckets)
        if (height, width) != (req_h, req_w):
            logger.info(
                "HunyuanImage-3.0: %dx%d snaps to the %dx%d bucket", req_h, req_w, height, width
            )
        latent_h, latent_w = height // _VAE_SCALE_FACTOR, width // _VAE_SCALE_FACTOR
        if self.instruct:
            # Prefix (system prompt, condition images, instruction) + the generated image.
            total = len(self.tokenizer.encode(self.system_prompt + prompt)) + 16
            total += sum(len(self.tokenizer.encode(text)) + 2 for _, text, _ in cot or [])
            total += sum(c.latent_hw[0] * c.latent_hw[1] + 1030 for c in conditions)
            total += latent_h * latent_w
            if total > self.config.max_position_embeddings:
                raise ValueError(
                    f"{len(conditions)} condition images and the output need about {total} "
                    f"positions; the model has {self.config.max_position_embeddings}"
                )
        num_steps = sp.num_inference_steps or self.default_steps
        guidance = (
            sp.guidance_scale
            if getattr(sp, "guidance_scale_provided", False)
            else self.default_guidance
        )
        do_cfg = guidance > 1.0
        model = self.transformer
        device = self.device

        t0 = time.perf_counter()
        if self.instruct:
            text_k, text_v, prefix_len, rows = self._prefix_instruct(
                prompt, conditions, encoded, ratio_index, do_cfg, cot
            )
        else:
            ids, prefix_len = self._prefix_ids(prompt, ratio_index, do_cfg)
            ids = self._own_rows(ids, do_cfg)
            rows = ids.shape[0]
            embeds = model.embed(ids).to(device)
            cos, sin = text_rope(self.max_text_tokens, model.head_dim, model.rope_theta)
            text_k, text_v = model.prefix(embeds, cos.to(device), sin.to(device))
        _wait_for(text_k)
        timings["prefix_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # As the official pipeline: bf16 noise (drawn after the condition latents'),
        # float32 latents from the first update on.
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
            rows=rows,
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
