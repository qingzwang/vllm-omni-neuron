# HunyuanImage-3.0 Model Card

<!-- meta: description: Model card for HunyuanImage-3.0 text-to-image and
HunyuanImage-3.0-Instruct image editing on AWS Trainium with the vLLM Omni Neuron plugin —
supported features, parallelism configurations, latency, accuracy against the official
reference, sample outputs, and known limitations. -->
<!-- meta: keywords: HunyuanImage-3.0, HunyuanImage-3.0-Instruct, Hunyuan, model card,
text-to-image, image editing, multi-image fusion, image generation,
diffusion, mixture of experts, MoE, vLLM, vLLM Omni, Neuron, Trainium, trn2, BF16, tensor
parallelism, expert parallelism, CFG parallelism, sequence parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-19 -->

## Introduction

[HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0) is a native multimodal
image generation model from Tencent Hunyuan. It generates an image with an 80B-parameter
mixture-of-experts language model (13B active per token: 32 layers, 64 experts with top-8
routing plus a shared expert): the image latents are tokens of the same sequence as the
prompt, attended causally by the text and bidirectionally among themselves, and every
denoising step runs the language model over them. Small convolutional heads embed the 32
latent channels into the model and project its output back to a velocity, and a 3-D
convolutional VAE (16x spatial) decodes the result.

HunyuanImage-3.0 **text-to-image** and HunyuanImage-3.0-Instruct **image editing** (one or
more reference images) are supported with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS
Trainium2 (`trn2`); the plugin replaces vLLM Omni's `HunyuanImage3Pipeline`
(`HunyuanImage3ForCausalMM`).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Precision |
|-------|-------------|----------|-----------|
| HunyuanImage-3.0 | [tencent/HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0) | Trn2 | BF16 |
| HunyuanImage-3.0-Instruct | [tencent/HunyuanImage-3.0-Instruct](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) | Trn2 | BF16 |

The checkpoints are released under the Tencent Hunyuan Community License; check its terms
(including its territory restrictions) before use.

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image | ✅ |
| | The 33 resolution buckets around 1024x1024 (512x2048 to 2048x512) | ✅ |
| | Classifier-free guidance (`guidance_scale`, default 5.0 / 2.5 for Instruct, as the checkpoints) | ✅ |
| | Image editing (Instruct; condition images in `multi_modal_data["image"]`) | ✅ |
| | Multi-image fusion (several condition images) | ✅ |
| | Think / recaption prompting (Instruct; `bot_task` `recaption`, `think_recaption`, `think`) | ✅ |
| | Automatic aspect ratio (predicted after the think / recaption text) | ✅ |
| **Precision** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP) of attention heads | ✅ |
| | Expert parallelism (EP) over the TP ranks | ✅ (always on) |
| | Megatron sequence parallelism (SP, within TP) | ✅ (always on) |
| | CFG parallelism (`cfg_parallel_size: 2`) | ✅ |
| | Parallel VAE decode (row strips over all ranks) | ✅ |
| **Compilation** | torch.compile | ✅ |
| **Serving** | Batch size | 1 |

## How it runs

For text-to-image with an explicit size the official `generate_image` builds one sequence
per guidance row:

```
<bos> prompt <boi> <img_size_1024> <img_ratio_k> <timestep> <img> x N <eoi>
```

(the unconditional row replaces each prompt token with `<cfg>`), and runs the model over
the timestep token and the N = (H/16)·(W/16) image tokens at every step. Text attends
causally and does not depend on the timestep, so its keys and values are computed once
(the official pipeline's KV cache). The Neuron pipeline runs three kinds of compiled
graphs:

1. **Prefix** — once per request, the prompt rows (static length `max_text_tokens`,
   default 1024) pass through the 32 layers and every layer's text K/V is kept.
2. **Step** — every denoising step embeds the latents (patch embed), runs the timestep
   token and the image tokens of both rows through the 32 layers against the cached text
   K/V (per-query attention bounds: image tokens see the image, the timestep token and the
   text; the timestep token only itself and the text), projects the velocity (final
   layer), applies the guidance and the flow-matching Euler update in-graph. The latents
   stay on device for the whole loop.
3. **VAE decoder** — the latent-resolution head (with its global attention) on every
   rank, then the up levels as row strips over all ranks with exact halos and
   all-reduced GroupNorm statistics (the last three levels in one graph, so their
   full-resolution intermediates are never gathered).

Parallel layout (TP=16 on one replica):

- **Attention**: 2 query heads per rank, each of the 8 KV heads on 2 ranks. One fused NKI
  kernel does the QKV projection, RoPE and the post-RoPE QK RMSNorm.
- **MoE**: the 64 experts are spread over the TP ranks (4 each). Router (float32
  logits) and blockwise token-to-expert mapping run in-graph; the NKI blockwise MoE kernel
  iterates only over the blocks that hold tokens. The shared expert runs
  sequence-parallel.
- **Sequence parallelism**: norms, residuals and the shared expert run on 1/TP of the
  tokens, with an all-gather before attention and the routed experts and a reduce-scatter
  after.
- **Diffusion heads**: their 3x3 convolutions (about 4 TFLOP per step) are split by input
  or output channels over the TP ranks.
- **CFG parallelism**: with `cfg_parallel_size: 2`, two TP groups run the conditional and
  the unconditional row, and exchange velocities each step.

The schedule is the checkpoint's own (`sigma = linspace(1, 0, steps + 1)` with shift 3,
50 steps); sizes snap to the nearest of the 33 resolution buckets by aspect ratio, as the
official image processor does.

**Image editing** (HunyuanImage-3.0-Instruct, direct generation as `bot_task="image"`)
changes only the prefix. The official Instruct sequence is

```
<bos> system prompt "User: "
  per condition image: <boi> <img_size_1024> <img_ratio_k> <timestep>
                       VAE latent tokens <joint_img_sep> 1024 SigLIP2 tokens <eoi>
  instruction "Assistant: " <answer> <boi> <img_size_1024> <img_ratio>
<timestep> <img> x N ...
```

with the checkpoint's unified system prompt (about 1200 tokens). Each condition image is
resized and centre-cropped to its resolution bucket and encoded by the **VAE encoder**
(latents sampled from the posterior, patch-embedded at timestep 0; the encoder's levels run
as row strips over all ranks, like the decoder), and its original pixels by the **SigLIP2
tower** and aligner (replicated, 1024 tokens with key padding). The prefix is causal
except inside each condition image (its VAE and SigLIP2 tokens attend to each other), via
per-query attention bounds; the unconditional row keeps the images and replaces the
instruction with `<cfg>`. Nothing in the prefix depends on the timestep, so its K/V are
computed once per request and the step graph is the text-to-image one. The output takes
the first condition image's bucket unless a size is given; the prefix is padded to a
multiple of 1024 tokens.

**Think / recaption** (Instruct, `extra_args={"bot_task": ...}`) first lets the model write
text, as the official `generate_image`: the reply starts with `<think>` (`think`,
`think_recaption`) or `<recaption>` (`recaption`); in `think_recaption` the model's
`</think>` is followed by `<recaption>`, and the text ends at `</recaption>` (at most 2048
tokens), sampled with the checkpoint's temperature 0.6, top-k 1024 and top-p 0.95 (seeded
by the request's seed). Without an explicit size, `<answer> <boi> <img_size_1024>` follow
and the most likely ratio token picks the output bucket. The image pass then uses the
sequence with the text after `"Assistant: "` (its unconditional row keeps the tags and
replaces the text with `<cfg>`). Two more graphs run the text: a prefill (the edit prefix
graph plus the final norm and the vocab-sharded LM head, filling a static KV cache of the
prefix + 2048 + 1024 positions) and a one-token decode step (attention over the cache,
the MoE layers at one token, and the LM head); the logits are sampled on the host. With
CFG parallelism the decode uses all 32 NeuronCores (see Performance).

## Running

```bash
# 1024x1024, 50 steps, guidance 5, TP=16 (examples/hunyuan_image3/hunyuan_image3_stage.yaml)
python examples/hunyuan_image3/run.py --prompt "A neon shop sign ..." --output out.png

# Other sizes snap to the nearest resolution bucket (here 768x1280)
python examples/hunyuan_image3/run.py --height 720 --width 1280

# Faster layouts: TP=16 x CFG=2 or TP=32 (32 NeuronCores)
python examples/hunyuan_image3/run.py --cfg-parallel-size 2
python examples/hunyuan_image3/run.py --tensor-parallel-size 32

# Warm latency: time 3 requests after the first (compiling) one
python examples/hunyuan_image3/run.py --profile --profile-runs 3
```

# Image editing with the Instruct checkpoint (TP=16 x CFG=2,
# examples/hunyuan_image3/hunyuan_image3_instruct_stage.yaml); refer to several images as
# image 1, image 2, ...
python examples/hunyuan_image3/run.py --image cat.png \
  --prompt "Put a red wizard hat on the cat and change the background to a starry night sky"
python examples/hunyuan_image3/run.py --image astronaut.jpg livingroom.png \
  --prompt "Place the astronaut from image 1 sitting on the orange sofa in the living room from image 2"
```

With the `Omni` API:

```python
omni.generate(
    {"prompt": "A neon shop sign ..."},
    OmniDiffusionSamplingParams(height=1024, width=1024, num_inference_steps=50,
                                guidance_scale=5.0, seed=42),
)
```

and, for an edit (with `model="tencent/HunyuanImage-3.0-Instruct"`), the condition image
(or a list of images) in the prompt:

```python
from PIL import Image
omni.generate(
    {"prompt": "Make it snow", "multi_modal_data": {"image": Image.open("in.png")}},
    OmniDiffusionSamplingParams(num_inference_steps=50, seed=42),
)
```

Think / recaption: `--bot-task recaption` (or `think_recaption`) with `run.py`, or
`extra_args={"bot_task": "think_recaption"}` in the sampling parameters (with the Instruct
checkpoint, with or without condition images); `max_new_tokens` in `extra_args` lowers the
2048-token limit.

`guidance_scale` and the step count default to the checkpoint's (5.0 / 2.5 and 50);
`guidance_scale <= 1` turns guidance off (one row).

Loading the 160 GB checkpoint takes about a minute. The first request compiles the
graphs for the requested size and parallel layout, about 5–8 minutes; later runs load
them from the compile cache.

## Performance

Measured on one `trn2.48xlarge` (Neuron SDK 2.32, `NEURON_LOGICAL_NC_CONFIG=2`), BF16,
1024x1024, 50 steps, guidance 5, batch size 1. Latency is the warm end-to-end time of
`omni.generate` (prefix, denoising, VAE decode and post-processing), averaged over the
profiled runs.

| Configuration | NeuronCores | End-to-end | Denoise (per step) | VAE decode |
|---|---|---|---|---|
| TP=16 | 16 | 20.78 s | 20.00 s (400 ms) | 0.59 s |
| TP=32 | 32 | 15.06 s | 14.37 s (287 ms) | 0.55 s |
| TP=16 × CFG=2 | 32 | 12.25 s | 11.57 s (231 ms) | 0.53 s |

**Image editing** (HunyuanImage-3.0-Instruct, TP=16 × CFG=2, 50 steps, guidance 2.5).
The output takes the first condition image's bucket; "prefix" is the VAE encoder, the
SigLIP2 tower and the prefix graph (system prompt, condition images and instruction).

| Condition images | Output | End-to-end | Denoise (per step) | Prefix |
|---|---|---|---|---|
| 1 (cat, 1024x704) | 832x1216 | 17.33 s | 16.00 s (320 ms) | 0.95 s |
| 1 (astronaut) | 768x1024 | 10.89 s | 9.86 s (197 ms) | 0.73 s |
| 1 (living room, 640x640) | 1024x1024 | 14.82 s | 13.37 s (267 ms) | 0.84 s |
| 2 (astronaut, living room) | 768x1024 | 12.55 s | 10.63 s (213 ms) | 1.58 s |
| 3 (cat, astronaut, living room) | 832x1216 | 20.52 s | 17.70 s (354 ms) | 2.33 s |

An edit step costs more than a text-to-image step of the same size because the image
tokens also attend to the cached prefix (about 1200 system-prompt tokens plus about 5000
per condition image).

**Think / recaption** (Instruct, 50 steps, guidance 2.5, seed 42, the output size predicted
by the model). "Text" is the prefill and the token-by-token decode of the think / recaption
text; "prefix" the image pass's prefix graph. With TP=16 × CFG=2 the text runs on all 32
NeuronCores (the two replicas split the experts, see below) and the image on TP=16 × CFG=2;
plain TP=32 is shown for comparison.

| Request | Layout | Text | Prefix | Denoise | VAE decode | End-to-end |
|---|---|---|---|---|---|---|
| text-to-image, `recaption` (1024x1024) | TP=16 × CFG=2 | 7.10 s (327 tokens, 21.7 ms/token) | 0.14 s | 12.42 s | 0.53 s | **20.23 s** |
| | TP=32 | 6.83 s (343 tokens, 19.9 ms/token) | 0.21 s | 16.66 s | 0.53 s | 24.27 s |
| text-to-image, `think_recaption` (1024x1024) | TP=16 × CFG=2 | 18.10 s (851 tokens, 21.3 ms/token) | 0.21 s | 12.49 s | 0.53 s | **31.37 s** |
| | TP=32 | 16.14 s (834 tokens, 19.3 ms/token) | 0.29 s | 16.87 s | 0.53 s | 33.87 s |
| edit (cat), `think_recaption` (832x1216) | TP=16 × CFG=2 | 8.46 s (365 tokens, 23.2 ms/token) | 0.46 s | 16.03 s | 0.31 s | **25.85 s** |

The edit also spends 0.50 s encoding the condition image. The texts differ slightly between
layouts (sampling is seeded, but the BF16 logits are not bit-identical across layouts);
the text length, set by the model, decides how much the text adds.

How the text decode was sped up (TP=16 × CFG=2, from 37 ms to about 21 ms per token):

- **Dense experts at one token.** The blockwise MoE kernel computes a whole 512-token block
  per selected expert; at one token every local expert is instead computed densely and
  weighted by its affinity (exact: unselected experts weigh 0), bound only by reading the
  expert weights.
- **The CFG replicas split the experts.** Both replicas hold the same weights, and a
  graph's collectives need every rank, so both run the text: each computes half of its
  local experts (and of the shared expert) and one all-reduce over the 32 ranks sums them,
  halving the expert weights each rank reads per token (as TP=32 would). The logits are
  averaged over the replicas in-graph, so every rank samples the same token.
- **Fewer host transfers.** The RoPE tables are on device and indexed by a device-side
  position counter; each step sends only the token's embedding.

What remains per token is about 14 ms on device (half of it the expert weights) and about
6 ms of host-side graph dispatch and sampling.

TP=16 is the smallest layout that fits the weights (about 13 GB per NeuronCore). On 32
NeuronCores CFG parallelism is faster than TP=32: each replica runs half the tokens, so
both its compute and its collectives halve, while TP=32 only splits the compute further
(every layer still all-gathers and reduce-scatters all tokens).

### How the text-to-image path was optimized

A step at 1024x1024 with CFG runs about 8,200 tokens through the 32 MoE layers (about
200 TFLOP). The port was profiled per part on a single NeuronCore at the TP=16 shapes
(no collectives) and the largest costs were removed in this order:

| Change | Effect |
|---|---|
| MoE kernel: `moe_cte` `shard_on_i_hybrid` with `num_static_block=0` instead of `shard_on_block` | `shard_on_block` ignores the occupied-block mask and computes every statically allocated block (about 130 at TP=16, of which about 16 hold tokens); the hybrid kernel's dynamic loop runs only the occupied ones. MoE per layer and step: 53 ms → 4.2 ms |
| MoE token count padded to a multiple of 2048 | the blockwise token-to-expert mapping then runs as NKI kernels instead of a slow PyTorch fallback (its `indexed_flatten` needs `T / 128` divisible by 16): 7 ms → 0.6 ms per layer |
| Diffusion heads sharded over TP (patch embed and final layer convolutions split by input / output channels; GroupNorm groups stay whole on a rank) | about 4 TFLOP per step that every rank used to repeat: about 60 ms → about 5 ms per step |
| Shared expert sequence-parallel (full weights on each rank's 1/TP of the tokens) instead of an intermediate dim sharded to 192 | an efficient matmul shape and no extra collective, about 0.8 ms per layer |
| Fused NKI QKV kernel (projection, RoPE and the post-RoPE QK RMSNorm in one kernel, head-major output) | 1.42 ms → 0.82 ms per layer |
| CFG parallelism (each of two TP=16 groups runs one CFG row) | halves both compute and collective volume per group; on 32 NeuronCores 12.25 s vs 15.06 s for TP=32 |

The first four changes took TP=16 from 25.65 s to 21.38 s end to end after the MoE fix
(the MoE kernel fix came first, before the first full-model run). What remains per layer
and step at TP=16 is roughly 4 ms of routed experts, 1.4 ms of attention, 0.8 ms of QKV
projection, and about 5 ms of collectives: the all-gather before attention and the
experts and the reduce-scatter after them each move all 8,200 tokens x 4096 channels,
about 40% of the step, which is why TP=32 only gains 1.4x over TP=16.

The VAE was reworked for compile time and memory rather than speed: compiling whole up
levels on one core (128 channels x 4 frames at 1024x1024) took over 40 minutes per level,
so the up levels run as row strips over all ranks (exact halos, all-reduced GroupNorm
statistics), which compiles in about 5 minutes and decodes in about 0.5 s; fusing the last
three levels into one strip graph keeps their full-resolution intermediates (up to 1 GB)
from ever being gathered.

## Accuracy

**Component level** (CPU, FP32, first 2 layers) against the official implementation in
the checkpoint (`hunyuan.py`, transformers 4.57.1): the 2-D RoPE tables match exactly, and
the first- and second-step velocities of both CFG rows match with a relative L2 error of
2e-7 at TP=1 and 1e-6 at TP=16 (16 CPU processes, exercising the KV-head replication,
expert parallelism, sequence parallelism and the sharded heads). The VAE decoder matches
the official `AutoencoderKLConv3D` with 2e-7, also split into row strips over 3 processes.

**End to end on Neuron.** With the same seed the Neuron pipeline starts from the same
noise as the official pipeline. At 1024x1024, 20 steps, guidance 5, the Neuron output
matches the official implementation run on CPU in BF16 with:

| Configuration | PSNR | SSIM |
|---|---|---|
| TP=16 | 25.5 dB | 0.849 |
| TP=32 | 26.2 dB | 0.866 |
| TP=16 × CFG=2 | 24.6 dB | 0.835 |

Two BF16 implementations of an 80B model do not match bit for bit: rounding differences
(and the occasional different top-8 expert choice they cause) compound over the steps;
the composition, text and lighting are the same.

![Neuron (left) vs official CPU reference (right), 1024x1024, 20 steps, seed 42](images/hunyuan-image-3.0/neuron_vs_reference_1024_s20.jpg)

*Neuron TP=16 (left) and the official implementation on CPU in BF16 (right): 1024x1024,
20 steps, seed 42.*

**Image editing.** The Instruct path is checked against the official Instruct code on CPU
in FP32 (first 2 layers): the token sequence and the prefix attention mask match exactly,
the SigLIP2 tower and aligner with a relative L2 error of 2e-7, the VAE encoder's posterior
mean with 4e-7 (also over 4 processes), and the first- and second-step velocities of an
edit (prefix with the condition image, then steps) with 3e-7 at TP=1 and 1e-6 at TP=16.

## Sample outputs

Generated on Neuron (TP=16 × CFG=2, 50 steps, guidance 5, seed 42), downscaled for this
page.

| | |
|---|---|
| ![portrait](images/hunyuan-image-3.0/portrait_1024.jpg) | ![landscape](images/hunyuan-image-3.0/landscape_1024.jpg) |
| *A close-up portrait of an elderly fisherman with a weathered face and a knitted wool cap, soft window light, 85mm photograph, highly detailed skin texture* | *A misty alpine lake at sunrise, pine forests and snow-capped peaks reflected in still water, golden light, wide-angle landscape photograph* |
| ![chinese text](images/hunyuan-image-3.0/chinese_text_1024.jpg) | ![illustration](images/hunyuan-image-3.0/illustration_1024.jpg) |
| *一张中国风海报，宣纸纹理背景，中间用毛笔书法写着“春风十里”四个大字，左下角有一枝盛开的梅花，右上角有红色印章* | *A cozy isometric illustration of a tiny bookstore cafe on a rainy evening, warm lights inside, a cat sleeping on a windowsill, soft pastel colors* |
| ![neon](images/hunyuan-image-3.0/neon_1024.jpg) | |
| *A neon shop sign that reads "HUNYUAN IMAGE 3", rainy night, reflections on wet pavement* | |

![city 768x1280](images/hunyuan-image-3.0/city_768x1280.jpg)

*A panoramic view of a futuristic city skyline at dusk, flying cars between glass towers,
holographic billboards, cinematic lighting* (768x1280 bucket)

### Think / recaption

![think / recaption samples](images/hunyuan-image-3.0/think_recaption_samples.jpg)

*Left to right: text-to-image with `recaption` and with `think_recaption` (prompt: A neon
shop sign that reads "HUNYUAN IMAGE 3", rainy night, reflections on wet pavement), and the
cat edit with `think_recaption` (Put a red wizard hat on the cat and change the background
to a starry night sky). The model's rewritten prompt, e.g. "Add a tall, pointed red wizard
hat made of velvet with a wide brim onto the cat's head, positioned slightly tilted to one
side. Replace the plain green background with a deep blue night sky filled with countless
bright stars and a soft, glowing nebula. ...", decides the details.*

### Image editing

HunyuanImage-3.0-Instruct on Neuron (TP=16 × CFG=2, 50 steps, guidance 2.5, seed 42):
condition images on the left (image 1 on top), output on the right. Inputs are from the
[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)
dataset (`diffusers/cat.png`, `diffusers/astronaut.jpg`, `diffusers/gligen/livingroom_modern.png`).

![edit cat wizard](images/hunyuan-image-3.0/edit_cat_wizard.jpg)

*Put a red wizard hat on the cat and change the background to a starry night sky*

![edit cat neon text](images/hunyuan-image-3.0/edit_cat_neon_text.jpg)

*Replace the green background with a dark brick wall with a pink neon sign that reads "HELLO NEURON"*

![edit astronaut](images/hunyuan-image-3.0/edit_astronaut_reef.jpg)

*Change the scene to an underwater coral reef with colorful fish, keep the astronaut*

![edit living room](images/hunyuan-image-3.0/edit_livingroom_winter.jpg)

*Turn this living room into a cozy winter cabin with a lit fireplace and snow falling outside the window*

![multi astronaut sofa](images/hunyuan-image-3.0/edit_multi_astronaut_sofa.jpg)

*Place the astronaut from image 1 sitting on the orange sofa in the living room from image 2*

![multi cat astronaut room](images/hunyuan-image-3.0/edit_multi_cat_astronaut_room.jpg)

*The cat from image 1 and the astronaut from image 2 sitting together on the orange sofa in
the living room from image 3*

The official multi-image demo of the checkpoint (`assets/demo_instruct_imgs/input_1_*.png`,
"基于图一的logo，参考图二中冰箱贴的材质，制作一个新的冰箱贴") also reproduces on Neuron: the
logo of image 1 becomes an enamel fridge magnet in the material of image 2 (16.53 s warm).

## Known limitations

- **Think / recaption text is sampled, not matched to the reference token for token**: it
  uses the request's seed (the official code samples from an unseeded global RNG), the
  condition latents are shared with the image pass, and the text differs slightly between
  parallel layouts (BF16 logits). `drop_think` and the automatic aspect
  ratio of direct generation (`bot_task="image"` without a size, a separate one-token
  pass officially) are not implemented: direct generation takes the first condition
  image's bucket, or the given size.
- **Editing needs the Instruct checkpoint**; the base checkpoint is text-to-image only.
- **Sizes are the checkpoint's resolution buckets.** Other sizes snap to the nearest
  bucket by aspect ratio, and each bucket compiles its own graphs on first use.
- **Prompt length is static** for the base checkpoint: prompts longer than
  `model_config.max_text_tokens` (default 1024 tokens including the special tokens) are
  rejected; raise it in the stage config (this recompiles). Instruct prefixes are padded
  to a multiple of 1024 tokens, and each new padded length, condition-image size and
  output size compiles its own graphs on first use (several minutes).
- **Each edit layout holds its own device memory.** A new combination of condition-image
  count and sizes and output size loads its own prefix, step and VAE graphs (about
  1.5–2 GB per NeuronCore on top of about 15 GB for the weights); one process fits about
  four such layouts before running out of device memory. Restart the server to serve
  other layouts.
- **Batch size is one** prompt and one image per request.
- **16 or 32 NeuronCores.** The 160 GB of weights need TP >= 16, and TP must divide the
  64 experts and the 32 attention heads; the tested layouts are TP=16, TP=32 and
  TP=16 × CFG=2.
