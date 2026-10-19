# HunyuanImage-3.0 Model Card

<!-- meta: description: Model card for HunyuanImage-3.0 text-to-image on AWS Trainium with
the vLLM Omni Neuron plugin — supported features, parallelism configurations, latency at
1024x1024, accuracy against the official reference, sample outputs, and known limitations. -->
<!-- meta: keywords: HunyuanImage-3.0, Hunyuan, model card, text-to-image, image generation,
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

HunyuanImage-3.0 **text-to-image** is supported with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS
Trainium2 (`trn2`); the plugin replaces vLLM Omni's `HunyuanImage3Pipeline`
(`HunyuanImage3ForCausalMM`).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Precision |
|-------|-------------|----------|-----------|
| HunyuanImage-3.0 | [tencent/HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0) | Trn2 | BF16 |

The checkpoint is released under the Tencent Hunyuan Community License; check its terms
(including its territory restrictions) before use.

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image | ✅ |
| | The 33 resolution buckets around 1024x1024 (512x2048 to 2048x512) | ✅ |
| | Classifier-free guidance (`guidance_scale`, default 5.0, as the checkpoint) | ✅ |
| | Automatic aspect ratio (`image_size="auto"`), think / recaption prompting | ❌ |
| | Image editing (HunyuanImage-3.0-Instruct) | ❌ |
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
   rank, then each up level as row strips over all ranks with exact halos and
   all-reduced GroupNorm statistics.

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

With the `Omni` API:

```python
omni.generate(
    {"prompt": "A neon shop sign ..."},
    OmniDiffusionSamplingParams(height=1024, width=1024, num_inference_steps=50,
                                guidance_scale=5.0, seed=42),
)
```

`guidance_scale` defaults to the checkpoint's 5.0; `guidance_scale <= 1` turns guidance
off (one row).

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
| TP=16 | 16 | 21.17 s | 20.01 s (400 ms) | 0.97 s |
| TP=32 | 32 | 15.26 s | 14.37 s (287 ms) | 0.74 s |
| TP=16 × CFG=2 | 32 | 12.45 s | 11.59 s (232 ms) | 0.74 s |

TP=16 is the smallest layout that fits the weights (about 13 GB per NeuronCore). On 32
NeuronCores CFG parallelism is faster than TP=32: each replica runs half the tokens, so
both its compute and its collectives halve, while TP=32 only splits the compute further
(every layer still all-gathers and reduce-scatters all tokens).

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

## Known limitations

- **Text-to-image only.** The automatic aspect ratio (`image_size="auto"`, which asks the
  model for a ratio token first), the think / recaption prompt modes and image editing
  (the separate HunyuanImage-3.0-Instruct checkpoint) are not implemented; pass an
  explicit size.
- **Sizes are the checkpoint's resolution buckets.** Other sizes snap to the nearest
  bucket by aspect ratio, and each bucket compiles its own graphs on first use.
- **Prompt length is static.** Prompts longer than `model_config.max_text_tokens`
  (default 1024 tokens including the special tokens) are rejected; raise it in the stage
  config (this recompiles).
- **Batch size is one** prompt and one image per request.
- **16 or 32 NeuronCores.** The 160 GB of weights need TP >= 16, and TP must divide the
  64 experts and the 32 attention heads; the tested layouts are TP=16, TP=32 and
  TP=16 × CFG=2.
