# Qwen-Image-2.1 Model Card

<!-- meta: description: Model card for Qwen-Image-2.1 text-to-image on AWS Trainium with the
vLLM Omni Neuron plugin — supported features, parallelism configurations, latency at
1024x1024 and 2048x2048, accuracy against the diffusers reference, sample outputs, and
known limitations. -->
<!-- meta: keywords: Qwen-Image-2.1, Qwen-Image, model card, text-to-image, image generation,
diffusion, DiT, Qwen3-VL, vLLM, vLLM Omni, Neuron, Trainium, trn2, BF16, tensor parallelism,
context parallelism, sequence parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-14 -->

## Introduction

[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) is a text-to-image and image
editing model from the Qwen team. Its visual generation component is a 7B-parameter,
32-layer single-stream DiT conditioned on a Qwen3-VL (8B) encoder, with a 16x spatial VAE
that produces RGBA images. The DiT attends over one joint text/image sequence with a
block-causal mask, and modulates the text tokens from `t = 0`, which makes their keys and
values independent of the denoising step.

Qwen-Image-2.1 **text-to-image** is supported with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS
Trainium2 (`trn2`).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Precision |
|-------|-------------|----------|-----------|
| Qwen-Image-2.1 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image | ✅ |
| | 512x512 to 2048x2048 (multiples of 32) | ✅ |
| | Image editing / reference images | ❌ |
| | Classifier-free guidance | ❌ (the model is sampled without it) |
| **Precision** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Megatron sequence parallelism (SP, within TP) | ✅ (on by default) |
| | Context parallelism (CP, `ring_degree`) | ✅ |
| | Parallel tiled VAE decode | ✅ (above 1024x1024) |
| **Compilation** | torch.compile | ✅ |
| **Serving** | Batch size | 1 |

## How it runs

Per request the pipeline runs four kinds of compiled graphs:

1. **Text encoder** — the Qwen3-VL language model (TP-sharded), at a static prompt length
   (`max_text_tokens`, default 512 tokens after the system prompt).
2. **DiT prefix** — once per request, the text tokens pass through the 32 blocks and every
   block's text K/V is kept (the reference pipeline's prefix KV cache).
3. **DiT step** — every denoising step runs the image tokens only, attending to
   `[image K/V, cached text K/V]`, and applies the flow-matching Euler update in-graph, so
   the latents stay on device for the whole loop.
4. **VAE decoder** — compiled per stage. Above a 64x64 latent (1024x1024 pixels) the
   up-sampling stages decode as exact overlapping tiles spread over all ranks.

The DiT projections use the fused NKI QKV kernel (projection, per-head RMS QK-norm and RoPE
in one kernel, head-major output) and the fused AdaLN kernel, with sequence parallelism
over the TP group.

## Running

```bash
# 1024x1024, 40 steps, TP=8 (examples/qwen_image_2_1/qwen_image_2_1_stage.yaml)
python examples/qwen_image_2_1/run.py --prompt "A neon shop sign ..." --output out.png

# Other configurations
python examples/qwen_image_2_1/run.py --tensor-parallel-size 16 --height 2048 --width 2048
python examples/qwen_image_2_1/run.py --tensor-parallel-size 4 --context-parallel-size 2

# Warm latency: time 3 requests after the first (compiling) one
python examples/qwen_image_2_1/run.py --profile --profile-runs 3
```

The first request compiles the graphs for the requested resolution and parallel layout:
roughly 4–5 minutes at 1024x1024 and 15–25 minutes at 2048x2048. Later runs load the
compiled graphs from the cache.

## Performance

Measured on one `trn2.48xlarge` (Neuron SDK 2.32, `NEURON_LOGICAL_NC_CONFIG=2`), BF16,
40 steps, batch size 1. Latency is the warm end-to-end time of `omni.generate` (text
encoding, denoising, VAE decode and post-processing), averaged over the profiled runs.

**1024x1024**

| Configuration | NeuronCores | End-to-end | Denoise (per step) | VAE decode |
|---|---|---|---|---|
| TP=4 | 4 | 10.64 s | 10.11 s (253 ms) | 0.51 s |
| TP=8 | 8 | 5.88 s | 5.36 s (134 ms) | 0.46 s |
| TP=16 | 16 | 3.89 s | 3.33 s (83 ms) | 0.46 s |

**2048x2048**

| Configuration | NeuronCores | End-to-end | Denoise (per step) | VAE decode |
|---|---|---|---|---|
| TP=4 | 4 | 64.3 s | 63.8 s (1.59 s) | 3.10 s |
| TP=8 | 8 | 30.8 s | 29.7 s (743 ms) | 1.31 s |
| TP=4 × CP=2 | 8 | 32.1 s | 30.8 s (770 ms) | 1.30 s |
| TP=16 | 16 | 16.0 s | 15.0 s (376 ms) | 0.74 s |

At 512x512 with 20 steps, TP=8 generates an image in 0.95 s.

Latency scales close to linearly with the core count, so per-core throughput is similar
across these layouts; TP=8 is slightly faster than TP=4 × CP=2 on the same 8 cores. For the
lowest latency use TP=16; for throughput, run several smaller replicas (for example TP=4)
on separate cores.

## Accuracy

**Component level** (CPU, FP32, TP=1 and CP=2) against the diffusers reference
(`QwenImage21Pipeline`, huggingface/diffusers@916a5791): the text encoder, the first-step
DiT noise prediction and the VAE decode match with a relative L2 error of about 1e-6.

**End to end on Neuron.** With the same seed, the Neuron pipeline starts from the same
noise as the diffusers pipeline in BF16. At 512x512 and 20 steps the Neuron output
matches the CPU BF16 diffusers output with:

| Configuration | PSNR | SSIM |
|---|---|---|
| TP=8 | 27.5 dB | 0.939 |
| TP=4 | 28.5 dB | 0.949 |
| TP=4 × CP=2 | 27.7 dB | 0.943 |

Two BF16 implementations do not match bit for bit: rounding differences compound over the
denoising steps (the diffusers documentation notes the same when toggling its KV cache).

![Neuron (left) vs CPU diffusers reference (right), 512x512, 20 steps, seed 42](images/qwen-image-2.1/neuron_vs_cpu_reference_512.jpg)

*Neuron TP=8 (left) and the CPU BF16 diffusers reference (right): 512x512, 20 steps,
seed 42.*

## Sample outputs

Generated on Neuron (TP=8, 1024x1024, 40 steps, seed 42), downscaled for this page.

| | |
|---|---|
| ![portrait](images/qwen-image-2.1/portrait_1024.jpg) | ![landscape](images/qwen-image-2.1/landscape_1024.jpg) |
| *A close-up portrait of an elderly fisherman with a weathered face and a knitted wool cap, soft window light, 85mm photograph, highly detailed skin texture* | *A misty alpine lake at sunrise, pine forests and snow-capped peaks reflected in still water, golden light, wide-angle landscape photograph* |
| ![chinese text](images/qwen-image-2.1/chinese_text_1024.jpg) | ![illustration](images/qwen-image-2.1/illustration_1024.jpg) |
| *一张中国风海报，宣纸纹理背景，中间用毛笔书法写着“春风十里”四个大字，左下角有一枝盛开的梅花，右上角有红色印章* | *A cozy isometric illustration of a tiny bookstore cafe on a rainy evening, warm lights inside, a cat sleeping on a windowsill, soft pastel colors* |
| ![neon 1024](images/qwen-image-2.1/neon_1024.jpg) | ![neon 2048](images/qwen-image-2.1/neon_2048.jpg) |
| *A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement* (1024x1024, TP=8) | Same prompt at 2048x2048 (TP=4 × CP=2) |

## Known limitations

- **Text-to-image only.** Image editing and reference-image conditioning are not
  implemented.
- **No classifier-free guidance.** `true_cfg_scale` and negative prompts are ignored.
- **Batch size is one** prompt and one image per request.
- **Prompt length is static.** Prompts longer than `model_config.max_text_tokens` (default
  512 tokens) are rejected; raise it in the stage config (this recompiles).
- **Context parallelism** needs the latent token count, `(H/16) * (W/16)`, to divide by
  the CP degree.
- **Fused QKV kernel width.** On this SDK the fused NKI QKV kernel returns wrong results
  above a 1536-wide output (more than 4 heads per rank), so ranks holding 8 heads (TP=4)
  project in two head groups.
