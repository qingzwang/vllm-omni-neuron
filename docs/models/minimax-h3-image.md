# MiniMax-H3-Image Model Card

<!-- meta: description: Model card for MiniMax-H3-Image on AWS Trainium with the
vLLM Omni Neuron plugin — supported features (text-to-image up to 4096x4096, BF16,
TP/CP parallelism, tile-parallel VAE decode), recommended 8- and 16-core
configurations, accuracy against the diffusers reference, latency, and known issues. -->
<!-- meta: keywords: MiniMax-H3-Image, MiniMax, H3, model card, text-to-image, image
generation, diffusion, DiT, PDD, step distillation, vLLM, vLLM Omni, Neuron, Trainium,
trn2, BF16, tensor parallelism, context parallelism, VAE tiling -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-11-11 -->

## Introduction

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) is a text-to-image and
image-editing model from the MiniMax H3 family. It generates images of up to about 16
megapixels (4096x4096) directly. Its diffusion transformer (DiT) has 33.1B parameters: 50
single-stream blocks (hidden size 5376, 56 heads of 128, SwiGLU FFN 14336) over one packed
sequence of text and image tokens with 3D rotary positions. The text encoder is the
MiniMax-H3 Qwen3-VL encoder, read at `hidden_states[50]`; the VAE has a 24-layer 2D ViT
decoder with 16x spatial compression.

The checkpoint is **guidance- and step-distilled** (Parallel Decoding Distillation): one
sample takes exactly **six DiT forwards on a fixed sigma grid**, with no classifier-free
guidance and no negative prompt. Each step has its own output head.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Precision |
|-------|-------------|----------|-----------|
| MiniMax-H3-Image | [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) | Trn2 | BF16 |

The pipeline reads `transformer_diffusers/`, `vae_diffusers/`, `text_encoder/`,
`tokenizer/` and `pdd_config.json` from the checkpoint.

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image | ✅ |
| | Image editing (reference images) | ❌ Not yet |
| | 1024x1024 to 4096x4096 (`min_generation_area` canvas policy, LANCZOS resize back) | ✅ |
| | Six-step PDD sampling, no CFG | ✅ |
| **Precision** | BF16 DiT and text encoder, BF16 VAE decoder (FP32 norms) | ✅ |
| | FP8 FFN (`quantization: fp8_ffn`, W8A8 ROW) | Experimental |
| **Parallelism** | Tensor parallelism (TP=8, 7 heads per rank) | ✅ |
| | Context parallelism over the packed sequence (`ring_degree`) | ✅ |
| | Tile-parallel VAE decode over all ranks | ✅ |
| **Acceleration** | Cache-DiT | ❌ Not yet |
| **Compilation** | `torch.compile`, one block-group graph reused for all 50 blocks | ✅ |

### How the Neuron implementation differs from the reference

The modeling code lives in
[`vllm_omni_neuron/diffusion/models/minimax_h3_image/`](https://github.com/aws-neuron/vllm-omni-neuron/tree/main/vllm_omni_neuron/diffusion/models/minimax_h3_image).
All changes are numerically equivalent to the diffusers reference up to bf16 rounding:

- **AdaLN tables on the host.** Every block's `adaln_proj` (about 40% of the DiT
  parameters, ~26 GB) depends only on the timestep. The six timesteps are fixed, so the
  modulation rows, the output-norm modulation and the per-step output heads are computed
  once on the CPU at load time; those weights never reach the device.
- **Packed layout `[image | text | padding]`.** Unmasked self-attention is
  permutation-equivariant, so image tokens sit at fixed positions and the text is padded
  to a bucket (`text_buckets`); padded keys are masked in the attention kernel through
  its runtime `kv_used_len`, so one compiled graph serves many prompt lengths.
- **Text stream once per request.** `context_embedder` + `token_refiner` depend on the
  prompt only and run once instead of once per step.
- **Layer groups.** The 50 blocks run as ten calls of one compiled 5-block graph, with each
  group's weights passed in as graph inputs (`layers_per_graph`, default 5). This bounds
  compiler-held activations (a single 50-block graph needs ~30 GB of HBM per core at
  4096x4096 on 8 cores) and cuts cold compile time 2-3x.
- **All-reduce overlap.** Each block's post-attention tail runs in two token chunks when a
  core holds at most 8704 tokens (`overlap_chunks`, 0 = automatic), so one chunk's TP
  all-reduce overlaps the other chunk's matmuls.
- **Text encoder.** Only the 50 layers that feed `hidden_states[50]` are built (TP over
  the same TP group as the DiT); the token embedding lookup runs on the host.
- **VAE decode.** The reference decodes in overlapping 256-pixel tiles of identical
  shape; tiles are batched across all ranks, gathered in bf16 (the decoder's output
  precision) and stitched on rank 0 with the reference blending. The image is returned to
  the engine as uint8.

### Recommended configurations

| Cores | Stage config | Layout |
|---|---|---|
| 8 | [`minimax_h3_image_stage_tp8.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/examples/minimax_h3_image/minimax_h3_image_stage_tp8.yaml) | `tensor_parallel_size=8` |
| 16 (recommended) | [`minimax_h3_image_stage_tp8cp2.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/examples/minimax_h3_image/minimax_h3_image_stage_tp8cp2.yaml) | `tensor_parallel_size=8`, `ring_degree=2` |

Weights take ~11 GB of HBM per core at TP=8 (DiT ~5 GB, text encoder ~6 GB, VAE <1 GB),
so 8 cores is the minimum. A 64-core configuration
([`minimax_h3_image_stage.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/examples/minimax_h3_image/minimax_h3_image_stage.yaml),
TP=8 x CP=8) is provided but gives little over 16 cores at 1024x1024.

Run:

```bash
python examples/minimax_h3_image/run.py \
  --model-path MiniMaxAI/MiniMax-H3-Image \
  --stage-config examples/minimax_h3_image/minimax_h3_image_stage_tp8cp2.yaml \
  --height 4096 --width 4096 --min-generation-area 16777216 \
  --output out.png
```

`--min-generation-area` follows the reference: a smaller requested size is generated on a
canvas of at least this area (default 2048x2048) and resized back. Pass `1024` to generate
at the requested size.

## Accuracy Evaluation

**Method.** Three tiers against the diffusers 0.42 (main) CPU reference, same prompt,
seed and initial noise:

1. **Components** — text encoder `hidden_states[50]`, one DiT step on identical inputs,
   VAE decode of identical latents.
2. **Per step** — each of the six steps fed the reference's own input latents
   (teacher-forced), comparing the predicted velocity.
3. **End to end** — PSNR of the final 8-bit image.

The six-step trajectory is very sensitive to last-bit differences (the reference scheduler
documents this), so free-running end-to-end outputs are expected to match visually, not
bit-exactly. As a calibration, the CPU bf16 reference itself differs from a CPU fp32 run by
the same per-step magnitudes as Neuron does.

| Tier | Metric | Result (Trn2) |
|---|---|---|
| Text encoder | relative L2 / per-token cosine (min) | 8.8e-4 / 0.99989 |
| DiT, per step (1024x1024, teacher-forced) | relative L2 of velocity, steps 0-5 | 0.8e-2 - 3.6e-2 (cosine >= 0.9993) |
| DiT, 2 layers | Neuron vs CPU bf16 / CPU bf16 vs CPU fp32 (step 0) | 6.0e-3 / 5.7e-3 |
| VAE decode (1024x1024) | PSNR of 8-bit image vs FP32 reference decode | 49.8 dB |

End-to-end image PSNR against the CPU reference (prompt *"A highland cow standing in a
misty meadow at sunrise, photorealistic, detailed fur"*, seed 42):

| Cores | 1024x1024 | 2048x2048 | 4096x4096 |
|---|---|---|---|
| 8 | 28.2 dB | 27.8 dB | 30.1 dB |
| 16 | 24.9 dB | 34.1 dB | 30.8 dB |

All outputs are visually indistinguishable from the reference (same composition and fine
detail at full resolution).

## Performance

trn2.48xlarge, `NEURON_LOGICAL_NC_CONFIG=2`, BF16, warm (compiled) runs, 20-token prompt,
average of three generations. "Denoise" is six DiT steps; "Request" is the wall time of
`Omni.generate`.

| Cores | Resolution | Text encode | Denoise (6 steps) | VAE decode | Request | First request (cold compile) |
|---|---|---|---|---|---|---|
| 8 | 1024x1024 | 8 ms | 1.01 s | 0.07 s | **1.14 s** | 55 s |
| 8 | 2048x2048 | 8 ms | 3.27 s | 0.39 s | **3.99 s** | 72 s |
| 8 | 4096x4096 | 8 ms | 14.8 s | 0.91 s | **16.2 s** | 298 s |
| 16 | 1024x1024 | 8 ms | 0.62 s | 0.06 s | **0.74 s** | 50 s |
| 16 | 2048x2048 | 8 ms | 1.91 s | 0.20 s | **2.23 s** | 54 s |
| 16 | 4096x4096 | 11 ms | 9.91 s | 0.68 s | **11.0 s** | 111 s |

**Where the time goes** (one 5-block group at 2048x2048 on 8 cores, 4352 tokens per core):
~11.7 ms per block, of which ~3.5 ms is the two TP all-reduces (47 MB each), ~6.4 ms the
projection matmuls (~40% of BF16 peak) and ~1.9 ms attention. Fusing communication and
elementwise work into the projection kernels is the next optimization step.

## Quantization (experimental)

`model_config.quantization: fp8_ffn` (or `run.py --quantization fp8_ffn`) runs every DiT
block's SwiGLU FFN as FP8 W8A8 through the nkilib MLP kernel's Trn2 `ROW` mode: weights are
quantized per output column at load time (e4m3, range +-240), activations per row in the
graph. The kernel itself is 2.1x faster than the BF16 FFN (190 vs 90 TFLOPS at 4352 tokens),
but the activation quantization still runs as separate ops, so the end-to-end gain is small
and image quality drops by ~3 dB PSNR against the reference. It is not the default.

| Config | BF16 | FP8 FFN | FP8 PSNR vs reference |
|---|---|---|---|
| 16 cores, 2048x2048 | 2.23 s | 2.33 s | 27.3 dB |
| 16 cores, 4096x4096 | 11.0 s | 10.9 s | 27.5 dB |
| 8 cores, 4096x4096 | 16.2 s | 15.1 s | 27.1 dB |
| 8 cores, 2048x2048 | 3.99 s | 11.9 s (known slow path) | 25.6 dB |

Teacher-forced per-step velocity error with FP8 FFN is 2.8e-2 - 6.4e-2 relative (cosine >=
0.998), about 2-3x the BF16 error.

## Known issues and limitations

- **Image editing is not supported yet.** Requests with reference images are rejected;
  the vision tower of the text encoder and the VAE encoder are not implemented.
- **Concurrent requests run serially.**
- **One compiled graph set per (canvas, text bucket).** The first request at a new
  resolution or a prompt longer than the current text bucket compiles new graphs.
- **`NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS`.** One step enqueues ~12 graph launches;
  the example runner raises the per-core queue to 63 (the runtime maximum) through the
  `MINIMAX_H3_IMAGE` environment profile. Without it, execution fails with
  `Execution Queue Full`.
- **Prompt expansion.** The optional MiniMax-H3-Image-PE prompt expander is not part of
  this pipeline; pass already expanded prompts.
