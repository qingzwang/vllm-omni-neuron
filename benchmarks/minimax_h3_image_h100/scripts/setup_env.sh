#!/bin/bash
# Build the two benchmark environments under $ROOT (default ~/minimax-h3-image).
# Revisions used for the published numbers:
#   vllm-omni ee44b9afc (2026-11-12) on vLLM 0.33.0, torch 2.14.0+cu130
#   sglang    12b727d3ef (2026-11-12),               torch 2.14.1+cu130
set -e
ROOT=${ROOT:-$HOME/minimax-h3-image}
mkdir -p $ROOT/src $ROOT/envs && cd $ROOT
pip install -U uv "huggingface_hub[hf_xet]" hf_transfer

git -C src clone https://github.com/vllm-project/vllm-omni.git
git -C src clone https://github.com/sgl-project/sglang.git
git -C src/vllm-omni checkout ee44b9afc
git -C src/sglang checkout 12b727d3ef

uv venv envs/vllm --python 3.12 --seed
(source envs/vllm/bin/activate && uv pip install vllm==0.33.0 --torch-backend=cu130 && cd src/vllm-omni && uv pip install -e .)

uv venv envs/sglang --python 3.12 --seed
# no Rust toolchain on the host -> skip the optional Rust extensions
(source envs/sglang/bin/activate && cd src/sglang && SGLANG_BUILD_RUST_EXTS=none uv pip install -e "python[diffusion]" --torch-backend=cu130)

# The host has no CUDA toolkit. Both serve_*.sh scripts point CUDA_HOME at the pip-installed
# nvidia/cu13 tree; JIT kernels (sglang fused QKNorm+RoPE, deep_gemm, flashinfer) also need
# lib64/ and an unversioned libcudart.so there, otherwise SGLang silently falls back to slower kernels.
for e in vllm sglang; do
  c=envs/$e/lib/python3.12/site-packages/nvidia/cu13
  [ -e $c/lib64 ] || ln -s lib $c/lib64
  [ -e $c/lib/libcudart.so ] || ln -s libcudart.so.13 $c/lib/libcudart.so
done

# Optional: per-stage timing instrumentation used for the 4096 breakdown (no-op unless H3_PROF_FILE is set)
HERE=$(cd "$(dirname "$0")" && pwd)
git -C src/vllm-omni apply $HERE/../patches/vllm-omni-h3prof.diff
git -C src/sglang apply $HERE/../patches/sglang-h3prof.diff
