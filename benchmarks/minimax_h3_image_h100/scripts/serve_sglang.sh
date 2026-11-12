#!/bin/bash
# SGLang diffusion, 2x H100, cookbook derived recipe: Ulysses2 + encoder fold + resident + AdaLN precompute
# override layout with LAYOUT="--tp-size 2 --ulysses-degree 1"
source ${ROOT:-$HOME/minimax-h3-image}/envs/sglang/bin/activate
export CUDA_VISIBLE_DEVICES=${GPUS:-4,5} HF_HUB_OFFLINE=1
export CUDA_HOME=${ROOT:-$HOME/minimax-h3-image}/envs/sglang/lib/python3.12/site-packages/nvidia/cu13 PATH=${ROOT:-$HOME/minimax-h3-image}/envs/sglang/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
exec numactl --cpunodebind=${NUMA:-1} --membind=${NUMA:-1} \
  sglang serve --model-path MiniMaxAI/MiniMax-H3-Image --backend sglang \
  --num-gpus 2 ${LAYOUT:---tp-size 1 --ulysses-degree 2} --ring-degree 1 \
  --component-residency all=resident --encoder-parallel fold --minimax-h3-adaln-online true \
  --host 127.0.0.1 --port ${PORT:-30000} "$@"
