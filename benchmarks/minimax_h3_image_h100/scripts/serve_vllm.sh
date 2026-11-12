#!/bin/bash
# vLLM-Omni, 2x H100, DiT TP2 resident (recipes/MiniMaxAI/MiniMax-H3-Image-H100.md)
source ${ROOT:-$HOME/minimax-h3-image}/envs/vllm/bin/activate
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1} HF_HUB_OFFLINE=1
export CUDA_HOME=${ROOT:-$HOME/minimax-h3-image}/envs/vllm/lib/python3.12/site-packages/nvidia/cu13 PATH=${ROOT:-$HOME/minimax-h3-image}/envs/vllm/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
exec numactl --cpunodebind=${NUMA:-0} --membind=${NUMA:-0} \
  vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 2 --tensor-parallel-size 2 --usp 1 --text-encoder-tp-size 2 \
  --host 127.0.0.1 --port ${PORT:-8000} "$@"
