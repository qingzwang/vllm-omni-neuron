#!/bin/bash
# Download MiniMax-H3-Image + the MiniMax-H3 text encoder into the HF cache (~320 GB on disk incl. *_diffusers copies).
# Xet transfers stalled on our host, so plain HTTP + hf_transfer is used.
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
hf download MiniMaxAI/MiniMax-H3-Image --max-workers 16
hf download MiniMaxAI/MiniMax-H3 --include "text_encoder/*" --max-workers 16
