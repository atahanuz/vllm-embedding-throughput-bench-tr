#!/usr/bin/env bash
# Reference launch command for the embedding server.
# Run on the GPU host (NOT inside this benchmark script).
#
# Qwen3-Embedding-8B is a decoder-style embedding model: vLLM needs --task embed.
# Tune --max-num-seqs and --max-model-len to your GPU; values below are starting points
# for a single H100 80GB. Lower them on smaller cards.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_UTIL="${GPU_UTIL:-0.90}"

exec vllm serve "$MODEL" \
    --task embed \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --disable-log-requests
