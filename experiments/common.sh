#!/bin/bash
# Shared settings for the experiment scripts. Run every script from the repository root.
# Pick the GPU with CUDA_VISIBLE_DEVICES; each training run uses one GPU.
set -euo pipefail

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD/data}"  # per-task LeRobot datasets live here
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_CPP_MIN_LOG_LEVEL=3 GLOG_minloglevel=3 TF_ENABLE_ONEDNN_OPTS=0  # quiet data-loader workers

CONFIG=pi05_libero-libero_goal_sequential
CKPT_ROOT="checkpoints/$CONFIG"

# train <experiment-name> <extra train_sequential.py flags...>
# Trains one run per task order. Orders come from the calling script's arguments (default: 0 1 2).
# Settings shared by all paper runs: full fine-tuning of pi0.5, 10 LIBERO-Goal tasks, 5,000 steps per
# task, 32 current-task samples per step (+32 replay samples from the second task on), seed 42.
# --resume continues an interrupted run, or starts a new one if none exists.
train() {
  local name=$1
  shift
  for order in "${ORDERS[@]}"; do
    uv run scripts/train_sequential.py "$CONFIG" \
      --exp-name="${name}_order${order}" \
      --task-order-index "$order" \
      --batch-size=32 \
      --fsdp-devices=1 \
      --resume \
      "$@"
  done
}

ORDERS=("$@")
if [ ${#ORDERS[@]} -eq 0 ]; then
  ORDERS=(0 1 2)
fi
