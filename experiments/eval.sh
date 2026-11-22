#!/bin/bash
# Evaluate runs in LIBERO: every end-of-task checkpoint is rolled out on each task learned so far
# (50 trials per task). Requires the LIBERO environment (see README). Results are written to
# checkpoints/<config>/<run>/eval/<step>/libero_goal/.
#
# Usage: bash experiments/eval.sh <run> [<run> ...]
#   e.g. bash experiments/eval.sh rander_1pct_order0 rander_1pct_order1 rander_1pct_order2
: "${1:?usage: bash experiments/eval.sh <run> [<run> ...]}"
CKPT_RUNS=("$@")
source "$(dirname "$0")/common.sh"

for run in "${CKPT_RUNS[@]}"; do
  uv run examples/libero/eval_sequential.py \
    --ckpt-dir "$CKPT_ROOT/$run" \
    --task-suite libero_goal \
    --num-trials-per-task 50 \
    --num-workers 4 \
    --parallel-mode episode \
    --video-save-fraction 0.1 \
    --wait-for-boot-s 3600
done
