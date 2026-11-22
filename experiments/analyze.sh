#!/bin/bash
# Success matrix and FWT/BWT/AUC metrics for each order of one experiment, then BWT per task pooled
# across orders (mean and SEM). Run after experiments/eval.sh.
# Outputs per run: checkpoints/<config>/<run>/eval/plots/libero_goal_success_matrix{.png,.npy,_metrics.json}
#
# Usage: bash experiments/analyze.sh <experiment> [ORDER ...]   (default orders: 0 1 2)
#   e.g. bash experiments/analyze.sh add_anchors_9010
EXPERIMENT=${1:?"usage: bash experiments/analyze.sh <experiment> [ORDER ...]"}
shift
source "$(dirname "$0")/common.sh"

metrics=()
for order in "${ORDERS[@]}"; do
  run="$CKPT_ROOT/${EXPERIMENT}_order${order}"
  uv run scripts/plot_sequential_success_matrix.py --ckpt-dir "$run" --task-suite libero_goal
  metrics+=("$run/eval/plots/libero_goal_success_matrix_metrics.json")
done

uv run scripts/aggregate_bwt_per_task.py "${metrics[@]}"
