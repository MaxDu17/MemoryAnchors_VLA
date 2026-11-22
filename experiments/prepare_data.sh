#!/bin/bash
# Download the LIBERO RLDS data (openvla/modified_libero_rlds) and convert LIBERO-Goal into one
# LeRobot dataset per task under data/physical-intelligence/libero_bytask/libero-libero_goal_no_noops/.
# Other suites: --idx 0 = libero_10, 2 = libero_object, 3 = libero_spatial.
#
# Usage: bash experiments/prepare_data.sh
source "$(dirname "$0")/common.sh"

uv run download_dataset.py
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir=data \
  --output-dir=data/physical-intelligence/libero_bytask \
  --idx=1

# Per-task norm stats for this config are already in assets/. To recompute them:
# uv run scripts/compute_norm_stats.py --config-name "$CONFIG"
