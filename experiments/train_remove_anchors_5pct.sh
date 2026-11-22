#!/bin/bash
# Remove memory anchors: 1,000-frame replay budget drawn at random after excluding the 5% of
# previous-task frames that are nearest to the new task's action-OOD frames (the strongest anchors).
# Compare with train_rander_1000.sh.
#
# Usage: bash experiments/train_remove_anchors_5pct.sh [ORDER ...]   (default orders: 0 1 2)
source "$(dirname "$0")/common.sh"

train remove_anchors_5pct \
  --cl er_anchor \
  --n-memories 1000 \
  --n-memories-total \
  --anchor-invert-selection \
  --anchor-invert-remove-proportion 0.05
