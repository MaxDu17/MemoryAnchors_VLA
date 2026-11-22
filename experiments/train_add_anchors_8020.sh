#!/bin/bash
# Add memory anchors: 1% replay budget, 20% memory anchors + 80% random frames.
# Before each new task, the anchors are the previous-task frames nearest to the new task's
# action-OOD frames (selected from all previous tasks pooled). Compare with train_rander_1pct.sh.
#
# Usage: bash experiments/train_add_anchors_8020.sh [ORDER ...]   (default orders: 0 1 2)
source "$(dirname "$0")/common.sh"

train add_anchors_8020 \
  --cl er_anchor \
  --n-memories-proportion=0.01 \
  --n-memories-total \
  --anchor-random-proportion 0.8
