#!/bin/bash
# Random experience replay (randER) with a 1% replay budget.
# Before each new task, the buffer holds 1% of all previous-task frames, split across the previous
# tasks in proportion to their size and sampled uniformly within each task. The buffer is re-drawn
# at every task boundary. Reference for the add-anchors experiments.
#
# Usage: bash experiments/train_rander_1pct.sh [ORDER ...]   (default orders: 0 1 2)
source "$(dirname "$0")/common.sh"

train rander_1pct \
  --cl er \
  --n-memories-proportion=0.01 \
  --n-memories-total
