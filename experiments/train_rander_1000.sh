#!/bin/bash
# Random experience replay (randER) with a fixed budget of 1,000 frames in total.
# The 1,000 frames are split across the previous tasks in proportion to their size and re-drawn at
# every task boundary. Reference for the remove-anchors experiments.
#
# Usage: bash experiments/train_rander_1000.sh [ORDER ...]   (default orders: 0 1 2)
source "$(dirname "$0")/common.sh"

train rander_1000 \
  --cl er \
  --n-memories 1000 \
  --n-memories-total
