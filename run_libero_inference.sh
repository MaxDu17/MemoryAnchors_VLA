#!/bin/bash

# uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
# only the first time 
# uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
# uv pip install -e packages/openpi-client
# uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

MUJOCO_GL=egl python examples/libero/main_parallel.py \
  --args.task-suite-names libero_goal \
  --args.max-tasks 1 \
  --args.num-trials-per-task 10 \
  --args.num-workers 2 \
#   --args.out-path my_eval_runs \
#   --args.run-id pi0_seq_task1_ckpt9999