# Memory Anchors for Continual Robot Learning

Maximilian Du<sup>1</sup>, Zhanyi Sun<sup>1</sup>, Chen Xu<sup>2</sup>, Paarth Shah<sup>2</sup>, Masha Itkina<sup>2</sup>, Shuran Song<sup>1</sup><br>
<sup>1</sup>Stanford University, <sup>2</sup>Toyota Research Institute

**CoRL 2026** · [Paper (arXiv)](https://arxiv.org/abs/2608.26545) · [Project page](https://robot-adaptation.github.io/MemoryAnchors)

This repository has the code for the paper's π0.5 (vision-language-action) experiments. The π0.5 policy is fine-tuned on the ten LIBERO-Goal tasks one after another, with experience
replay, and we measure how adding or removing memory anchors in the replay buffer changes forgetting.

This repository is built on [openpi](https://github.com/Physical-Intelligence/openpi) from Physical
Intelligence, starting at upstream commit `15a9616`. The git history keeps the upstream commits, so
`git diff 15a9616 HEAD` shows everything we added. openpi's original README is in
[OPENPI_README.md](OPENPI_README.md).

## What we added to openpi

| Area | Files |
|---|---|
| Sequential training with replay | [scripts/train_sequential.py](scripts/train_sequential.py); [replay_memory_budget.py](src/openpi/training/replay_memory_budget.py), [libero_task_order.py](src/openpi/training/libero_task_order.py); replay buffers in [data_loader.py](src/openpi/training/data_loader.py); replay state in [checkpoints.py](src/openpi/training/checkpoints.py) |
| Configs | `*_sequential` configs, `LeRobotLiberoSeqDataConfig` and the continual-learning fields of `TrainConfig` in [config.py](src/openpi/training/config.py) |
| Memory-anchor selection | [find_anchor_memories.py](scripts/find_anchor_memories.py), [anchor_memory_pipeline.py](scripts/anchor_memory_pipeline.py), [anchor_selection_utils.py](scripts/anchor_selection_utils.py), [action_disagreement_utils.py](scripts/action_disagreement_utils.py), [extract_obs_representations.py](scripts/extract_obs_representations.py), [obs_representation_utils.py](scripts/obs_representation_utils.py) |
| Per-task data and normalization | [convert_libero_data_to_lerobot.py](examples/libero/convert_libero_data_to_lerobot.py) (one LeRobot dataset per task), [compute_norm_stats.py](scripts/compute_norm_stats.py) (stats per task), per-task norm stats in [assets/](assets/), policy server option `--task-idx` |
| Evaluation and analysis | [eval_sequential.py](examples/libero/eval_sequential.py), [main_parallel.py](examples/libero/main_parallel.py), [plot_sequential_success_matrix.py](scripts/plot_sequential_success_matrix.py), [aggregate_bwt_per_task.py](scripts/aggregate_bwt_per_task.py) |
| Experiments | [experiments/](experiments/): one script per experiment, plus data, eval and analysis wrappers |

## Setup

Everything runs from the repository root. Training needs one GPU large enough for full fine-tuning of
π0.5; we used a 96 GB RTX PRO 6000 Blackwell.

```bash
git clone --recurse-submodules https://github.com/MaxDu17/MemoryAnchors_VLA.git
cd MemoryAnchors_VLA

# Training and policy-server environment (Python 3.11)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# LIBERO simulator environment, needed only for evaluation (Python 3.8)
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
deactivate

# Data: download the LIBERO RLDS data and convert LIBERO-Goal into one LeRobot dataset per task
bash experiments/prepare_data.sh
```

If you cloned without `--recurse-submodules`, run `git submodule update --init --recursive`.
The π0.5 base weights (`gs://openpi-assets/checkpoints/pi05_base`) download automatically on the first
training run into `$OPENPI_DATA_HOME` (default `~/.cache/openpi`).

## Experiments

Each training script runs task orders 0, 1 and 2 by default (pass orders as arguments to choose).
Select the GPU with `CUDA_VISIBLE_DEVICES`. Checkpoints go to
`checkpoints/pi05_libero-libero_goal_sequential/<experiment>_order<N>/`.

| Script | Replay buffer | Compare with |
|---|---|---|
| `train_rander_1pct.sh` | random replay, 1% of previous-task frames | – |
| `train_add_anchors_8020.sh` | 1% budget: 20% memory anchors + 80% random | `rander_1pct` |
| `train_add_anchors_9010.sh` | 1% budget: 10% memory anchors + 90% random | `rander_1pct` |
| `train_rander_1000.sh` | random replay, 1,000 frames in total | – |
| `train_remove_anchors_5pct.sh` | 1,000 random frames, after excluding the nearest 5% (strongest anchors) | `rander_1000` |
| `train_remove_anchors_10pct.sh` | 1,000 random frames, after excluding the nearest 10% | `rander_1000` |

Memory anchors are the previous-task frames nearest to the new task's action-OOD frames, recomputed
before each new task with the current model. One training run (10 tasks × 5,000 steps) took about
3.5 days on one GPU. Add `--debug` to any `train_sequential.py` command for a short dry run.

**Variance across task orders.** Results depend on the order in which the tasks are learned, and different
orders can give noticeably different numbers. The add-anchors experiments (`add_anchors_8020`,
`add_anchors_9010`) vary more across orders than the others, so compare methods on the same task orders.

**Removing anchors gives a stronger signal than adding them.** A random replay buffer already contains
some memory anchors, so adding more helps in some task orders but not in others. Removing them has a
consistent effect: forgetting increases in every task order we ran.

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/train_rander_1pct.sh          # orders 0, 1, 2
CUDA_VISIBLE_DEVICES=1 bash experiments/train_add_anchors_9010.sh 0   # order 0 only
```

Training details (optimizer, EMA, batch composition, how each buffer is built) are in
[docs/experiments.md](docs/experiments.md). The complete logged config of each experiment is in
[docs/resolved_configs/](docs/resolved_configs/).

## Evaluation and analysis

```bash
# 1. Roll out every end-of-task checkpoint on each task learned so far (50 trials per task)
bash experiments/eval.sh rander_1pct_order0 rander_1pct_order1 rander_1pct_order2

# 2. Success matrices and FWT/BWT/AUC per run, then BWT per task pooled across orders (mean, SEM)
bash experiments/analyze.sh rander_1pct
```

Each run's results go to `<run>/eval/<step>/libero_goal/libero_goal.yaml`. Plots and metrics go to
`<run>/eval/plots/`. With S[i][k] the success rate on task k after training on task i:
- **FWT** (`fwt`): mean of S[k][k], the success on each task right after learning it.
- **BWT** (`bwt`): for each task k, the mean of S[k][k] − S[i][k] over the later stages i > k, then averaged
  over tasks. Positive values mean forgetting (higher is worse).
- **AUC** (`auc`): for each task k, the mean of S[i][k] over stages i ≥ k, then averaged over tasks.

## Changes to openpi files

Under the Apache License 2.0, modified upstream files must be marked as changed. Compared with
openpi commit `15a9616`, we modified:

- `src/openpi/training/config.py`, `data_loader.py`, `checkpoints.py`
- `src/openpi/policies/policy_config.py`, `src/openpi/serving/websocket_policy_server.py`
- `scripts/train.py`, `scripts/serve_policy.py`, `scripts/compute_norm_stats.py`
- `examples/libero/convert_libero_data_to_lerobot.py`
- `pyproject.toml`, `uv.lock`, `.gitignore`

We removed openpi's GitHub CI workflows and `CODEOWNERS` file, and moved its README to
`OPENPI_README.md`. All other
additions are new files (see the table above, and `git diff --stat 15a9616 HEAD`).

## Citation

```bibtex
@misc{du2026memoryanchorscontinualrobot,
      title={Memory Anchors for Continual Robot Learning},
      author={Maximilian Du and Zhanyi Sun and Chen Xu and Paarth Shah and Masha Itkina and Shuran Song},
      year={2026},
      eprint={2608.26545},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.26545},
}
```

Please also cite [openpi](https://github.com/Physical-Intelligence/openpi) and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

## License

Apache License 2.0 ([LICENSE](LICENSE)), as for openpi. The PaliGemma/Gemma weights used by π0.5 are
under the Gemma terms of use ([LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)). LIBERO is included as a git
submodule under its own license.
