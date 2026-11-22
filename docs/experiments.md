# Training details

All experiments use the `pi05_libero-libero_goal_sequential` config in
[src/openpi/training/config.py](../src/openpi/training/config.py) with the flags in the
[experiments/](../experiments/) scripts. The complete logged config of each experiment is in
[resolved_configs/](resolved_configs/).

## Shared setup

| Setting | Value |
|---|---|
| Model | π0.5, initialized from `pi05_base`; full fine-tuning (no LoRA), JAX, bf16 compute |
| Action horizon / tokens | 10-step action chunks; state not tokenized (`discrete_state_input=False`); `max_token_len=200` |
| Tasks | 10 LIBERO-Goal tasks (no-op actions removed), one LeRobot dataset per task |
| Task orders | Orders 0, 1 and 2 from [libero_task_order.py](../src/openpi/training/libero_task_order.py) |
| Steps | 5,000 per task, 50,000 total |
| Batch | 32 current-task samples per step; from the second task on, plus 32 replay samples (1:1, 64 total) |
| Optimizer | AdamW (β1 0.9, β2 0.95, ε 1e-8, weight decay 1e-10), gradient-norm clip 1.0 |
| Learning rate | Linear warmup to 5e-5 over the first 10,000 steps of the run (the first two tasks), then constant |
| EMA | Decay 0.999, updated every step; checkpoints store the EMA weights, so evaluation uses them |
| Across tasks | Model, optimizer state, LR schedule and EMA carry over; nothing is reset at task boundaries |
| Normalization | Per-task quantile norm stats ([assets/](../assets/pi05_libero-libero_goal_sequential/)); the policy server loads the stats of the evaluated task |
| Checkpoints | One per task, saved at the end of the task |
| Seed | 42 |

## Replay buffers

Buffer sizes are counted in frames (timesteps; each comes with its 10-step action chunk), not episodes.
The buffer for the next task is built at each task boundary from the full data of all previous tasks.

**Random replay (`--cl er`).** The total budget is either a fraction of all previous-task frames
(`--n-memories-proportion 0.01 --n-memories-total`) or a fixed count (`--n-memories 1000 --n-memories-total`).
It is split across the previous tasks in proportion to their frame counts (largest-remainder rounding) and
sampled uniformly without replacement within each task. With the 1% budget on LIBERO-Goal this is 42–72
frames per task, and 458–475 frames in total before the last task. The buffer is re-drawn at every task
boundary with seed `42 + 1009 × (number of previous tasks)`, so the stored frames of an old task change from
stage to stage. See [replay_memory_budget.py](../src/openpi/training/replay_memory_budget.py) and
`resample_replay_buffer_lst` in [data_loader.py](../src/openpi/training/data_loader.py).

**Memory-anchor replay (`--cl er_anchor`).** Before each new task, the current model selects anchors in
four steps (see [anchor_memory_pipeline.py](../scripts/anchor_memory_pipeline.py)):

1. Embed every previous-task frame and every new-task frame (`prefix_embed` representations).
2. Keep the new-task frames that look in-distribution: for each new-task frame, take its distances to the
   nearest 10% of previous-task frames, and split the new-task frames into two clusters on those distances.
3. Score the kept frames by action disagreement: noise the demonstrated action chunk (flow noise level 0.2),
   denoise it with the model (10 flow steps), and measure the gap (rotations by geodesic distance, weight 0.1).
   The top 25% are the action-OOD frames.
4. Give each previous-task frame the median distance to its 5 nearest action-OOD frames. The previous-task
   frames with the smallest distance are the memory anchors.

Selection uses all previous tasks pooled, without per-task quotas (`finalize_global_anchor_buffers` in
[anchor_selection_utils.py](../scripts/anchor_selection_utils.py)).

- *Add anchors* (`--anchor-random-proportion p`): a fraction (1 − p) of the budget goes to the nearest
  anchors; the rest is sampled at random from the remaining frames. p = 0.8 and 0.9 give 20% and 10% anchors.
- *Remove anchors* (`--anchor-invert-selection --anchor-invert-remove-proportion r`): the nearest fraction r
  of all previous-task frames is excluded, then the whole budget is sampled at random from what remains.

All other anchor settings (`anchor_*` fields of `TrainConfig`) are at their defaults, which are the values
given in steps 1–4.

**Batches.** Each step concatenates one batch from the current task and one batch drawn uniformly from the
pooled replay buffer (no per-task balancing). The first task trains without replay.
