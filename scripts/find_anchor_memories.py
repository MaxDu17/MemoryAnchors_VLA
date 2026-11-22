"""Find anchor memories: state-space ID/OOD partition + flow-matching action disagreement + KNN anchor selection.

Usage:
  export HF_LEROBOT_HOME=/path/to/lerobot

  uv run scripts/find_anchor_memories.py \\
    --config-name pi05_libero-libero_goal_sequential \\
    --checkpoint-dir checkpoints/pi05_libero-libero_goal_sequential/LiberoTrainPi05_100mem/4999
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import tyro

import openpi.training.config as _config
from action_disagreement_utils import ActionOodMode, Consolidation, RotMetric
from anchor_memory_pipeline import AnchorMemoryConfig, run_anchor_memory_pipeline
from obs_representation_utils import (
    ReprMode,
    infer_task_idx_from_checkpoint,
    init_logging,
    load_jax_model,
    n_sequential_tasks,
    resolve_checkpoint_output_dir,
    resolve_checkpoint_path,
)


def main(
    config_name: str,
    checkpoint_dir: str,
    *,
    output_dir: str | None = None,
    output_base: str = "outputs",
    repr_mode: ReprMode = ReprMode.PREFIX_EMBED,
    trained_task_idx: int | None = None,
    next_task_idx: int | None = None,
    max_samples_per_task: int | None = None,
    chunk_size: int = 512,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int = 42,
    flow_noise_level: float = 0.2,
    num_flow_steps: int = 10,
    action_ood_mode: ActionOodMode = ActionOodMode.BASELINE,
    std_cutoff: float = 2.0,
    ood_top_percent: float = 25.0,
    rot_metric: RotMetric = RotMetric.GEODESIC,
    rot_weight: float = 0.1,
    n_memories_per_task: int = 1000,
    n_memories_total: bool = False,
    n_memories_proportion: float | None = None,
    n_neighbors: int = 5,
    consolidation: Consolidation = Consolidation.MEDIAN,
    random_proportion: float = 0.6,
    invert_selection: bool = False,
    invert_remove_proportion: float = 0.05,
) -> None:
    init_logging()
    config = _config.get_config(config_name)
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir)

    if trained_task_idx is None:
        trained_task_idx = infer_task_idx_from_checkpoint(checkpoint_path)
        if trained_task_idx is None:
            raise ValueError("Could not infer trained_task_idx from checkpoint; pass --trained-task-idx.")
        logging.info("Inferred trained_task_idx=%d from checkpoint", trained_task_idx)

    n_tasks = n_sequential_tasks(config)
    if next_task_idx is None:
        next_task_idx = trained_task_idx + 1
    if next_task_idx < 0 or next_task_idx >= n_tasks:
        raise ValueError(f"next_task_idx {next_task_idx} out of range [0, {n_tasks}).")

    reference_task_ids = list(range(trained_task_idx + 1))
    output_path = resolve_checkpoint_output_dir(
        checkpoint_path,
        output_dir=output_dir,
        output_base=output_base,
        suffix=f"anchor_memories_{repr_mode.value}",
    )

    model = load_jax_model(config, checkpoint_path)
    anchor_cfg = AnchorMemoryConfig(
        repr_mode=repr_mode,
        action_ood_mode=action_ood_mode,
        ood_top_percent=ood_top_percent,
        std_cutoff=std_cutoff,
        flow_noise_level=flow_noise_level,
        num_flow_steps=num_flow_steps,
        rot_metric=rot_metric,
        rot_weight=rot_weight,
        n_memories_per_task=n_memories_per_task,
        n_memories_total=n_memories_total,
        n_memories_proportion=n_memories_proportion,
        n_neighbors=n_neighbors,
        consolidation=consolidation,
        random_proportion=random_proportion,
        invert_selection=invert_selection,
        invert_remove_proportion=invert_remove_proportion,
        baseline_max_samples_per_task=max_samples_per_task,
        chunk_size=chunk_size,
        seed=seed,
    )

    result = run_anchor_memory_pipeline(
        model,
        config,
        reference_task_ids=reference_task_ids,
        next_task_idx=next_task_idx,
        output_dir=output_path,
        anchor_cfg=anchor_cfg,
        norm_stats_root=checkpoint_path / "assets",
        batch_size=batch_size,
        num_workers=num_workers,
        save_artifacts=True,
    )
    logging.info(
        "Anchor memories: %d frames selected across %d tasks -> %s",
        int(result.anchor_selected_global_indices.shape[0]),
        len(result.buffer_lst),
        result.output_dir,
    )


if __name__ == "__main__":
    tyro.cli(main)
