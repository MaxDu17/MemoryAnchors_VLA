"""Extract observation representations from a trained checkpoint and plot t-SNE.

Usage:
  export HF_LEROBOT_HOME=/path/to/lerobot  # parent of physical-intelligence/...

  uv run scripts/extract_obs_representations.py \\
    --config-name pi05_libero-libero_goal_sequential \\
    --checkpoint-dir checkpoints/pi05_libero-libero_goal_sequential/LiberoTrainPi05_100mem/4999 \\
    --repr-mode prefix_out \\
    --all-tasks --max-samples 2000

Outputs are written to outputs/{exp_name}/{step}/ by default, e.g.:
  outputs/LiberoTrainPi05_100mem/4999/tsne_prefix_out.png
"""

from __future__ import annotations

import dataclasses
import json
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

import jax
import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold
import tqdm
import tyro

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
from obs_representation_utils import (
    ReprMode,
    create_loader,
    decode_prompts,
    infer_task_idx_from_checkpoint,
    init_logging,
    load_jax_model,
    make_extract_fn,
    resolve_checkpoint_output_dir,
    resolve_checkpoint_path,
    task_name_from_repo_id,
)


def _task_indices(config: _config.TrainConfig, *, task_idx: int | None, all_tasks: bool) -> list[int]:
    data_factory = config.data
    if hasattr(data_factory, "repo_ids") and data_factory.repo_ids:
        n_tasks = len(data_factory.repo_ids)
        if all_tasks:
            return list(range(n_tasks))
        if task_idx is None:
            raise ValueError("Sequential config requires --task-idx or --all-tasks.")
        if task_idx < 0 or task_idx >= n_tasks:
            raise ValueError(f"task_idx {task_idx} out of range [0, {n_tasks})")
        return [task_idx]
    if all_tasks:
        raise ValueError("--all-tasks is only supported for sequential configs with repo_ids.")
    return [0]


def _run_tsne(embeddings: np.ndarray, seed: int) -> np.ndarray:
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")
    perplexity = min(30.0, max(1.0, (n - 1) / 3.0))
    tsne = sklearn.manifold.TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(embeddings)


def _plot_tsne(
    coords: np.ndarray,
    task_names: np.ndarray,
    output_path: pathlib.Path,
    *,
    repr_mode: ReprMode,
) -> None:
    unique_names = []
    seen: set[str] = set()
    for name in task_names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)

    cmap = plt.get_cmap("tab10" if len(unique_names) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, name in enumerate(unique_names):
        mask = task_names == name
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.7,
            color=cmap(i % cmap.N),
            label=name,
        )
    ax.set_title(f"t-SNE of observation representations ({repr_mode.value})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if len(unique_names) > 1:
        ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(
    config_name: str,
    checkpoint_dir: str,
    *,
    output_dir: str | None = None,
    output_base: str = "outputs",
    repr_mode: ReprMode = ReprMode.PREFIX_EMBED,
    task_idx: int | None = None,
    all_tasks: bool = False,
    max_samples: int = 2000,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int = 42,
) -> None:
    init_logging()
    config = _config.get_config(config_name)
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir)
    output_path = resolve_checkpoint_output_dir(checkpoint_path, output_dir=output_dir, output_base=output_base)
    output_path.mkdir(parents=True, exist_ok=True)

    if task_idx is None and not all_tasks:
        task_idx = infer_task_idx_from_checkpoint(checkpoint_path)
        if task_idx is not None:
            logging.info("Inferred task_idx=%d from checkpoint", task_idx)

    batch_size = batch_size or config.batch_size
    num_workers = config.num_workers if num_workers is None else num_workers

    model = load_jax_model(config, checkpoint_path)
    extract_fn = make_extract_fn(model, repr_mode)
    logging.info("Representation mode: %s", repr_mode.value)

    task_ids = _task_indices(config, task_idx=task_idx, all_tasks=all_tasks)
    logging.info("Extracting embeddings for task(s): %s", task_ids)

    all_embeddings: list[np.ndarray] = []
    all_task_idx: list[int] = []
    all_task_names: list[str] = []
    all_frame_idx: list[int] = []
    all_prompts: list[str] = []
    all_repo_ids: list[str] = []
    global_frame_offset = 0

    for tid in task_ids:
        if isinstance(config.data, _config.LeRobotLiberoSeqDataConfig):
            data_config = config.data.create(config.assets_dirs, config.model, task_id=tid)
        else:
            data_config = config.data.create(config.assets_dirs, config.model)

        if data_config.asset_id is None:
            raise ValueError("asset_id is required to load norm stats from checkpoint.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_path / "assets", data_config.asset_id)
        data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

        loader, n_samples = create_loader(
            config,
            data_config,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_samples=max_samples,
        )
        n_to_extract = min(n_samples, max_samples)
        task_name = task_name_from_repo_id(data_config.repo_id or f"task_{tid}")
        logging.info("Task %d (%s): extracting first %d / %d frames", tid, task_name, n_to_extract, n_samples)

        frame_idx = 0
        for observation, _ in tqdm.tqdm(
            loader, desc=task_name, total=(n_to_extract + batch_size - 1) // batch_size
        ):
            if frame_idx >= n_to_extract:
                break

            embeddings = np.asarray(jax.device_get(extract_fn(observation))).astype(np.float32)
            prompts = decode_prompts(observation)

            frames_remaining = n_to_extract - frame_idx
            if embeddings.shape[0] > frames_remaining:
                embeddings = embeddings[:frames_remaining]
                prompts = prompts[:frames_remaining]

            batch_size_actual = embeddings.shape[0]
            all_embeddings.append(embeddings)
            all_task_idx.extend([tid] * batch_size_actual)
            all_task_names.extend([task_name] * batch_size_actual)
            all_frame_idx.extend(
                range(global_frame_offset + frame_idx, global_frame_offset + frame_idx + batch_size_actual)
            )
            all_prompts.extend(prompts)
            all_repo_ids.extend([data_config.repo_id or ""] * batch_size_actual)
            frame_idx += batch_size_actual

        global_frame_offset += n_to_extract

    embeddings = np.concatenate(all_embeddings, axis=0)
    task_indices = np.asarray(all_task_idx, dtype=np.int32)
    task_names = np.asarray(all_task_names, dtype=object)
    frame_indices = np.asarray(all_frame_idx, dtype=np.int64)
    prompts = np.asarray(all_prompts, dtype=object)
    repo_ids = np.asarray(all_repo_ids, dtype=object)

    repr_suffix = repr_mode.value
    npz_path = output_path / f"representations_{repr_suffix}.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        task_idx=task_indices,
        task_name=task_names,
        frame_idx=frame_indices,
        prompts=prompts,
        repo_id=repo_ids,
    )
    logging.info("Saved %d embeddings (%d-d) to %s", embeddings.shape[0], embeddings.shape[1], npz_path)

    tsne_coords = _run_tsne(embeddings, seed)
    tsne_path = output_path / f"tsne_{repr_suffix}.png"
    _plot_tsne(tsne_coords, task_names, tsne_path, repr_mode=repr_mode)
    logging.info("Saved t-SNE plot to %s", tsne_path)

    metadata = {
        "config_name": config_name,
        "checkpoint_dir": str(checkpoint_path),
        "representation": repr_mode.value,
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "max_samples_per_task": max_samples,
        "task_indices": task_ids,
        "seed": seed,
        "output_dir": str(output_path),
    }
    metadata_path = output_path / f"metadata_{repr_suffix}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logging.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    tyro.cli(main)
