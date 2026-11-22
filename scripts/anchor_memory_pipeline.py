"""Shared anchor-memory selection pipeline for offline analysis and er_anchor training."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance
import sklearn.cluster
import tqdm

import openpi.policies.libero_policy as libero_policy
import openpi.training.config as _config
import openpi.training.replay_memory_budget as _replay_memory_budget
from action_disagreement_utils import (
    ActionOodMode,
    Consolidation,
    RotMetric,
    compute_baseline_stats,
    export_min_max_action_reconstruction_videos,
    extra_delta_transform_enabled,
    make_flow_denoise_fn,
    plot_action_disagreement_scores_histogram,
    plot_action_ood_step_histogram,
    save_action_ood_image_tile,
    score_loader_action_disagreement,
    select_action_ood_indices,
    select_action_ood_top_percent,
)
from anchor_selection_utils import ReferenceFrameIndex, run_anchor_selection_stage
from obs_representation_utils import (
    ReprMode,
    create_loader,
    create_subset_loader,
    extract_task_embeddings,
    make_data_config,
    make_extract_fn,
    resolve_norm_stats_root,
    task_name_from_repo_id,
)


@dataclasses.dataclass
class AnchorMemoryConfig:
    repr_mode: ReprMode = ReprMode.PREFIX_EMBED
    action_ood_mode: ActionOodMode = ActionOodMode.TOP_PERCENT
    ood_top_percent: float = 25.0
    std_cutoff: float = 2.0
    flow_noise_level: float = 0.2
    num_flow_steps: int = 10
    rot_metric: RotMetric = RotMetric.GEODESIC
    rot_weight: float = 0.1
    n_memories_per_task: int = 1000
    n_memories_total: bool = False
    n_memories_proportion: float | None = None
    n_neighbors: int = 5
    consolidation: Consolidation = Consolidation.MEDIAN
    random_proportion: float = 0.6
    invert_selection: bool = False
    invert_remove_proportion: float = 0.05
    baseline_max_samples_per_task: int | None = None
    max_embedding_samples_per_task: int | None = None
    chunk_size: int = 512
    seed: int = 42


@dataclasses.dataclass
class AnchorMemoryResult:
    buffer_lst: list[list[int]]
    anchor_selected_global_indices: np.ndarray
    action_ood_indices: np.ndarray
    reference_task_ids: list[int]
    metadata: dict
    output_dir: pathlib.Path


def create_raw_lerobot_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
) -> lerobot_dataset.LeRobotDataset:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    repo_id = data_config.repo_id
    assert repo_id is not None
    root = pathlib.Path(lerobot_dataset.HF_LEROBOT_HOME) / repo_id
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    return lerobot_dataset.LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )


def collect_frame_metadata(
    raw_dataset: lerobot_dataset.LeRobotDataset,
    n_frames: int,
) -> dict[str, np.ndarray]:
    frame_indices = np.zeros(n_frames, dtype=np.int64)
    episode_indices = np.zeros(n_frames, dtype=np.int64)
    for idx in range(n_frames):
        row = raw_dataset.hf_dataset[idx]
        frame_indices[idx] = int(row["frame_index"])
        episode_indices[idx] = int(row["episode_index"])
    return {"frame_index": frame_indices, "episode_index": episode_indices}


def load_agentview_image(raw_dataset: lerobot_dataset.LeRobotDataset, dataset_idx: int) -> np.ndarray:
    image = raw_dataset.hf_dataset[dataset_idx]["image"]
    return libero_policy._parse_image(np.asarray(image))


def _cluster_and_group_by_distance_features(topk: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    k = topk.shape[1]
    q1 = topk[:, max(int(0.1 * k), 0)]
    q5 = topk[:, max(int(0.5 * k), 0)]
    q10 = topk[:, -1]
    nn = topk[:, 0]
    spread = q10 - nn
    local_std = topk.std(axis=1)
    features_raw = np.stack([nn, q1, q5, q10, spread, local_std], axis=1)
    features = (features_raw - features_raw.mean(axis=0)) / features_raw.std(axis=0)
    labels = sklearn.cluster.KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(features)
    in_cluster_idx = 0 if features[labels == 0, 0].mean() < features[labels == 1, 0].mean() else 1
    return labels, in_cluster_idx, features_raw


def partition_new_data(
    reference_reprs: np.ndarray,
    target_reprs: np.ndarray,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    k = int(0.1 * reference_reprs.shape[0])
    assert k > 0, "partition_new_data requires at least 10 samples in the reference set."
    n_target = target_reprs.shape[0]
    topk = np.full((n_target, k), np.inf, dtype=np.float32)
    for i in tqdm.tqdm(range(0, reference_reprs.shape[0], chunk_size), desc="Partitioning new data"):
        ref_block = reference_reprs[i : i + chunk_size]
        dist_block = scipy.spatial.distance.cdist(ref_block, target_reprs, metric="euclidean")
        merged = np.concatenate([topk, dist_block.T], axis=1)
        topk = np.partition(merged, k - 1, axis=1)[:, :k]
    topk = np.sort(topk, axis=1)
    labels, in_cluster_idx, features_raw = _cluster_and_group_by_distance_features(topk)
    id_indices = np.where(labels == in_cluster_idx)[0]
    return labels, in_cluster_idx, id_indices, features_raw


def _plot_data_partition_histogram(
    labels: np.ndarray,
    frame_indices: np.ndarray,
    in_cluster_idx: int,
    output_path: pathlib.Path,
) -> None:
    out_cluster_idx = 0 if in_cluster_idx == 1 else 1
    steps_by_label: dict[int, list[int]] = {0: [], 1: []}
    for i, label in enumerate(labels):
        steps_by_label[int(label)].append(int(frame_indices[i]))
    plt.figure(figsize=(10, 5))
    plt.hist(steps_by_label[out_cluster_idx], bins=30, alpha=0.7, label="Out of Distribution")
    plt.hist(steps_by_label[in_cluster_idx], bins=30, alpha=0.7, label="In Distribution")
    plt.xlabel("Step in Trajectory")
    plt.ylabel("Count")
    plt.title("Histogram of Step in Trajectory by cluster label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_id_image_tile(
    id_indices: np.ndarray,
    raw_dataset: lerobot_dataset.LeRobotDataset,
    frame_indices: np.ndarray,
    output_path: pathlib.Path,
    *,
    seed: int,
) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    images_needed = 100
    if id_indices.shape[0] > images_needed:
        sample_positions = rng.choice(id_indices.shape[0], size=images_needed, replace=False)
        chosen = id_indices[sample_positions]
    else:
        chosen = id_indices
    elements: list[np.ndarray] = []
    placeholder_shape: tuple[int, int, int] | None = None
    for i in range(images_needed):
        if i < chosen.shape[0]:
            dataset_idx = int(chosen[i])
            img = load_agentview_image(raw_dataset, dataset_idx)
            img = np.ascontiguousarray(img)
            if placeholder_shape is None:
                placeholder_shape = img.shape
            step = int(frame_indices[dataset_idx])
            cv2.putText(img, f"step {step}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            if placeholder_shape is None:
                placeholder_shape = (256, 256, 3)
            img = np.zeros(placeholder_shape, dtype=np.uint8)
        elements.append(img)
    stacked = np.stack(elements)
    grid_rows = [np.hstack([stacked[row * 10 + col] for col in range(10)]) for row in range(10)]
    plt.imsave(output_path, np.vstack(grid_rows))


def _run_action_disagreement_stage(
    model,
    config: _config.TrainConfig,
    norm_stats_root: pathlib.Path,
    *,
    reference_task_ids: list[int],
    id_indices: np.ndarray,
    target_data_config: _config.DataConfig,
    frame_metadata: dict[str, np.ndarray],
    raw_dataset: lerobot_dataset.LeRobotDataset,
    task_slug: str,
    vis_dir: pathlib.Path,
    batch_size: int,
    num_workers: int,
    anchor_cfg: AnchorMemoryConfig,
) -> dict:
    flow_denoise_fn = make_flow_denoise_fn(model)
    extra_delta = extra_delta_transform_enabled(config)
    baseline: dict[str, float] | None = None

    if anchor_cfg.action_ood_mode == ActionOodMode.BASELINE:
        baseline_chunks: list[np.ndarray] = []
        for task_id in reference_task_ids:
            data_config = make_data_config(config, norm_stats_root, task_id)
            loader, _ = create_loader(
                config,
                data_config,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=anchor_cfg.seed,
                max_samples=anchor_cfg.baseline_max_samples_per_task,
            )
            task_name = task_name_from_repo_id(data_config.repo_id or f"task_{task_id}")
            scores, _ = score_loader_action_disagreement(
                flow_denoise_fn,
                loader,
                data_config,
                extra_delta_transform=extra_delta,
                flow_t=anchor_cfg.flow_noise_level,
                num_flow_steps=anchor_cfg.num_flow_steps,
                rot_metric=anchor_cfg.rot_metric,
                rot_weight=anchor_cfg.rot_weight,
                seed=anchor_cfg.seed + task_id,
                desc=f"Baseline action disagreement ({task_name})",
                return_viz=False,
            )
            baseline_chunks.append(scores)
        baseline_scores = np.concatenate(baseline_chunks, axis=0)
        baseline = compute_baseline_stats(baseline_scores)
        logging.info("Action disagreement baseline: mean=%.4f std=%.4f", baseline["mean"], baseline["std"])

    id_loader, _ = create_subset_loader(
        config,
        target_data_config,
        id_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=anchor_cfg.seed,
    )
    id_scores, id_viz = score_loader_action_disagreement(
        flow_denoise_fn,
        id_loader,
        target_data_config,
        extra_delta_transform=extra_delta,
        flow_t=anchor_cfg.flow_noise_level,
        num_flow_steps=anchor_cfg.num_flow_steps,
        rot_metric=anchor_cfg.rot_metric,
        rot_weight=anchor_cfg.rot_weight,
        seed=anchor_cfg.seed + 1000,
        desc=f"Action disagreement ID ({task_slug})",
    )

    if anchor_cfg.action_ood_mode == ActionOodMode.BASELINE:
        assert baseline is not None
        action_ood_indices = select_action_ood_indices(
            id_indices,
            id_scores,
            baseline["mean"],
            baseline["std"],
            anchor_cfg.std_cutoff,
        )
    else:
        action_ood_indices = select_action_ood_top_percent(id_indices, id_scores, anchor_cfg.ood_top_percent)

    plot_action_ood_step_histogram(
        frame_metadata["frame_index"],
        action_ood_indices,
        vis_dir / f"{task_slug}_hist_action_ood_steps.png",
    )
    if anchor_cfg.action_ood_mode == ActionOodMode.BASELINE:
        assert baseline is not None
        plot_action_disagreement_scores_histogram(
            id_scores,
            vis_dir / f"{task_slug}_hist_action_ood_scores.png",
            action_ood_mode=ActionOodMode.BASELINE,
            baseline_mean=baseline["mean"],
            baseline_std=baseline["std"],
            std_cutoff=anchor_cfg.std_cutoff,
        )
    else:
        plot_action_disagreement_scores_histogram(
            id_scores,
            vis_dir / f"{task_slug}_hist_action_ood_scores.png",
            action_ood_mode=ActionOodMode.TOP_PERCENT,
            top_percent=anchor_cfg.ood_top_percent,
        )
    export_min_max_action_reconstruction_videos(id_scores, id_indices, id_viz, vis_dir, task_slug)
    if action_ood_indices.shape[0] > 0:
        save_action_ood_image_tile(
            action_ood_indices,
            id_scores,
            id_indices,
            id_scores,
            raw_dataset,
            frame_metadata["frame_index"],
            vis_dir / f"{task_slug}_top100_action_ood.png",
            load_agentview_image,
        )

    result: dict = {
        "action_disagreement_scores": id_scores,
        "action_ood_indices": action_ood_indices,
        "num_action_ood_frames": int(action_ood_indices.shape[0]),
        "action_ood_mode": anchor_cfg.action_ood_mode.value,
        "flow_noise_level": anchor_cfg.flow_noise_level,
        "num_flow_steps": anchor_cfg.num_flow_steps,
        "rot_metric": anchor_cfg.rot_metric.value,
        "rot_weight": anchor_cfg.rot_weight,
        "extra_delta_transform": extra_delta,
    }
    if baseline is not None:
        result["baseline_mean"] = baseline["mean"]
        result["baseline_std"] = baseline["std"]
        result["std_cutoff"] = anchor_cfg.std_cutoff
    else:
        result["ood_top_percent"] = anchor_cfg.ood_top_percent
    return result


def run_anchor_memory_pipeline(
    model,
    config: _config.TrainConfig,
    *,
    reference_task_ids: list[int],
    next_task_idx: int,
    output_dir: pathlib.Path,
    anchor_cfg: AnchorMemoryConfig,
    norm_stats_root: pathlib.Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
    save_artifacts: bool = True,
) -> AnchorMemoryResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    batch_size = batch_size or config.batch_size
    num_workers = config.num_workers if num_workers is None else num_workers
    extract_fn = make_extract_fn(model, anchor_cfg.repr_mode)
    max_emb = anchor_cfg.max_embedding_samples_per_task

    reference_chunks: list[np.ndarray] = []
    ref_task_id_chunks: list[np.ndarray] = []
    ref_dataset_index_chunks: list[np.ndarray] = []
    ref_frame_index_chunks: list[np.ndarray] = []
    ref_raw_datasets: dict[int, lerobot_dataset.LeRobotDataset] = {}
    ref_task_slugs: dict[int, str] = {}

    for task_id in reference_task_ids:
        embeddings, ref_data_config, n_ref_frames = extract_task_embeddings(
            model,
            extract_fn,
            config,
            norm_stats_root,
            task_id,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=anchor_cfg.seed,
            max_samples=max_emb,
        )
        reference_chunks.append(embeddings)
        ref_task_id_chunks.append(np.full(embeddings.shape[0], task_id, dtype=np.int32))
        ref_dataset_index_chunks.append(np.arange(embeddings.shape[0], dtype=np.int64))
        ref_raw = create_raw_lerobot_dataset(ref_data_config, config.model.action_horizon)
        ref_raw_datasets[task_id] = ref_raw
        ref_task_slugs[task_id] = task_name_from_repo_id(ref_data_config.repo_id or f"task_{task_id}")
        ref_metadata = collect_frame_metadata(ref_raw, n_ref_frames)
        ref_frame_index_chunks.append(ref_metadata["frame_index"][: embeddings.shape[0]])

    reference_reprs = np.concatenate(reference_chunks, axis=0)
    ref_task_id_per_frame = np.concatenate(ref_task_id_chunks)
    ref_dataset_index_per_frame = np.concatenate(ref_dataset_index_chunks)
    ref_frame_index_per_frame = np.concatenate(ref_frame_index_chunks)

    target_reprs, target_data_config, n_target_frames = extract_task_embeddings(
        model,
        extract_fn,
        config,
        norm_stats_root,
        next_task_idx,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=anchor_cfg.seed,
        max_samples=max_emb,
    )
    raw_dataset = create_raw_lerobot_dataset(target_data_config, config.model.action_horizon)
    frame_metadata = collect_frame_metadata(raw_dataset, n_target_frames)
    n_target_emb = target_reprs.shape[0]
    for key in frame_metadata:
        frame_metadata[key] = frame_metadata[key][:n_target_emb]

    labels, in_cluster_idx, id_indices, features_raw = partition_new_data(
        reference_reprs,
        target_reprs,
        chunk_size=anchor_cfg.chunk_size,
    )
    ood_cluster_idx = 0 if in_cluster_idx == 1 else 1
    id_mask = labels == in_cluster_idx
    ood_mask = labels == ood_cluster_idx
    task_slug = task_name_from_repo_id(target_data_config.repo_id or f"task_{next_task_idx}")

    if save_artifacts:
        np.save(output_dir / "partition_labels.npy", labels)
        np.save(output_dir / "id_indices.npy", id_indices)
        np.savez(
            output_dir / "representations_reference.npz",
            embeddings=reference_reprs,
            task_ids=np.asarray(reference_task_ids, dtype=np.int32),
            task_id_per_frame=ref_task_id_per_frame,
            dataset_index=ref_dataset_index_per_frame,
            frame_index=ref_frame_index_per_frame,
        )
        np.savez(
            output_dir / "representations_target.npz",
            embeddings=target_reprs,
            frame_index=frame_metadata["frame_index"],
            episode_index=frame_metadata["episode_index"],
        )
        _plot_data_partition_histogram(
            labels,
            frame_metadata["frame_index"],
            in_cluster_idx,
            vis_dir / f"{task_slug}_hist_data_partition.png",
        )
        _save_id_image_tile(
            id_indices,
            raw_dataset,
            frame_metadata["frame_index"],
            vis_dir / f"{task_slug}_top100_ID.png",
            seed=anchor_cfg.seed,
        )

    action_results = _run_action_disagreement_stage(
        model,
        config,
        norm_stats_root,
        reference_task_ids=reference_task_ids,
        id_indices=id_indices,
        target_data_config=target_data_config,
        frame_metadata=frame_metadata,
        raw_dataset=raw_dataset,
        task_slug=task_slug,
        vis_dir=vis_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        anchor_cfg=anchor_cfg,
    )

    if save_artifacts:
        np.save(output_dir / "action_disagreement_scores.npy", action_results["action_disagreement_scores"])
        np.save(output_dir / "action_ood_indices.npy", action_results["action_ood_indices"])

    ref_index = ReferenceFrameIndex(
        task_id=ref_task_id_per_frame,
        dataset_index=ref_dataset_index_per_frame,
        frame_index=ref_frame_index_per_frame,
        task_slugs=ref_task_slugs,
        raw_datasets=ref_raw_datasets,
    )
    pool_sizes = [int((ref_task_id_per_frame == task_id).sum()) for task_id in reference_task_ids]
    memory_budget = _replay_memory_budget.resolve_replay_memory_budget(
        pool_sizes,
        n_memories=anchor_cfg.n_memories_per_task,
        n_memories_total=anchor_cfg.n_memories_total,
        n_memories_proportion=anchor_cfg.n_memories_proportion,
    )
    anchor_results = run_anchor_selection_stage(
        reference_reprs,
        target_reprs,
        action_results["action_ood_indices"],
        ref_index,
        output_dir,
        vis_dir,
        task_slug,
        memory_budget=memory_budget,
        reference_task_ids=reference_task_ids,
        n_neighbors=anchor_cfg.n_neighbors,
        consolidation=anchor_cfg.consolidation,
        random_proportion=anchor_cfg.random_proportion,
        invert_selection=anchor_cfg.invert_selection,
        invert_remove_proportion=anchor_cfg.invert_remove_proportion,
        seed=anchor_cfg.seed,
    )

    metadata = {
        "reference_task_ids": reference_task_ids,
        "next_task_idx": next_task_idx,
        "task_slug": task_slug,
        "memory_budget": memory_budget,
        "n_memories_total": anchor_cfg.n_memories_total,
        "n_memories_proportion": anchor_cfg.n_memories_proportion,
        "representation": anchor_cfg.repr_mode.value,
        "num_reference_frames": int(reference_reprs.shape[0]),
        "num_target_frames": int(n_target_emb),
        "num_id_frames": int(id_indices.shape[0]),
        "num_ood_frames": int(ood_mask.sum()),
        "in_cluster_idx": int(in_cluster_idx),
        "id_cluster_mean_nn_distance": float(features_raw[id_mask, 0].mean()),
        "ood_cluster_mean_nn_distance": float(features_raw[ood_mask, 0].mean()),
        "output_dir": str(output_dir),
        **{k: v for k, v in action_results.items() if k not in ("action_disagreement_scores", "action_ood_indices")},
        **{
            k: v
            for k, v in anchor_results.items()
            if k not in ("anchor_selected_indices", "anchor_indices_count", "anchor_retrieval_dict", "buffer_lst")
        },
    }
    if save_artifacts:
        (output_dir / "metadata_anchor_memories.json").write_text(json.dumps(metadata, indent=2, default=str))

    return AnchorMemoryResult(
        buffer_lst=anchor_results["buffer_lst"],
        anchor_selected_global_indices=anchor_results["anchor_selected_indices"],
        action_ood_indices=action_results["action_ood_indices"],
        reference_task_ids=reference_task_ids,
        metadata=metadata,
        output_dir=output_dir,
    )


def anchor_config_from_train_config(config: _config.TrainConfig) -> AnchorMemoryConfig:
    return AnchorMemoryConfig(
        repr_mode=ReprMode(config.anchor_repr_mode),
        action_ood_mode=ActionOodMode(config.anchor_action_ood_mode),
        ood_top_percent=config.anchor_ood_top_percent,
        std_cutoff=config.anchor_std_cutoff,
        flow_noise_level=config.anchor_flow_noise_level,
        num_flow_steps=config.anchor_num_flow_steps,
        rot_metric=RotMetric(config.anchor_rot_metric),
        rot_weight=config.anchor_rot_weight,
        n_memories_per_task=config.n_memories,
        n_memories_total=config.n_memories_total,
        n_memories_proportion=config.n_memories_proportion,
        n_neighbors=config.anchor_n_neighbors,
        consolidation=Consolidation(config.anchor_consolidation),
        random_proportion=config.anchor_random_proportion,
        invert_selection=config.anchor_invert_selection,
        invert_remove_proportion=config.anchor_invert_remove_proportion,
        baseline_max_samples_per_task=config.anchor_baseline_max_samples_per_task,
        max_embedding_samples_per_task=config.anchor_max_embedding_samples_per_task,
        chunk_size=config.anchor_chunk_size,
        seed=config.seed,
    )
