"""Latent-space KNN anchor selection (SelectiveER nearest_neighbor_ood port)."""

from __future__ import annotations

import json
import logging
import math
import pathlib
from dataclasses import dataclass

import cv2
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors

import openpi.policies.libero_policy as libero_policy
from action_disagreement_utils import Consolidation


@dataclass
class ReferenceFrameIndex:
    task_id: np.ndarray
    dataset_index: np.ndarray
    frame_index: np.ndarray
    task_slugs: dict[int, str]
    raw_datasets: dict[int, lerobot_dataset.LeRobotDataset]

    def load_image(self, global_index: int) -> np.ndarray:
        task_id = int(self.task_id[global_index])
        dataset_idx = int(self.dataset_index[global_index])
        image = self.raw_datasets[task_id].hf_dataset[dataset_idx]["image"]
        return libero_policy._parse_image(np.asarray(image))


def score_reference_distances_to_ood(
    pretrain_reprs: np.ndarray,
    target_ood_reprs: np.ndarray,
    *,
    n_neighbors: int = 5,
    consolidation: Consolidation = Consolidation.MEDIAN,
) -> np.ndarray:
    """Consolidated distance from each reference frame to the action-OOD target set."""
    assert target_ood_reprs.shape[0] > 0
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(target_ood_reprs)
    distances, _ = nbrs.kneighbors(pretrain_reprs)
    if consolidation == Consolidation.MEDIAN:
        return np.median(distances, axis=1)
    return np.mean(distances, axis=1)


def _global_indices_to_buffer_lst(
    chosen_global: np.ndarray,
    ref_task_id_per_frame: np.ndarray,
    ref_dataset_index_per_frame: np.ndarray,
    reference_task_ids: list[int],
) -> list[list[int]]:
    """Map globally selected reference frames to per-task local index lists for ER replay."""
    buffer_lst: list[list[int]] = [[] for _ in reference_task_ids]
    task_to_slot = {task_id: i for i, task_id in enumerate(reference_task_ids)}
    for global_idx in chosen_global:
        task_id = int(ref_task_id_per_frame[global_idx])
        slot = task_to_slot[task_id]
        buffer_lst[slot].append(int(ref_dataset_index_per_frame[global_idx]))
    return buffer_lst


def finalize_global_anchor_buffers(
    consolidated_distances: np.ndarray,
    ref_task_id_per_frame: np.ndarray,
    ref_dataset_index_per_frame: np.ndarray,
    reference_task_ids: list[int],
    *,
    memory_budget: int,
    random_proportion: float,
    invert_selection: bool,
    invert_remove_proportion: float,
    seed: int,
) -> tuple[list[list[int]], np.ndarray]:
    """Build ER buffers by selecting from the pooled past-task reference set (no per-task quota).

    ``memory_budget`` is the total number of frames to select from the pooled reference set.
    KNN / random / invert selection all operate on the full reference pool; results are split
    into per-task local indices.
    """
    rng = np.random.default_rng(seed)
    n_pool = int(consolidated_distances.shape[0])
    k_select = min(memory_budget, n_pool)

    if k_select == 0:
        return [[] for _ in reference_task_ids], np.array([], dtype=np.int64)

    if invert_selection:
        n_exclude = max(1, int(invert_remove_proportion * n_pool))
        n_exclude = min(n_exclude, n_pool - k_select) if n_pool > k_select else min(n_exclude, n_pool)
        exclude_order = np.argsort(consolidated_distances)[:n_exclude]
        pool_mask = np.ones(n_pool, dtype=bool)
        pool_mask[exclude_order] = False
        pool_global = np.where(pool_mask)[0]
        if pool_global.shape[0] < k_select:
            raise ValueError(
                f"invert_selection leaves {pool_global.shape[0]} frames in the pooled reference set, "
                f"need {k_select}"
            )
        chosen_global = rng.choice(pool_global, size=k_select, replace=False)
    else:
        n_knn = int((1.0 - random_proportion) * k_select)
        order = np.argsort(consolidated_distances)
        knn_global = order[:n_knn]
        chosen_set = set(int(i) for i in knn_global)
        n_random = k_select - n_knn
        if n_random > 0:
            pool_global = np.array([i for i in range(n_pool) if i not in chosen_set], dtype=np.int64)
            if pool_global.shape[0] < n_random:
                raise ValueError(
                    f"Not enough unselected frames in pooled reference set for random fill "
                    f"(need {n_random}, have {pool_global.shape[0]})"
                )
            extra = rng.choice(pool_global, size=n_random, replace=False)
            chosen_global = np.concatenate([knn_global, extra])
        else:
            chosen_global = knn_global

    chosen_global = np.asarray(chosen_global, dtype=np.int64)
    buffer_lst = _global_indices_to_buffer_lst(
        chosen_global, ref_task_id_per_frame, ref_dataset_index_per_frame, reference_task_ids
    )
    return buffer_lst, chosen_global


def find_nearest_neighbors(
    pretrain_reprs: np.ndarray,
    target_ood_reprs: np.ndarray,
    num_nearest_neighbors: int,
    *,
    n_neighbors: int = 5,
    consolidation: Consolidation = Consolidation.MEDIAN,
) -> tuple[np.ndarray, dict[int, int], dict[int, list[int]], np.ndarray]:
    """Select reference indices closest to action-OOD target representations.

    Mirrors SelectiveER.find_nearest_neighbors: fit on target OOD set, query pretrain set.
    Returns (k_nearest_indices, indices_count, counterexample_dict, consolidated_distances).
    """
    assert target_ood_reprs.shape[0] > 0, "find_nearest_neighbors requires at least one OOD target point."
    assert 0 < num_nearest_neighbors <= pretrain_reprs.shape[0]

    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(target_ood_reprs)
    distances, indices_in_target_list = nbrs.kneighbors(pretrain_reprs)

    if consolidation == Consolidation.MEDIAN:
        consolidated = np.median(distances, axis=1)
    else:
        consolidated = np.mean(distances, axis=1)

    k_nearest_indices = np.argpartition(consolidated, num_nearest_neighbors)[:num_nearest_neighbors]
    k_nearest_indices = k_nearest_indices[np.argsort(consolidated[k_nearest_indices])]

    indices_in_target_list = indices_in_target_list[k_nearest_indices]
    indices_count: dict[int, int] = {}
    for row in indices_in_target_list:
        for idx in row:
            indices_count[int(idx)] = indices_count.get(int(idx), 0) + 1

    counterexample_dict: dict[int, list[int]] = {}
    for old_set_idx, new_set_idxes in zip(k_nearest_indices, indices_in_target_list, strict=True):
        for new_set_idx in new_set_idxes:
            key = int(new_set_idx)
            if key not in counterexample_dict:
                counterexample_dict[key] = []
            counterexample_dict[key].append(int(old_set_idx))

    return k_nearest_indices, indices_count, counterexample_dict, consolidated


def log_knn_results(
    counterexample_dict: dict[int, list[int]],
    indices_count: dict[int, int],
    vis_dir: pathlib.Path,
    task_slug: str,
) -> None:
    counterexample_counts = [len(v) for v in counterexample_dict.values()]
    plt.figure(figsize=(8, 5))
    plt.hist(counterexample_counts, bins=30, alpha=0.7, color="green")
    plt.xlabel("Number of counterexamples per anchor")
    plt.ylabel("Count")
    plt.title("Histogram of Counterexample Counts")
    plt.tight_layout()
    plt.savefig(vis_dir / f"{task_slug}_hist_counterexample_counts.png")
    plt.close()

    values = np.array(list(indices_count.values()), dtype=float)
    values = np.sort(values)
    cum_vals = np.cumsum(values)
    cum_vals = np.insert(cum_vals, 0, 0)
    cum_vals = cum_vals / cum_vals[-1]
    pop = np.arange(0, len(values) + 1) / len(values)

    plt.figure(figsize=(6, 6))
    plt.plot(pop, cum_vals, label="Lorenz curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Equality line")
    plt.xlabel("Fraction of items")
    plt.ylabel("Fraction of total value")
    plt.title("Lorenz Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / f"{task_slug}_lorentzcurveknn.png")
    plt.close()


def log_selection_results(
    selected_indices: np.ndarray,
    consolidated_distances: np.ndarray,
    ref_index: ReferenceFrameIndex,
    vis_dir: pathlib.Path,
    task_slug: str,
    *,
    images_needed: int = 100,
) -> None:
    task_distribution: dict[str, list[int]] = {}
    for global_idx in selected_indices:
        task_id = int(ref_index.task_id[global_idx])
        name = ref_index.task_slugs[task_id]
        step = int(ref_index.frame_index[global_idx])
        task_distribution.setdefault(name, []).append(step)

    n_tasks = len(task_distribution)
    n_cols = min(3, n_tasks)
    n_rows = math.ceil(n_tasks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_tasks == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])

    axes_flat = axes.flatten()
    for idx, (name, steps) in enumerate(task_distribution.items()):
        ax = axes_flat[idx]
        ax.hist(steps, bins=20, alpha=0.7)
        ax.set_title(f"Histogram for {name}")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Count")

    for i in range(len(task_distribution), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout()
    plt.savefig(vis_dir / f"{task_slug}_selected_data_hist.png")
    plt.close()

    order = np.argsort(consolidated_distances[selected_indices])
    ranked = selected_indices[order]

    elements: list[np.ndarray] = []
    placeholder_shape: tuple[int, int, int] | None = None
    for i in range(images_needed):
        if i < ranked.shape[0]:
            global_idx = int(ranked[i])
            img = ref_index.load_image(global_idx)
            img = np.ascontiguousarray(img)
            if placeholder_shape is None:
                placeholder_shape = img.shape
            task_id = int(ref_index.task_id[global_idx])
            step = int(ref_index.frame_index[global_idx])
            dist = float(consolidated_distances[global_idx])
            slug = ref_index.task_slugs[task_id]
            cv2.putText(img, slug[:30], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            cv2.putText(img, f"step {step}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(img, f"d={dist:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        else:
            if placeholder_shape is None:
                placeholder_shape = (256, 256, 3)
            img = np.zeros(placeholder_shape, dtype=np.uint8)
        elements.append(img)

    stacked = np.stack(elements)
    grid_rows = [np.hstack([stacked[row * 10 + col] for col in range(10)]) for row in range(10)]
    out_path = vis_dir / f"{task_slug}_top100_selected.png"
    plt.imsave(out_path, np.vstack(grid_rows))
    logging.info("Saved selected anchor image tile to %s", out_path)


def run_anchor_selection_stage(
    reference_reprs: np.ndarray,
    target_reprs: np.ndarray,
    action_ood_indices: np.ndarray,
    ref_index: ReferenceFrameIndex,
    output_path: pathlib.Path,
    vis_dir: pathlib.Path,
    task_slug: str,
    *,
    memory_budget: int,
    reference_task_ids: list[int],
    n_neighbors: int,
    consolidation: Consolidation,
    random_proportion: float = 0.6,
    invert_selection: bool = False,
    invert_remove_proportion: float = 0.05,
    seed: int = 42,
) -> dict:
    if action_ood_indices.shape[0] == 0:
        logging.warning("No action-OOD frames; skipping anchor KNN selection.")
        return {
            "anchor_selected_indices": np.array([], dtype=np.int64),
            "buffer_lst": [[] for _ in reference_task_ids],
            "num_selected_anchors": 0,
        }

    target_ood_reprs = target_reprs[action_ood_indices]
    consolidated = score_reference_distances_to_ood(
        reference_reprs,
        target_ood_reprs,
        n_neighbors=n_neighbors,
        consolidation=consolidation,
    )
    buffer_lst, selected_indices = finalize_global_anchor_buffers(
        consolidated,
        ref_index.task_id,
        ref_index.dataset_index,
        reference_task_ids,
        memory_budget=memory_budget,
        random_proportion=random_proportion,
        invert_selection=invert_selection,
        invert_remove_proportion=invert_remove_proportion,
        seed=seed,
    )

    _, indices_count, counterexample_dict = _counterexample_stats(
        reference_reprs,
        target_ood_reprs,
        selected_indices,
        n_neighbors=n_neighbors,
        consolidation=consolidation,
    )

    logging.info(
        "Anchor global selection: %d / %d reference frames across %d tasks (OOD targets=%d)",
        int(selected_indices.shape[0]),
        memory_budget,
        len(reference_task_ids),
        int(action_ood_indices.shape[0]),
    )

    np.save(output_path / "anchor_selected_indices.npy", selected_indices)
    np.save(output_path / "anchor_ood_target_indices.npy", action_ood_indices)
    (output_path / "anchor_indices_count.json").write_text(json.dumps(indices_count, indent=2))
    retrieval_serializable = {str(k): v for k, v in counterexample_dict.items()}
    (output_path / "anchor_retrieval_dict.json").write_text(json.dumps(retrieval_serializable, indent=2))

    log_knn_results(counterexample_dict, indices_count, vis_dir, task_slug)
    log_selection_results(selected_indices, consolidated, ref_index, vis_dir, task_slug)

    return {
        "anchor_selected_indices": selected_indices,
        "buffer_lst": buffer_lst,
        "anchor_indices_count": indices_count,
        "anchor_retrieval_dict": counterexample_dict,
        "num_selected_anchors": int(selected_indices.shape[0]),
        "memory_budget": memory_budget,
        "n_neighbors": n_neighbors,
        "consolidation": consolidation.value,
        "random_proportion": random_proportion,
        "invert_selection": invert_selection,
    }


def _counterexample_stats(
    pretrain_reprs: np.ndarray,
    target_ood_reprs: np.ndarray,
    selected_global: np.ndarray,
    *,
    n_neighbors: int,
    consolidation: Consolidation,
) -> tuple[np.ndarray, dict[int, int], dict[int, list[int]]]:
    if selected_global.shape[0] == 0:
        return selected_global, {}, {}

    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(target_ood_reprs)
    distances, indices_in_target_list = nbrs.kneighbors(pretrain_reprs[selected_global])

    indices_count: dict[int, int] = {}
    for row in indices_in_target_list:
        for idx in row:
            indices_count[int(idx)] = indices_count.get(int(idx), 0) + 1

    counterexample_dict: dict[int, list[int]] = {}
    for local_i, new_set_idxes in enumerate(indices_in_target_list):
        old_set_idx = int(selected_global[local_i])
        for new_set_idx in new_set_idxes:
            key = int(new_set_idx)
            counterexample_dict.setdefault(key, []).append(old_set_idx)

    return selected_global, indices_count, counterexample_dict
