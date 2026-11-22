"""Flow-matching action disagreement scoring for Libero anchor memory selection."""

from __future__ import annotations

import enum
import logging
import pathlib
import types

import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform
import tqdm

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models.pi0 import make_attn_mask

LIBERO_ACTION_DIM = 7


class RotMetric(enum.Enum):
    GEODESIC = "geodesic"
    L2 = "l2"


class ActionOodMode(enum.Enum):
    BASELINE = "baseline"
    TOP_PERCENT = "top_percent"


class Consolidation(enum.Enum):
    MEDIAN = "median"
    MEAN = "mean"


def extra_delta_transform_enabled(config: _config.TrainConfig) -> bool:
    data = config.data
    return bool(getattr(data, "extra_delta_transform", False))


def to_libero_physical_actions(
    actions: np.ndarray,
    state: np.ndarray,
    data_config: _config.DataConfig,
    *,
    extra_delta_transform: bool,
) -> np.ndarray:
    """Map normalized padded model actions to physical Libero 7D OSC deltas."""
    unnorm = _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    data = unnorm({"actions": np.asarray(actions), "state": np.asarray(state)})
    if extra_delta_transform:
        mask = _transforms.make_bool_mask(6, -1)
        data = _transforms.AbsoluteActions(mask)(data)
    return np.asarray(data["actions"][..., :LIBERO_ACTION_DIM])


def rotation_delta_geodesic(aa_gt: np.ndarray, aa_recon: np.ndarray) -> np.ndarray:
    """Geodesic SO(3) angle (radians) between axis-angle delta rotations."""
    if aa_gt.shape != aa_recon.shape:
        raise ValueError(f"Shape mismatch: {aa_gt.shape} vs {aa_recon.shape}")
    flat_gt = aa_gt.reshape(-1, 3)
    flat_recon = aa_recon.reshape(-1, 3)
    r_gt = scipy.spatial.transform.Rotation.from_rotvec(flat_gt)
    r_recon = scipy.spatial.transform.Rotation.from_rotvec(flat_recon)
    r_rel = r_gt.inv() * r_recon
    angles = np.linalg.norm(r_rel.as_rotvec(), axis=-1)
    return angles.reshape(aa_gt.shape[:-1])


def compute_disagreement_physical(
    gt_physical: np.ndarray,
    recon_physical: np.ndarray,
    *,
    rot_metric: RotMetric,
    rot_weight: float,
) -> np.ndarray:
    """Per-batch-element disagreement in physical Libero action space. Shape: (batch,)."""
    d_trans = np.linalg.norm(gt_physical[..., :3] - recon_physical[..., :3], axis=-1).mean(axis=-1)
    if rot_metric == RotMetric.GEODESIC:
        d_rot = rotation_delta_geodesic(gt_physical[..., 3:6], recon_physical[..., 3:6]).mean(axis=-1)
    else:
        d_rot = np.linalg.norm(gt_physical[..., 3:6] - recon_physical[..., 3:6], axis=-1).mean(axis=-1)
    return d_trans + rot_weight * d_rot


def _flow_denoise_from_ground_truth(
    self: _model.BaseModel,
    rng: jax.Array,
    observation: _model.Observation,
    gt_actions: jax.Array,
    flow_t: jax.Array,
    num_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Corrupt gt actions at flow_t, integrate to t=0. Returns (recon, noise, x_noisy)."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    noise = jax.random.normal(rng, gt_actions.shape)
    x_noisy = flow_t * noise + (1.0 - flow_t) * gt_actions

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_b = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_b, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (x_noisy, flow_t))
    return x_0, noise, x_noisy


def make_flow_denoise_fn(model: _model.BaseModel):
    bound = types.MethodType(_flow_denoise_from_ground_truth, model)
    return nnx_utils.module_jit(bound)


def score_action_disagreement_batch(
    flow_denoise_fn,
    rng: jax.Array,
    observation: _model.Observation,
    gt_actions: jax.Array,
    data_config: _config.DataConfig,
    *,
    extra_delta_transform: bool,
    flow_t: float,
    num_flow_steps: int,
    rot_metric: RotMetric,
    rot_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    flow_t_arr = jnp.asarray(flow_t, dtype=jnp.float32)
    recon, _noise, _x_noisy = flow_denoise_fn(rng, observation, gt_actions, flow_t_arr, num_flow_steps)

    recon_np = np.asarray(jax.device_get(recon))
    gt_np = np.asarray(jax.device_get(gt_actions))
    state_np = np.asarray(jax.device_get(observation.state))

    gt_phys = to_libero_physical_actions(gt_np, state_np, data_config, extra_delta_transform=extra_delta_transform)
    recon_phys = to_libero_physical_actions(recon_np, state_np, data_config, extra_delta_transform=extra_delta_transform)

    scores = compute_disagreement_physical(gt_phys, recon_phys, rot_metric=rot_metric, rot_weight=rot_weight)
    return scores, {"gt_physical": gt_phys, "recon_physical": recon_phys}


def score_loader_action_disagreement(
    flow_denoise_fn,
    loader,
    data_config: _config.DataConfig,
    *,
    extra_delta_transform: bool,
    flow_t: float,
    num_flow_steps: int,
    rot_metric: RotMetric,
    rot_weight: float,
    seed: int,
    desc: str,
    return_viz: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = jax.random.key(seed)
    scores_list: list[np.ndarray] = []
    viz_lists: dict[str, list[np.ndarray]] | None = None
    if return_viz:
        viz_lists = {"gt_physical": [], "recon_physical": []}

    for observation, actions in tqdm.tqdm(loader, desc=desc):
        rng, step_rng = jax.random.split(rng)
        batch_scores, viz = score_action_disagreement_batch(
            flow_denoise_fn,
            step_rng,
            observation,
            actions,
            data_config,
            extra_delta_transform=extra_delta_transform,
            flow_t=flow_t,
            num_flow_steps=num_flow_steps,
            rot_metric=rot_metric,
            rot_weight=rot_weight,
        )
        scores_list.append(batch_scores)
        if viz_lists is not None:
            for key in viz_lists:
                viz_lists[key].append(viz[key])

    stacked_viz: dict[str, np.ndarray] = {}
    if viz_lists is not None:
        stacked_viz = {key: np.concatenate(chunks, axis=0) for key, chunks in viz_lists.items()}
    return np.concatenate(scores_list, axis=0), stacked_viz


def compute_baseline_stats(scores: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


def select_action_ood_indices(
    id_indices: np.ndarray,
    scores: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    std_cutoff: float,
) -> np.ndarray:
    assert scores.shape[0] == id_indices.shape[0]
    threshold = baseline_mean + std_cutoff * baseline_std
    mask = scores > threshold
    return id_indices[mask]


def select_action_ood_top_percent(
    id_indices: np.ndarray,
    scores: np.ndarray,
    top_percent: float,
) -> np.ndarray:
    assert scores.shape[0] == id_indices.shape[0]
    if id_indices.shape[0] == 0:
        return id_indices
    n_select = max(1, int(np.ceil(id_indices.shape[0] * top_percent / 100.0)))
    order = np.argsort(-scores)
    return id_indices[order[:n_select]]


def top_percent_score_threshold(scores: np.ndarray, top_percent: float) -> float:
    n_select = max(1, int(np.ceil(scores.shape[0] * top_percent / 100.0)))
    sorted_scores = np.sort(scores)[::-1]
    return float(sorted_scores[min(n_select - 1, sorted_scores.shape[0] - 1)])


def plot_action_ood_step_histogram(
    frame_indices: np.ndarray,
    action_ood_indices: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    steps = frame_indices[action_ood_indices]
    plt.figure(figsize=(8, 5))
    plt.hist(steps, bins=30, alpha=0.7, color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Count")
    plt.title("Histogram OOD point selection (action disagreement)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_action_disagreement_scores_histogram(
    id_scores: np.ndarray,
    output_path: pathlib.Path,
    *,
    action_ood_mode: ActionOodMode,
    baseline_mean: float | None = None,
    baseline_std: float | None = None,
    std_cutoff: float | None = None,
    top_percent: float | None = None,
) -> None:
    if action_ood_mode == ActionOodMode.BASELINE:
        assert baseline_mean is not None and baseline_std is not None and std_cutoff is not None
        threshold = baseline_mean + std_cutoff * baseline_std
        below = id_scores[id_scores <= threshold]
        above = id_scores[id_scores > threshold]
        cutoff_label = f"threshold = mean + {std_cutoff}·std"
    else:
        assert top_percent is not None
        threshold = top_percent_score_threshold(id_scores, top_percent)
        above = id_scores[id_scores >= threshold]
        below = id_scores[id_scores < threshold]
        cutoff_label = f"top {top_percent}% cutoff"

    plt.figure(figsize=(10, 5))
    if below.shape[0] > 0:
        plt.hist(below, bins=30, alpha=0.6, color="steelblue", label="ID (below cutoff)")
    if above.shape[0] > 0:
        plt.hist(above, bins=30, alpha=0.7, color="orange", label="Action-OOD (above cutoff)")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=cutoff_label)
    plt.xlabel("Action disagreement score")
    plt.ylabel("Count")
    plt.title("Action disagreement scores on state-ID subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _cumulative_xyz_path(actions_physical: np.ndarray) -> np.ndarray:
    return np.cumsum(actions_physical[:, :3], axis=0)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    import io

    import imageio

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    frame = imageio.imread(buf)
    return frame[:, :, :3]


def save_action_reconstruction_mp4(
    gt_physical: np.ndarray,
    recon_physical: np.ndarray,
    output_path: pathlib.Path,
    *,
    score: float,
    dataset_idx: int,
    label: str,
    fps: int = 24,
    n_rotation_frames: int = 72,
) -> None:
    """Rotating 3D MP4 of integrated gt vs denoised trajectories."""
    import imageio

    gt_path = _cumulative_xyz_path(gt_physical)
    recon_path = _cumulative_xyz_path(recon_physical)
    points = np.concatenate([gt_path, recon_path], axis=0)
    mins = points.min(axis=0) - 0.05
    maxs = points.max(axis=0) + 0.05

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    frames: list[np.ndarray] = []
    for angle in np.linspace(0, 360, n_rotation_frames):
        ax.cla()
        ax.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], color="black", linewidth=2, label="gt")
        ax.scatter(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], color="black", s=30)
        ax.plot(recon_path[:, 0], recon_path[:, 1], recon_path[:, 2], color="red", linewidth=2, label="denoised")
        ax.scatter(recon_path[:, 0], recon_path[:, 1], recon_path[:, 2], color="red", s=30)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Action reconstruction ({label}, score={score:.4f}, idx={dataset_idx})")
        ax.legend()
        ax.view_init(elev=30, azim=angle)
        frames.append(_fig_to_rgb(fig))

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps)
    logging.info("Saved action reconstruction video to %s", output_path)


def export_min_max_action_reconstruction_videos(
    id_scores: np.ndarray,
    id_indices: np.ndarray,
    viz: dict[str, np.ndarray],
    vis_dir: pathlib.Path,
    task_slug: str,
) -> None:
    """Export rotating 3D reconstructions for smallest and largest ID disagreement scores."""
    if id_scores.shape[0] == 0:
        return

    selections: list[tuple[int, str]] = [
        (int(np.argmin(id_scores)), "min_score"),
        (int(np.argmax(id_scores)), "max_score"),
    ]
    seen: set[int] = set()
    for pos, tag in selections:
        if pos in seen:
            continue
        seen.add(pos)
        save_action_reconstruction_mp4(
            viz["gt_physical"][pos],
            viz["recon_physical"][pos],
            vis_dir / f"{task_slug}_action_recon_{tag}.mp4",
            score=float(id_scores[pos]),
            dataset_idx=int(id_indices[pos]),
            label=tag,
        )


def save_action_ood_image_tile(
    action_ood_indices: np.ndarray,
    scores: np.ndarray,
    id_indices: np.ndarray,
    all_scores: np.ndarray,
    raw_dataset,
    frame_indices: np.ndarray,
    output_path: pathlib.Path,
    load_image_fn,
    *,
    images_needed: int = 100,
) -> None:
    import cv2

    score_by_dataset_idx = {int(id_indices[i]): float(all_scores[i]) for i in range(len(id_indices))}
    sorted_ood = action_ood_indices[np.argsort(-np.array([score_by_dataset_idx[int(i)] for i in action_ood_indices]))]

    elements: list[np.ndarray] = []
    placeholder_shape: tuple[int, int, int] | None = None
    for i in range(images_needed):
        if i < sorted_ood.shape[0]:
            dataset_idx = int(sorted_ood[i])
            img = load_image_fn(raw_dataset, dataset_idx)
            img = np.ascontiguousarray(img)
            if placeholder_shape is None:
                placeholder_shape = img.shape
            step = int(frame_indices[dataset_idx])
            score = score_by_dataset_idx[dataset_idx]
            cv2.putText(img, f"{score:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(img, f"step {step}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        else:
            if placeholder_shape is None:
                placeholder_shape = (256, 256, 3)
            img = np.zeros(placeholder_shape, dtype=np.uint8)
        elements.append(img)

    stacked = np.stack(elements)
    grid_rows = [np.hstack([stacked[row * 10 + col] for col in range(10)]) for row in range(10)]
    plt.imsave(output_path, np.vstack(grid_rows))
    logging.info("Saved action-OOD image tile to %s", output_path)
