"""Plot continual-learning success matrix from sequential eval YAML results.

Loads per-checkpoint eval stats written by ``examples/libero/eval_sequential.py``
under ``{ckpt_dir}/eval/{step}/{task_suite}/{task_suite}.yaml`` and produces a
lower-triangular heatmap: x-axis = training progress (stage), y-axis = evaluated
task in *learning order*.

If ``{ckpt_dir}/task_order.json`` exists, matrix rows/cols are training stages
(0..N-1) mapped to canonical LIBERO ids via that order. Otherwise the canonical
identity order is assumed.

Usage:
  uv run scripts/plot_sequential_success_matrix.py \\
    --ckpt-dir checkpoints/pi05_libero-libero_goal_sequential/LiberoTrainPi05_1000mem \\
    --task-suite libero_goal
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tyro
import yaml

import openpi.libero_eval_yaml as _eval_yaml


@dataclasses.dataclass
class Args:
    """Arguments for plotting sequential eval success matrices."""

    ckpt_dir: str
    task_suite: str = "libero_goal"
    output_dir: str | None = None
    # Write success matrix as .npy alongside the figure.
    save_npy: bool = True
    # Write FWT/BWT/AUC metrics JSON alongside the figure.
    save_metrics: bool = True
    show_colorbar: bool = False
    # 1-indexed axis labels (matches paper figures).
    one_indexed_labels: bool = True
    annot_fmt: str = ".2f"


def _load_eval_yaml(yaml_path: pathlib.Path) -> dict:
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "tasks" in data
    return data


def _resolve_task_order(experiment_dir: pathlib.Path, n_hint: int | None = None) -> list[int]:
    order = _eval_yaml.load_task_order(experiment_dir)
    if order is not None:
        return order
    assert n_hint is not None and n_hint > 0, "Cannot infer identity task order without n_hint"
    return list(range(n_hint))


def _trained_stage_for_yaml(
    yaml_path: pathlib.Path,
    task_order: list[int],
    data: dict,
) -> int:
    """Return training stage index for this checkpoint (max stage evaluated)."""
    done_path = yaml_path.parent / "done.txt"
    if done_path.is_file():
        fields = _eval_yaml.parse_done_sentinel(done_path)
        return int(fields["trained_task_idx"])

    # Fallback: stages of tasks that appear in the curriculum prefix.
    canonical_to_stage = {canonical: stage for stage, canonical in enumerate(task_order)}
    stages = [
        canonical_to_stage[int(entry["task_id"])]
        for entry in data["tasks"]
        if int(entry["task_id"]) in canonical_to_stage
    ]
    assert stages, f"No curriculum tasks found in {yaml_path}"
    return max(stages)


def discover_eval_results(experiment_dir: pathlib.Path, task_suite: str) -> list[tuple[int, pathlib.Path]]:
    """Return sorted (step, yaml_path) for each eval checkpoint."""
    eval_root = experiment_dir / "eval"
    assert eval_root.is_dir(), f"No eval/ directory under {experiment_dir}"

    results: list[tuple[int, pathlib.Path]] = []
    for step_dir in sorted(eval_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not step_dir.name.isdigit():
            continue
        step = int(step_dir.name)
        yaml_path = step_dir / task_suite / f"{task_suite}.yaml"
        if not yaml_path.is_file():
            continue
        results.append((step, yaml_path))

    assert results, f"No eval YAML files found under {eval_root} for task_suite={task_suite!r}"
    return results


def build_success_matrix(
    eval_results: list[tuple[int, pathlib.Path]],
    task_order: list[int],
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Build S[stage_i, stage_j] = success rate on curriculum task j after training
    through stage i.

    Cells with stage_j > stage_i remain NaN.
    ``task_order[stage]`` is the canonical LIBERO id learned at that stage.
    """
    canonical_to_stage = {canonical: stage for stage, canonical in enumerate(task_order)}

    stage_to_yaml: dict[int, pathlib.Path] = {}
    stage_to_step: dict[int, int] = {}
    stage_to_data: dict[int, dict] = {}
    for step, yaml_path in eval_results:
        data = _load_eval_yaml(yaml_path)
        trained_stage = _trained_stage_for_yaml(yaml_path, task_order, data)
        assert trained_stage < len(task_order), (
            f"{yaml_path}: trained_stage={trained_stage} out of range for task_order={task_order}"
        )
        if trained_stage in stage_to_yaml:
            prev_step = stage_to_step[trained_stage]
            assert step >= prev_step, (
                f"Duplicate trained_stage={trained_stage} at steps {prev_step} and {step}"
            )
        stage_to_yaml[trained_stage] = yaml_path
        stage_to_step[trained_stage] = step
        stage_to_data[trained_stage] = data

    stages = sorted(stage_to_yaml)
    assert stages == list(range(stages[-1] + 1)), (
        f"Expected contiguous training stages 0..{stages[-1]}, got {stages}"
    )
    n = len(stages)

    task_labels = [f"task_{task_order[i]}" for i in range(n)]
    final_tasks = stage_to_data[stages[-1]]["tasks"]
    for task_entry in final_tasks:
        canonical_id = int(task_entry["task_id"])
        if canonical_id not in canonical_to_stage:
            continue
        stage_j = canonical_to_stage[canonical_id]
        if stage_j < n:
            task_labels[stage_j] = str(task_entry.get("task", f"task_{canonical_id}"))

    matrix = np.full((n, n), np.nan, dtype=float)

    for stage_i in stages:
        expected = set(task_order[: stage_i + 1])
        tasks = stage_to_data[stage_i]["tasks"]
        for task_entry in tasks:
            canonical_id = int(task_entry["task_id"])
            if canonical_id not in expected:
                # Stale entry from a prior incorrect eval run; ignore.
                continue
            stage_j = canonical_to_stage[canonical_id]
            assert stage_j <= stage_i, (
                f"Step {stage_to_step[stage_i]} (stage {stage_i}) yaml contains "
                f"canonical task_id={canonical_id} (stage {stage_j}) which should "
                f"not be evaluated yet"
            )
            rate = float(task_entry["success_rate"])
            assert 0.0 <= rate <= 1.0
            matrix[stage_i, stage_j] = rate

        missing = [
            task_order[j]
            for j in range(stage_i + 1)
            if np.isnan(matrix[stage_i, j])
        ]
        assert not missing, (
            f"Step {stage_to_step[stage_i]} (stage {stage_i}) missing success rates "
            f"for canonical tasks {missing}"
        )

    return matrix, task_labels, [stage_to_step[s] for s in stages]


def _step_index_at_or_before(steps: list[int], boundary: int) -> int:
    """Largest index ``k`` with ``steps[k] <= boundary`` (falls back to 0)."""
    if not steps:
        return 0
    pos = bisect.bisect_right(steps, boundary) - 1
    return max(0, pos)


def build_task_trajectories(
    eval_results: list[tuple[int, pathlib.Path]],
    task_labels: list[str],
    task_order: list[int],
) -> tuple[list[int], dict[str, list[float]]]:
    """Per-task success rate at every eval checkpoint (NaN before a task is evaluated)."""
    sorted_results = sorted(eval_results, key=lambda item: item[0])
    steps = [step for step, _ in sorted_results]
    task_to_series: dict[str, list[float]] = {label: [] for label in task_labels}
    canonical_to_stage = {canonical: stage for stage, canonical in enumerate(task_order)}

    for _step, yaml_path in sorted_results:
        data = _load_eval_yaml(yaml_path)
        trained_stage = _trained_stage_for_yaml(yaml_path, task_order, data)
        expected = set(task_order[: trained_stage + 1])
        rates_by_stage: dict[int, float] = {}
        for entry in data["tasks"]:
            canonical_id = int(entry["task_id"])
            if canonical_id not in expected:
                continue
            rates_by_stage[canonical_to_stage[canonical_id]] = float(entry["success_rate"])
        for stage_j, label in enumerate(task_labels):
            rate = rates_by_stage.get(stage_j)
            task_to_series[label].append(rate if rate is not None else float("nan"))

    return steps, task_to_series


def compute_continual_metrics(
    conf_mat: np.ndarray,
    steps: list[int],
    task_to_series: dict[str, list[float]],
    task_ends: list[tuple[int, str]],
) -> dict[str, Any]:
    """
    Continual-learning metrics from a stage-end success matrix.

    Matches ``umi_day.plotting_codes.ckpt_rollout_viz.compute_metrics`` for the
    non-``_precise`` fields (matrix-sampled FWT/BWT/AUC and moving-anchor BWT).
    """
    n_tasks = conf_mat.shape[0]

    fwt_per_task = [float(conf_mat[k, k]) for k in range(n_tasks)]
    fwt = float(np.mean(fwt_per_task))

    bwt_per_task: list[float] = []
    bwt_task_to_task_deltas: list[list[float]] = []
    for k in range(n_tasks):
        if k == n_tasks - 1:
            bwt_task_to_task_deltas.append([])
            continue
        val_k = conf_mat[k, k]
        vals_future = conf_mat[k + 1 :, k]
        deltas = val_k - vals_future
        bwt_task_to_task_deltas.append(deltas.astype(float).tolist())
        bwt_per_task.append(float(np.mean(deltas)))
    bwt = float(np.mean(bwt_per_task))

    auc_per_task: list[float] = []
    for k in range(n_tasks):
        col_vals = conf_mat[k:, k]
        auc_per_task.append(float(np.mean(col_vals)))
    auc = float(np.mean(auc_per_task))

    resolved_task_ends: list[tuple[int, str]] = []
    end_indices: list[int] = []
    for end_step, task_name in task_ends:
        idx = _step_index_at_or_before(steps, int(end_step))
        assert steps[idx] == int(end_step), (
            f"No eval checkpoint at task boundary for {task_name!r}: "
            f"end_step={int(end_step)}, nearest preceding checkpoint at "
            f"step={steps[idx]} (idx={idx})"
        )
        resolved_task_ends.append((int(end_step), task_name))
        end_indices.append(idx)

    n_resolved = len(resolved_task_ends)
    curriculum_resolved = [name for _, name in resolved_task_ends]
    start_indices = [0] + [end_indices[i - 1] + 1 for i in range(1, n_resolved)]

    bwt_task_to_task_moving_anchors: list[list[float]] = []
    bwt_moving_anchors_per_task: list[float] = []
    for source_idx in range(n_resolved):
        if source_idx == n_resolved - 1:
            bwt_task_to_task_moving_anchors.append([])
            continue
        source_task = curriculum_resolved[source_idx]
        source_series = np.asarray(task_to_series[source_task], dtype=float)
        pair_values: list[float] = []
        for trained_idx in range(source_idx + 1, n_resolved):
            win_start = start_indices[trained_idx] - 1
            win_end = end_indices[trained_idx]
            assert win_start < win_end, (
                f"Moving-anchor window must contain at least two checkpoints for "
                f"trained_idx={trained_idx}: start={win_start}, end={win_end}"
            )
            window = source_series[win_start : win_end + 1]
            assert window.size > 1, (
                f"Moving-anchor window sliced to a single value for "
                f"source_idx={source_idx}, trained_idx={trained_idx}: "
                f"start={win_start}, end={win_end}, size={window.size}"
            )
            assert np.all(np.isfinite(window)), (
                f"Non-finite values in moving-anchor window for source={source_task!r}, "
                f"trained_idx={trained_idx}: {window.tolist()}"
            )
            pair_values.append(float(window[0]) - float(window[-1]))
        bwt_task_to_task_moving_anchors.append(pair_values)
        if pair_values:
            bwt_moving_anchors_per_task.append(float(np.mean(pair_values)))
    bwt_moving_anchors = (
        float(np.mean(bwt_moving_anchors_per_task)) if bwt_moving_anchors_per_task else 0.0
    )

    metric_task_order = list(curriculum_resolved)
    metric_task_order_normalized = [
        "_".join(name.strip().lower().split()) for name in metric_task_order
    ]
    assert len(set(metric_task_order_normalized)) == len(metric_task_order_normalized), (
        f"Normalized metric_task_order has duplicates: {metric_task_order_normalized}"
    )

    return {
        "fwt": fwt,
        "bwt": bwt,
        "auc": auc,
        "fwt_per_task": fwt_per_task,
        "bwt_per_task": bwt_per_task,
        "bwt_task_to_task_deltas": bwt_task_to_task_deltas,
        "auc_per_task": auc_per_task,
        "bwt_moving_anchors": bwt_moving_anchors,
        "bwt_moving_anchors_per_task": bwt_moving_anchors_per_task,
        "bwt_task_to_task_moving_anchors": bwt_task_to_task_moving_anchors,
        "metric_task_order": metric_task_order,
        "metric_task_order_normalized": metric_task_order_normalized,
    }


def _text_color(value: float) -> str:
    return "white" if value > 0.55 else "#333333"


def plot_success_matrix(
    matrix: np.ndarray,
    output_path: pathlib.Path,
    *,
    show_colorbar: bool,
    one_indexed_labels: bool,
    annot_fmt: str,
) -> None:
    n = matrix.shape[0]
    # Transpose so x = training stage, y = evaluated task.
    plot_mat = matrix.T
    mask = np.isnan(plot_mat)

    label_offset = 1 if one_indexed_labels else 0
    x_labels = [str(i + label_offset) for i in range(n)]
    y_labels = [str(i + label_offset) for i in range(n)]

    fig_w = max(4.0, n * 0.9 + 1.5)
    fig_h = max(4.0, n * 0.9 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="white")
    display = np.ma.masked_where(mask, plot_mat)
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    for row in range(n):
        for col in range(n):
            if mask[row, col]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor="white",
                        linewidth=1.0,
                        zorder=2,
                    )
                )
            else:
                val = float(plot_mat[row, col])
                ax.text(
                    col,
                    row,
                    format(val, annot_fmt),
                    ha="center",
                    va="center",
                    color=_text_color(val),
                    fontsize=max(10, 24 - n),
                    fontweight="bold",
                    zorder=3,
                )

    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis="both", which="major", labelsize=max(10, 18 - n // 2))
    ax.invert_yaxis()

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Success rate")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args: Args) -> None:
    experiment_dir = pathlib.Path(args.ckpt_dir).resolve()
    assert experiment_dir.is_dir(), f"Checkpoint dir not found: {experiment_dir}"

    eval_results = discover_eval_results(experiment_dir, args.task_suite)
    saved_order = _eval_yaml.load_task_order(experiment_dir)
    if saved_order is not None:
        task_order = saved_order
        print(f"Task order (stage -> canonical): {task_order}")
    else:
        # Peek at final yaml size for identity order.
        final_data = _load_eval_yaml(eval_results[-1][1])
        n_hint = max(int(t["task_id"]) for t in final_data["tasks"]) + 1
        task_order = _resolve_task_order(experiment_dir, n_hint=n_hint)
        print(f"Task order: canonical identity {task_order}")

    matrix, task_labels, steps = build_success_matrix(eval_results, task_order)

    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).resolve()
    else:
        out_dir = experiment_dir / "eval" / "plots"

    stem = f"{args.task_suite}_success_matrix"
    plot_path = out_dir / f"{stem}.png"
    plot_success_matrix(
        matrix,
        plot_path,
        show_colorbar=args.show_colorbar,
        one_indexed_labels=args.one_indexed_labels,
        annot_fmt=args.annot_fmt,
    )
    print(f"Wrote {plot_path}")

    if args.save_npy:
        npy_path = out_dir / f"{stem}.npy"
        np.save(npy_path, matrix)
        print(f"Wrote {npy_path}")

    task_ends = [(step, task_labels[i]) for i, step in enumerate(steps)]
    traj_steps, task_to_series = build_task_trajectories(eval_results, task_labels, task_order)
    metrics = compute_continual_metrics(matrix, traj_steps, task_to_series, task_ends)
    print(
        f"FWT={metrics['fwt']:.4f}  BWT={metrics['bwt']:.4f}  AUC={metrics['auc']:.4f}  "
        f"BWT_moving_anchors={metrics['bwt_moving_anchors']:.4f}"
    )
    if args.save_metrics:
        metrics_path = out_dir / f"{stem}_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {metrics_path}")

    print(f"Matrix shape: {matrix.shape} ({len(steps)} training stages)")
    for stage_i, step in enumerate(steps):
        rates = matrix[stage_i, : stage_i + 1]
        rates_str = ", ".join(f"{r:.2f}" for r in rates)
        print(
            f"  stage {stage_i} (canonical {task_order[stage_i]}, step {step}): [{rates_str}]"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
