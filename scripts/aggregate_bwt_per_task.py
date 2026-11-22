"""Aggregate success-matrix metrics JSONs: pool ``bwt_per_task``, print mean/SEM.

Usage:
  uv run scripts/aggregate_bwt_per_task.py \\
    checkpoints/.../order0/eval/plots/libero_goal_success_matrix_metrics.json \\
    checkpoints/.../order1/eval/plots/libero_goal_success_matrix_metrics.json \\
    checkpoints/.../order2/eval/plots/libero_goal_success_matrix_metrics.json
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Annotated, Any

import tyro


@dataclasses.dataclass
class Args:
    """Pool ``bwt_per_task`` values across one or more metrics JSONs."""

    jsons: Annotated[tuple[str, ...], tyro.conf.Positional]
    """Paths to ``{suite}_success_matrix_metrics.json`` files."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    assert isinstance(data, dict), f"Expected object json at {path}"
    return data


def mean_sem(values: list[float]) -> tuple[float, float]:
    """Return mean and standard error of the mean (sample std / sqrt(n))."""
    assert values, "Cannot compute mean/SEM of empty value list"
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    return mean, std / math.sqrt(n)


def main(args: Args) -> None:
    assert args.jsons, "Provide at least one metrics json path"

    all_values: list[float] = []
    per_file: list[tuple[Path, int, float | None]] = []

    for raw in args.jsons:
        path = Path(raw).resolve()
        assert path.is_file(), f"Missing metrics json: {path}"
        data = load_json(path)
        assert "bwt_per_task" in data, f"Missing bwt_per_task in {path}"
        bwt_per_task = data["bwt_per_task"]
        assert isinstance(bwt_per_task, list), f"bwt_per_task must be a list in {path}"

        file_values: list[float] = []
        for idx, value in enumerate(bwt_per_task):
            assert isinstance(value, (int, float)), (
                f"bwt_per_task[{idx}] must be numeric in {path}"
            )
            file_values.append(float(value))

        all_values.extend(file_values)
        file_mean = (sum(file_values) / len(file_values)) if file_values else None
        per_file.append((path, len(file_values), file_mean))

    assert all_values, "No bwt_per_task values found across the provided jsons"
    mean, sem = mean_sem(all_values)

    print(f"Files: {len(per_file)}")
    for path, n_vals, file_mean in per_file:
        mean_str = f"{file_mean:.4f}" if file_mean is not None else "n/a"
        print(f"  {path.name}: n_tasks={n_vals}, mean={mean_str}")
        print(f"    {path}")

    print(
        f"Pooled bwt_per_task: n={len(all_values)}  "
        f"mean={mean:.6f}  sem={sem:.6f}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
