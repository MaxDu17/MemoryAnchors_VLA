"""Helpers for sequential-eval YAML results (filter + learning-order rewrite)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import yaml

TASK_ORDER_FILENAME = "task_order.json"


def load_task_order(experiment_dir: pathlib.Path) -> list[int] | None:
    path = experiment_dir / TASK_ORDER_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return [int(x) for x in payload["order"]]


def parse_done_sentinel(done_path: pathlib.Path) -> dict[str, str]:
    assert done_path.is_file(), f"Missing done sentinel: {done_path}"
    fields: dict[str, str] = {}
    for line in done_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def expected_canonical_ids(
    *,
    task_order: list[int] | None,
    trained_stage: int,
    done_fields: dict[str, str] | None = None,
) -> list[int]:
    """Canonical LIBERO ids that should appear, in learning order."""
    if done_fields and "canonical_tasks_evaluated" in done_fields:
        raw = done_fields["canonical_tasks_evaluated"]
        return [int(x) for x in raw.split(",") if x.strip()]
    if task_order:
        assert 0 <= trained_stage < len(task_order), (
            f"trained_stage={trained_stage} out of range for task_order={task_order}"
        )
        return list(task_order[: trained_stage + 1])
    return list(range(trained_stage + 1))


def reorder_eval_yaml_data(
    data: dict[str, Any],
    canonical_ids_in_order: list[int],
) -> dict[str, Any]:
    """Keep only ``canonical_ids_in_order``, sorted in that learning order; recompute totals."""
    by_id = {int(entry["task_id"]): entry for entry in data["tasks"]}
    missing = [tid for tid in canonical_ids_in_order if tid not in by_id]
    assert not missing, (
        f"Eval YAML missing expected canonical task ids {missing}; "
        f"have {sorted(by_id)}"
    )

    ordered_tasks: list[dict[str, Any]] = []
    for stage, canonical_id in enumerate(canonical_ids_in_order):
        entry = dict(by_id[canonical_id])
        entry["task_id"] = canonical_id
        entry["stage"] = stage
        ordered_tasks.append(entry)

    total_episodes = sum(int(t["episodes"]) for t in ordered_tasks)
    total_successes = sum(int(t["successes"]) for t in ordered_tasks)
    assert total_episodes > 0
    success_rate = float(total_successes) / float(total_episodes)

    out = dict(data)
    out["total_episodes"] = total_episodes
    out["total_successes"] = total_successes
    out["success_rate"] = round(success_rate, 4)
    out["tasks"] = ordered_tasks
    return out


def rewrite_eval_yaml(
    yaml_path: pathlib.Path,
    canonical_ids_in_order: list[int],
    *,
    dry_run: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Filter+reorder an eval YAML in place.

    Returns (kept_ids, removed_ids).
    """
    assert yaml_path.is_file(), f"Missing eval YAML: {yaml_path}"
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict) and "tasks" in data

    before_ids = [int(t["task_id"]) for t in data["tasks"]]
    removed = [tid for tid in before_ids if tid not in set(canonical_ids_in_order)]
    rewritten = reorder_eval_yaml_data(data, canonical_ids_in_order)
    kept = [int(t["task_id"]) for t in rewritten["tasks"]]

    if not dry_run:
        with yaml_path.open("w") as f:
            yaml.dump(rewritten, f, default_flow_style=False, sort_keys=False)
    return kept, removed
