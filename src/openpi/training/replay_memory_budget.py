"""Resolve replay memory budgets from count / total / proportion settings."""

from __future__ import annotations

from collections.abc import Sequence


def _validate_proportion(n_memories_proportion: float | None) -> None:
    if n_memories_proportion is None:
        return
    if not 0 < n_memories_proportion <= 1:
        raise ValueError(
            f"n_memories_proportion must be in (0, 1], got {n_memories_proportion}"
        )


def resolve_replay_memory_budget(
    pool_sizes: Sequence[int],
    *,
    n_memories: int,
    n_memories_total: bool,
    n_memories_proportion: float | None,
) -> int:
    """Total number of replay frames to select across all tasks in ``pool_sizes``."""
    _validate_proportion(n_memories_proportion)
    pool_sizes = list(pool_sizes)
    n_pool = sum(pool_sizes)
    if n_pool == 0:
        return 0

    if n_memories_proportion is not None:
        if n_memories_total:
            return min(int(n_memories_proportion * n_pool), n_pool)
        per_task = [int(n_memories_proportion * size) for size in pool_sizes]
        return min(sum(per_task), n_pool)

    if n_memories == -1:
        return n_pool
    if n_memories_total:
        return min(n_memories, n_pool)
    return min(n_memories * len(pool_sizes), n_pool)


def distribute_budget_across_tasks(total: int, pool_sizes: Sequence[int]) -> list[int]:
    """Split ``total`` selections across tasks, weighted by each task's pool size."""
    pool_sizes = list(pool_sizes)
    n_pool = sum(pool_sizes)
    if n_pool == 0:
        return [0] * len(pool_sizes)
    if total >= n_pool:
        return list(pool_sizes)

    raw = [total * size / n_pool for size in pool_sizes]
    budgets = [int(value) for value in raw]
    remainder = total - sum(budgets)
    if remainder > 0:
        fractional = sorted(
            enumerate(raw),
            key=lambda item: item[1] - int(item[1]),
            reverse=True,
        )
        for i in range(remainder):
            budgets[fractional[i % len(fractional)][0]] += 1
    return [min(budget, size) for budget, size in zip(budgets, pool_sizes, strict=True)]


def resolve_per_task_memory_budgets(
    pool_sizes: Sequence[int],
    *,
    n_memories: int,
    n_memories_total: bool,
    n_memories_proportion: float | None,
) -> list[int]:
    """Per-task replay frame counts for independent random subsampling (plain ``er``)."""
    _validate_proportion(n_memories_proportion)
    pool_sizes = list(pool_sizes)
    if not pool_sizes:
        return []

    if n_memories_proportion is not None:
        if n_memories_total:
            total = resolve_replay_memory_budget(
                pool_sizes,
                n_memories=n_memories,
                n_memories_total=True,
                n_memories_proportion=n_memories_proportion,
            )
            return distribute_budget_across_tasks(total, pool_sizes)
        return [min(int(n_memories_proportion * size), size) for size in pool_sizes]

    if n_memories == -1:
        return list(pool_sizes)
    if n_memories_total:
        total = min(n_memories, sum(pool_sizes))
        return distribute_budget_across_tasks(total, pool_sizes)
    return [min(n_memories, size) for size in pool_sizes]
