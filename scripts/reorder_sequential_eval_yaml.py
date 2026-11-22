"""Filter and reorder sequential-eval YAMLs into learning order.

Removes stale canonical task entries left over from incorrect (canonical-order)
eval runs, and rewrites each ``{task_suite}.yaml`` so ``tasks`` follows the
training curriculum (stage 0, 1, ...).

Usage:
  # Dry-run (show what would change):
  uv run scripts/reorder_sequential_eval_yaml.py \\
    --ckpt-dir checkpoints/pi05_libero-libero_goal_sequential/LiberoTrainPi05_RandER_01_order2 \\
    --dry-run

  # Apply in place:
  uv run scripts/reorder_sequential_eval_yaml.py \\
    --ckpt-dir checkpoints/pi05_libero-libero_goal_sequential/LiberoTrainPi05_RandER_01_order2
"""

from __future__ import annotations

import dataclasses
import pathlib

import tyro
import yaml

import openpi.libero_eval_yaml as _eval_yaml


@dataclasses.dataclass
class Args:
    ckpt_dir: str
    task_suite: str = "libero_goal"
    dry_run: bool = False


def main(args: Args) -> None:
    experiment_dir = pathlib.Path(args.ckpt_dir).resolve()
    assert experiment_dir.is_dir(), f"Checkpoint dir not found: {experiment_dir}"

    task_order = _eval_yaml.load_task_order(experiment_dir)
    if task_order is not None:
        print(f"Task order (stage -> canonical): {task_order}")
    else:
        print("No task_order.json; assuming canonical identity order")

    eval_root = experiment_dir / "eval"
    assert eval_root.is_dir(), f"No eval/ directory under {experiment_dir}"

    n_rewritten = 0
    n_unchanged = 0
    for step_dir in sorted(eval_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not step_dir.name.isdigit():
            continue
        suite_dir = step_dir / args.task_suite
        yaml_path = suite_dir / f"{args.task_suite}.yaml"
        done_path = suite_dir / "done.txt"
        if not yaml_path.is_file():
            continue
        assert done_path.is_file(), (
            f"Missing done.txt for {yaml_path}; cannot determine expected tasks safely"
        )

        done_fields = _eval_yaml.parse_done_sentinel(done_path)
        trained_stage = int(done_fields["trained_task_idx"])
        expected = _eval_yaml.expected_canonical_ids(
            task_order=task_order,
            trained_stage=trained_stage,
            done_fields=done_fields,
        )
        if task_order is not None:
            assert expected == task_order[: trained_stage + 1], (
                f"step {step_dir.name}: done.txt canonical list {expected} != "
                f"task_order[:{trained_stage + 1}]={task_order[: trained_stage + 1]}"
            )

        with yaml_path.open() as f:
            before = yaml.safe_load(f)
        before_ids = [int(t["task_id"]) for t in before["tasks"]]
        needs_rewrite = before_ids != expected

        kept, removed = _eval_yaml.rewrite_eval_yaml(
            yaml_path, expected, dry_run=args.dry_run
        )
        assert kept == expected, f"step {step_dir.name}: rewrite produced {kept}, expected {expected}"
        if needs_rewrite:
            action = "Would rewrite" if args.dry_run else "Rewrote"
            print(
                f"  {action} step {step_dir.name}: {before_ids} -> {kept}"
                + (f" (removed {removed})" if removed else "")
            )
            n_rewritten += 1
        else:
            n_unchanged += 1
            print(f"  ok step {step_dir.name}: {kept}")

    print(
        f"Done: {n_rewritten} rewritten, {n_unchanged} already correct"
        + (" (dry-run)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
