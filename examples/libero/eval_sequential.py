#!/usr/bin/env python3
"""
eval_sequential.py

Orchestrates LIBERO evaluation for sequential (per-task norm stats) checkpoints.

For each training stage, starts the policy server with the matching canonical
``--task-idx`` (norm stats), runs evaluation on that LIBERO task in the LIBERO
venv, then shuts the server down before moving to the next stage.

Task order (stage -> canonical LIBERO id) is resolved as:
  1. ``--task-order-index`` / ``--task-order-permutation`` if provided
  2. else auto-load ``{experiment_dir}/task_order.json`` written by training
  3. else canonical identity (0, 1, 2, ...)

The policy server (uv / repo root venv) and LIBERO sim (examples/libero/.venv)
run in separate Python environments.
"""

from __future__ import annotations

import dataclasses
import datetime
import glob
import http.client
import json
import os
import pathlib
import re
import select
import signal
import socket
import subprocess
import time
from typing import Literal

import tyro

import openpi.libero_eval_yaml as _eval_yaml
import openpi.training.libero_task_order as _libero_task_order

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LIBERO_PYTHON = REPO_ROOT / "examples/libero/.venv/bin/python"
TASK_ORDER_FILENAME = "task_order.json"


@dataclasses.dataclass
class Args:
    """Arguments for sequential LIBERO evaluation with per-task norm stats."""

    # Experiment dir (e.g. .../ExtendedSingleLiberoTrainPi05/) or a single step dir (.../9999/).
    # Experiment dirs: all matching step subdirs are discovered and evaluated in order.
    ckpt_dir: str = ""
    # Training config name (e.g. pi05_libero-libero_goal_sequential). Inferred from path when empty.
    policy_config: str = ""
    task_suite: str = "libero_goal"
    # Stage indices (training order), not necessarily canonical LIBERO ids.
    task_start: int = 0
    task_end: int | None = None
    # Match train_sequential: index into TASK_ORDERS, or an explicit permutation.
    # If both omitted, auto-loads {experiment_dir}/task_order.json when present; else canonical 0..N-1.
    task_order_index: int | None = None
    task_order_permutation: tuple[int, ...] | None = None

    num_trials_per_task: int = 50
    num_workers: int = 4
    # Parallelize episodes within each task (recommended with eval_sequential).
    parallel_mode: Literal["task", "episode"] = "episode"
    seed: int = 7
    # Fraction of episodes to save rollout videos for (0.0–1.0).
    video_save_fraction: float = 1.0

    repo_root: pathlib.Path = REPO_ROOT
    libero_python: pathlib.Path = DEFAULT_LIBERO_PYTHON
    # Write results under {experiment_dir}/eval/{step}/{task_suite}/ when True (default).
    out_in_ckpt_dir: bool = True
    # Used only when out_in_ckpt_dir=False.
    out_path: str = "data/libero/runs"
    run_id: str = ""

    port: int | None = None
    cuda_visible_devices: str = ""
    wait_for_boot_s: float = 240.0
    shutdown_timeout_s: float = 240.0
    port_release_timeout_s: float = 240.0
    poll_interval_s: float = 1.0
    # Stream policy-server stdout/stderr to this process (sets PYTHONUNBUFFERED on the server).
    show_server_output: bool = False

    # If set, only step dirs whose name ends with this string are evaluated (e.g. "9999").
    # Leave empty to include all numeric step subdirs.
    ckpt_step_suffix: str = ""
    # Skip checkpoints whose eval/{step}/{task_suite}/done.txt sentinel already exists.
    skip_done: bool = True
    # Re-evaluate even when done.txt exists (equivalent to --no-skip-done).
    overwrite: bool = False


DONE_SENTINEL = "done.txt"


def _load_task_order_or_none(experiment_dir: pathlib.Path) -> list[int] | None:
    path = experiment_dir / TASK_ORDER_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return [int(x) for x in payload["order"]]


def resolve_eval_task_order(args: Args, experiment_dir: str) -> list[int]:
    """Return stage -> canonical LIBERO id mapping used for evaluation.

    Precedence:
      1. Explicit --task-order-index / --task-order-permutation (must match saved order if present).
      2. Auto-load {experiment_dir}/task_order.json from training.
      3. Canonical identity order sized to the highest trained stage seen later.
    """
    saved = _load_task_order_or_none(pathlib.Path(experiment_dir))
    has_flag = args.task_order_index is not None or args.task_order_permutation is not None

    if has_flag:
        if args.task_order_permutation is not None:
            n_tasks = len(args.task_order_permutation)
        elif saved is not None:
            n_tasks = len(saved)
        else:
            n_tasks = len(_libero_task_order.TASK_ORDERS[0])
        order = _libero_task_order.resolve_task_order(
            n_tasks,
            task_order_index=args.task_order_index,
            task_order_permutation=args.task_order_permutation,
        )
        if saved is not None:
            assert saved == order, (
                f"Requested task order {order} does not match {experiment_dir}/{TASK_ORDER_FILENAME} "
                f"({saved})."
            )
        return order

    if saved is not None:
        return saved

    # Identity; length is finalized once we know trained_task_idx.
    return []


def _canonical_task_ids(task_order: list[int], stage_start: int, stage_end: int) -> list[int]:
    """Map inclusive stage range to canonical LIBERO task ids."""
    if not task_order:
        return list(range(stage_start, stage_end + 1))
    assert stage_end < len(task_order), (
        f"stage_end={stage_end} out of range for task_order of length {len(task_order)}: {task_order}"
    )
    return task_order[stage_start : stage_end + 1]


def _suite_eval_dir(experiment_dir: str, step: int, task_suite: str) -> pathlib.Path:
    return pathlib.Path(experiment_dir) / "eval" / str(step) / task_suite


def _done_sentinel_path(experiment_dir: str, step: int, task_suite: str) -> pathlib.Path:
    return _suite_eval_dir(experiment_dir, step, task_suite) / DONE_SENTINEL


def _is_eval_done(experiment_dir: str, step: int, task_suite: str) -> bool:
    return _done_sentinel_path(experiment_dir, step, task_suite).is_file()


def _mark_eval_done(
    shared_run_dir: pathlib.Path,
    *,
    task_suite: str,
    trained_task_idx: int,
    task_start: int,
    task_end: int,
    canonical_task_ids: list[int],
    task_order: list[int],
    result_path: pathlib.Path,
    elapsed_s: float,
) -> None:
    shared_run_dir.mkdir(parents=True, exist_ok=True)
    sentinel = shared_run_dir / DONE_SENTINEL
    order_str = ",".join(str(x) for x in task_order) if task_order else "canonical"
    canonical_str = ",".join(str(x) for x in canonical_task_ids)
    sentinel.write_text(
        "\n".join(
            [
                f"completed_at={datetime.datetime.now().isoformat(timespec='seconds')}",
                f"task_suite={task_suite}",
                f"trained_task_idx={trained_task_idx}",
                f"stages_evaluated={task_start}..{task_end}",
                f"canonical_tasks_evaluated={canonical_str}",
                f"task_order={order_str}",
                f"elapsed_s={elapsed_s:.1f}",
                f"results={result_path}",
            ]
        )
        + "\n"
    )
    print(f"[driver] Wrote sentinel: {sentinel}")


def _infer_policy_config(path: str) -> str:
    parts = pathlib.Path(path).resolve().parts
    for marker in ("checkpoints", "runs"):
        for i, part in enumerate(parts):
            if part == marker and i + 1 < len(parts):
                return parts[i + 1]
    raise ValueError(f"Cannot infer policy_config from path: {path}")


def find_unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def healthz_ok(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except OSError:
        return False


def wait_for_port_state(
    port: int, *, should_be_healthy: bool, timeout_s: float, state_name: str, poll_interval_s: float
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if healthz_ok(port) == should_be_healthy:
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out waiting for port {port} to become {state_name}")


def discover_checkpoints(base_dir: str, ckpt_step_suffix: str) -> list[tuple[str, int, int]]:
    """Return sorted list of (ckpt_dir, task_idx, step)."""
    step_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    ckpts = []
    for d in step_dirs:
        step_name = os.path.basename(d)
        if not step_name.isdigit():
            continue
        if ckpt_step_suffix and not step_name.endswith(ckpt_step_suffix):
            continue
        idx_files = glob.glob(os.path.join(d, "task_idx_*.txt"))
        if not idx_files:
            print(f"[skip] {d} – no task_idx_*.txt")
            continue
        task_idx = int(re.search(r"task_idx_(\d+)\.txt", idx_files[0]).group(1))
        ckpts.append((d, task_idx, int(step_name)))
    ckpts.sort(key=lambda x: x[2])
    return ckpts


def _is_checkpoint_step_dir(path: pathlib.Path) -> bool:
    return path.is_dir() and (
        bool(list(path.glob("task_idx_*.txt"))) or (path / "params").is_dir()
    )


def resolve_checkpoints(
    ckpt_dir: str, ckpt_step_suffix: str
) -> tuple[str, list[tuple[str, int, int]]]:
    """Return (experiment_dir, checkpoints) for an experiment or single-step path."""
    path = pathlib.Path(ckpt_dir).resolve()
    if _is_checkpoint_step_dir(path):
        trained_task_idx = _read_trained_task_idx(str(path))
        assert trained_task_idx is not None, f"No task_idx_*.txt in checkpoint step dir: {path}"
        step = int(path.name)
        return str(path.parent), [(str(path), trained_task_idx, step)]
    ckpts = discover_checkpoints(str(path), ckpt_step_suffix)
    return str(path), ckpts


def _read_trained_task_idx(ckpt_dir: str) -> int | None:
    idx_files = glob.glob(os.path.join(ckpt_dir, "task_idx_*.txt"))
    if not idx_files:
        return None
    return int(re.search(r"task_idx_(\d+)\.txt", idx_files[0]).group(1))


def _poll_server_output(srv: subprocess.Popen, server_log: list[str]) -> None:
    """Read any available policy-server stdout (non-blocking) into ``server_log``."""
    if srv.stdout is None:
        return
    while True:
        ready, _, _ = select.select([srv.stdout], [], [], 0)
        if not ready:
            return
        chunk = srv.stdout.read()
        if not chunk:
            return
        server_log.append(chunk)


def _format_server_log(server_log: list[str]) -> str:
    if not server_log:
        return "(no server output captured; pass --show-server-output to stream live)"
    return "".join(server_log)


def launch_server(
    args: Args,
    *,
    ckpt_dir: str,
    policy_config: str,
    task_idx: int,
    port: int,
) -> subprocess.Popen:
    print(f"[driver] Launching policy server: ckpt={ckpt_dir} task_idx={task_idx} port={port}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        "uv",
        "run",
        "scripts/serve_policy.py",
        "--port",
        str(port),
        "--task-idx",
        str(task_idx),
        "policy:checkpoint",
        "--policy.config",
        policy_config,
        "--policy.dir",
        ckpt_dir,
    ]
    print(f"[driver] Server cmd: {' '.join(cmd)}")
    if args.show_server_output:
        print("[driver] --- policy server output (live) ---")
        popen_kwargs = {"stdout": None, "stderr": None}
    else:
        popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=args.repo_root,
        start_new_session=True,
        **popen_kwargs,
    )


def wait_for_server_boot(
    args: Args, srv: subprocess.Popen, port: int, server_log: list[str]
) -> None:
    print(f"[driver] Waiting up to {args.wait_for_boot_s}s for server on port {port}")
    deadline = time.time() + args.wait_for_boot_s
    while time.time() < deadline:
        _poll_server_output(srv, server_log)

        exit_code = srv.poll()
        if exit_code is not None:
            _poll_server_output(srv, server_log)
            log = _format_server_log(server_log)
            raise RuntimeError(
                f"Policy server exited during boot with code {exit_code}\n\n--- server output ---\n{log}"
            )
        if healthz_ok(port):
            print(f"[driver] Server is up on port {port} (healthz OK)")
            if args.show_server_output:
                print("[driver] --- end policy server boot output ---")
            return
        time.sleep(args.poll_interval_s)

    _poll_server_output(srv, server_log)
    log = _format_server_log(server_log)
    raise TimeoutError(
        f"Policy server did not become healthy on port {port} within {args.wait_for_boot_s}s\n\n"
        f"--- server output ---\n{log}"
    )


def stop_server(args: Args, srv: subprocess.Popen, port: int) -> None:
    if srv.poll() is not None:
        print(f"[driver] Server already exited with code {srv.returncode}")
    else:
        pgid = os.getpgid(srv.pid)
        print(f"[driver] Stopping policy server (process group {pgid})")
        os.killpg(pgid, signal.SIGINT)
        try:
            srv.wait(timeout=args.shutdown_timeout_s)
            exit_code = srv.returncode
            if exit_code in (0, -signal.SIGINT, 128 + signal.SIGINT):
                print("[driver] Server stopped")
            else:
                print(f"[driver] Server exited with code {exit_code}")
        except subprocess.TimeoutExpired:
            print(f"[driver] SIGINT timed out; sending SIGKILL to pgid {pgid}")
            os.killpg(pgid, signal.SIGKILL)
            srv.wait(timeout=5)
            print(f"[driver] Server killed; exit code {srv.returncode}")

    print(f"[driver] Waiting for port {port} to be released")
    wait_for_port_state(
        port,
        should_be_healthy=False,
        timeout_s=args.port_release_timeout_s,
        state_name="unreachable (no healthz)",
        poll_interval_s=args.poll_interval_s,
    )


def _libero_env(args: Args) -> dict[str, str]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    libero_path = str(args.repo_root / "third_party/libero")
    env["PYTHONPATH"] = f"{libero_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def run_eval_parallel(
    args: Args,
    *,
    task_idx: int,
    port: int,
    shared_run_dir: pathlib.Path,
) -> None:
    cmd = [
        str(args.libero_python),
        str(args.repo_root / "examples/libero/main_parallel.py"),
        "--args.task-suite-names",
        args.task_suite,
        "--args.task-ids",
        str(task_idx),
        "--args.num-trials-per-task",
        str(args.num_trials_per_task),
        "--args.num-workers",
        str(args.num_workers),
        "--args.parallel-mode",
        args.parallel_mode,
        "--args.port",
        str(port),
        "--args.seed",
        str(args.seed),
        "--args.video-save-fraction",
        str(args.video_save_fraction),
        "--args.shared-run-dir",
        str(shared_run_dir),
    ]
    print(f"[driver] Eval cmd: {' '.join(cmd)}")
    subprocess.run(cmd, env=_libero_env(args), cwd=args.repo_root, check=True)


def eval_checkpoint(
    args: Args,
    *,
    ckpt_dir: str,
    policy_config: str,
    task_start: int,
    task_end: int,
    task_order: list[int],
    port: int,
    shared_run_dir: pathlib.Path,
) -> list[tuple[int, int, float]]:
    """Evaluate stages ``task_start..task_end``; returns (stage, canonical_id, elapsed_s)."""
    per_task_times: list[tuple[int, int, float]] = []
    canonical_ids = _canonical_task_ids(task_order, task_start, task_end)

    for stage, canonical_task_id in zip(
        range(task_start, task_end + 1), canonical_ids, strict=True
    ):
        print(
            f"\n=== stage {stage} (canonical task {canonical_task_id}) / ckpt {ckpt_dir} ==="
        )
        assert not healthz_ok(port), f"Port {port} already healthy before launch"
        task_start_time = time.time()
        server_log: list[str] = []
        srv = launch_server(
            args,
            ckpt_dir=ckpt_dir,
            policy_config=policy_config,
            task_idx=canonical_task_id,
            port=port,
        )
        wait_for_server_boot(args, srv, port, server_log)
        try:
            run_eval_parallel(
                args,
                task_idx=canonical_task_id,
                port=port,
                shared_run_dir=shared_run_dir,
            )
        finally:
            stop_server(args, srv, port)
        elapsed = time.time() - task_start_time
        per_task_times.append((stage, canonical_task_id, elapsed))
        print(
            f"[driver] Completed stage {stage} (canonical task {canonical_task_id}) "
            f"in {elapsed:.1f}s"
        )

    return per_task_times


def _make_shared_run_dir(args: Args, ckpt_dir: str, experiment_dir: str) -> pathlib.Path:
    step = pathlib.Path(ckpt_dir).name
    if args.out_in_ckpt_dir:
        return pathlib.Path(experiment_dir) / "eval" / step / args.task_suite
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{timestamp}_step{step}"
    if args.run_id:
        label = f"{label}_{args.run_id}"
    return pathlib.Path(args.out_path) / label / args.task_suite


def main(args: Args) -> None:
    if not args.libero_python.is_file():
        raise FileNotFoundError(
            f"LIBERO python not found at {args.libero_python}. "
            "Create it with: uv venv --python 3.8 examples/libero/.venv"
        )

    port = args.port if args.port is not None else find_unused_port()
    print(f"[driver] Using port {port}")
    print(f"[driver] Repo root: {args.repo_root}")
    print(f"[driver] LIBERO python: {args.libero_python}")

    if not args.ckpt_dir:
        raise ValueError("Provide --ckpt-dir (experiment dir or single step dir).")

    experiment_dir, ckpts = resolve_checkpoints(args.ckpt_dir, args.ckpt_step_suffix)

    if not ckpts:
        raise ValueError(f"No checkpoints found under {args.ckpt_dir}")

    policy_config = args.policy_config or _infer_policy_config(experiment_dir)
    task_order = resolve_eval_task_order(args, experiment_dir)
    print(f"[driver] Found {len(ckpts)} checkpoint(s); policy_config={policy_config}")
    print(f"[driver] Experiment dir: {experiment_dir}")
    if task_order:
        print(f"[driver] Task order (stage -> canonical): {task_order}")
    else:
        print("[driver] Task order: canonical (identity)")

    overall_start = time.time()
    result_paths: list[pathlib.Path] = []
    skipped = 0
    for ckpt_dir, trained_task_idx, step in ckpts:
        if args.skip_done and not args.overwrite and _is_eval_done(experiment_dir, step, args.task_suite):
            print(f"[skip] step {step} – {_done_sentinel_path(experiment_dir, step, args.task_suite)} exists")
            skipped += 1
            continue

        print(
            f"\n{'=' * 60}\nCheckpoint step {step} "
            f"(trained through stage {trained_task_idx})\n{'=' * 60}"
        )
        task_start = args.task_start
        task_end = args.task_end if args.task_end is not None else trained_task_idx
        assert task_end <= trained_task_idx, (
            f"task_end={task_end} exceeds checkpoint trained_task_idx={trained_task_idx}"
        )
        if task_order:
            assert trained_task_idx < len(task_order), (
                f"trained_task_idx={trained_task_idx} out of range for task_order "
                f"of length {len(task_order)}: {task_order}"
            )

        canonical_task_ids = _canonical_task_ids(task_order, task_start, task_end)
        print(
            f"[driver] Evaluating stages {task_start}..{task_end} -> "
            f"canonical tasks {canonical_task_ids}"
        )

        shared_run_dir = _make_shared_run_dir(args, ckpt_dir, experiment_dir)
        print(f"[driver] Run directory: {shared_run_dir}")
        result_path = shared_run_dir / f"{args.task_suite}.yaml"
        if args.overwrite:
            # Drop stale merged results from prior (possibly wrong-order) runs.
            if result_path.is_file():
                result_path.unlink()
                print(f"[driver] Removed stale results: {result_path}")
            done_path = shared_run_dir / DONE_SENTINEL
            if done_path.is_file():
                done_path.unlink()

        ckpt_start = time.time()
        per_task_times = eval_checkpoint(
            args,
            ckpt_dir=ckpt_dir,
            policy_config=policy_config,
            task_start=task_start,
            task_end=task_end,
            task_order=task_order,
            port=port,
            shared_run_dir=shared_run_dir,
        )
        kept, removed = _eval_yaml.rewrite_eval_yaml(result_path, canonical_task_ids)
        print(
            f"[driver] Finalized {result_path.name} in learning order: {kept}"
            + (f" (removed stale {removed})" if removed else "")
        )
        result_paths.append(result_path)
        ckpt_elapsed = time.time() - ckpt_start
        _mark_eval_done(
            shared_run_dir,
            task_suite=args.task_suite,
            trained_task_idx=trained_task_idx,
            task_start=task_start,
            task_end=task_end,
            canonical_task_ids=canonical_task_ids,
            task_order=task_order,
            result_path=result_path,
            elapsed_s=ckpt_elapsed,
        )
        print(f"\n[driver] Checkpoint step {step} finished in {ckpt_elapsed:.1f}s")
        for stage, canonical_task_id, elapsed in per_task_times:
            print(f"  stage {stage} (canonical {canonical_task_id}): {elapsed:.1f}s")

    evaluated = len(result_paths)
    print(
        f"\n[driver] Eval complete ({evaluated} evaluated, {skipped} skipped, "
        f"{time.time() - overall_start:.1f}s total)"
    )
    for result_path in result_paths:
        print(f"  Results: {result_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
