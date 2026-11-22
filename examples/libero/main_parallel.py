from __future__ import annotations

import collections
import dataclasses
import datetime
import logging
import math
import multiprocessing
import pathlib
from typing import Literal, Tuple

import imageio
import yaml
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


def _configure_worker_logging() -> None:
    """Spawned workers never run ``if __name__ == "__main__"``; enable INFO logs there too."""
    logging.basicConfig(level=logging.INFO, force=True)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_names: Tuple[str, ...] = (
        "libero_goal",
        "libero_spatial",
        "libero_10",
        "libero_object",
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = 0  # If set (> 0), only run the first N tasks in the suite
    task_ids: tuple[int, ...] = ()  # If non-empty, evaluate only these task ids (ignores max_tasks)
    # "task": parallelize across tasks; "episode": parallelize episodes within each task
    parallel_mode: Literal["task", "episode"] = "task"

    #################################################################################################################
    # Utils
    #################################################################################################################
    out_path: str = "data/libero/runs"  # Base directory; each run saves to a timestamped subfolder
    shared_run_dir: str = ""  # If set, write directly here (no new timestamp subfolder); used by eval orchestrators
    run_id: str = ""  # Optional label appended to the run folder name (e.g. "pi05_baseline")
    num_workers: int = 20  # Number of parallel worker processes

    seed: int = 7  # Random Seed (for reproducibility)
    # Fraction of episodes to save rollout videos for (0.0–1.0). Sampling is deterministic per (seed, task, episode).
    video_save_fraction: float = 1.0


def _should_save_video(args: Args, task_id: int, episode_idx: int) -> bool:
    frac = args.video_save_fraction
    if frac >= 1.0:
        return True
    if frac <= 0.0:
        return False
    rng = np.random.RandomState(args.seed + task_id * 10000 + episode_idx)
    return rng.random() < frac


def _rollout_episode(
    env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    *,
    initial_state,
    task_description: str,
    max_steps: int,
    args: Args,
    save_video: bool,
) -> tuple[bool, list[np.ndarray]]:
    """Run one episode; return (success, replay_images)."""
    env.reset()
    action_plan: collections.deque = collections.deque()
    obs = env.set_init_state(initial_state)

    replay_images: list[np.ndarray] = []
    done = False
    t = 0

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            if save_video:
                replay_images.append(img)

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
            t += 1
        except Exception as e:
            logging.error(f"Episode rollout exception ({task_description}): {e}")
            break

    return done, replay_images


def _save_episode_video(
    video_out_path: pathlib.Path,
    *,
    task_id: int,
    task_segment: str,
    episode_idx: int,
    success: bool,
    replay_images: list[np.ndarray],
) -> None:
    if not replay_images:
        return
    outcome = "success" if success else "failure"
    video_path = video_out_path / f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{outcome}.mp4"
    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)


def run_episode(episode_args: tuple) -> dict:
    """Run a single episode in its own process. Top-level for multiprocessing pickling."""
    episode_idx, task_id, args, max_steps, run_dir = episode_args
    try:
        return _run_episode_inner(episode_idx, task_id, args, max_steps, run_dir)
    except Exception as e:
        raise RuntimeError(f"[Task {task_id} ep {episode_idx}] failed: {e}") from None


def _run_episode_inner(
    episode_idx: int, task_id: int, args: Args, max_steps: int, run_dir: pathlib.Path
) -> dict:
    np.random.seed(args.seed + task_id + episode_idx)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_names[0]]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, args.seed + task_id + episode_idx
    )
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    logging.info(
        f"[Task {task_id}] Episode {episode_idx + 1}/{args.num_trials_per_task}: {task_description}"
    )

    save_video = _should_save_video(args, task_id, episode_idx)
    done, replay_images = _rollout_episode(
        env,
        client,
        initial_state=initial_states[episode_idx],
        task_description=task_description,
        max_steps=max_steps,
        args=args,
        save_video=save_video,
    )

    if save_video:
        video_out_path = run_dir / "videos"
        task_segment = task_description.replace(" ", "_")
        _save_episode_video(
            video_out_path,
            task_id=task_id,
            task_segment=task_segment,
            episode_idx=episode_idx,
            success=done,
            replay_images=replay_images,
        )

    logging.info(f"[Task {task_id}] Episode {episode_idx + 1}: {'success' if done else 'failure'}")
    return {
        "episode_idx": episode_idx,
        "task_id": task_id,
        "task": task_description,
        "success": done,
    }


def run_task(task_args: tuple) -> dict:
    """Unpack args and run a single task. Top-level for multiprocessing pickling."""
    task_id, args, max_steps, run_dir = task_args
    try:
        return _run_task_inner(task_id, args, max_steps, run_dir)
    except Exception as e:
        raise RuntimeError(f"[Task {task_id}] failed: {e}") from None


def _run_task_inner(task_id: int, args: Args, max_steps: int, run_dir: pathlib.Path) -> dict:
    np.random.seed(args.seed + task_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_names[0]]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + task_id)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    video_out_path = run_dir / "videos"
    task_segment = task_description.replace(" ", "_")
    task_episodes, task_successes = 0, 0

    for episode_idx in range(args.num_trials_per_task):
        logging.info(
            f"[Task {task_id}] Episode {episode_idx + 1}/{args.num_trials_per_task}: {task_description}"
        )

        save_video = _should_save_video(args, task_id, episode_idx)
        done, replay_images = _rollout_episode(
            env,
            client,
            initial_state=initial_states[episode_idx],
            task_description=task_description,
            max_steps=max_steps,
            args=args,
            save_video=save_video,
        )

        task_episodes += 1
        if done:
            task_successes += 1

        if save_video:
            _save_episode_video(
                video_out_path,
                task_id=task_id,
                task_segment=task_segment,
                episode_idx=episode_idx,
                success=done,
                replay_images=replay_images,
            )

        logging.info(f"[Task {task_id}] Episode {episode_idx + 1}: {'success' if done else 'failure'}")

    task_success_rate = float(task_successes) / float(task_episodes)
    logging.info(f"[Task {task_id}] Success rate: {task_success_rate:.2%}")
    return {
        "task_id": task_id,
        "task": task_description,
        "episodes": task_episodes,
        "successes": task_successes,
        "success_rate": round(task_success_rate, 4),
    }


def _aggregate_episode_results(task_id: int, episode_results: list[dict]) -> dict:
    assert episode_results, f"No episode results for task {task_id}"
    task_description = episode_results[0]["task"]
    successes = sum(1 for r in episode_results if r["success"])
    episodes = len(episode_results)
    success_rate = float(successes) / float(episodes)
    logging.info(f"[Task {task_id}] Success rate: {success_rate:.2%}")
    return {
        "task_id": task_id,
        "task": task_description,
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(success_rate, 4),
    }


def _run_tasks_episode_parallel(
    task_ids: list[int],
    suite_args: Args,
    max_steps: int,
    run_dir: pathlib.Path,
) -> list[dict]:
    ctx = multiprocessing.get_context("spawn")
    per_task_stats: list[dict] = []

    for task_id in task_ids:
        episode_args = [
            (episode_idx, task_id, suite_args, max_steps, run_dir)
            for episode_idx in range(suite_args.num_trials_per_task)
        ]
        num_workers = min(suite_args.num_workers, suite_args.num_trials_per_task)
        logging.info(
            f"Task {task_id}: running {suite_args.num_trials_per_task} episodes with {num_workers} workers"
        )
        with ctx.Pool(processes=num_workers, initializer=_configure_worker_logging) as pool:
            episode_results = list(
                pool.imap_unordered(run_episode, episode_args)
            )
        per_task_stats.append(_aggregate_episode_results(task_id, episode_results))

    return per_task_stats


def eval_suite(args: Args, task_suite_name: str, run_dir: pathlib.Path) -> dict:
    """Evaluate a single task suite and return its stats."""
    if task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    max_steps = MAX_STEPS[task_suite_name]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {task_suite_name}")

    (run_dir / "videos").mkdir(parents=True, exist_ok=True)

    suite_args = dataclasses.replace(args, task_suite_names=(task_suite_name,))

    if args.task_ids:
        for task_id in args.task_ids:
            assert 0 <= task_id < num_tasks_in_suite, (
                f"task_id {task_id} must be in [0, {num_tasks_in_suite})"
            )
        task_ids = list(args.task_ids)
    else:
        num_tasks = min(num_tasks_in_suite, args.max_tasks) if args.max_tasks > 0 else num_tasks_in_suite
        task_ids = list(range(num_tasks))

    num_tasks = len(task_ids)

    if args.parallel_mode == "episode":
        logging.info(f"Episode-parallel mode: {num_tasks} task(s)")
        per_task_stats = _run_tasks_episode_parallel(task_ids, suite_args, max_steps, run_dir)
    else:
        task_args = [(task_id, suite_args, max_steps, run_dir) for task_id in task_ids]
        num_workers = min(args.num_workers, num_tasks)
        logging.info(f"Task-parallel mode: {num_tasks} tasks with {num_workers} workers")

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=_configure_worker_logging) as pool:
            per_task_stats = list(
                tqdm.tqdm(pool.imap_unordered(run_task, task_args), total=num_tasks, desc=task_suite_name)
            )

    per_task_stats.sort(key=lambda s: s["task_id"])

    stats_path = run_dir / f"{task_suite_name}.yaml"
    if stats_path.is_file():
        with open(stats_path) as f:
            stats = yaml.safe_load(f)
        existing_by_id = {t["task_id"]: t for t in stats["tasks"]}
        for task_stat in per_task_stats:
            existing_by_id[task_stat["task_id"]] = task_stat
        per_task_stats = sorted(existing_by_id.values(), key=lambda s: s["task_id"])
    else:
        stats = {"split": task_suite_name}

    total_episodes = sum(s["episodes"] for s in per_task_stats)
    total_successes = sum(s["successes"] for s in per_task_stats)
    total_success_rate = float(total_successes) / float(total_episodes)

    logging.info(f"[{task_suite_name}] Success rate: {total_success_rate:.2%} ({total_successes}/{total_episodes})")

    stats.update(
        {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": round(total_success_rate, 4),
            "tasks": per_task_stats,
        }
    )
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Stats saved to {stats_path}")
    return stats


def eval_libero(args: Args) -> None:
    if args.shared_run_dir:
        suite_run_dir = pathlib.Path(args.shared_run_dir)
        suite_run_dir.mkdir(parents=True, exist_ok=True)
        task_suite_name = args.task_suite_names[0]
        print(f"\n{'=' * 60}\nRun directory: {suite_run_dir}\n{'=' * 60}\n")
        eval_suite(args, task_suite_name, suite_run_dir)
        return

    script_start = datetime.datetime.now()
    timestamp = script_start.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{args.run_id}" if args.run_id else timestamp
    run_dir = pathlib.Path(args.out_path) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nRun directory: {run_dir}\n{'=' * 60}\n")

    all_stats = []
    split_times = {}
    for task_suite_name in args.task_suite_names:
        suite_start = datetime.datetime.now()
        suite_run_dir = run_dir / task_suite_name
        stats = eval_suite(args, task_suite_name, suite_run_dir)
        all_stats.append(stats)
        split_times[task_suite_name] = (datetime.datetime.now() - suite_start).total_seconds() / 3600

    if len(all_stats) > 1:
        total_episodes = sum(s["total_episodes"] for s in all_stats)
        total_successes = sum(s["total_successes"] for s in all_stats)
        logging.info(f"Overall success rate: {total_successes / total_episodes:.2%} across {len(all_stats)} suites")

    total_time = (datetime.datetime.now() - script_start).total_seconds() / 3600
    print(f"\n{'=' * 60}\nTiming Summary\n{'=' * 60}")
    for split, hours in split_times.items():
        print(f"  {split}: {hours:.4f} hours")
    print(f"  Total: {total_time:.4f} hours")
    print("=" * 60)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
