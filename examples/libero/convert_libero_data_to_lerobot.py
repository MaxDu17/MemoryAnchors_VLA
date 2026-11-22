"""
Convert one Libero RLDS suite into per-task LeRobot datasets.

Each task gets its own LeRobot repo under:
  {output_dir or HF_LEROBOT_HOME}/libero-{suite_name}/{task_slug}/

This layout matches `LeRobotLiberoSeqDataConfig` in sequential training configs
(e.g. `pi0_libero_low_mem_finetune-libero_goal_sequential`).

Usage:
  # Convert libero_goal (idx=1) into per-task datasets:
  uv run examples/libero/convert_libero_data_to_lerobot.py \\
    --data-dir /path/to/libero_noops \\
    --output-dir /path/to/libero_lerobot \\
    --idx 1

  # Convert all four suites (run once per idx):
  for idx in 0 1 2 3; do
    uv run examples/libero/convert_libero_data_to_lerobot.py \\
      --data-dir /path/to/libero_noops \\
      --output-dir /path/to/libero_lerobot \\
      --idx $idx
  done

Suite indices (--idx):
  0 = libero_10_no_noops
  1 = libero_goal_no_noops
  2 = libero_object_no_noops
  3 = libero_spatial_no_noops

For a single combined dataset (all suites or all tasks in one repo), see:
  - convert_libero_data_to_lerobot_combined.py  (all 4 suites → one repo)
  - convert_libero_data_to_lerobot_multitask.py (one suite → one repo)

Requires tensorflow_datasets:
  `uv pip install tensorflow tensorflow_datasets`

Download raw Libero RLDS data from:
  https://huggingface.co/datasets/openvla/modified_libero_rlds
"""

import pathlib
import re

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "libero"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def slugify(txt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", txt.lower()).strip("_")


def main(data_dir: str, *, push_to_hub: bool = False, idx: int = 0, output_dir: str | None = None):
    lerobot_home = output_dir or str(HF_LEROBOT_HOME)
    suite_name = RAW_DATASET_NAMES[idx]

    datasets: dict[str, LeRobotDataset] = {}

    def get_ds(task_slug: str) -> LeRobotDataset:
        if task_slug not in datasets:
            repo_id = f"{REPO_NAME}-{suite_name}/{task_slug}"
            out_dir = pathlib.Path(lerobot_home) / repo_id
            datasets[task_slug] = LeRobotDataset.create(
                repo_id=repo_id,
                root=out_dir,
                robot_type="panda",
                fps=10,
                features={
                    "image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "state": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["actions"],
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
            )
        return datasets[task_slug]

    raw_dataset = tfds.load(suite_name, data_dir=data_dir, split="train")

    for episode in raw_dataset:
        dataset = None
        for step in episode["steps"].as_numpy_iterator():
            if dataset is None:
                task_text = step["language_instruction"].decode()
                task_slug = slugify(task_text)
                dataset = get_ds(task_slug)

            dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                    "task": step["language_instruction"].decode(),
                }
            )
        dataset.save_episode()

    print(f"Converted {suite_name} into {len(datasets)} task datasets under {lerobot_home}/libero-{suite_name}/")

    if push_to_hub:
        for dataset in datasets.values():
            dataset.push_to_hub(
                tags=["libero", "panda", "rlds"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    tyro.cli(main)
