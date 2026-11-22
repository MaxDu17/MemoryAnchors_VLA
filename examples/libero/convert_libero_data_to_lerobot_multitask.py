"""
Convert one Libero RLDS suite into a single LeRobot dataset (all tasks merged).

For sequential training with one repo per task, use `convert_libero_data_to_lerobot.py`.

Usage:
  uv run examples/libero/convert_libero_data_to_lerobot_multitask.py \\
    --data_dir /path/to/rlds --idx 1

Suite indices (--idx):
  0 = libero_10_no_noops
  1 = libero_goal_no_noops
  2 = libero_object_no_noops
  3 = libero_spatial_no_noops
"""

import shutil

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


def main(data_dir: str, *, push_to_hub: bool = False, idx: int = 0, output_dir: str | None = None):
    del output_dir
    suite_name = RAW_DATASET_NAMES[idx]
    repo_id = f"{REPO_NAME}-{suite_name}"
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
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

    raw_dataset = tfds.load(suite_name, data_dir=data_dir, split="train")
    for episode in raw_dataset:
        for step in episode["steps"].as_numpy_iterator():
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

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
