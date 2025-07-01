import dataclasses
import os
from pathlib import Path
import pickle

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    root: str = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="galaxea",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=dataset_config.image_writer_threads,
        image_writer_processes=dataset_config.image_writer_processes,
    )


def populate_dataset(dataset: LeRobotDataset, pkl_files: list[Path]) -> LeRobotDataset:
    for pkl_file in tqdm.tqdm(pkl_files):
        with pkl_file.open("rb") as f:
            episode_data = pickle.load(f)

        for step_data in episode_data:
            arm = "right_arm"
            if arm not in step_data["qpos"]:
                continue

            state = np.concatenate([step_data["qpos"][arm], step_data["gripper_qpos"][arm]]).astype(np.float32)
            action = np.concatenate([step_data["qpos_des"][arm], step_data["gripper_qpos_des"][arm]]).astype(np.float32)

            frame = {
                "image": step_data["image"]["base"]["color"],
                "wrist_image": step_data["image"]["right_wrist"]["color"],
                "state": state,
                "actions": action,
                "task": "place the green block on the inside corner of the blue T",
            }
            dataset.add_frame(frame)

        dataset.save_episode()
    return dataset


def port_galaxea(
    data_dir: Path,
    repo_id: str = "kelvinzhaozg/galaxea_single_arm_block_pick_and_place",
    root_dir: str = "/home/bytedance/datasets/",
    *,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    pkl_files = sorted(data_dir.glob("**/*.pkl"))
    root = os.path.join(root_dir, repo_id)

    dataset = create_empty_dataset(
        repo_id,
        root,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        pkl_files,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_galaxea)
