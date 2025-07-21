import dataclasses
import os
from pathlib import Path
import pickle
import shutil

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
    use_videos: bool = True,
) -> LeRobotDataset:
    vision_dtype = "video" if use_videos else "image"
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="arx",
        fps=50,
        features={
            "base_image": {
                "dtype": vision_dtype,
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": vision_dtype,
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": vision_dtype,
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["actions"],
            },
        },
        image_writer_threads=dataset_config.image_writer_threads,
        image_writer_processes=dataset_config.image_writer_processes,
        use_videos=True,
    )


def populate_dataset(dataset: LeRobotDataset, pkl_files: list[Path]) -> LeRobotDataset:
    for pkl_file in tqdm.tqdm(pkl_files):
        print(f"Processing file: {pkl_file}")
        # filename = pkl_file.name
        # if filename.startswith("teleop_log_20250711"):
        #     task_prompt = "fold the carpet in half along the short edge"
        # elif filename.startswith("teleop_log_20250716"):
        #     task_prompt = "fold the carpet again along the long edge"
        # else:
        #     raise ValueError(f"Unexpected: {filename}")
        task_prompt = "fold the carpet in half along the short edge, and then fold it again along the long edge"

        with pkl_file.open("rb") as f:
            episode_data = pickle.load(f)

        for step_data in episode_data:
            state = np.concatenate([step_data["qpos"]["left_arm"], step_data["qpos"]["right_arm"]]).astype(np.float32)
            # print(f"state6: {state[6]}, state13: {state[13]}")
            # left_gripper_target = 0.0 if state[6] > 4.79 else 1.0
            # right_gripper_target = 0.0 if state[13] > 4.79 else 1.0
            # convert gripper state to 0 open and 1 closed
            state[6] = 1.0 - state[6] / 4.9
            state[13] = 1.0 - state[13] / 4.9
            # gripper_target = step_data["gripper_target"][arm]["joint7"]
            left_gripper_target = 1.0 - step_data["gripper_target"]["left_arm"]["left_joint7"] / 4.9
            right_gripper_target = 1.0 - step_data["gripper_target"]["right_arm"]["right_joint7"] / 4.9
            action = np.concatenate(
                [
                    step_data["qpos_des"]["left_arm"],
                    [left_gripper_target],
                    step_data["qpos_des"]["right_arm"],
                    [right_gripper_target],
                ]
            ).astype(np.float32)

            frame = {
                "base_image": step_data["image"]["base"]["color"],
                "left_wrist_image": step_data["image"]["left_wrist"]["color"],
                "right_wrist_image": step_data["image"]["right_wrist"]["color"],
                "state": state,
                "actions": action,
                "task": task_prompt,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
    return dataset


def port_arx(
    data_dir: Path,
    repo_id: str = "kelvinzhaozg/arx_dual_arm_carpet_fold_combined",
    root_dir: str = None,
    *,
    use_videos: bool = True,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    num_episodes: int = None,
):
    pkl_files = sorted(data_dir.glob("**/*.pkl"))
    # root = os.path.join(root_dir, repo_id)

    # if os.path.exists(root):
    #     shutil.rmtree(root)
    if num_episodes is not None:
        num_episodes = min(num_episodes, len(pkl_files))
        pkl_files = pkl_files[:num_episodes]

    dataset = create_empty_dataset(
        repo_id,
        root_dir,
        dataset_config=dataset_config,
        use_videos=use_videos,
    )
    dataset = populate_dataset(
        dataset,
        pkl_files,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_arx)
