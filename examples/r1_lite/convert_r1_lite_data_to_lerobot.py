import dataclasses
import os
from pathlib import Path
import pickle
import shutil

import cv2
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


def decompress_jpg_to_image(jpg_bytes: bytes) -> np.ndarray:
    """
    Decompress JPG bytes back to numpy image array.

    Args:
        jpg_bytes: JPG compressed image bytes

    Returns:
        numpy.ndarray: Decompressed image array, or None if decompression fails
    """
    if jpg_bytes is None:
        return None

    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return image
    except Exception as e:
        print(f"Warning: Failed to decompress JPG bytes: {e}")
        return None


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
        robot_type="galaxea_r1_lite",
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
                "shape": (17,),
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
        task_prompt = "drive to the white table"

        with pkl_file.open("rb") as f:
            episode_data = pickle.load(f)

        for step_data in episode_data:
            # Extract joint positions for both arms (6 DOF each)
            left_arm_qpos = np.array(step_data["qpos"]["left_arm"], dtype=np.float32)
            right_arm_qpos = np.array(step_data["qpos"]["right_arm"], dtype=np.float32)

            # Extract gripper positions (1 DOF each, normalized to 0-1 range)
            left_gripper_qpos = np.array(step_data["gripper_qpos"]["left_arm"], dtype=np.float32)
            right_gripper_qpos = np.array(step_data["gripper_qpos"]["right_arm"], dtype=np.float32)

            # Normalize gripper positions (assuming range -2.9 to 0, convert to 0-1)
            left_gripper_normalized = np.clip((left_gripper_qpos + 2.9) / 2.9, 0, 1)
            right_gripper_normalized = np.clip((right_gripper_qpos + 2.9) / 2.9, 0, 1)

            # Extract chassis velocity (3 DOF)
            chassis_vel = np.array(step_data["chassis_velocity_cmd"], dtype=np.float32)

            # Combine into state vector: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1) = 14
            state = np.concatenate(
                [left_arm_qpos, left_gripper_normalized, right_arm_qpos, right_gripper_normalized]
            ).astype(np.float32)

            # Extract desired positions for actions
            left_arm_qpos_des = np.array(step_data["qpos_des"]["left_arm"], dtype=np.float32)
            right_arm_qpos_des = np.array(step_data["qpos_des"]["right_arm"], dtype=np.float32)

            # Extract desired gripper positions
            left_gripper_des = np.array(step_data["gripper_qpos_des"]["left_arm"], dtype=np.float32)
            right_gripper_des = np.array(step_data["gripper_qpos_des"]["right_arm"], dtype=np.float32)

            # Normalize desired gripper positions
            left_gripper_des_normalized = np.clip((left_gripper_des + 2.9) / 2.9, 0, 1)
            right_gripper_des_normalized = np.clip((right_gripper_des + 2.9) / 2.9, 0, 1)

            # Combine into action vector: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1) + chassis(3) = 17
            action = np.concatenate(
                [
                    left_arm_qpos_des,
                    left_gripper_des_normalized,
                    right_arm_qpos_des,
                    right_gripper_des_normalized,
                    chassis_vel,
                ]
            ).astype(np.float32)

            # Map camera names from R1 Lite to expected format and decompress JPG bytes
            # R1 Lite has: 'right', 'head_left', 'head_right', 'left'
            # We'll map: head_left -> base_image, left -> left_wrist_image, right -> right_wrist_image
            frame = {
                "base_image": decompress_jpg_to_image(step_data["image"]["head_left"]["color"]),
                "left_wrist_image": decompress_jpg_to_image(step_data["image"]["left"]["color"]),
                "right_wrist_image": decompress_jpg_to_image(step_data["image"]["right"]["color"]),
                "state": state,
                "actions": action,
                "task": task_prompt,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
    return dataset


def port_r1_lite(
    data_dir: Path,
    repo_id: str = "kelvinzhaozg/r1_lite_drive_to_table",
    root_dir: str = None,
    *,
    use_videos: bool = True,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    num_episodes: int = None,
):
    pkl_files = sorted(data_dir.glob("**/*.pkl"))

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
    tyro.cli(port_r1_lite)
