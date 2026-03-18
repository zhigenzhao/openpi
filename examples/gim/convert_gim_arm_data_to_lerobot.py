"""
Script to convert GIM dual arm data to the LeRobot dataset v3.0 format.

Example usage:
conda run -n openpi python examples/gim/convert_gim_arm_data_to_lerobot.py \
    --data-dir /path/to/pkl/data \
    --repo-id <org>/<dataset-name> \
    --push-to-hub
"""

import collections
import dataclasses
from pathlib import Path
import pickle

import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


def decompress_jpg_to_image(jpg_bytes: bytes) -> np.ndarray | None:
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
        # Decode image and return directly
        return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Warning: Failed to decompress JPG bytes: {e}")
        return None


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    image_writer_processes: int = 10
    image_writer_threads: int = 5


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    root: str | None = None,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    use_videos: bool = True,
) -> LeRobotDataset:
    vision_dtype = "video" if use_videos else "image"
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="gim",
        fps=50,
        features={
            "observation.images.top_image": {
                "dtype": vision_dtype,
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist_image": {
                "dtype": vision_dtype,
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist_image": {
                "dtype": vision_dtype,
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["action"],
            },
        },
        image_writer_threads=dataset_config.image_writer_threads,
        image_writer_processes=dataset_config.image_writer_processes,
        use_videos=use_videos,
    )


def _normalize_gripper(value: float) -> float:
    """Normalize gripper value: [-0.2 (closed) -> 1.0, 0.0 (open) -> 0.0]."""
    return value / -0.2


# Measured lag between gripper command and physical gripper position (~160ms at 50 FPS).
# Determined by cross-correlation of gripper_target vs measured qpos on 7-dim data.
GRIPPER_LAG_STEPS = 8


def _process_step(step_data: dict, task_prompt: str, gripper_history: collections.deque | None = None) -> dict:
    """Process a single step and return a frame dict."""
    left_qpos = np.array(step_data["qpos"]["left_arm"], dtype=np.float32)
    right_qpos = np.array(step_data["qpos"]["right_arm"], dtype=np.float32)

    if len(left_qpos) == 6:
        # 6-dim: use gripper_target from GRIPPER_LAG_STEPS ago as gripper state.
        # At 50 FPS the gripper at time t has approximately reached the position
        # commanded at t-8 (~160ms lag, measured via cross-correlation on 7-dim data).
        grip_src = (
            gripper_history[0]
            if gripper_history and len(gripper_history) == GRIPPER_LAG_STEPS
            else step_data["gripper_target"]
        )
        left_grip = _normalize_gripper(grip_src["left_arm"])
        right_grip = _normalize_gripper(grip_src["right_arm"])
        state = np.concatenate([left_qpos, [left_grip], right_qpos, [right_grip]]).astype(np.float32)
    elif len(left_qpos) == 7:
        # 7-dim: 7th element is gripper, normalize it
        state = np.concatenate([left_qpos, right_qpos]).astype(np.float32)
        state[6] = _normalize_gripper(state[6])
        state[13] = _normalize_gripper(state[13])
    else:
        raise ValueError(f"Unexpected qpos dimension: {len(left_qpos)}")

    # Action from qpos_target + gripper_target
    left_target = np.array(step_data["qpos_target"]["left_arm"], dtype=np.float32)
    right_target = np.array(step_data["qpos_target"]["right_arm"], dtype=np.float32)
    left_grip_target = _normalize_gripper(step_data["gripper_target"]["left_arm"])
    right_grip_target = _normalize_gripper(step_data["gripper_target"]["right_arm"])
    action = np.concatenate([left_target, [left_grip_target], right_target, [right_grip_target]]).astype(np.float32)

    return {
        "observation.images.top_image": decompress_jpg_to_image(step_data["image"]["top"]["color"]),
        "observation.images.left_wrist_image": decompress_jpg_to_image(step_data["image"]["left_wrist"]["color"]),
        "observation.images.right_wrist_image": decompress_jpg_to_image(step_data["image"]["right_wrist"]["color"]),
        "observation.state": state,
        "action": action,
        "task": task_prompt,
    }


def populate_dataset(
    dataset: LeRobotDataset,
    pkl_files: list[Path],
    *,
    active_only: bool = True,
) -> LeRobotDataset:
    for pkl_file in tqdm.tqdm(pkl_files):
        print(f"Processing file: {pkl_file}")
        task_prompt = "pick up a tshirt and load flatly onto the board"

        with pkl_file.open("rb") as f:
            episode_data = pickle.load(f)

        skip_inactive = active_only and pkl_file.stem.startswith("teleop")
        frame_count = 0
        gripper_history: collections.deque = collections.deque(maxlen=GRIPPER_LAG_STEPS)
        for step_data in episode_data:
            if skip_inactive:
                grip = step_data["grip_active"]
                if not grip["left_arm"] and not grip["right_arm"]:
                    # Update history even for skipped steps so gaps don't create stale values
                    gripper_history.append(step_data["gripper_target"])
                    continue

            frame = _process_step(step_data, task_prompt, gripper_history)
            gripper_history.append(step_data["gripper_target"])
            dataset.add_frame(frame)
            frame_count += 1

        if frame_count > 0:
            dataset.save_episode()
            print(f"  Saved episode with {frame_count} frames")
        else:
            print(f"  Skipping {pkl_file.name}: no active frames")

    return dataset


def port_gim(
    data_dir: Path,
    repo_id: str = "kelvinzhaozg/gim_tshirt",
    root_dir: str | None = None,
    *,
    use_videos: bool = True,
    push_to_hub: bool = True,
    private: bool = True,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    num_episodes: int | None = None,
    active_only: bool = True,
):
    data_dir = data_dir.expanduser()
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
        active_only=active_only,
    )
    print("Finalizing dataset (closing parquet writers)...")
    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(private=private)


if __name__ == "__main__":
    tyro.cli(port_gim)
