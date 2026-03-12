"""GIM Dual Arm OpenPI Teleoperation Controller.

Implementation of OpenPITeleopController for GIM 6-DOF bimanual arm robots
using OpenPI's GIM dual arm environment system.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpi_client.runtime import environment as _environment
from openpi_client.runtime.openpi_teleop_controller import OpenPITeleopController

R_HEADSET_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
ASSET_PATH = str(Path(__file__).resolve().parent / "assets")


class GIMDualArmOpenPITeleopController(OpenPITeleopController):
    """GIM 6-DOF dual arm teleoperation controller for OpenPI environment."""

    def __init__(
        self,
        environment: _environment.Environment,
        robot_urdf_path: str | None = None,
        scale_factor: float = 1.0,
        *,
        enable_log_data: bool = False,
        log_dir: str = "logs/gim_dual_openpi_teleop",
    ):
        if robot_urdf_path is None:
            robot_urdf_path = self._get_default_urdf_path()

        manipulator_config = {
            "right_arm": {
                "link_name": "right_tool0",
                "pose_source": "right_controller",
                "control_trigger": "right_grip",
                "gripper_config": {
                    "type": "parallel",
                    "gripper_trigger": "right_trigger",
                    "joint_names": ["right_gripper"],
                    "open_pos": [-0.20],
                    "close_pos": [0.0],
                },
            },
            "left_arm": {
                "link_name": "left_tool0",
                "pose_source": "left_controller",
                "control_trigger": "left_grip",
                "gripper_config": {
                    "type": "parallel",
                    "gripper_trigger": "left_trigger",
                    "joint_names": ["left_gripper"],
                    "open_pos": [-0.20],
                    "close_pos": [0.0],
                },
            },
        }

        q_init = None

        super().__init__(
            environment=environment,
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            floating_base=False,
            R_headset_world=R_HEADSET_TO_WORLD,
            scale_factor=scale_factor,
            q_init=q_init,
            dt=1.0 / 50.0,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
        )

        logging.info("GIMDualArmOpenPITeleopController initialized")

    def _placo_setup(self):
        """Setup Placo robot and compute joint slices for each arm."""
        super()._placo_setup()
        self.placo_arm_joint_slice: dict[str, slice] = {}
        for arm_name, config in self.manipulator_config.items():
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("tool0", "")
            arm_joint_names = [f"{arm_prefix}joint{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _get_default_urdf_path(self) -> str:
        """Get default URDF path for GIM dual arm."""
        urdf_path = Path(ASSET_PATH) / "gim_arm_6dof/dual_gim_arm_6dof_2.urdf"

        if urdf_path.exists():
            logging.info(f"Using URDF path: {urdf_path}")
            return str(urdf_path)

        logging.warning(f"URDF not found at {urdf_path}")
        return None

    def _update_robot_state(self):
        """Update robot state from GIM dual arm environment observations."""
        try:
            obs = self._environment.get_observation()
            joint_positions = np.array(obs["state"])

            if len(joint_positions) == 14:
                left_arm_joints = joint_positions[:6]
                right_arm_joints = joint_positions[7:13]

                self.placo_robot.state.q[self.placo_arm_joint_slice["left_arm"]] = left_arm_joints
                self.placo_robot.state.q[self.placo_arm_joint_slice["right_arm"]] = right_arm_joints

        except Exception as e:
            logging.debug(f"Error updating robot state from environment: {e}")

    def _placo_to_env_action(self) -> dict[str, Any]:
        """Convert Placo joint targets to GIM dual arm environment action format."""
        try:
            action_array = np.zeros(14)

            # Extract arm joint targets
            action_array[:6] = self.placo_robot.state.q[self.placo_arm_joint_slice["left_arm"]]
            action_array[7:13] = self.placo_robot.state.q[self.placo_arm_joint_slice["right_arm"]]

            # Add gripper commands with GIM-specific normalization
            for arm_name, gripper_idx in [("left_arm", 6), ("right_arm", 13)]:
                if arm_name in self.gripper_pos_target:
                    gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                    joint_name = gripper_config["joint_names"][0]
                    if joint_name in self.gripper_pos_target[arm_name]:
                        target = self.gripper_pos_target[arm_name][joint_name]
                        open_pos = gripper_config["open_pos"][0]  # -0.20
                        close_pos = gripper_config["close_pos"][0]  # 0.0
                        # (target - (-0.2)) / (0.0 - (-0.2)) -> 0=open, 1=closed
                        normalized = (target - open_pos) / (close_pos - open_pos)
                        action_array[gripper_idx] = np.clip(normalized, 0.0, 1.0)

            return {"actions": action_array}

        except Exception as e:
            logging.error(f"Error converting placo to environment action: {e}")
            return {"actions": np.zeros(14)}

    def _compress_image_to_jpg(self, image: np.ndarray, quality: int = 85) -> bytes:
        """Compress a numpy image array to JPG bytes."""
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode(".jpg", image, encode_param)
            return encoded_img.tobytes()
        except Exception as e:
            logging.error(f"Error compressing image: {e}")
            return None

    def _get_robot_state_for_logging(self) -> dict[str, Any]:
        """Returns a dictionary of robot-specific data for logging."""
        try:
            obs = self._environment.get_observation()
            joint_positions = np.array(obs["state"])

            if len(joint_positions) == 14:
                robot_state = {
                    "qpos": {
                        "left_arm": joint_positions[:6],
                        "left_gripper": joint_positions[6:7],
                        "right_arm": joint_positions[7:13],
                        "right_gripper": joint_positions[13:14],
                    },
                    "qpos_des": {
                        "left_arm": self.placo_robot.state.q[self.placo_arm_joint_slice["left_arm"]].copy(),
                        "right_arm": self.placo_robot.state.q[self.placo_arm_joint_slice["right_arm"]].copy(),
                    },
                    "gripper_target": {
                        "left_arm": (
                            self.gripper_pos_target.get("left_arm", {}).copy()
                            if "left_arm" in self.gripper_pos_target
                            else None
                        ),
                        "right_arm": (
                            self.gripper_pos_target.get("right_arm", {}).copy()
                            if "right_arm" in self.gripper_pos_target
                            else None
                        ),
                    },
                }
            else:
                logging.warning(f"Unexpected joint positions length: {len(joint_positions)}")
                robot_state = {"qpos": {}, "qpos_des": {}, "gripper_target": {}}

            robot_state["images"] = {
                "top_image": self._compress_image_to_jpg(obs["top_image"]),
                "right_wrist_image": self._compress_image_to_jpg(obs["right_wrist_image"]),
                "left_wrist_image": self._compress_image_to_jpg(obs["left_wrist_image"]),
            }

            return robot_state

        except Exception as e:
            logging.error(f"Error getting robot state for logging: {e}")
            return {"error": str(e)}
