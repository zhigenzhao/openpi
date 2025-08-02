"""ARX R5 Dual Arm OpenPI Teleoperation Controller.

Implementation of OpenPITeleopController specifically for ARX R5 dual arm robots
using OpenPI's ARX dual arm environment system.
"""

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
from openpi_client.runtime import environment as _environment
from openpi_client.runtime.openpi_teleop_controller import OpenPITeleopController
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class ARXDualArmOpenPITeleopController(OpenPITeleopController):
    """ARX R5 dual arm teleoperation controller for OpenPI environment."""

    def __init__(
        self,
        environment: _environment.Environment,
        robot_urdf_path: str | None = None,
        scale_factor: float = 1.0,
        enable_log_data: bool = False,
        log_dir: str = "logs/arx_dual_openpi_teleop",
    ):
        """Initialize ARX dual arm OpenPI teleoperation controller.

        Args:
            environment: OpenPI ARX dual arm environment
            robot_urdf_path: Path to ARX dual arm URDF (uses default if None)
            scale_factor: Scaling factor for controller movements
            enable_log_data: Whether to enable data logging
            log_dir: Directory for log data
        """
        # Use default URDF path if not provided
        if robot_urdf_path is None:
            robot_urdf_path = self._get_default_urdf_path()

        # ARX R5 dual arm manipulator configuration
        manipulator_config = {
            "right_arm": {
                "link_name": "right_link6",
                "pose_source": "right_controller",
                "control_trigger": "right_grip",
                "gripper_config": {
                    "type": "parallel",
                    "gripper_trigger": "right_trigger",
                    "joint_names": ["right_joint7"],
                    "open_pos": [4.9],
                    "close_pos": [0.0],
                },
            },
            "left_arm": {
                "link_name": "left_link6",
                "pose_source": "left_controller",
                "control_trigger": "left_grip",
                "gripper_config": {
                    "type": "parallel",
                    "gripper_trigger": "left_trigger",
                    "joint_names": ["left_joint7"],
                    "open_pos": [4.9],
                    "close_pos": [0.0],
                },
            },
        }

        # Initialize with default dual arm joint configuration
        # Placo state has 23 values: [xyz (3), quat (4), left_arm (6), left_gripper (2), right_arm (6), right_gripper (2)]
        q_init = None

        super().__init__(
            environment=environment,
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            floating_base=False,  # ARX R5 is fixed base
            R_headset_world=R_HEADSET_TO_WORLD,
            scale_factor=scale_factor,
            q_init=q_init,
            dt=1.0 / 50.0,  # 50Hz control rate
            enable_log_data=enable_log_data,
            log_dir=log_dir,
        )

        logging.info("ARXDualArmOpenPITeleopController initialized")

    def _get_default_urdf_path(self) -> str:
        """Get default URDF path for ARX R5 dual arm."""
        urdf_path = Path(ASSET_PATH) / "arx/R5a/dual_R5a.urdf"

        if urdf_path.exists():
            logging.info(f"Using URDF path: {urdf_path}")
            return str(urdf_path)

        logging.warning(f"URDF not found at {urdf_path}")
        return None

    def _update_robot_state(self):
        """Update robot state from ARX dual arm environment observations."""
        try:
            obs = self._environment.get_observation()
            joint_positions = np.array(obs["state"])

            # Map 14-element real robot state to 23-element Placo state
            # Real: [left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
            # Placo: [xyz+quat(7), left_arm (6), left_gripper (2), right_arm (6), right_gripper (2)]
            if len(joint_positions) == 14 and len(self.placo_robot.state.q) == 23:
                # Extract components from real robot state
                left_arm_joints = joint_positions[:6]
                left_gripper_value = joint_positions[6]
                right_arm_joints = joint_positions[7:13]
                right_gripper_value = joint_positions[13]

                # Map to Placo state, skipping the first 7 (floating base)
                self.placo_robot.state.q[7:13] = left_arm_joints
                self.placo_robot.state.q[13:15] = [left_gripper_value, left_gripper_value]
                self.placo_robot.state.q[15:21] = right_arm_joints
                self.placo_robot.state.q[21:23] = [right_gripper_value, right_gripper_value]

        except Exception as e:
            logging.debug(f"Error updating robot state from environment: {e}")

    def _placo_to_env_action(self) -> dict[str, Any]:
        """Convert Placo joint targets to ARX dual arm environment action format."""
        try:
            # Get joint targets from placo (arm joints only, no grippers)
            joint_targets = self.placo_robot.state.q.copy()

            # Expected action format for ARX dual arm environment:
            # action["actions"] = [left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
            # Total: 14 elements
            action_array = np.zeros(14)

            # Extract arm joint targets from placo (23 elements)
            # Placo state: [xyz+quat(7), left_arm (6), left_gripper (2), right_arm (6), right_gripper (2)]
            # Fill action array: [left_arm, left_gripper, right_arm, right_gripper]
            action_array[:6] = joint_targets[7:13]  # left arm
            action_array[7:13] = joint_targets[15:21]  # right arm

            # Add gripper commands
            for arm_name, gripper_idx in [("left_arm", 6), ("right_arm", 13)]:
                if arm_name in self.gripper_pos_target:
                    gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                    joint_name = gripper_config["joint_names"][0]
                    if joint_name in self.gripper_pos_target[arm_name]:
                        target = self.gripper_pos_target[arm_name][joint_name]
                        open_pos = gripper_config["open_pos"][0]
                        close_pos = gripper_config["close_pos"][0]
                        normalized = (target - open_pos) / (close_pos - open_pos)
                        action_array[gripper_idx] = np.clip(normalized, 0.0, 1.0)

            # Return action in the format expected by ARX dual arm environment
            return {"actions": action_array}

        except Exception as e:
            logging.error(f"Error converting placo to environment action: {e}")
            # Return safe zero action for dual arm (14 elements)
            return {"actions": np.zeros(14)}

    def _get_robot_state_for_logging(self) -> dict[str, Any]:
        """Returns a dictionary of robot-specific data for logging."""
        try:
            obs = self._environment.get_observation()
            joint_positions = np.array(obs["state"])

            # Split the 14-element state into arm-specific data
            if len(joint_positions) == 14:
                robot_state = {
                    "qpos": {
                        "left_arm": joint_positions[:6],
                        "left_gripper": joint_positions[6:7],
                        "right_arm": joint_positions[7:13],
                        "right_gripper": joint_positions[13:14],
                    },
                    "qpos_des": {
                        "left_arm": self.placo_robot.state.q[7:13].copy(),
                        "right_arm": self.placo_robot.state.q[15:21].copy(),
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
                "base_image": obs["base_image"],
                "right_wrist_image": obs["right_wrist_image"],
                "left_wrist_image": obs["left_wrist_image"],
            }

            return robot_state

        except Exception as e:
            logging.error(f"Error getting robot state for logging: {e}")
            return {"error": str(e)}
