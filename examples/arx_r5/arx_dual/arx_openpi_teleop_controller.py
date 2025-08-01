"""ARX R5 Dual Arm OpenPI Teleoperation Controller.

Implementation of OpenPITeleopController specifically for ARX R5 dual arm robots
using OpenPI's ARX dual arm environment system.
"""

import logging
import os
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
        # Use ASSET_PATH to find URDF
        urdf_path = os.path.join(ASSET_PATH, "arx/R5a/dual_R5a.urdf")

        if os.path.exists(urdf_path):
            logging.info(f"Using URDF path: {urdf_path}")
            return urdf_path

        logging.warning(f"URDF not found at {urdf_path}")
        return None

    def _update_robot_state(self):
        """Update robot state from ARX dual arm environment observations."""
        try:
            # Get current observation from environment
            obs = self._environment.get_observation()

            # Extract joint positions from ARX dual arm environment observation
            # Based on env.py: obs["state"] contains the qpos from real_env
            # Based on real_env.py: qpos is concatenated [left_qpos (7), right_qpos (7)] = 14 total
            joint_positions = obs["state"]

            if joint_positions is not None:
                joint_positions = np.array(joint_positions)

                # Expected format: [left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
                if len(joint_positions) == 14:
                    # Map 14-element real robot state to 23-element Placo state
                    # Real: [left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
                    # Placo: [xyz+quat(7), left_arm (6), left_gripper (2), right_arm (6), right_gripper (2)]
                    if len(self.placo_robot.state.q) == 23:
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
                    else:
                        # Fallback: update whatever we can
                        logging.debug(f"Unexpected Placo state size: {len(self.placo_robot.state.q)}")
                else:
                    logging.debug(f"Unexpected joint position size: expected 14, got {len(joint_positions)}")
            else:
                logging.debug("Could not extract joint positions from observation")

        except Exception as e:
            logging.debug(f"Error updating robot state from environment: {e}")

    def set_robot_state(self, action):
        """Set the robot's internal state from a policy action.

        Args:
            action: Policy action in ARX dual arm environment format.
                   Expected: dict with "actions" key containing 14-element array
                   [left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
        """
        try:
            # Extract action array - handle different action formats
            action_array = None
            if isinstance(action, dict):
                if "actions" in action:
                    action_array = action["actions"]
                elif "action" in action:
                    action_array = action["action"]
                else:
                    # Try to find any array-like value in the dict
                    for key, value in action.items():
                        if isinstance(value, (list, tuple)) or (hasattr(value, "__array__")):
                            action_array = value
                            break
            elif isinstance(action, (list, tuple)) or (hasattr(action, "__array__")):
                action_array = action

            if action_array is None:
                logging.warning(f"Could not extract action array from action: {type(action)}")
                return

            # Convert to numpy array
            action_array = np.array(action_array)

            # Update placo robot state based on action format
            # Expected action format: [left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)] = 14 elements
            # Placo state format: [xyz+quat(7), left_arm (6), left_gripper (2), right_arm (6), right_gripper (2)] = 23 elements

            if len(action_array) == 14 and len(self.placo_robot.state.q) == 23:
                logging.info("Setting placo robot state from 14-element action")

                # Extract components from action
                left_arm_joints = action_array[:6]
                left_gripper_value = action_array[6]
                right_arm_joints = action_array[7:13]
                right_gripper_value = action_array[13]

                # Map to Placo state (skip first 7 elements which are base pose)
                self.placo_robot.state.q[7:13] = left_arm_joints
                self.placo_robot.state.q[13:15] = [left_gripper_value, left_gripper_value]
                self.placo_robot.state.q[15:21] = right_arm_joints
                self.placo_robot.state.q[21:23] = [right_gripper_value, right_gripper_value]

                logging.info("Successfully set placo robot state from policy action")

            else:
                logging.warning(
                    f"Action/state size mismatch: action={len(action_array)}, placo_state={len(self.placo_robot.state.q)}"
                )
                # Fallback: try to map what we can
                min_len = min(len(action_array), len(self.placo_robot.state.q))
                if min_len > 0:
                    self.placo_robot.state.q[:min_len] = action_array[:min_len]
                    logging.info(f"Fallback: set first {min_len} elements of placo robot state")

        except Exception as e:
            logging.error(f"Error setting robot state from action: {e}")
            logging.debug(f"Action type: {type(action)}, Action content: {action}")

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
            if len(joint_targets) == 23:
                left_arm_joints = joint_targets[7:13]
                right_arm_joints = joint_targets[15:21]

                # Fill action array: [left_arm, left_gripper, right_arm, right_gripper]
                action_array[:6] = left_arm_joints
                action_array[7:13] = right_arm_joints
            else:
                logging.warning(f"Unexpected placo joint target size: {len(joint_targets)}")

            # Add gripper commands
            # Left gripper at index 6
            if "left_arm" in self.gripper_pos_target:
                left_gripper_config = self.manipulator_config["left_arm"]["gripper_config"]
                left_joint_name = left_gripper_config["joint_names"][0]
                if left_joint_name in self.gripper_pos_target["left_arm"]:
                    left_gripper_target = self.gripper_pos_target["left_arm"][left_joint_name]
                    # Convert gripper position to normalized value (0=open, 1=closed)
                    left_open_pos = left_gripper_config["open_pos"][0]
                    left_close_pos = left_gripper_config["close_pos"][0]
                    left_normalized = (left_gripper_target - left_open_pos) / (left_close_pos - left_open_pos)
                    action_array[6] = np.clip(left_normalized, 0.0, 1.0)

            # Right gripper at index 13
            if "right_arm" in self.gripper_pos_target:
                right_gripper_config = self.manipulator_config["right_arm"]["gripper_config"]
                right_joint_name = right_gripper_config["joint_names"][0]
                if right_joint_name in self.gripper_pos_target["right_arm"]:
                    right_gripper_target = self.gripper_pos_target["right_arm"][right_joint_name]
                    # Convert gripper position to normalized value (0=open, 1=closed)
                    right_open_pos = right_gripper_config["open_pos"][0]
                    right_close_pos = right_gripper_config["close_pos"][0]
                    right_normalized = (right_gripper_target - right_open_pos) / (right_close_pos - right_open_pos)
                    action_array[13] = np.clip(right_normalized, 0.0, 1.0)

            # Return action in the format expected by ARX dual arm environment
            return {"actions": action_array}

        except Exception as e:
            logging.error(f"Error converting placo to environment action: {e}")
            # Return safe zero action for dual arm (14 elements)
            return {"actions": np.zeros(14)}

    def get_current_joint_positions(self) -> dict[str, np.ndarray]:
        """Get current joint positions for both arms."""
        try:
            obs = self._environment.get_observation()

            if "state" in obs and len(obs["state"]) == 14:
                joint_positions = np.array(obs["state"])
                return {
                    "left_arm": joint_positions[:6],  # left arm joints
                    "left_gripper": joint_positions[6:7],  # left gripper
                    "right_arm": joint_positions[7:13],  # right arm joints
                    "right_gripper": joint_positions[13:14],  # right gripper
                }
            logging.warning("Could not extract joint positions from observation")
            return {
                "left_arm": np.zeros(6),
                "left_gripper": np.zeros(1),
                "right_arm": np.zeros(6),
                "right_gripper": np.zeros(1),
            }
        except Exception as e:
            logging.error(f"Error getting current joint positions: {e}")
            return {
                "left_arm": np.zeros(6),
                "left_gripper": np.zeros(1),
                "right_arm": np.zeros(6),
                "right_gripper": np.zeros(1),
            }

    def get_current_end_effector_poses(self) -> dict[str, dict[str, np.ndarray]]:
        """Get current end-effector poses for both arms."""
        poses = {}

        for arm_name, config in self.manipulator_config.items():
            link_name = config["link_name"]
            try:
                position, quaternion = self._get_link_pose(link_name)
                poses[arm_name] = {
                    "position": position,
                    "quaternion": quaternion,
                }
            except Exception as e:
                logging.error(f"Error getting pose for {arm_name}: {e}")
                poses[arm_name] = {
                    "position": np.zeros(3),
                    "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
                }

        return poses
