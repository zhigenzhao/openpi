import collections
import time
from typing import List, Optional

import dm_env
import numpy as np

from examples.arx_r5.robot_utils import DEFAULT_BASE_CAM_SERIAL
from examples.arx_r5.robot_utils import DEFAULT_CAN_PORTS
from examples.arx_r5.robot_utils import DEFAULT_GRIPPER_CLOSE
from examples.arx_r5.robot_utils import DEFAULT_GRIPPER_OPEN
from examples.arx_r5.robot_utils import DEFAULT_LEFT_WRIST_CAM_SERIAL
from examples.arx_r5.robot_utils import DEFAULT_RIGHT_WRIST_CAM_SERIAL
from examples.arx_r5.robot_utils import ARXR5Interface
from examples.arx_r5.robot_utils import RealSenseCameraInterface

DT = 0.02  # Control frequency


class RealEnv:
    """
    Environment for real robot single-arm manipulation with ARX R5
    Action space:      [arm_qpos (6), gripper_position (1)] # absolute joint position

    Observation space: {"qpos": Concat[arm_qpos (6), gripper_position (1)],
                        "qvel": Concat[arm_qvel (6), gripper_velocity (1)],
                        "effort": Concat[arm_effort (6), gripper_effort (1)],
                        "images": {"cam_right_wrist": (H, W, C), "cam_top": (H, W, C)}}
    """

    def __init__(
        self,
        can_port_left: str = "can1",
        can_port_right: str = "can3",
        camera_serial_numbers: Optional[List[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        enable_depth: bool = False,
    ):
        self.can_port_left = can_port_left
        self.can_port_right = can_port_right

        # Initialize camera interface for single arm (right wrist and top cameras)
        self.camera_interface = RealSenseCameraInterface(
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            serial_numbers=camera_serial_numbers,
            enable_depth=enable_depth,
        )
        self.camera_interface.start()

        # Initialize the ARX R5 interface
        self.robot_left = ARXR5Interface(can_port=can_port_left, dt=DT)
        self.robot_right = ARXR5Interface(can_port=can_port_right, dt=DT)

        self.reset()
        time.sleep(1.0)  # Allow time for the robot to initialize

    def get_qpos(self):
        """Get current joint positions including gripper"""
        left_qpos = np.array(self.robot_left.get_joint_positions(), dtype=np.float32)
        left_qpos[6] = 1 - left_qpos[6] / (DEFAULT_GRIPPER_OPEN - DEFAULT_GRIPPER_CLOSE)
        right_qpos = np.array(self.robot_right.get_joint_positions(), dtype=np.float32)
        right_qpos[6] = 1 - right_qpos[6] / (DEFAULT_GRIPPER_OPEN - DEFAULT_GRIPPER_CLOSE)
        return np.concatenate([left_qpos, right_qpos])

    def get_qvel(self):
        """Get current joint velocities including gripper"""
        left_qvel = np.array(self.robot_left.get_joint_velocities(), dtype=np.float32)
        right_qvel = np.array(self.robot_right.get_joint_velocities(), dtype=np.float32)
        return np.concatenate([left_qvel, right_qvel])

    def get_effort(self):
        """Get current joint efforts including gripper"""
        left_effort = np.array(self.robot_left.get_joint_currents(), dtype=np.float32)
        right_effort = np.array(self.robot_right.get_joint_currents(), dtype=np.float32)
        return np.concatenate([left_effort, right_effort])

    def get_images(self):
        """Get camera images from RealSense cameras"""
        self.camera_interface.update_frames()
        frames = self.camera_interface.get_frames()

        # Map camera serial numbers to standard camera names for single arm
        # Assuming the first camera is the wrist camera and second is the top camera
        images = {}

        # Map first camera to right wrist camera
        images["right_wrist"] = frames[DEFAULT_RIGHT_WRIST_CAM_SERIAL]["color"]
        images["left_wrist"] = frames[DEFAULT_LEFT_WRIST_CAM_SERIAL]["color"]

        # Map second camera to base camera
        images["base"] = frames[DEFAULT_BASE_CAM_SERIAL]["color"]

        return images

    def set_gripper_pose(self, left_gripper_desired_pos_normalized: float, right_gripper_desired_pos_normalized: float):
        left_gripper_desired_pos = (
            left_gripper_desired_pos_normalized * (DEFAULT_GRIPPER_CLOSE - DEFAULT_GRIPPER_OPEN)
        ) + DEFAULT_GRIPPER_OPEN
        right_gripper_desired_pos = (
            right_gripper_desired_pos_normalized * (DEFAULT_GRIPPER_CLOSE - DEFAULT_GRIPPER_OPEN)
        ) + DEFAULT_GRIPPER_OPEN
        self.robot_left.set_catch_pos(left_gripper_desired_pos)
        self.robot_right.set_catch_pos(right_gripper_desired_pos)

    def _reset_joints(self):
        """Move arm to reset position"""
        self.robot_left.go_home()
        self.robot_right.go_home()
        time.sleep(1.0)  # Wait for movement to complete

    def _reset_gripper(self):
        """Reset gripper to open position"""
        self.robot_left.set_catch_pos(DEFAULT_GRIPPER_OPEN)
        self.robot_right.set_catch_pos(DEFAULT_GRIPPER_OPEN)
        time.sleep(1.0)

    def get_observation(self):
        """Get full observation including state and images"""
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        return obs

    def get_reward(self):
        """Get reward (placeholder)"""
        return 0

    def reset(self, *, fake=False):
        """Reset the environment"""
        if not fake:
            self._reset_joints()
            self._reset_gripper()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        """Execute one step with the given action"""
        if len(action) != 14:
            raise ValueError(f"Action must have 14 elements (12 joints + 2 grippers), got {len(action)}")

        # print(f"Action received: {action}")

        # Split action into arm and gripper components
        left_arm_action = action[:6]
        right_arm_action = action[7:13]
        # left_gripper_action_normalized = 0 if action[6] < 0.99 else 1
        # right_gripper_action_normalized = 0 if action[13] < 0.99 else 1
        left_gripper_action_normalized = action[6]
        right_gripper_action_normalized = action[13]

        # Send commands to robot
        self.robot_left.set_joint_positions(left_arm_action)
        self.robot_right.set_joint_positions(right_arm_action)
        self.set_gripper_pose(left_gripper_action_normalized, right_gripper_action_normalized)

        # Wait for control step
        time.sleep(DT)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, "camera_interface"):
            self.camera_interface.stop()


def make_real_env(
    can_port_left: str = DEFAULT_CAN_PORTS["left_arm"],
    can_port_right: str = DEFAULT_CAN_PORTS["right_arm"],
    camera_serial_numbers: Optional[List[str]] = [
        DEFAULT_LEFT_WRIST_CAM_SERIAL,
        DEFAULT_RIGHT_WRIST_CAM_SERIAL,
        DEFAULT_BASE_CAM_SERIAL,
    ],
    camera_width: int = 424,
    camera_height: int = 240,
    camera_fps: int = 60,
    enable_depth: bool = False,
) -> RealEnv:
    """Factory function to create RealEnv instance"""
    return RealEnv(
        can_port_left=can_port_left,
        can_port_right=can_port_right,
        camera_serial_numbers=camera_serial_numbers,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        enable_depth=enable_depth,
    )
