import collections
import time
from typing import List, Optional

import dm_env
import numpy as np

from examples.arx_real.robot_utils import DEFAULT_BASE_CAM_SERIAL
from examples.arx_real.robot_utils import DEFAULT_CAN_PORTS
from examples.arx_real.robot_utils import DEFAULT_GRIPPER_OPEN
from examples.arx_real.robot_utils import DEFAULT_RIGHT_WRIST_CAM_SERIAL
from examples.arx_real.robot_utils import ARXR5Interface
from examples.arx_real.robot_utils import RealSenseCameraInterface

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
        can_port: str = "can0",
        camera_serial_numbers: Optional[List[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        enable_depth: bool = False,
    ):
        self.can_port = can_port

        # Initialize the ARX R5 interface
        self.robot = ARXR5Interface(can_port=can_port, dt=DT)

        # Initialize camera interface for single arm (right wrist and top cameras)
        self.camera_interface = RealSenseCameraInterface(
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            serial_numbers=camera_serial_numbers,
            enable_depth=enable_depth,
        )
        self.camera_interface.start()

        self.setup_robot()

    def setup_robot(self):
        """Initialize the robot to a ready state"""
        self.robot.go_home()
        time.sleep(1.0)

    def get_qpos(self):
        """Get current joint positions including gripper"""
        return np.array(self.robot.get_joint_positions(), dtype=np.float32)

    def get_qvel(self):
        """Get current joint velocities including gripper"""
        return np.array(self.robot.get_joint_velocities(), dtype=np.float32)

    def get_effort(self):
        """Get current joint efforts including gripper"""
        return np.array(self.robot.get_joint_currents(), dtype=np.float32)

    def get_images(self):
        """Get camera images from RealSense cameras"""
        self.camera_interface.update_frames()
        frames = self.camera_interface.get_frames()

        # Map camera serial numbers to standard camera names for single arm
        # Assuming the first camera is the wrist camera and second is the top camera
        images = {}

        # Map first camera to right wrist camera
        images["right_wrist"] = frames[DEFAULT_RIGHT_WRIST_CAM_SERIAL]["color"]

        # Map second camera to base camera
        images["base"] = frames[DEFAULT_BASE_CAM_SERIAL]["color"]

        return images

    def set_gripper_pose(self, gripper_desired_pos):
        self.robot.set_catch_pos(gripper_desired_pos)

    def _reset_joints(self):
        """Move arm to reset position"""
        self.robot.go_home()
        time.sleep(1.0)  # Wait for movement to complete

    def _reset_gripper(self):
        """Reset gripper to open position"""
        self.set_gripper_pose(DEFAULT_GRIPPER_OPEN)
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
        if len(action) != 7:
            raise ValueError(f"Action must have 7 elements (6 joints + 1 gripper), got {len(action)}")

        print(f"Action received: {action}")

        # Split action into arm and gripper components
        arm_action = action[:6]
        gripper_action = action[6]

        # Send commands to robot
        self.robot.set_joint_positions(arm_action)
        self.set_gripper_pose(gripper_action)

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
    can_port: str = DEFAULT_CAN_PORTS["right_arm"],
    camera_serial_numbers: Optional[List[str]] = [DEFAULT_RIGHT_WRIST_CAM_SERIAL, DEFAULT_BASE_CAM_SERIAL],
    camera_width: int = 424,
    camera_height: int = 240,
    camera_fps: int = 60,
    enable_depth: bool = False,
) -> RealEnv:
    """Factory function to create RealEnv instance"""
    return RealEnv(
        can_port=can_port,
        camera_serial_numbers=camera_serial_numbers,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        enable_depth=enable_depth,
    )
