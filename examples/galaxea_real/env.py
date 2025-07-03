from typing import Any, Dict, List, Optional

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
import rospy
from typing_extensions import override

from examples.galaxea_real import robot_utils


class GalaxeaRealEnv:
    """
    Environment for real robot single-arm manipulation
    Action space:      [arm_qpos (6), gripper_qpos (1)] # absolute joint position
    Observation space: {"state": Concat[arm_qpos (6), gripper_qpos (1)],
                        "images": {"cam_1": (H, W, C), "cam_2": ...}
                       }
    """

    def __init__(self, camera_serials: Optional[List[str]] = None):
        rospy.init_node("galaxea_real_env", anonymous=True)
        self.robot = robot_utils.A1XController()
        self.cameras = robot_utils.RealSenseCameraInterface(serial_numbers=camera_serials)
        self.cameras.start()
        # Allow some time for ROS connections and first messages
        rospy.sleep(1.0)

    def get_observation(self) -> Dict[str, Any]:
        """
        Get the current observation from the robot and cameras.
        """
        # Get robot state
        qpos = np.array(self.robot.qpos, dtype=np.float32)
        gripper_qpos = np.array(self.robot.qpos_gripper, dtype=np.float32)
        state = np.concatenate([qpos, gripper_qpos])

        # Get images
        self.cameras.update_frames()
        frames = self.cameras.get_frames()
        images = {f"cam_{serial.replace('-', '_')}": frame["color"] for serial, frame in frames.items()}

        return {"state": state, "images": images}

    def apply_action(self, action: np.ndarray) -> None:
        """
        Apply an action to the robot.
        The action is a 7-element array: 6 for arm joint positions, 1 for gripper position.
        """
        if not isinstance(action, np.ndarray) or action.shape != (7,):
            raise ValueError("Action must be a 7-element numpy array.")

        arm_action = action[:6]
        gripper_action = [action[6]]

        self.robot.q_des = arm_action.tolist()
        self.robot.q_des_gripper = gripper_action

        self.robot.publish_arm_control()
        self.robot.publish_gripper_control()

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment. For a real robot, this might involve moving to a home position.
        For now, it just returns the current observation.
        """
        # In a real scenario, you might want to move the robot to a known reset position.
        # self.robot.q_des = [0.0] * 6
        # self.robot.q_des_gripper = [-2.1] # Open gripper
        # self.robot.publish_arm_control()
        # self.robot.publish_gripper_control()
        # rospy.sleep(2.0) # Wait for robot to reach home
        return self.get_observation()

    def close(self):
        self.cameras.stop()


class GalaxeaEnvironment(_environment.Environment):
    """An environment for a Galaxea robot on real hardware."""

    def __init__(
        self,
        camera_serials: Optional[List[str]] = None,
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._env = GalaxeaRealEnv(camera_serials=camera_serials)
        self._render_height = render_height
        self._render_width = render_width
        self._obs = None

    @override
    def reset(self) -> None:
        self._obs = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        # This might be determined by a specific task condition in a real scenario
        return False

    @override
    def get_observation(self) -> dict:
        if self._obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        # Process images for the policy
        images = {}
        for cam_name, img in self._obs["images"].items():
            img_resized = image_tools.resize_with_pad(img, self._render_height, self._render_width)
            img_uint8 = image_tools.convert_to_uint8(img_resized)
            images[cam_name] = einops.rearrange(img_uint8, "h w c -> c h w")

        return {
            "state": self._obs["state"],
            "images": images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        # The policy outputs a dictionary, we extract the 'actions' part for the environment
        self._env.apply_action(action["actions"])
        # Get the next observation after applying the action
        self._obs = self._env.get_observation()
