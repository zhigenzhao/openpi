import os
import threading
import time
from typing import List, Optional, Tuple, Union

import arx_r5_python.arx_r5_python as arx
import meshcat.transformations as tf
import numpy as np
import pyrealsense2 as rs

# Default camera configuration
DEFAULT_RIGHT_WRIST_CAM_SERIAL = "218622272014"
DEFAULT_LEFT_WRIST_CAM_SERIAL = "218622272499"
DEFAULT_BASE_CAM_SERIAL = "215222077461"

CAM_SERIAL_DICT = {
    "left_wrist": DEFAULT_LEFT_WRIST_CAM_SERIAL,
    "right_wrist": DEFAULT_RIGHT_WRIST_CAM_SERIAL,
    "base": DEFAULT_BASE_CAM_SERIAL,
}

DEFAULT_CAN_PORTS = {
    "left_arm": "can1",
    "right_arm": "can3",
}
# Default reset position for the ARX R5 arm
DEFAULT_GRIPPER_OPEN = 4.8
DEFAULT_GRIPPER_CLOSE = 0.0


class ARXR5Interface:
    """
    Base class for a single robot arm.

    Args:
        config (Dict[str, sAny]): Configuration dictionary for the robot arm

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the robot arm
        num_joints (int): Number of joints in the arm
    """

    def __init__(
        self,
        can_port: str = "can0",
        dt: float = 0.01,
    ):
        file_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(file_path, "assets/R5a/R5a.urdf")
        self.arm = arx.InterfacesPy(urdf_path, can_port, 0)
        self.arm.arx_x(500, 2000, 10)

    def get_joint_names(self) -> List[str]:
        """
        Get the names of all joints in the arm.

        Returns:
            List[str]: List of joint names. Shape: (num_joints,)
        """
        return NotImplementedError

    def go_home(self) -> bool:
        """
        Move the robot arm to a pre-defined home pose.

        Returns:
            bool: True if the action was successful, False otherwise
        """
        self.arm.set_arm_status(1)
        return True

    def gravity_compensation(self) -> bool:
        self.arm.set_arm_status(3)
        return True

    def protect_mode(self) -> bool:
        self.arm.set_arm_status(2)
        return True

    def set_joint_positions(
        self,
        positions: Union[float, List[float], np.ndarray],
        **kwargs,  # Shape: (num_joints,)
    ) -> bool:
        """
        Move the arm to the given joint position(s).

        Args:
            positions: Desired joint position(s). Shape: (6)
            **kwargs: Additional arguments

        """
        self.arm.set_joint_positions(positions)
        self.arm.set_arm_status(5)

    def set_ee_pose(
        self,
        pos: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (3,)
        quat: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (4,)
        **kwargs,
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            pos: Desired position [x, y, z]. Shape: (3,)
            ori: Desired orientation (quaternion).
                 Shape: (4,) (w, x, y, z)
            **kwargs: Additional arguments

        """

        self.arm.set_ee_pose([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
        self.arm.set_arm_status(4)

    def set_ee_pose_xyzrpy(
        self,
        xyzrpy: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs,  # Shape: (6,)
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            xyzrpy: Desired position [x, y, z, rol, pitch, yaw]. Shape: (6,)
            **kwargs: Additional arguments

        """
        quat = tf.quaternion_from_euler(xyzrpy[3], xyzrpy[4], xyzrpy[5])

        self.arm.set_ee_pose([xyzrpy[0], xyzrpy[1], xyzrpy[2], quat[0], quat[1], quat[2], quat[3]])
        self.arm.set_arm_status(4)

    def set_catch_pos(self, pos: float):
        self.arm.set_catch(pos)
        self.arm.set_arm_status(5)

    def get_joint_positions(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        """
        Get the current joint position(s) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get positions for. Shape: (num_joints,) or single string. If None,
                            return positions for all joints.

        """
        return self.arm.get_joint_positions()

    def get_joint_velocities(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        """
        Get the current joint velocity(ies) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get velocities for. Shape: (num_joints,) or single string. If None,
                            return velocities for all joints.

        """
        return self.arm.get_joint_velocities()

    def get_joint_currents(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        return self.arm.get_joint_currents()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current end effector pose of the arm.

        Returns:
            End effector pose as (position, quaternion)
            Shapes: position (3,), quaternion (4,) [w, x, y, z]
        """
        xyzwxyz = self.arm.get_ee_pose()

        return xyzwxyz

    def get_ee_pose_xyzrpy(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xyzwxyz = self.arm.get_ee_pose()

        array = np.array([xyzwxyz[3], xyzwxyz[4], xyzwxyz[5], xyzwxyz[6]])

        roll, pitch, yaw = tf.quaternion_from_euler(array)

        xyzrpy = np.array([xyzwxyz[0], xyzwxyz[1], xyzwxyz[2], roll, pitch, yaw])

        return xyzrpy

    def __del__(self):
        print("ARXR5Interface is being deleted")


class RealSenseCameraInterface:
    """
    An interface to handle one or more Intel RealSense cameras.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_numbers: list[str] = None,
        enable_depth: bool = True,
    ):
        """
        Initializes the RealSense camera interface.

        Args:
            width (int): The width of the camera streams.
            height (int): The height of the camera streams.
            fps (int): The frames per second of the camera streams.
            serial_numbers (list[str], optional): A list of serial numbers of the cameras to use.
                                                  If None, all connected cameras will be used.
            enable_depth (bool): Whether to enable the depth stream.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_numbers = serial_numbers
        self.enable_depth = enable_depth
        self.pipelines = {}
        self.configs = {}
        self.align = {}

        self.frames_dict = {}
        self.frames_lock = threading.Lock()  # Thread-safe access to frames
        self.last_update_time = {}  # Track last successful frame update per camera

        self.context = rs.context()
        devices = self.context.query_devices()

        if not devices:
            raise RuntimeError("No Intel RealSense devices connected.")

        device_serials = [d.get_info(rs.camera_info.serial_number) for d in devices]

        if self.serial_numbers:
            # Filter for specified serial numbers
            self.active_serials = [s for s in self.serial_numbers if s in device_serials]
            if not self.active_serials:
                raise RuntimeError(f"Specified RealSense devices with serials {self.serial_numbers} not found.")
        else:
            # Use all connected devices
            self.active_serials = device_serials

        for serial in self.active_serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            if self.enable_depth:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            self.pipelines[serial] = pipeline
            self.configs[serial] = config
            # Align depth frames to color frames
            if self.enable_depth:
                self.align[serial] = rs.align(rs.stream.color)
            self.last_update_time[serial] = 0
            print(f"Initialized RealSense camera: {serial}")

        self.started = False

    def start(self):
        """Starts the camera pipelines."""
        for serial, pipeline in self.pipelines.items():
            pipeline.start(self.configs[serial])
            print(f"Started pipeline for camera: {serial}")
        self.started = True

    def update_frames(self):
        """
        Fetches and returns frames from all cameras with improved error handling and timeout management.

        Returns:
            dict: A dictionary where keys are camera serial numbers and values are
                  another dictionary containing 'color' and 'depth' numpy arrays,
                  the 'timestamp_us' of the frame, and stream format information.
        """
        current_time = time.time()
        frames_dict = {}

        for serial, pipeline in self.pipelines.items():
            try:
                # Use shorter timeout and exponential backoff on failures
                timeout_ms = 500  # Reduced timeout to prevent blocking
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)

                color_frame = None
                depth_frame = None

                if self.enable_depth:
                    aligned_frames = self.align[serial].process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()

                if not color_frame:
                    print(f"Warning: No color frame available from camera {serial}")
                    continue

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None

                frames_dict[serial] = {
                    "color": color_image,
                    "depth": depth_image,
                    "timestamp_us": color_frame.get_timestamp(),  # microseconds
                    "color_format": color_frame.get_profile().format(),
                    "depth_format": depth_frame.get_profile().format() if depth_frame else None,
                }

                self.last_update_time[serial] = current_time

            except RuntimeError as e:
                # Handle timeout more gracefully
                if "timeout" in str(e).lower():
                    print(
                        f"Frame timeout for camera {serial} (last successful: {current_time - self.last_update_time[serial]:.2f}s ago)"
                    )
                else:
                    print(f"Error getting frames from {serial}: {e}")
                continue

        # Thread-safe update of frames dictionary
        with self.frames_lock:
            self.frames_dict = frames_dict

    def get_frames(self):
        """
        Fetches frames from all initialized cameras (thread-safe).

        Returns:
            dict: A dictionary where keys are camera serial numbers and values are
                  another dictionary containing 'color' and 'depth' numpy arrays,
                  the 'timestamp_us' of the frame, and stream format information.
        """
        with self.frames_lock:
            return self.frames_dict.copy()

    def get_frame(self, serial: str):
        """
        Fetches frames from a specific camera (thread-safe).

        Args:
            serial (str): The serial number of the camera.

        Returns:
            dict: A dictionary containing 'color' and 'depth' numpy arrays,
                  the 'timestamp_us' of the frame, and stream format information.
        """
        with self.frames_lock:
            return self.frames_dict[serial].copy() if serial in self.frames_dict else None

    def stop(self):
        """Stops the camera pipelines."""
        for serial, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
                print(f"Stopped pipeline for camera: {serial}")
            except RuntimeError as e:
                print(f"Error stopping pipeline for {serial}: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
