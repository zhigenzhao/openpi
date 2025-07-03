import threading
import time

from hdas_msg.msg import motor_control  # Import the custom message
import numpy as np
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header  # For the header field

DEFAULT_LEFT_WRIST_CAM_SERIAL = "218622272014"
DEFAULT_RIGHT_WRIST_CAM_SERIAL = "218622272499"
DEFAULT_BASE_CAM_SERIAL = "215222077461"

CAM_SERIAL_DICT = {
    "left_wrist": DEFAULT_LEFT_WRIST_CAM_SERIAL,
    "right_wrist": DEFAULT_RIGHT_WRIST_CAM_SERIAL,
    "base": DEFAULT_BASE_CAM_SERIAL,
}


class A1XController:
    def __init__(
        self,
        arm_control_topic: str = "/motion_control/control_arm",
        gripper_control_topic: str = "/motion_control/control_gripper",
        arm_state_topic: str = "/hdas/feedback_arm",
        rate_hz: float = 100,
    ):
        self.pub = rospy.Publisher(arm_control_topic, motor_control, queue_size=1)
        self.gripper_pub = rospy.Publisher(gripper_control_topic, motor_control, queue_size=1)
        self.sub = rospy.Subscriber(arm_state_topic, JointState, self.arm_state_callback)
        self.rate = rospy.Rate(rate_hz)

        self.qpos = [0.0] * 6
        self.qvel = [0.0] * 6
        self.qpos_gripper = 0.0
        self.qvel_gripper = 0.0
        self.timestamp = 0.0

        # motor control parameters
        self.q_des = None
        self.v_des = [0.0] * 6
        # self.kp = [2000, 2000, 1000, 200, 200, 200]
        # self.kd = [200.0, 500.0, 500, 200, 200, 200]
        # self.kp = [200, 200, 150, 100, 100, 100]
        # self.kd = [20.0, 30.0, 10, 10, 10, 10]
        self.kp = [140, 200, 120, 80, 80, 80]
        self.kd = [10, 30, 5, 10, 10, 10]
        self.t_ff = [0.0] * 6
        self.arm_ctrl_msg = motor_control()

        self.q_des_gripper = [-2.1]
        self.v_des_gripper = [0.0]
        self.kp_gripper = [1]
        self.kd_gripper = [0.05]
        self.t_ff_gripper = [0.0]
        self.gripper_ctrl_msg = motor_control()

    def arm_state_callback(self, msg: JointState):
        """
        Callback function to handle joint state updates.
        """
        self.qpos = msg.position[:6]
        self.qvel = msg.velocity[:6]
        self.qpos_gripper = [msg.position[6]]
        self.qvel_gripper = [msg.velocity[6]]
        if self.q_des is None:
            self.q_des = self.qpos
        self.timestamp = msg.header.stamp.to_sec()
        # print(f"Received joint state: {self.qpos}, {self.qvel} at time {self.timestamp}")
        # print(f"Gripper state: {self.qpos_gripper}, {self.qvel_gripper}")

    def publish_arm_control(self):
        """
        Publishes motor control messages to the /motion_control/control_arm topic.
        """
        if self.q_des is None:
            return

        self.arm_ctrl_msg.header = Header()
        self.arm_ctrl_msg.header.stamp = rospy.Time.now()
        self.arm_ctrl_msg.header.frame_id = "base_link"
        self.arm_ctrl_msg.kp = self.kp
        self.arm_ctrl_msg.kd = self.kd
        self.arm_ctrl_msg.t_ff = self.t_ff
        self.arm_ctrl_msg.p_des = self.q_des
        self.arm_ctrl_msg.v_des = self.v_des

        self.pub.publish(self.arm_ctrl_msg)

    def publish_gripper_control(self):
        self.gripper_ctrl_msg.header = Header()
        self.gripper_ctrl_msg.header.stamp = rospy.Time.now()
        self.gripper_ctrl_msg.header.frame_id = "gripper_link"
        self.gripper_ctrl_msg.kp = self.kp_gripper
        self.gripper_ctrl_msg.kd = self.kd_gripper
        self.gripper_ctrl_msg.t_ff = self.t_ff_gripper
        self.gripper_ctrl_msg.p_des = self.q_des_gripper
        self.gripper_ctrl_msg.v_des = self.v_des_gripper

        self.gripper_pub.publish(self.gripper_ctrl_msg)

    def run(self):
        """
        Main loop to run the controller.
        """
        while not rospy.is_shutdown():
            self.publish_motor_control()
            self.rate.sleep()


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

    def start(self):
        """Starts the camera pipelines."""
        for serial, pipeline in self.pipelines.items():
            pipeline.start(self.configs[serial])
            print(f"Started pipeline for camera: {serial}")

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
