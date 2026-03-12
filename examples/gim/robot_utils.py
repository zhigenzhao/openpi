import threading
import time

import numpy as np
import pyrealsense2 as rs

DEFAULT_CAN_PORTS = {"left_arm": "can1", "right_arm": "can0"}
DEFAULT_LEFT_WRIST_CAM_SERIAL = "335122271855"
DEFAULT_RIGHT_WRIST_CAM_SERIAL = "335122271104"
DEFAULT_TOP_CAM_SERIAL = "315122272794"
DEFAULT_GRIPPER_OPEN = -0.20  # joint space open
DEFAULT_GRIPPER_CLOSE = 0.0  # joint space close


class RealSenseCameraInterface:
    """
    An interface to handle one or more Intel RealSense cameras.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_numbers: list[str] | None = None,
        *,
        enable_depth: bool = True,
        auto_exposure: bool = True,
        exposure: int | None = None,
        gain: int | None = None,
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
            auto_exposure (bool): Enable automatic exposure control (default: True).
            exposure (int, optional): Manual exposure value (1-10000). Only used if auto_exposure is False.
            gain (int, optional): Camera gain value (16-248 typical). Only used if auto_exposure is False.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_numbers = serial_numbers
        self.enable_depth = enable_depth
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        self.gain = gain
        self.pipelines = {}
        self.configs = {}
        self.align = {}
        self.sensors = {}  # Store RGB sensors for exposure control

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

        # Apply exposure settings after starting pipelines
        self._set_exposure_settings()

    def _set_exposure_settings(self):
        """
        Configures exposure settings for all cameras after pipelines have started.
        This method retrieves the RGB sensor for each camera and applies the specified
        exposure, gain, and auto-exposure settings.
        """
        for serial, pipeline in self.pipelines.items():
            try:
                # Get the device from the pipeline's active profile
                profile = pipeline.get_active_profile()
                device = profile.get_device()

                # Find the RGB sensor
                rgb_sensor = None
                for sensor in device.sensors:
                    # Check if this sensor supports exposure options
                    if rs.option.exposure in sensor.get_supported_options():
                        rgb_sensor = sensor
                        break

                if rgb_sensor is None:
                    print(f"Warning: Could not find RGB sensor with exposure control for camera {serial}")
                    continue

                # Store sensor reference for runtime adjustments
                self.sensors[serial] = rgb_sensor

                # Set auto-exposure
                if rs.option.enable_auto_exposure in rgb_sensor.get_supported_options():
                    rgb_sensor.set_option(rs.option.enable_auto_exposure, self.auto_exposure)
                    print(f"Camera {serial}: Auto-exposure set to {self.auto_exposure}")

                # Set manual exposure if auto-exposure is disabled
                if not self.auto_exposure:
                    if self.exposure is not None and rs.option.exposure in rgb_sensor.get_supported_options():
                        rgb_sensor.set_option(rs.option.exposure, self.exposure)
                        print(f"Camera {serial}: Exposure set to {self.exposure}")

                    if self.gain is not None and rs.option.gain in rgb_sensor.get_supported_options():
                        rgb_sensor.set_option(rs.option.gain, self.gain)
                        print(f"Camera {serial}: Gain set to {self.gain}")

            except Exception as e:
                print(f"Error setting exposure for camera {serial}: {e}")

    def set_exposure(
        self,
        serial: str | None = None,
        *,
        auto_exposure: bool | None = None,
        exposure: int | None = None,
        gain: int | None = None,
    ) -> bool:
        """
        Dynamically adjust exposure settings for one or all cameras at runtime.

        Args:
            serial (str, optional): Serial number of the camera to adjust. If None, applies to all cameras.
            auto_exposure (bool, optional): Enable/disable automatic exposure.
            exposure (int, optional): Manual exposure value (1-10000).
            gain (int, optional): Camera gain value.

        Returns:
            bool: True if settings were applied successfully, False otherwise.
        """
        target_serials = [serial] if serial else list(self.sensors.keys())

        success = True
        for cam_serial in target_serials:
            if cam_serial not in self.sensors:
                print(f"Warning: Camera {cam_serial} sensor not found. Cannot adjust exposure.")
                success = False
                continue

            try:
                rgb_sensor = self.sensors[cam_serial]

                # Set auto-exposure if specified
                if auto_exposure is not None:
                    if rs.option.enable_auto_exposure in rgb_sensor.get_supported_options():
                        rgb_sensor.set_option(rs.option.enable_auto_exposure, auto_exposure)
                        print(f"Camera {cam_serial}: Auto-exposure set to {auto_exposure}")
                    else:
                        print(f"Warning: Auto-exposure not supported on camera {cam_serial}")

                # Set manual exposure if specified and auto-exposure is off
                current_auto = rgb_sensor.get_option(rs.option.enable_auto_exposure)
                if not current_auto:
                    if exposure is not None and rs.option.exposure in rgb_sensor.get_supported_options():
                        rgb_sensor.set_option(rs.option.exposure, exposure)
                        print(f"Camera {cam_serial}: Exposure set to {exposure}")

                    if gain is not None and rs.option.gain in rgb_sensor.get_supported_options():
                        rgb_sensor.set_option(rs.option.gain, gain)
                        print(f"Camera {cam_serial}: Gain set to {gain}")

            except Exception as e:
                print(f"Error adjusting exposure for camera {cam_serial}: {e}")
                success = False

        return success

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


__all__ = [
    "DEFAULT_CAN_PORTS",
    "DEFAULT_GRIPPER_CLOSE",
    "DEFAULT_GRIPPER_OPEN",
    "DEFAULT_LEFT_WRIST_CAM_SERIAL",
    "DEFAULT_RIGHT_WRIST_CAM_SERIAL",
    "DEFAULT_TOP_CAM_SERIAL",
    "RealSenseCameraInterface",
]
