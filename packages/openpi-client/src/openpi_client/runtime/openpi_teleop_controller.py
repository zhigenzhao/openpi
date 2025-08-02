"""OpenPI Teleoperation Controller.

A teleoperation controller that extends BaseTeleopController but sends actions
to OpenPI's environment system instead of directly controlling robot hardware.
This avoids conflicts with robot control and camera systems.
"""

import abc
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

from openpi_client.runtime import environment as _environment


class OpenPITeleopController(BaseTeleopController):
    """Teleoperation controller that sends actions to OpenPI environment.

    This controller extends BaseTeleopController but instead of directly controlling
    robot hardware, it sends computed actions to the OpenPI environment system.
    This allows seamless integration with existing OpenPI workflows while avoiding
    hardware and camera conflicts.
    """

    def __init__(
        self,
        environment: _environment.Environment,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base: bool = False,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = 1.0,
        q_init: Optional[np.ndarray] = None,
        dt: float = 1.0 / 50.0,  # 50Hz control rate
        enable_log_data: bool = False,
        log_dir: str = "logs/openpi_teleop",
        log_freq: float = 50,
    ):
        """Initialize OpenPI teleoperation controller.

        Args:
            environment: OpenPI environment to send actions to
            robot_urdf_path: Path to robot URDF file for IK computation
            manipulator_config: Configuration for manipulator end-effectors
            floating_base: Whether robot has floating base
            R_headset_world: Rotation matrix from headset to world frame
            scale_factor: Scaling factor for controller movements
            q_init: Initial joint configuration
            dt: Control timestep
            enable_log_data: Whether to enable data logging
            log_dir: Directory for log data
            log_freq: Logging frequency
        """
        self._environment = environment
        self._action_lock = threading.Lock()
        self._latest_action = None

        # Thread management for backward compatibility
        self._running = False
        self._thread = None

        # Initialize start time for logging
        self._start_time = time.time()

        self._is_logging = False
        self._prev_b_button_state = False

        # Initialize base teleop controller
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            floating_base=floating_base,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=q_init,
            dt=dt,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
        )

        logging.info("OpenPITeleopController initialized")

    def _robot_setup(self):
        """Setup robot connection - no-op since we use OpenPI environment."""
        logging.info("Robot setup: Using OpenPI environment (no direct hardware connection)")

    @abc.abstractmethod
    def _update_robot_state(self):
        """Update robot state from environment observations.

        This method should extract joint positions from the environment's
        observation and update self.placo_robot.state.q accordingly.
        Implementation is environment-specific.
        """
        raise NotImplementedError("Subclasses must implement _update_robot_state")

    def _send_command(self):
        """Send computed action to OpenPI environment instead of robot hardware."""
        try:
            # Convert placo joint targets to environment action format
            action = self._placo_to_env_action()

            # Store the latest action for the environment to consume
            with self._action_lock:
                self._latest_action = action

        except Exception as e:
            logging.error(f"Error sending command to environment: {e}")

    @abc.abstractmethod
    def _placo_to_env_action(self) -> Dict[str, Any]:
        """Convert Placo joint targets to environment action format.

        This method should convert the computed joint targets in
        self.placo_robot.state.q and gripper targets in self.gripper_pos_target
        to the specific action format expected by the environment.

        Returns:
            Dictionary containing actions in environment-specific format
        """
        raise NotImplementedError("Subclasses must implement _placo_to_env_action")

    @abc.abstractmethod
    def _get_robot_state_for_logging(self) -> Dict[str, Any]:
        """Returns a dictionary of robot-specific data for logging.

        This method should extract current robot state information from the
        environment and return it in a structured format for data logging.

        Returns:
            Dictionary containing robot state data (joint positions, targets, etc.)
        """
        raise NotImplementedError("Subclasses must implement _get_robot_state_for_logging")

    def _get_link_pose(self, link_name: str):
        """Get current world pose for a given link name."""
        try:
            # Update kinematics first
            self._update_robot_state()
            self.placo_robot.update_kinematics()

            # Get link pose from placo
            T_world_link = self.placo_robot.get_T_world_frame(link_name)

            # Extract position and quaternion
            position = T_world_link[:3, 3]

            # Convert rotation matrix to quaternion (w, x, y, z)
            import meshcat.transformations as tf

            quaternion = tf.quaternion_from_matrix(T_world_link)

            return position, quaternion

        except Exception as e:
            logging.error(f"Error getting pose for link {link_name}: {e}")
            # Return identity pose as fallback
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def get_latest_action(self) -> Optional[Dict[str, Any]]:
        """Get the latest computed action for the environment to apply."""
        with self._action_lock:
            action = self._latest_action
            self._latest_action = None  # Consume the action
            return action

    def is_active(self) -> bool:
        """Check if any manipulator is currently being controlled."""
        return any(self.active.values()) if hasattr(self, "active") else False

    def execute_step(self):
        """Execute one teleoperation step.

        This method performs the core teleop operations that were previously
        managed in a separate thread: update robot state, update IK, update
        gripper targets, and generate commands.

        This should be called from the main control loop when teleop is active.
        """
        try:
            # Update robot state from environment
            self._update_robot_state()

            # Update IK computations
            self._update_ik()

            # Update gripper targets
            self._update_gripper_target()

            # Generate and store command
            self._send_command()

            # Log data if enabled
            if self.enable_log_data:
                self._log_data()

        except Exception as e:
            logging.error(f"Error in teleoperation step execution: {e}")

    def run(self):
        """Run the teleoperation control loop in a separate thread.

        This method runs the traditional threaded control loop for backward compatibility
        with scripts that use start()/stop(). For direct integration, use execute_step().
        """
        self._running = True
        logging.info("OpenPI teleoperation control loop started")

        while self._running and not self._stop_event.is_set():
            try:
                # Execute one teleop step
                self.execute_step()

                # Sleep for control rate
                time.sleep(self.dt)

            except Exception as e:
                logging.error(f"Error in teleoperation control loop: {e}")
                time.sleep(0.1)  # Brief pause on error

        logging.info("OpenPI teleoperation control loop stopped")

    def start(self):
        """Start the teleoperation controller in a separate thread."""
        if self._running:
            logging.warning("Teleoperation controller already running")
            return

        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        logging.info("OpenPI teleoperation controller started")

    def stop(self):
        """Stop the teleoperation controller."""
        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        logging.info("OpenPI teleoperation controller stopped")

    def _log_data(self):
        """Logs the current state of the robot and environment data."""
        if not self.enable_log_data or not self._is_logging:
            return

        timestamp = time.time() - self._start_time
        data_entry = {"timestamp": timestamp}

        # Get robot-specific state data from subclass
        robot_state = self._get_robot_state_for_logging()
        data_entry.update(robot_state)

        # Log the data entry using base controller's logger
        if hasattr(self, "data_logger") and self.data_logger is not None:
            self.data_logger.add_entry(data_entry)
            print(f"[Data Log] Saved entry with keys: {list(data_entry.keys())}")
            logging.info(f"Logged data entry with keys: {list(data_entry.keys())}")
        else:
            print("[Data Log] ERROR: Data logger not available")
            logging.warning("Data logger not available - cannot save data")

    def _data_logging_thread(self, stop_event: threading.Event):
        """Dedicated thread for data logging."""
        while not stop_event.is_set():
            start_time = time.time()
            self._check_logging_button()
            if self._is_logging:
                self._log_data()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.log_freq) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("Data logging thread has stopped.")

    def _check_logging_button(self):
        """Checks for the 'B' button press to toggle data logging."""
        b_button_state = self.xr_client.get_button_state_by_name("B")
        right_axis_click = self.xr_client.get_button_state_by_name("right_axis_click")

        if b_button_state and not self._prev_b_button_state:
            self._is_logging = not self._is_logging
            if self._is_logging:
                print("--- Started data logging ---")
            else:
                print("--- Stopped data logging. Saving data... ---")
                self.data_logger.save()
                self.data_logger.reset()

        if right_axis_click and self._is_logging:
            print("--- Stopped data logging. Discarding data... ---")
            self.data_logger.reset()
            self._is_logging = False

        self._prev_b_button_state = b_button_state
