import logging
import pickle
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from examples.gim.robot_utils import DEFAULT_GRIPPER_OPEN
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
from openpi_client.runtime.openpi_teleop_controller import OpenPITeleopController as TeleopController


class XRInteractiveRuntime:
    """Interactive runtime controlled by XR controller buttons.

    Supports the following XR controller commands:
    - 'X' button: Pause/Resume toggle
    - 'Y' button: Start new episode (restart current episode)
    - 'Menu' button: Quit current episode

    Requires xrobotoolkit_sdk to be installed and XR device connected.
    """

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        teleop_controller: TeleopController,
        max_hz: float = 50,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
        enable_log_data: bool = False,
        log_dir: str = "logs/xr_rollouts",
        log_freq: float = 50.0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps
        self._teleop_controller = teleop_controller

        # Simple pause/resume state
        self._is_paused = False
        self._should_quit = False
        self._should_restart_episode = False
        self._state_lock = threading.Lock()

        # Episode tracking
        self._in_episode = False
        self._episode_steps = 0

        # XR input handling
        self._xr_monitor_thread = None
        self._stop_xr_monitor = False

        # Button state tracking for both controllers to detect button press events (not held)
        self._last_left_x_button_state = False
        self._last_left_y_button_state = False
        self._last_left_menu_button_state = False
        self._last_right_a_button_state = False
        self._last_right_b_button_state = False
        self._last_right_menu_button_state = False

        # Teleoperation state tracking
        self._teleop_active = False
        self._teleop_lock = threading.Lock()
        self._grip_threshold = 0.5  # Threshold for grip activation

        # Data logging
        self._enable_log_data = enable_log_data
        self._log_dir = Path(log_dir)
        self._log_freq = log_freq
        self._is_logging = False
        self._log_start_time = None
        self._episode_data: list[dict] = []

        # Threaded data logging
        self._data_logging_thread = None
        self._stop_data_logging = threading.Event()
        self._latest_step_data: dict | None = None
        self._step_data_lock = threading.Lock()

        logging.info("XRInteractiveRuntime initialized for both controllers.")
        logging.info("XR Controls: Left(X/Y/Menu) or Right(A/B/Menu) - pause/resume, new episode, quit")
        logging.info("XR Teleoperation: Hold grip buttons to take manual control")
        if enable_log_data:
            logging.info(f"Data logging enabled at {log_freq}Hz - press 'B' to toggle, logs saved to {log_dir}")

    def run(self) -> None:
        """Run the interactive runtime with XR controller controls."""
        print("XR Interactive Runtime: Left(X/Y/Menu) or Right(A/B/Menu) for pause/resume, new episode, quit")
        print("Hold grip buttons for teleoperation control")
        if self._enable_log_data:
            print("Data logging enabled - press 'B' to start/stop, right stick click to discard")

        try:
            # Use teleop controller's XR client (which initializes the SDK)
            print("[XR] Using teleop controller's XR client...")
            self._start_xr_monitor()

            # Don't start teleop controller in separate thread - we manage steps directly
            # But we need to ensure XR client is available for input monitoring
            self._initialize_teleop_controller()

            # Start data logging thread if enabled
            if self._enable_log_data:
                self._start_data_logging_thread()

            episode = 0
            while episode < self._num_episodes and not self._should_quit:
                self._run_episode()

                # Check if we should restart the same episode
                if self._should_restart_episode:
                    with self._state_lock:
                        self._should_restart_episode = False
                    # Don't increment episode counter, run the same episode again
                    continue

                episode += 1

        finally:
            self._stop_xr_monitor_func()
            # Stop data logging thread
            self._stop_data_logging_thread()
            # Save any remaining logged data
            if self._is_logging:
                self._save_episode_data()

    def _run_episode(self) -> None:
        """Run a single episode with XR controller controls."""
        logging.info("Starting episode...")

        # Reset environment and agent
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode and not self._should_quit:
            # Check if episode restart requested
            if self._should_restart_episode:
                with self._state_lock:
                    self._should_restart_episode = False
                break  # Exit current episode to restart

            # Check if paused
            if self._is_paused:
                time.sleep(0.1)  # Sleep while paused
                continue

            # Execute one step
            should_continue = self._step()
            if not should_continue:
                break

            # Maintain frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        # Episode completed
        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

        # Reset agent after episode ends
        self._agent.reset()

    def _step(self) -> bool:
        """Execute a single step. Returns False if episode should end."""
        try:
            observation = self._environment.get_observation()
            action = None

            # Check if teleoperation is active
            with self._teleop_lock:
                teleop_active = self._teleop_active

            if teleop_active:
                # When teleop is active, run teleop steps and get action
                action = self._run_teleop_step()
            else:
                # Otherwise, use policy agent
                action = self._agent.get_action(observation)

            # If an action was determined (from either source), apply it
            if action is not None:
                self._environment.apply_action(action)
                self._episode_steps += 1

                # Store step data for logging thread (non-blocking)
                source = "teleop" if teleop_active else "policy"
                self._store_step_data(observation, action, source)

                # Notify subscribers
                for subscriber in self._subscribers:
                    subscriber.on_step(observation, action)

            # Check termination conditions
            if self._environment.is_episode_complete() or (
                self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
            ):
                self._in_episode = False
                return False

            return True

        except Exception as e:
            logging.error(f"Error during step execution: {e}")
            return False

    def _start_xr_monitor(self) -> None:
        """Start the XR controller input monitoring thread."""
        self._stop_xr_monitor = False
        self._xr_monitor_thread = threading.Thread(target=self._xr_input_monitor, daemon=True)
        self._xr_monitor_thread.start()

    def _stop_xr_monitor_func(self) -> None:
        """Stop the XR controller input monitoring thread."""
        self._stop_xr_monitor = True
        if self._xr_monitor_thread and self._xr_monitor_thread.is_alive():
            self._xr_monitor_thread.join(timeout=1.0)

    def _xr_input_monitor(self) -> None:
        """Monitor XR controller input in a separate thread."""
        print("[XR] Starting input monitor for both controllers...")

        while not self._stop_xr_monitor:
            try:
                # Check if teleop controller has XR client
                if not hasattr(self._teleop_controller, "xr_client") or self._teleop_controller.xr_client is None:
                    if not hasattr(self, "_xr_client_warning_shown"):
                        logging.error("Teleop controller XR client not available - XR input monitoring disabled")
                        print("[XR Error] XR client not available in teleop controller")
                        print("[XR Error] This may be due to missing XRobot SDK or initialization issues")
                        self._xr_client_warning_shown = True
                    time.sleep(5.0)  # Wait longer before retrying
                    continue

                # Use teleop controller's XR client
                xr_client = self._teleop_controller.xr_client
                # Get current button states for both controllers
                left_x_button = xr_client.get_button_state_by_name("X")
                left_y_button = xr_client.get_button_state_by_name("Y")

                right_a_button = xr_client.get_button_state_by_name("A")
                right_b_button = xr_client.get_button_state_by_name("B")

                # Get grip states for teleoperation
                left_grip = xr_client.get_key_value_by_name("left_grip")
                right_grip = xr_client.get_key_value_by_name("right_grip")

                # Update teleoperation state
                self._update_teleop_state(left_grip, right_grip)

                # Detect button press events (transition from False to True) for LEFT controller
                if left_x_button and not self._last_left_x_button_state:
                    print("\n[XR] Left X button pressed (pause/resume)")
                    self._handle_pause_command()

                if left_y_button and not self._last_left_y_button_state:
                    print("\n[XR] Left Y button pressed (new episode)")
                    self._handle_quit_command()

                # Handle B button for data logging toggle
                if right_b_button and not self._last_right_b_button_state:
                    print("\n[XR] Right B button pressed (toggle logging)")
                    self._toggle_logging()

                # Handle right stick click to discard logged data
                right_axis_click = xr_client.get_button_state_by_name("right_axis_click")
                if right_axis_click and self._is_logging:
                    self._discard_logging()

                # Update button state history for both controllers
                self._last_left_x_button_state = left_x_button
                self._last_left_y_button_state = left_y_button
                self._last_right_a_button_state = right_a_button
                self._last_right_b_button_state = right_b_button

                time.sleep(0.05)  # Check buttons at 20Hz

            except Exception as e:
                logging.error(f"XR input monitoring error: {e}")
                print(f"[XR Error] {e}")
                time.sleep(0.5)  # Longer sleep on error

    def _handle_pause_command(self) -> None:
        """Handle pause/resume command from XR controller."""
        self._toggle_pause()

    def _handle_new_episode_command(self) -> None:
        """Handle new episode command from XR controller."""
        self._new_episode()

    def _handle_quit_command(self) -> None:
        """Handle quit command from XR controller."""
        self._quit()

    def _toggle_pause(self) -> None:
        """Toggle pause/resume state."""
        with self._state_lock:
            self._is_paused = not self._is_paused

        if self._is_paused:
            print("Runtime PAUSED (XR)")
            logging.info("Runtime PAUSED (XR)")
        else:
            print("Runtime RESUMED (XR)")
            logging.info("Runtime RESUMED (XR)")

    def _new_episode(self) -> None:
        """Restart the current episode."""
        with self._state_lock:
            self._should_restart_episode = True
            self._in_episode = False  # End current episode

        print("Starting new episode... (XR)")
        logging.info("Starting new episode... (XR)")

    def _quit(self) -> None:
        """Quit the runtime."""
        with self._state_lock:
            self._should_quit = True
            self._in_episode = False

        print("Runtime STOPPED (XR)")
        logging.info("Runtime STOPPED (XR)")

    def _update_teleop_state(self, left_grip: float, right_grip: float) -> None:
        """Update teleoperation state based on grip button values."""
        with self._teleop_lock:
            # Check if grips are pressed above threshold
            left_grip_pressed = left_grip > self._grip_threshold
            right_grip_pressed = right_grip > self._grip_threshold

            # Teleoperation is active if either grip is pressed
            was_active = self._teleop_active
            self._teleop_active = left_grip_pressed or right_grip_pressed

            # Log state changes
            if self._teleop_active and not was_active:
                print("[XR Teleop] Teleoperation ACTIVATED - Syncing end effector poses.")
                logging.info("Teleoperation activated via grip buttons - syncing end effector poses to placo tasks.")
                # Reset filters to avoid velocity/acceleration spike from source transition
                if hasattr(self._environment, "reset_filter_state"):
                    self._environment.reset_filter_state()
                # Sync end effector poses to placo tasks for smooth teleop initialization
                self._teleop_controller.sync_end_effector_poses_to_placo_tasks()
                self._agent.reset()
            elif not self._teleop_active and was_active:
                print("[XR Teleop] Teleoperation DEACTIVATED - Policy RESUMED")
                logging.info("Teleoperation deactivated - policy resumed")
                # Reset filters to avoid velocity/acceleration spike from source transition
                if hasattr(self._environment, "reset_filter_state"):
                    self._environment.reset_filter_state()

    def _run_teleop_step(self):
        """Execute one teleoperation step and return the computed action.

        This method runs the teleop controller's execute_step method and then
        retrieves the computed action.

        Returns:
            Action computed by teleop controller, or None if no action available
        """
        try:
            # Execute one teleop step (update robot state, IK, grippers, generate command)
            self._teleop_controller.execute_step()

            # Get the latest computed action
            action = self._teleop_controller.get_latest_action()

            return action

        except Exception as e:
            logging.error(f"Error during teleop step execution: {e}")
            return None

    def _initialize_teleop_controller(self) -> None:
        """Initialize the teleoperation controller for XR input access.

        Ensures the XR client is available for input monitoring without starting
        the separate thread execution.
        """
        if self._teleop_controller is None:
            logging.warning("No teleop controller available")
            return

        try:
            # Check if XR client is available after construction
            if hasattr(self._teleop_controller, "xr_client") and self._teleop_controller.xr_client is not None:
                logging.info("Teleop controller XR client is available")
            else:
                logging.warning("Teleop controller XR client not available - XR input may not work")

        except Exception as e:
            logging.error(f"Error checking teleop controller initialization: {e}")

    # ==================== Data Logging Methods ====================

    def _compress_image_to_jpg(self, image: np.ndarray | None, quality: int = 85) -> bytes | None:
        """Compress a numpy image array to JPG bytes."""
        if image is None:
            return None
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode(".jpg", image, encode_param)
            return encoded_img.tobytes()
        except Exception as e:
            logging.error(f"Error compressing image: {e}")
            return None

    def _store_step_data(self, observation: dict, action: dict, source: str) -> None:
        """Store latest step data for the logging thread to consume (non-blocking).

        Args:
            observation: Environment observation dict with state and images
            action: Action dict with 'actions' array
            source: Either "teleop" or "policy"
        """
        if not self._enable_log_data:
            return

        with self._step_data_lock:
            self._latest_step_data = {
                "observation": observation.copy(),
                "action": action.copy(),
                "source": source,
                "timestamp": time.time(),
            }

    def _data_logging_thread_func(self, stop_event: threading.Event) -> None:
        """Dedicated thread for data logging at consistent frequency."""
        print("[Data Logging] Thread started")

        while not stop_event.is_set():
            start_time = time.time()

            # Log data if logging is active
            if self._is_logging:
                with self._step_data_lock:
                    step_data = self._latest_step_data

                if step_data is not None:
                    self._log_step_from_data(step_data)

            # Maintain consistent frequency
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self._log_freq) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[Data Logging] Thread stopped")

    def _log_step_from_data(self, step_data: dict) -> None:
        """Log a step from stored data (called from logging thread)."""
        try:
            observation = step_data["observation"]
            action = step_data["action"]
            source = step_data["source"]
            capture_time = step_data["timestamp"]

            timestamp = capture_time - self._log_start_time
            joint_positions = np.array(observation["state"])
            action_array = np.array(action["actions"])

            # Denormalize gripper values: observation has normalized (0-1), log needs raw
            left_qpos = np.concatenate([joint_positions[:6], [joint_positions[6] * DEFAULT_GRIPPER_OPEN]])
            right_qpos = np.concatenate([joint_positions[7:13], [joint_positions[13] * DEFAULT_GRIPPER_OPEN]])

            # Get per-arm grip active state from teleop controller
            grip_active = {"left_arm": False, "right_arm": False}
            if source == "teleop" and self._teleop_controller is not None and hasattr(self._teleop_controller, "active"):
                grip_active = {
                    arm_name: self._teleop_controller.active.get(arm_name, False)
                    for arm_name in grip_active
                }

            log_entry = {
                "timestamp": timestamp,
                "source": source,
                "qpos": {
                    "left_arm": left_qpos.tolist(),
                    "right_arm": right_qpos.tolist(),
                },
                "qpos_target": {
                    "left_arm": action_array[:6].tolist(),
                    "right_arm": action_array[7:13].tolist(),
                },
                "gripper_target": {
                    "left_arm": float(action_array[6] * DEFAULT_GRIPPER_OPEN),
                    "right_arm": float(action_array[13] * DEFAULT_GRIPPER_OPEN),
                },
                "grip_active": grip_active,
                "image": {
                    "top": {"color": self._compress_image_to_jpg(observation.get("top_image"))},
                    "left_wrist": {"color": self._compress_image_to_jpg(observation.get("left_wrist_image"))},
                    "right_wrist": {"color": self._compress_image_to_jpg(observation.get("right_wrist_image"))},
                },
            }
            self._episode_data.append(log_entry)

        except Exception as e:
            logging.error(f"Error logging step from data: {e}")

    def _start_data_logging_thread(self) -> None:
        """Start the data logging thread."""
        self._stop_data_logging.clear()
        self._data_logging_thread = threading.Thread(
            target=self._data_logging_thread_func,
            args=(self._stop_data_logging,),
            daemon=True,
        )
        self._data_logging_thread.start()

    def _stop_data_logging_thread(self) -> None:
        """Stop the data logging thread."""
        if self._data_logging_thread is not None:
            self._stop_data_logging.set()
            self._data_logging_thread.join(timeout=2.0)
            self._data_logging_thread = None

    def _toggle_logging(self) -> None:
        """Toggle data logging on/off via B button."""
        if not self._enable_log_data:
            return

        self._is_logging = not self._is_logging
        if self._is_logging:
            self._log_start_time = time.time()
            self._episode_data = []
            print("--- Started data logging ---")
            logging.info("Data logging started")
        else:
            self._save_episode_data()
            print("--- Stopped data logging ---")
            logging.info("Data logging stopped")

    def _discard_logging(self) -> None:
        """Discard current logged data without saving."""
        if self._is_logging:
            self._episode_data = []
            self._is_logging = False
            print("--- Discarded data logging ---")
            logging.info("Data logging discarded")

    def _save_episode_data(self) -> None:
        """Save current episode data to pkl file."""
        if not self._episode_data:
            logging.info("No data to save")
            return

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = self._log_dir / f"rollout_{timestamp}.pkl"

            with open(filename, "wb") as f:
                pickle.dump(self._episode_data, f)

            print(f"Saved {len(self._episode_data)} steps to {filename}")
            logging.info(f"Saved {len(self._episode_data)} steps to {filename}")
            self._episode_data = []

        except Exception as e:
            logging.error(f"Error saving episode data: {e}")
            print(f"[Data Logging Error] Failed to save: {e}")
