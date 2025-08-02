import logging
import threading
import time

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

        # Data logging thread management
        self._data_logging_thread = None
        self._stop_data_logging = threading.Event()

        logging.info("XRInteractiveRuntime initialized for both controllers.")
        logging.info("XR Controls: Left(X/Y/Menu) or Right(A/B/Menu) - pause/resume, new episode, quit")
        logging.info("XR Teleoperation: Hold grip buttons to take manual control")

    def run(self) -> None:
        """Run the interactive runtime with XR controller controls."""
        print("🥽 XR Interactive Runtime: Left(X/Y/Menu) or Right(A/B/Menu) for pause/resume, new episode, quit")
        print("🕹️  Hold grip buttons for teleoperation control")

        try:
            # Use teleop controller's XR client (which initializes the SDK)
            print("[XR] Using teleop controller's XR client...")
            self._start_xr_monitor()

            # Don't start teleop controller in separate thread - we manage steps directly
            # But we need to ensure XR client is available for input monitoring
            self._initialize_teleop_controller()

            # Start data logging thread if enabled
            self._start_data_logging_if_enabled()

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
            self._stop_data_logging_thread()

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
            print("⏸️  Runtime PAUSED (XR)")
            logging.info("⏸️  Runtime PAUSED (XR)")
        else:
            print("▶️  Runtime RESUMED (XR)")
            logging.info("▶️  Runtime RESUMED (XR)")

    def _new_episode(self) -> None:
        """Restart the current episode."""
        with self._state_lock:
            self._should_restart_episode = True
            self._in_episode = False  # End current episode

        print("🔄 Starting new episode... (XR)")
        logging.info("🔄 Starting new episode... (XR)")

    def _quit(self) -> None:
        """Quit the runtime."""
        with self._state_lock:
            self._should_quit = True
            self._in_episode = False

        print("⏹️  Runtime STOPPED (XR)")
        logging.info("⏹️  Runtime STOPPED (XR)")

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
                # Sync end effector poses to placo tasks for smooth teleop initialization
                self._teleop_controller.sync_end_effector_poses_to_placo_tasks()
                self._agent.reset()
            elif not self._teleop_active and was_active:
                print("[XR Teleop] Teleoperation DEACTIVATED - Policy RESUMED")
                logging.info("Teleoperation deactivated - policy resumed")

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

    def _start_data_logging_if_enabled(self) -> None:
        """Start data logging thread if enabled in teleop controller."""
        if (
            self._teleop_controller is not None
            and hasattr(self._teleop_controller, "enable_log_data")
            and self._teleop_controller.enable_log_data
        ):
            try:
                logging.info("Starting data logging thread")
                print("[Data Logging] Starting data logging thread - press 'B' button to toggle logging")
                self._data_logging_thread = threading.Thread(
                    target=self._teleop_controller._data_logging_thread,
                    args=(self._stop_data_logging,),
                    daemon=True,
                )
                self._data_logging_thread.start()
            except Exception as e:
                logging.error(f"Error starting data logging thread: {e}")
                print(f"[Data Logging Error] Failed to start logging: {e}")

    def _stop_data_logging_thread(self) -> None:
        """Stop the data logging thread."""
        if self._data_logging_thread is not None:
            try:
                logging.info("Stopping data logging thread")
                self._stop_data_logging.set()
                self._data_logging_thread.join(timeout=2.0)
                self._data_logging_thread = None
            except Exception as e:
                logging.error(f"Error stopping data logging thread: {e}")
