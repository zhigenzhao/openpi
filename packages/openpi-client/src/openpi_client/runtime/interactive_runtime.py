import logging
import threading
import time

from pynput import keyboard

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class InteractiveRuntime:
    """Interactive runtime with pause/resume controls.

    Supports the following keyboard commands:
    - 'p': Pause/Resume toggle
    - 'n': Start new episode (restart current episode)
    - 'q': Quit current episode
    """

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        # Simple pause/resume state
        self._is_paused = False
        self._should_quit = False
        self._should_restart_episode = False
        self._state_lock = threading.Lock()

        # Episode tracking
        self._in_episode = False
        self._episode_steps = 0

        # Keyboard input handling
        self._keyboard_listener = None

        logging.info("InteractiveRuntime initialized. Press 'p' to pause/resume, 'n' for new episode, 'q' to quit.")

    def run(self) -> None:
        """Run the interactive runtime with keyboard controls."""
        print("🤖 Interactive Runtime: Press 'p' to pause/resume, 'n' for new episode, 'q' to quit")

        try:
            self._start_keyboard_listener()

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
            self._stop_keyboard_listener()

    def _run_episode(self) -> None:
        """Run a single episode with interactive controls."""
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
            action = self._agent.get_action(observation)
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

    def _start_keyboard_listener(self) -> None:
        """Start the keyboard listener."""
        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self._keyboard_listener.start()

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener.join()

    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        try:
            # Handle character keys
            if hasattr(key, "char") and key.char:
                char = key.char.lower()
                print(f"\n[Received key: '{char}']")  # Debug output
                self._handle_command(char)
        except AttributeError:
            # Handle special keys (escape, space, etc.)
            pass

    def _handle_command(self, command: str) -> None:
        """Handle keyboard commands."""
        if command == "p":  # Pause/Resume toggle
            self._toggle_pause()
        elif command == "n":  # New episode (restart)
            self._new_episode()
        elif command == "q":  # Quit
            self._quit()

    def _toggle_pause(self) -> None:
        """Toggle pause/resume state."""
        with self._state_lock:
            self._is_paused = not self._is_paused

        if self._is_paused:
            print("⏸️  Runtime PAUSED")
            logging.info("⏸️  Runtime PAUSED")
        else:
            print("▶️  Runtime RESUMED")
            logging.info("▶️  Runtime RESUMED")

    def _new_episode(self) -> None:
        """Restart the current episode."""
        with self._state_lock:
            self._should_restart_episode = True
            self._in_episode = False  # End current episode
            
        print("🔄 Starting new episode...")
        logging.info("🔄 Starting new episode...")

    def _quit(self) -> None:
        """Quit the runtime."""
        with self._state_lock:
            self._should_quit = True
            self._in_episode = False

        print("⏹️  Runtime STOPPED")
        logging.info("⏹️  Runtime STOPPED")
