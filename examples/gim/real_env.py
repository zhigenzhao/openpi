import collections
import logging
import threading
import time

import dm_env
from gim_arm_control import ButterworthFilter
from gim_arm_control import ControllerConfig
from gim_arm_control import ControlMode
from gim_arm_control import GimArmController
import numpy as np

from examples.gim.robot_utils import DEFAULT_CAN_PORTS
from examples.gim.robot_utils import DEFAULT_GRIPPER_OPEN
from examples.gim.robot_utils import DEFAULT_LEFT_WRIST_CAM_SERIAL
from examples.gim.robot_utils import DEFAULT_RIGHT_WRIST_CAM_SERIAL
from examples.gim.robot_utils import DEFAULT_TOP_CAM_SERIAL
from examples.gim.robot_utils import RealSenseCameraInterface

POLICY_HZ = 50
CONTROL_HZ = 200
UPDATE_RATIO = CONTROL_HZ // POLICY_HZ  # 4


class RealEnv:
    """Environment for GIM 6-DOF bimanual arm with dual-frequency control.

    Policy targets arrive at 50Hz. A background control thread runs at 200Hz,
    linearly interpolating between targets and computing Butterworth-filtered
    velocity/acceleration for feedforward control.

    Action space:      [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]  # 14D
    Observation space: {"qpos": 14D, "images": {"left_wrist", "right_wrist", "top"}}
    """

    def __init__(
        self,
        can_port_left: str = "can1",
        can_port_right: str = "can0",
        camera_serial_numbers: list[str] | None = None,
        camera_width: int = 424,
        camera_height: int = 240,
        camera_fps: int = 60,
        *,
        enable_depth: bool = False,
        auto_exposure: bool = True,
        # exposure: int = 10000,
        gain: int = 16,
    ):
        self.can_port_left = can_port_left
        self.can_port_right = can_port_right

        # Initialize cameras
        self.camera_interface = RealSenseCameraInterface(
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            serial_numbers=camera_serial_numbers,
            enable_depth=enable_depth,
            auto_exposure=auto_exposure,
            # exposure=exposure,
            gain=gain,
        )
        self.camera_interface.start()

        # Initialize GIM arm controllers
        left_config = ControllerConfig(
            can_interface=can_port_left,
            enable_gripper=True,
            gripper_type="single_side",
            feedforward_mode="gravity",
            control_rate_hz=float(CONTROL_HZ),
        )
        right_config = ControllerConfig(
            can_interface=can_port_right,
            enable_gripper=True,
            gripper_type="single_side",
            feedforward_mode="gravity",
            control_rate_hz=float(CONTROL_HZ),
        )

        self.controller_left = GimArmController(left_config)
        self.controller_right = GimArmController(right_config)
        self.controller_left.start(return_to_zero=True, start_thread=False)
        self.controller_right.start(return_to_zero=True, start_thread=False)
        self.controller_left.set_mode(ControlMode.MOMENTUM_OBSERVER)
        self.controller_right.set_mode(ControlMode.MOMENTUM_OBSERVER)

        # Butterworth filters for velocity and acceleration (per arm)
        self.vel_filter_left = ButterworthFilter(cutoff_hz=4.0, dt=1.0 / CONTROL_HZ, size=7)
        self.vel_filter_right = ButterworthFilter(cutoff_hz=4.0, dt=1.0 / CONTROL_HZ, size=7)
        self.accel_filter_left = ButterworthFilter(cutoff_hz=6.0, dt=1.0 / CONTROL_HZ, size=7)
        self.accel_filter_right = ButterworthFilter(cutoff_hz=6.0, dt=1.0 / CONTROL_HZ, size=7)

        # Previous state for finite-difference velocity/accel
        dt = 1.0 / CONTROL_HZ
        self.dt = dt
        self.prev_joints_left = np.zeros(7)
        self.prev_joints_right = np.zeros(7)
        self.prev_velocities_left = np.zeros(7)
        self.prev_velocities_right = np.zeros(7)

        # Shared target buffer (thread-safe)
        self._lock = threading.Lock()
        self._target_left = np.zeros(7)  # 6 joints + 1 gripper
        self._target_right = np.zeros(7)
        self._prev_target_left = np.zeros(7)
        self._prev_target_right = np.zeros(7)
        self._step_counter = 0
        self._control_running = False
        self._control_thread = None

        self.reset()
        time.sleep(1.0)

    def _control_loop(self):
        """200Hz control thread with linear interpolation and feedforward."""
        dt = self.dt
        while self._control_running:
            loop_start = time.monotonic()

            with self._lock:
                local_step = self._step_counter % UPDATE_RATIO
                alpha = local_step / UPDATE_RATIO
                target_left = (1 - alpha) * self._prev_target_left + alpha * self._target_left
                target_right = (1 - alpha) * self._prev_target_right + alpha * self._target_right
                self._step_counter += 1

            # Compute filtered velocity and acceleration for left arm
            raw_vel_left = (target_left - self.prev_joints_left) / dt
            vel_left = self.vel_filter_left.process(raw_vel_left)
            raw_accel_left = (vel_left - self.prev_velocities_left) / dt
            accel_left = self.accel_filter_left.process(raw_accel_left)
            self.prev_joints_left = target_left.copy()
            self.prev_velocities_left = vel_left.copy()

            # Compute filtered velocity and acceleration for right arm
            raw_vel_right = (target_right - self.prev_joints_right) / dt
            vel_right = self.vel_filter_right.process(raw_vel_right)
            raw_accel_right = (vel_right - self.prev_velocities_right) / dt
            accel_right = self.accel_filter_right.process(raw_accel_right)
            self.prev_joints_right = target_right.copy()
            self.prev_velocities_right = vel_right.copy()

            # Send feedforward commands (joints + gripper together in 7D)
            self.controller_left.set_feedforward_target(target_left[:6], vel_left[:6], accel_left[:6])
            self.controller_left.set_gripper(target_left[6])
            self.controller_left.step(dt)

            self.controller_right.set_feedforward_target(target_right[:6], vel_right[:6], accel_right[:6])
            self.controller_right.set_gripper(target_right[6])
            self.controller_right.step(dt)

            # Maintain 200Hz timing
            elapsed = time.monotonic() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _start_control_thread(self):
        """Start the 200Hz control thread."""
        if self._control_running:
            return
        self._control_running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def _stop_control_thread(self):
        """Stop the 200Hz control thread."""
        self._control_running = False
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
            self._control_thread = None

    def get_qpos(self):
        """Get current joint positions: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]."""
        left_reading = self.controller_left.get_reading()
        right_reading = self.controller_right.get_reading()

        left_pos = np.array(left_reading.position, dtype=np.float32)
        left_grip = left_reading.gripper_position
        left_grip_normalized = left_grip / DEFAULT_GRIPPER_OPEN  # -0.2 -> 1.0, 0.0 -> 0.0

        right_pos = np.array(right_reading.position, dtype=np.float32)
        right_grip = right_reading.gripper_position
        right_grip_normalized = right_grip / DEFAULT_GRIPPER_OPEN

        return np.concatenate([left_pos, [left_grip_normalized], right_pos, [right_grip_normalized]]).astype(np.float32)

    def get_images(self):
        """Get camera images from RealSense cameras."""
        self.camera_interface.update_frames()
        frames = self.camera_interface.get_frames()

        images = {}
        images["left_wrist"] = frames[DEFAULT_LEFT_WRIST_CAM_SERIAL]["color"]
        images["right_wrist"] = frames[DEFAULT_RIGHT_WRIST_CAM_SERIAL]["color"]
        images["top"] = frames[DEFAULT_TOP_CAM_SERIAL]["color"]

        return images

    def get_observation(self):
        """Get full observation including state and images."""
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["images"] = self.get_images()
        return obs

    def get_hardware_readings(self):
        """Get raw hardware readings (velocity, effort, external torque) from both arms."""
        result = {}
        for arm_name, controller in [("left_arm", self.controller_left), ("right_arm", self.controller_right)]:
            reading = controller.get_reading()
            if reading is not None:
                result[arm_name] = {
                    "velocity": reading.velocity.tolist(),
                    "effort": reading.effort.tolist(),
                    "external_torque": reading.external_torque.tolist() if reading.external_torque is not None else None,
                }
        return result

    def reset_filter_state(self):
        """Reset filters and previous-state trackers to current targets.

        Call on control source transitions (policy<->teleop) to avoid
        velocity/acceleration spikes from the step change in targets.
        """
        with self._lock:
            self.prev_joints_left = self._target_left.copy()
            self.prev_joints_right = self._target_right.copy()
        self.prev_velocities_left = np.zeros(7)
        self.prev_velocities_right = np.zeros(7)
        # ButterworthFilter has no reset(); zero out internal state directly
        for filt in (self.vel_filter_left, self.vel_filter_right,
                     self.accel_filter_left, self.accel_filter_right):
            filt.x1[:] = 0.0
            filt.x2[:] = 0.0
            filt.y1[:] = 0.0
            filt.y2[:] = 0.0

    def get_reward(self):
        return 0

    def _reset_joints(self):
        """Move arms to zero position."""
        # Pump one step to populate the reading cache before move_to_position
        self.controller_left.step(self.dt)
        self.controller_right.step(self.dt)
        self.controller_left.move_to_position(np.zeros(6))
        self.controller_right.move_to_position(np.zeros(6))
        time.sleep(2.0)

    def _reset_gripper(self):
        """Open grippers."""
        self.controller_left.set_gripper(DEFAULT_GRIPPER_OPEN)
        self.controller_right.set_gripper(DEFAULT_GRIPPER_OPEN)
        time.sleep(1.0)

    def reset(self, *, fake=False):
        """Reset the environment."""
        self._stop_control_thread()

        if not fake:
            self._reset_joints()
            self._reset_gripper()

        # Initialize targets from current position
        qpos = self.get_qpos()
        with self._lock:
            self._target_left = np.concatenate([qpos[:6], [qpos[6] * DEFAULT_GRIPPER_OPEN]])
            self._target_right = np.concatenate([qpos[7:13], [qpos[13] * DEFAULT_GRIPPER_OPEN]])
            self._prev_target_left = self._target_left.copy()
            self._prev_target_right = self._target_right.copy()
            self._step_counter = 0

        # Initialize filter states
        self.prev_joints_left = self._target_left.copy()
        self.prev_joints_right = self._target_right.copy()
        self.prev_velocities_left = np.zeros(7)
        self.prev_velocities_right = np.zeros(7)

        self._start_control_thread()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        """Execute one step with the given action (14D, called at 50Hz)."""
        if len(action) != 14:
            raise ValueError(f"Action must have 14 elements, got {len(action)}")

        # Parse 14D action
        left_arm = action[:6]
        left_gripper_normalized = action[6]
        right_arm = action[7:13]
        right_gripper_normalized = action[13]

        # Denormalize gripper: normalized * -0.20 gives joint space position
        left_gripper_joint = left_gripper_normalized * DEFAULT_GRIPPER_OPEN
        right_gripper_joint = right_gripper_normalized * DEFAULT_GRIPPER_OPEN

        # Update shared target buffer
        with self._lock:
            # Shift current to previous when new target arrives
            self._prev_target_left = self._target_left.copy()
            self._prev_target_right = self._target_right.copy()
            self._target_left = np.concatenate([left_arm, [left_gripper_joint]])
            self._target_right = np.concatenate([right_arm, [right_gripper_joint]])
            self._step_counter = 0  # Reset local step for interpolation

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def shutdown(self):
        """Return arms to zero and release all hardware resources."""
        self._stop_control_thread()
        try:
            # Pump one step to populate reading cache
            self.controller_left.step(self.dt)
            self.controller_right.step(self.dt)
            # Smooth return to zero (quintic interpolation, blocks ~2s each)
            self.controller_left.return_to_zero(duration=2.0)
            self.controller_right.return_to_zero(duration=2.0)
            # Wait for motion to fully settle
            time.sleep(5)
            # Hold position before releasing hardware
            self.controller_left.set_mode(ControlMode.IDLE)
            self.controller_right.set_mode(ControlMode.IDLE)
        except Exception as e:
            logging.warning(f"Error during return-to-zero: {e}")
        # Release hardware
        if hasattr(self, "controller_left"):
            self.controller_left.stop()
        if hasattr(self, "controller_right"):
            self.controller_right.stop()
        if hasattr(self, "camera_interface"):
            self.camera_interface.stop()

    def __del__(self):
        """Clean up resources."""
        self.shutdown()


def make_real_env(
    can_port_left: str = DEFAULT_CAN_PORTS["left_arm"],
    can_port_right: str = DEFAULT_CAN_PORTS["right_arm"],
    camera_serial_numbers: list[str] | None = None,
    camera_width: int = 424,
    camera_height: int = 240,
    camera_fps: int = 60,
    *,
    enable_depth: bool = False,
    auto_exposure: bool = True,
    # exposure: int = 10000,
    gain: int = 16,
) -> RealEnv:
    """Factory function to create RealEnv instance."""
    if camera_serial_numbers is None:
        camera_serial_numbers = [DEFAULT_LEFT_WRIST_CAM_SERIAL, DEFAULT_RIGHT_WRIST_CAM_SERIAL, DEFAULT_TOP_CAM_SERIAL]
    return RealEnv(
        can_port_left=can_port_left,
        can_port_right=can_port_right,
        camera_serial_numbers=camera_serial_numbers,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        enable_depth=enable_depth,
        auto_exposure=auto_exposure,
        # exposure=exposure,
        gain=gain,
    )
