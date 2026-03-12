from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.gim import real_env as _real_env


class GIMRealEnvironment(_environment.Environment):
    """An environment for a GIM 6-DOF bimanual arm on real hardware."""

    def __init__(
        self,
        can_port_left: str,
        can_port_right: str,
        camera_serial_numbers: list[str] | None,
    ) -> None:
        self._env = _real_env.make_real_env(
            can_port_left=can_port_left, can_port_right=can_port_right, camera_serial_numbers=camera_serial_numbers
        )
        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation

        return {
            "state": obs["qpos"],
            "top_image": obs["images"]["top"],
            "right_wrist_image": obs["images"]["right_wrist"],
            "left_wrist_image": obs["images"]["left_wrist"],
            "prompt": "",
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
