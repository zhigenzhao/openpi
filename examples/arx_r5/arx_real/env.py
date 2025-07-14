from typing import List, Optional

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.arx_r5.arx_real import real_env as _real_env


class ARXRealEnvironment(_environment.Environment):
    """An environment for an ARX R5 robot on real hardware."""

    def __init__(
        self,
        can_port: str,
        camera_serial_numbers: Optional[List[str]],
    ) -> None:
        self._env = _real_env.make_real_env(can_port=can_port, camera_serial_numbers=camera_serial_numbers)
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

        # # Process images if available
        # for cam_name in obs["images"]:
        #     img = image_tools.convert_to_uint8(
        #         image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
        #     )
        #     obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "image": obs["images"]["base"],
            "wrist_image": obs["images"]["right_wrist"],
            "prompt": "align the green block with the inside of the blue corner",
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
