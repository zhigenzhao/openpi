import dataclasses
import logging
import pathlib
from typing import List, Optional

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.arx_r5.arx_dual import env as _env
from examples.arx_r5.robot_utils import DEFAULT_BASE_CAM_SERIAL
from examples.arx_r5.robot_utils import DEFAULT_CAN_PORTS
from examples.arx_r5.robot_utils import DEFAULT_LEFT_WRIST_CAM_SERIAL
from examples.arx_r5.robot_utils import DEFAULT_RIGHT_WRIST_CAM_SERIAL
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Command line arguments for the ARX Real Robot client."""

    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 50
    execution_horizon: int = 35
    inference_delay: int = 5

    num_episodes: int = 1
    max_episode_steps: int = 10000000

    # Robot specific arguments
    can_port_right: str = DEFAULT_CAN_PORTS["right_arm"]
    can_port_left: str = DEFAULT_CAN_PORTS["left_arm"]
    camera_serials: List[str] = dataclasses.field(
        default_factory=lambda: [DEFAULT_LEFT_WRIST_CAM_SERIAL, DEFAULT_RIGHT_WRIST_CAM_SERIAL, DEFAULT_BASE_CAM_SERIAL]
    )

    checkpoint_dir: str = "checkpoints/pi0_arx_dual_low_mem_finetune_multi_step/exp_carpet_fold_0718/79999"
    policy_config: str = "pi0_arx_dual_low_mem_finetune_multi_step"


def main(args: Args) -> None:
    """Main function to run the ARX real robot environment with a policy server."""
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    config = _config.get_config(args.policy_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print(f"norm stats keys: {list(norm_stats.keys())}")

    runtime = _runtime.Runtime(
        environment=_env.ARXRealEnvironment(
            can_port_left=args.can_port_left,
            can_port_right=args.can_port_right,
            camera_serial_numbers=args.camera_serials,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.RTCActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                execution_horizon=args.execution_horizon,
                inference_delay=args.inference_delay,
                action_norm_stats=norm_stats["actions"],
            )
        ),
        subscribers=[],
        max_hz=50,  # Set a safe frequency for the real robot
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
