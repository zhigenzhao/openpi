import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import xr_interactive_runtime as _xr_interactive_runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.gim import env as _env
from examples.gim.gim_openpi_teleop_controller import GIMDualArmOpenPITeleopController
from examples.gim.robot_utils import DEFAULT_CAN_PORTS
from examples.gim.robot_utils import DEFAULT_LEFT_WRIST_CAM_SERIAL
from examples.gim.robot_utils import DEFAULT_RIGHT_WRIST_CAM_SERIAL
from examples.gim.robot_utils import DEFAULT_TOP_CAM_SERIAL


@dataclasses.dataclass
class Args:
    """Command line arguments for the GIM dual arm XR Interactive client."""

    host: str = "localhost"
    port: int = 8000

    action_horizon: int = 50
    execution_horizon: int = 10
    inference_delay: int = 6

    num_episodes: int = 1
    max_episode_steps: int = 10000000

    # Teleoperation options
    enable_teleop: bool = True
    enable_log_data: bool = True
    log_dir: str = "logs/pi0_gim_tshirt_dagger"
    log_freq: float = 50.0

    # Robot specific arguments
    can_port_right: str = DEFAULT_CAN_PORTS["right_arm"]
    can_port_left: str = DEFAULT_CAN_PORTS["left_arm"]
    camera_serials: list[str] = dataclasses.field(
        default_factory=lambda: [DEFAULT_LEFT_WRIST_CAM_SERIAL, DEFAULT_RIGHT_WRIST_CAM_SERIAL, DEFAULT_TOP_CAM_SERIAL]
    )


def create_teleop_controller(args: Args, environment: _env.GIMRealEnvironment):
    """Create GIMDualArmOpenPITeleopController for XR teleoperation if enabled."""
    if not args.enable_teleop:
        return None

    try:
        teleop_controller = GIMDualArmOpenPITeleopController(
            environment=environment,
            scale_factor=1.5,
            enable_log_data=False,
        )

        logging.info("GIMDualArmOpenPITeleopController created successfully")
        return teleop_controller

    except Exception as e:
        logging.error(f"Failed to create teleoperation controller: {e}")
        logging.info("XR teleoperation will not be available")
        return None


def main(args: Args) -> None:
    """Main function to run the GIM dual arm environment with XR controller controls."""
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    environment = _env.GIMRealEnvironment(
        can_port_left=args.can_port_left,
        can_port_right=args.can_port_right,
        camera_serial_numbers=args.camera_serials,
    )

    teleop_controller = create_teleop_controller(args, environment)

    xr_interactive_runtime = _xr_interactive_runtime.XRInteractiveRuntime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.RealTimeActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                execution_horizon=args.execution_horizon,
                inference_delay=args.inference_delay,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        teleop_controller=teleop_controller,
        enable_log_data=args.enable_log_data,
        log_dir=args.log_dir,
        log_freq=args.log_freq,
    )

    try:
        xr_interactive_runtime.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        logging.info("Shutting down, returning to zero...")
        environment.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
