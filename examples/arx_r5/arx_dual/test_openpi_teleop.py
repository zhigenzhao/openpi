#!/usr/bin/env python3
"""Standalone test for ARX R5 dual arm OpenPI teleoperation controller.

This script tests the OpenPI teleoperation controller independently
without the full XR interactive runtime integration.
"""

import logging
import time
import signal
import sys
from typing import Optional

from examples.arx_r5.arx_dual.arx_openpi_teleop_controller import ARXDualArmOpenPITeleopController
from examples.arx_r5.arx_dual.env import ARXRealEnvironment
from examples.arx_r5.robot_utils import DEFAULT_CAN_PORTS, DEFAULT_LEFT_WRIST_CAM_SERIAL, DEFAULT_RIGHT_WRIST_CAM_SERIAL, DEFAULT_BASE_CAM_SERIAL


class TeleopTest:
    """Test harness for OpenPI teleoperation controller."""
    
    def __init__(self):
        self.teleop_controller: Optional[ARXDualArmOpenPITeleopController] = None
        self.environment: Optional[ARXRealEnvironment] = None
        self.running = False
        
        # Set up signal handler for graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.shutdown_requested = True
    
    def setup(self):
        """Initialize environment and teleoperation controller."""
        try:
            print("Setting up ARX dual arm environment...")
            
            # Create ARX dual arm environment
            self.environment = ARXRealEnvironment(
                can_port_left=DEFAULT_CAN_PORTS["left_arm"],
                can_port_right=DEFAULT_CAN_PORTS["right_arm"],
                camera_serial_numbers=[
                    DEFAULT_LEFT_WRIST_CAM_SERIAL,
                    DEFAULT_RIGHT_WRIST_CAM_SERIAL,
                    DEFAULT_BASE_CAM_SERIAL,
                ],
            )
            
            # Reset environment
            print("Resetting environment...")
            self.environment.reset()
            
            # Create OpenPI teleoperation controller
            print("Creating OpenPI teleoperation controller...")
            self.teleop_controller = ARXDualArmOpenPITeleopController(
                environment=self.environment,
                scale_factor=0.5,  # Reduce scale for safety
                enable_log_data=False,
            )
            
            print("Setup complete!")
            return True
            
        except Exception as e:
            print(f"Setup failed: {e}")
            logging.error(f"Setup failed: {e}")
            return False
    
    def run_test(self):
        """Run the teleoperation test."""
        if not self.teleop_controller:
            print("Error: Teleoperation controller not initialized")
            return
        
        try:
            print("\n" + "="*60)
            print("ARX DUAL ARM OPENPI TELEOPERATION TEST")
            print("="*60)
            print("Controls:")
            print("  - Hold LEFT GRIP button to control left arm")
            print("  - Hold RIGHT GRIP button to control right arm")
            print("  - Move controllers to control end-effector poses")
            print("  - Use triggers to control grippers")
            print("  - Press Ctrl+C to stop")
            print("="*60)
            
            # Start teleoperation controller
            print("Starting teleoperation controller...")
            self.teleop_controller.start()
            
            self.running = True
            step_count = 0
            last_status_time = time.time()
            
            while self.running and not self.shutdown_requested:
                try:
                    # Get latest teleoperation action
                    teleop_action = self.teleop_controller.get_latest_action()

                    if teleop_action is not None:
                        print(f"[Step {step_count}] Applying teleoperation action")
                        # Show first few joint values for debugging
                        actions = teleop_action['actions']
                        print(f"  Left arm: [{actions[0]:.3f}, {actions[1]:.3f}, {actions[2]:.3f}...]")
                        print(f"  Right arm: [{actions[7]:.3f}, {actions[8]:.3f}, {actions[9]:.3f}...]")
                        print(f"  Grippers: left={actions[6]:.3f}, right={actions[13]:.3f}")
                        
                        # Apply action to environment
                        self.environment.apply_action(teleop_action)
                        step_count += 1
                    else:
                        # Check if teleoperation should be active but no action generated
                        is_active = self.teleop_controller.is_active()
                        if is_active:
                            print(f"[WARNING] Teleoperation active but no action generated!")

                    # Print status periodically
                    current_time = time.time()
                    if current_time - last_status_time > 2.0:
                        is_active = self.teleop_controller.is_active()
                        status = "ACTIVE" if is_active else "IDLE"
                        print(f"[Status] Teleoperation: {status} | Steps: {step_count}")
                        last_status_time = current_time

                    # Sleep for control rate
                    time.sleep(0.02)  # 50Hz

                except Exception as e:
                    print(f"Error in main loop: {e}")
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(0.1)
            
            print("\nTest completed!")
            
        except Exception as e:
            print(f"Test failed: {e}")
            logging.error(f"Test failed: {e}")
    
    def stop(self):
        """Stop the test and clean up resources."""
        print("Stopping teleoperation test...")
        self.running = False
        
        if self.teleop_controller:
            self.teleop_controller.stop()
            self.teleop_controller = None
        
        if self.environment:
            try:
                # Reset to safe position
                self.environment.reset()
            except Exception as e:
                print(f"Error resetting environment: {e}")
            self.environment = None
        
        print("Cleanup complete")


def main():
    """Main function to run teleoperation test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test = TeleopTest()
    
    try:
        if test.setup():
            test.run_test()
        else:
            print("Setup failed, exiting...")
            return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
        return 1
    finally:
        test.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())