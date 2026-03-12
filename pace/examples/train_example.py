#!/usr/bin/env python3
"""Example training script using PACE Runtime.

Demonstrates how to integrate pace_runtime for robust checkpointing
and completion marking.
"""

import argparse
import os
import time

# In your actual code, you would import your ML framework
# import torch

from pace_runtime import (
    checkpoint_transaction,
    mark_training_done,
    find_latest_checkpoint,
)
from pace_runtime.signals import GracefulShutdown


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    # Check environment variables (set by PACE)
    run_name = os.environ.get("PACE_RUN_NAME", "unknown")
    checkpoint_dir = os.environ.get("PACE_CHECKPOINT_DIR", args.checkpoint_dir)
    resume_dir = os.environ.get("PACE_RESUME_DIR", args.resume_from)

    print(f"Run: {run_name}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Resume from: {resume_dir}")

    # Load checkpoint if resuming
    start_step = 0
    if resume_dir:
        print(f"Loading checkpoint from: {resume_dir}")
        # checkpoint = torch.load(f"{resume_dir}/model.pt")
        # model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # start_step = checkpoint["step"]
        start_step = 5000  # Simulated

    # Setup graceful shutdown handling
    with GracefulShutdown() as shutdown:
        for step in range(start_step, args.max_steps):
            # Simulate training step
            time.sleep(0.01)

            # Check for shutdown signal
            if shutdown.should_stop:
                print(f"Received SIGTERM at step {step}, saving checkpoint...")
                save_checkpoint(checkpoint_dir, step)
                print("Exiting gracefully")
                return

            # Check for checkpoint request (SIGUSR1)
            if shutdown.should_checkpoint:
                print(f"Checkpoint requested at step {step}")
                save_checkpoint(checkpoint_dir, step)
                shutdown.clear_checkpoint_request()

            # Regular checkpoint saving
            if step > 0 and step % args.checkpoint_every == 0:
                save_checkpoint(checkpoint_dir, step)

            if step % 100 == 0:
                print(f"Step {step}/{args.max_steps}")

    # Training complete
    print("Training complete!")
    run_dir = os.path.dirname(checkpoint_dir)
    mark_training_done(run_dir, metadata={"final_step": args.max_steps})


def save_checkpoint(checkpoint_dir: str, step: int):
    """Save a checkpoint using PACE transaction wrapper."""
    tag = f"global_step_{step}"

    with checkpoint_transaction(
        checkpoint_root=checkpoint_dir,
        tag=tag,
        required_files=["model.pt", "optimizer.pt"],
    ) as tx_dir:
        # Save your model files into tx_dir
        # torch.save({
        #     "model": model.state_dict(),
        #     "step": step,
        # }, f"{tx_dir}/model.pt")
        #
        # torch.save({
        #     "optimizer": optimizer.state_dict(),
        # }, f"{tx_dir}/optimizer.pt")

        # Simulated save
        (tx_dir / "model.pt").write_text(f"model at step {step}")
        (tx_dir / "optimizer.pt").write_text(f"optimizer at step {step}")

    print(f"Saved checkpoint: {tag}")


if __name__ == "__main__":
    main()
