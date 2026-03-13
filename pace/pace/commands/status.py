"""PACE status command implementation."""

from __future__ import annotations

import time
from pathlib import Path

import click

from pace.backends.scheduler import SlurmBackend
from pace.commands.remote import (
    execution_mode,
    make_executor,
    remote_attempt_manifest_path,
    remote_manifest_path,
    remote_run_dir,
    scheduler_log_dir,
    remote_state_path,
    ssh_read_json,
    ssh_read_yaml,
)
from pace.config import load_config
from pace.config.loader import find_config
from pace.core.registry import RunRegistry


def _resolve_path_template(
    template: str,
    run_name: str,
    project: str,
    registry_root: str,
) -> str:
    """Resolve basic run path template variables."""
    return (
        template.replace("{run_name}", run_name)
        .replace("{project}", project)
        .replace("{registry_root}", registry_root)
    )


def run_status(
    run_name: str,
    config_path: str | None = None,
    force_remote: bool = False,
    force_local: bool = False,
) -> None:
    """Show status of a run.

    Args:
        run_name: Name of the run.
        config_path: Path to pace.yaml.
    """
    # Find and load config
    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = find_config()
        if not cfg_path:
            click.echo("No pace.yaml found.", err=True)
            raise SystemExit(1)

    config = load_config(cfg_path)
    mode = execution_mode(config, force_remote=force_remote, force_local=force_local)

    checkpoints_output = config.get_persistent_output_by_role("checkpoints")

    if mode == "local_pc":
        executor = make_executor(config)
        manifest = ssh_read_yaml(executor, remote_manifest_path(config, run_name))
        if not manifest:
            click.echo(f"Run not found remotely: {run_name}", err=True)
            raise SystemExit(1)

        state = ssh_read_json(executor, remote_state_path(config, run_name)) or {}
        latest_attempt = int(manifest.get("latest_attempt", 0))

        click.echo(f"\n{'='*50}")
        click.echo(f"Run: {run_name}")
        click.echo(f"Project: {manifest.get('project', config.project)}")
        click.echo(f"Created: {manifest.get('created_at', 'unknown')}")
        click.echo(f"{'='*50}")
        click.echo(f"\nAttempts: {latest_attempt}")

        if latest_attempt > 0:
            attempt = ssh_read_yaml(
                executor,
                remote_attempt_manifest_path(config, run_name, latest_attempt),
            ) or {}
            click.echo(f"\nLatest attempt (#{latest_attempt}):")
            click.echo(f"  Submitted: {attempt.get('submitted_at', 'unknown')}")
            click.echo(f"  Status: {attempt.get('status', 'unknown')}")
            slurm_job_id = attempt.get("slurm_job_id")
            if slurm_job_id:
                click.echo(f"  SLURM Job ID: {slurm_job_id}")
                state_cmd = (
                    f"squeue -j {slurm_job_id} -h -o %T 2>/dev/null | grep . || "
                    f"sacct -j {slurm_job_id} -n -o State -P 2>/dev/null | head -1"
                )
                _state_retries = 3
                _state_delay = 5
                job_state = "UNKNOWN"
                for _i in range(_state_retries):
                    completed = executor.run(
                        ["bash", "-lc", state_cmd],
                        remote_cwd=config.remote.project_dir,
                        stream_output=False,
                    )
                    if completed.returncode == 0:
                        job_state = (completed.stdout or "").strip() or "UNKNOWN"
                        break
                    if _i < _state_retries - 1:
                        click.echo(
                            f"  Warning: failed to query SLURM state, retrying in {_state_delay}s..."
                            f" ({_i + 1}/{_state_retries})",
                            err=True,
                        )
                        time.sleep(_state_delay)
                click.echo(f"  SLURM State: {job_state}")

        done_marker = f"{remote_run_dir(config, run_name)}/markers/{config.completion.marker_file}"
        done_check = executor.run(
            ["bash", "-lc", f"test -f {done_marker} && echo yes || echo no"],
            remote_cwd=config.remote.project_dir,
            stream_output=False,
        )
        if (done_check.stdout or "").strip() == "yes":
            click.echo("\nTraining: COMPLETE")
        else:
            click.echo("\nTraining: IN PROGRESS")

        click.echo("\nPaths:")
        click.echo(f"  Run dir: {remote_run_dir(config, run_name)}")
        click.echo(f"  Scheduler logs: {scheduler_log_dir(config, run_name)}")
        if checkpoints_output:
            checkpoint_path = _resolve_path_template(
                checkpoints_output.host_path,
                run_name=run_name,
                project=config.project,
                registry_root=config.registry_root,
            )
            click.echo(f"  Checkpoints: {checkpoint_path}")
        else:
            click.echo(f"  Checkpoints: {remote_run_dir(config, run_name)}/checkpoints")
        if state.get("latest_checkpoint"):
            click.echo(f"  Latest checkpoint: {state['latest_checkpoint']}")
        return

    registry = RunRegistry.from_config(config)

    # Check if run exists
    if not registry.run_exists(config.project, run_name):
        click.echo(f"Run not found: {run_name}", err=True)
        raise SystemExit(1)

    # Load manifest and state
    manifest = registry.load_manifest(config.project, run_name)
    state = registry.load_state(config.project, run_name)

    click.echo(f"\n{'='*50}")
    click.echo(f"Run: {run_name}")
    click.echo(f"Project: {manifest.project}")
    click.echo(f"Created: {manifest.created_at}")
    click.echo(f"{'='*50}")

    # Attempt info
    click.echo(f"\nAttempts: {manifest.latest_attempt}")

    if manifest.latest_attempt > 0:
        latest = registry.get_latest_attempt(config.project, run_name)
        if latest:
            click.echo(f"\nLatest attempt (#{latest.attempt_id}):")
            click.echo(f"  Submitted: {latest.submitted_at}")
            click.echo(f"  Status: {latest.status.value}")

            if latest.slurm_job_id:
                click.echo(f"  SLURM Job ID: {latest.slurm_job_id}")

                # Get live job state
                slurm = SlurmBackend()
                job_state = slurm.get_job_state(latest.slurm_job_id)
                click.echo(f"  SLURM State: {job_state}")

            if latest.resume_from:
                click.echo(f"  Resumed from: {latest.resume_from}")

    # Checkpoint info
    checkpoints_dir = registry.checkpoints_dir(config.project, run_name)
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.iterdir())
        click.echo(f"\nCheckpoints: {len(checkpoints)}")
        if state.latest_checkpoint:
            click.echo(f"  Latest: {state.latest_checkpoint}")

    # Completion status
    markers_dir = registry.markers_dir(config.project, run_name)
    done_marker = markers_dir / config.completion.marker_file
    if done_marker.exists():
        click.echo("\nTraining: COMPLETE")
    else:
        click.echo("\nTraining: IN PROGRESS")

    # Paths
    click.echo("\nPaths:")
    click.echo(f"  Run dir: {registry.run_dir(config.project, run_name)}")
    click.echo(f"  Scheduler logs: {scheduler_log_dir(config, run_name)}")
    if checkpoints_output:
        local_checkpoint_path = _resolve_path_template(
            checkpoints_output.host_path,
            run_name=run_name,
            project=config.project,
            registry_root=config.registry_root,
        )
        click.echo(f"  Checkpoints: {local_checkpoint_path}")
    else:
        click.echo(f"  Checkpoints: {checkpoints_dir}")
