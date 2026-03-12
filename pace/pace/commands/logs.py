"""PACE logs command implementation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from pace.commands.remote import (
    execution_mode,
    make_executor,
    remote_attempt_manifest_path,
    remote_manifest_path,
    scheduler_log_dir,
    ssh_read_yaml,
)
from pace.config import load_config
from pace.config.loader import find_config
from pace.core.registry import RunRegistry


def run_logs(
    run_name: str,
    config_path: str | None = None,
    force_remote: bool = False,
    force_local: bool = False,
    attempt_id: int | None = None,
    follow: bool = False,
) -> None:
    """View logs for a run.

    Args:
        run_name: Name of the run.
        config_path: Path to pace.yaml.
        attempt_id: Specific attempt number.
        follow: Follow log output (like tail -f).
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

    if mode == "local_pc":
        executor = make_executor(config)
        manifest = ssh_read_yaml(executor, remote_manifest_path(config, run_name))
        if not manifest:
            click.echo(f"Run not found remotely: {run_name}", err=True)
            raise SystemExit(1)

        if attempt_id is None:
            attempt_id = int(manifest.get("latest_attempt", 0))
        if attempt_id == 0:
            click.echo("No attempts found for this run.")
            return

        attempt = ssh_read_yaml(
            executor, remote_attempt_manifest_path(config, run_name, attempt_id)
        )
        if not attempt:
            click.echo(f"Attempt not found: {attempt_id}", err=True)
            raise SystemExit(1)

        slurm_job_id = attempt.get("slurm_job_id")
        remote_logs_dir = scheduler_log_dir(config, run_name)
        click.echo(f"Logs for {run_name}, attempt #{attempt_id}")
        click.echo(f"SLURM Job ID: {slurm_job_id}")
        click.echo(f"Logs directory: {remote_logs_dir}")
        click.echo("")

        if not slurm_job_id:
            click.echo("No SLURM job ID found for this attempt.")
            return

        out_file = f"{remote_logs_dir}/slurm_{slurm_job_id}.out"
        err_file = f"{remote_logs_dir}/slurm_{slurm_job_id}.err"

        if follow:
            executor.run(
                ["bash", "-lc", f"test -f {out_file} && tail -f {out_file} || echo 'Log file not found: {out_file}'"],
                remote_cwd=config.remote.project_dir,
                stream_output=True,
            )
            return

        click.echo(f"=== STDOUT: {out_file} ===")
        executor.run(
            ["bash", "-lc", f"test -f {out_file} && tail -n 50 {out_file} || echo 'Log file not found: {out_file}'"],
            remote_cwd=config.remote.project_dir,
            stream_output=True,
        )
        click.echo(f"\n=== STDERR: {err_file} ===")
        executor.run(
            ["bash", "-lc", f"test -f {err_file} && tail -n 20 {err_file} || true"],
            remote_cwd=config.remote.project_dir,
            stream_output=True,
        )
        return

    registry = RunRegistry.from_config(config)

    # Check if run exists
    if not registry.run_exists(config.project, run_name):
        click.echo(f"Run not found: {run_name}", err=True)
        raise SystemExit(1)

    manifest = registry.load_manifest(config.project, run_name)

    # Determine which attempt to show
    if attempt_id is None:
        attempt_id = manifest.latest_attempt

    if attempt_id == 0:
        click.echo("No attempts found for this run.")
        return

    # Load attempt to get job ID
    attempt = registry.load_attempt(config.project, run_name, attempt_id)
    logs_dir = Path(scheduler_log_dir(config, run_name))

    click.echo(f"Logs for {run_name}, attempt #{attempt_id}")
    click.echo(f"SLURM Job ID: {attempt.slurm_job_id}")
    click.echo(f"Logs directory: {logs_dir}")
    click.echo("")

    # Find SLURM log files
    if attempt.slurm_job_id:
        out_pattern = f"slurm_{attempt.slurm_job_id}.out"
        err_pattern = f"slurm_{attempt.slurm_job_id}.err"

        out_file = logs_dir / out_pattern
        err_file = logs_dir / err_pattern

        if out_file.exists():
            click.echo(f"=== STDOUT: {out_file} ===")
            if follow:
                # Use tail -f
                subprocess.run(["tail", "-f", str(out_file)])
            else:
                # Show last 50 lines
                subprocess.run(["tail", "-n", "50", str(out_file)])
        else:
            click.echo(f"Log file not found: {out_file}")

        if err_file.exists() and not follow:
            click.echo(f"\n=== STDERR: {err_file} ===")
            subprocess.run(["tail", "-n", "20", str(err_file)])
    else:
        click.echo("No SLURM job ID found for this attempt.")

        # List available log files
        if logs_dir.exists():
            log_files = list(logs_dir.glob("slurm_*.out"))
            if log_files:
                click.echo("\nAvailable log files:")
                for f in sorted(log_files)[-5:]:
                    click.echo(f"  {f.name}")
