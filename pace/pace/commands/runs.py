"""PACE runs command implementation."""

from __future__ import annotations

from pathlib import Path

import click

from pace.commands.remote import execution_mode, make_executor, ssh_read_yaml
from pace.config import load_config
from pace.config.loader import find_config
from pace.core.registry import RunRegistry


def run_list(
    config_path: str | None = None,
    force_remote: bool = False,
    force_local: bool = False,
) -> None:
    """List all runs for the project.

    Args:
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

    if mode == "local_pc":
        executor = make_executor(config)
        project_dir = f"{config.registry_root}/{config.project}"
        completed = executor.run(
            ["bash", "-lc", f"ls -1 {project_dir} 2>/dev/null || true"],
            remote_cwd=config.remote.project_dir,
            stream_output=False,
        )
        runs = [r.strip() for r in (completed.stdout or "").splitlines() if r.strip()]
        if not runs:
            click.echo(f"No runs found for project: {config.project}")
            click.echo(f"Registry root: {config.registry_root}")
            return

        click.echo(f"\nRuns for project: {config.project}")
        click.echo(f"{'='*60}")
        click.echo(f"{'Name':<30} {'Attempts':<10} {'Created'}")
        click.echo(f"{'-'*60}")
        for run_name in sorted(runs):
            manifest = ssh_read_yaml(
                executor, f"{config.registry_root}/{config.project}/{run_name}/manifest.yaml"
            ) or {}
            created = (manifest.get("created_at") or "unknown")[:10]
            attempts = manifest.get("latest_attempt", 0)
            click.echo(f"{run_name:<30} {attempts:<10} {created}")

        click.echo(f"\nTotal: {len(runs)} run(s)")
        return

    registry = RunRegistry.from_config(config)

    # List runs
    runs = registry.list_runs(config.project)

    if not runs:
        click.echo(f"No runs found for project: {config.project}")
        click.echo(f"Registry root: {config.registry_root}")
        return

    click.echo(f"\nRuns for project: {config.project}")
    click.echo(f"{'='*60}")
    click.echo(f"{'Name':<30} {'Attempts':<10} {'Created'}")
    click.echo(f"{'-'*60}")

    for run_name in sorted(runs):
        manifest = registry.load_manifest(config.project, run_name)
        created = manifest.created_at[:10] if manifest.created_at else "unknown"
        click.echo(f"{run_name:<30} {manifest.latest_attempt:<10} {created}")

    click.echo(f"\nTotal: {len(runs)} run(s)")
