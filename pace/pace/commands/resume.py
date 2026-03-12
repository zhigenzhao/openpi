"""PACE resume command implementation."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import click

from pace.commands.remote import (
    execution_mode,
    make_executor,
    remote_run_dir,
    resolve_template_path,
    ssh_read_yaml,
)
from pace.config import load_config
from pace.config.loader import find_config
from pace.config.models import ProjectConfig, ResumeStrategy
from pace.core.resume import (
    discover_latest_checkpoint_local,
    map_checkpoint_host_to_container,
    resolve_resume_output_paths,
)
from pace.core.registry import RunRegistry


def get_checkpoint_host_path(config: ProjectConfig, run_name: str) -> str | None:
    """Get the resolved checkpoint host path from persistent_outputs config.

    Args:
        config: Project configuration.
        run_name: Run name for template resolution.

    Returns:
        Resolved host path to checkpoints directory, or None if not found.
    """
    output_paths = resolve_resume_output_paths(config, run_name)
    if output_paths:
        return output_paths.host_root
    # Backward-compat fallback when no explicit checkpoint output is configured.
    checkpoint_output = config.get_persistent_output_by_role("checkpoints")
    if checkpoint_output is None:
        return None
    return resolve_template_path(config, checkpoint_output.host_path, run_name)


def _discover_latest_checkpoint_remote(
    executor,
    root_dir: str,
    run_name: str,
    *,
    strategy: ResumeStrategy,
    search_recursive: bool,
    checkpoint_marker: str,
    sort_pattern: str | None,
    remote_cwd: str,
) -> str | None:
    """Discover latest checkpoint directory on remote host using Python."""
    payload = {
        "root_dir": root_dir,
        "run_name": run_name,
        "strategy": strategy.value,
        "search_recursive": search_recursive,
        "checkpoint_marker": checkpoint_marker,
        "sort_pattern": sort_pattern,
    }
    script = (
        "python3 - <<'PY'\n"
        "import json\n"
        "import re\n"
        "from pathlib import Path\n"
        f"cfg = json.loads({json.dumps(payload)!r})\n"
        "root = Path(cfg['root_dir'])\n"
        "run_root = root / cfg['run_name']\n"
        "if run_root.exists() and run_root.is_dir():\n"
        "    root = run_root\n"
        "if not root.exists():\n"
        "    print('')\n"
        "    raise SystemExit(0)\n"
        "strategy = cfg['strategy']\n"
        "recursive = cfg['search_recursive']\n"
        "marker = cfg['checkpoint_marker']\n"
        "sort_pattern = cfg['sort_pattern']\n"
        "\n"
        "def marker_dirs():\n"
        "    if recursive:\n"
        "        iterator = root.rglob(marker)\n"
        "    else:\n"
        "        iterator = ((d / marker) for d in root.iterdir() if d.is_dir())\n"
        "    out = []\n"
        "    for marker_path in iterator:\n"
        "        if marker_path.is_file():\n"
        "            out.append(marker_path.parent)\n"
        "    return out\n"
        "\n"
        "def candidate_dirs():\n"
        "    if recursive:\n"
        "        return [d for d in root.rglob('*') if d.is_dir()]\n"
        "    return [d for d in root.iterdir() if d.is_dir()]\n"
        "\n"
        "def sorted_paths(candidates):\n"
        "    deduped = sorted({p for p in candidates if p.name and not p.name.startswith('.')}, key=lambda p: p.as_posix())\n"
        "    if not deduped:\n"
        "        return []\n"
        "    if not sort_pattern:\n"
        "        return deduped\n"
        "    pattern = re.compile(sort_pattern)\n"
        "    sortable = []\n"
        "    for p in deduped:\n"
        "        m = pattern.search(p.name) or pattern.search(p.as_posix())\n"
        "        if not m:\n"
        "            continue\n"
        "        k = m.group('key')\n"
        "        try:\n"
        "            key = (0, int(k))\n"
        "        except ValueError:\n"
        "            key = (1, k)\n"
        "        sortable.append((key, p.as_posix(), p))\n"
        "    sortable.sort(key=lambda item: (item[0], item[1]))\n"
        "    return [item[2] for item in sortable]\n"
        "\n"
        "markers = marker_dirs()\n"
        "if strategy == 'latest_safe':\n"
        "    ranked = sorted_paths(markers)\n"
        "elif strategy == 'latest':\n"
        "    ranked = sorted_paths(markers + candidate_dirs())\n"
        "else:\n"
        "    ranked = []\n"
        "print(ranked[-1].as_posix() if ranked else '')\n"
        "PY"
    )
    completed = executor.run(
        ["bash", "-lc", script],
        remote_cwd=remote_cwd,
        stream_output=False,
    )
    if completed.returncode != 0:
        raise click.ClickException(
            "Failed to discover remote checkpoint: "
            f"{completed.stderr or completed.stdout}"
        )
    discovered = (completed.stdout or "").strip()
    return discovered or None


def _map_resume_container_path(
    config: ProjectConfig,
    run_name: str,
    checkpoint_host: str,
) -> str:
    """Resolve resume path inside container from host checkpoint path."""
    output_paths = resolve_resume_output_paths(config, run_name)
    if output_paths is None:
        # Backward-compat fallback for old configs.
        return checkpoint_host
    return map_checkpoint_host_to_container(checkpoint_host, output_paths)


def run_resume(
    run_name: str,
    config_path: str | None = None,
    force_remote: bool = False,
    force_local: bool = False,
    checkpoint_path: str | None = None,
    dry_run: bool = False,
) -> None:
    """Resume an existing run from checkpoint.

    Args:
        run_name: Name of the run to resume.
        config_path: Path to pace.yaml.
        checkpoint_path: Specific checkpoint to resume from.
        dry_run: Print what would be done without executing.
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
    if not config.resume.enabled:
        raise click.ClickException(
            "Resume is disabled in config (resume.enabled=false)."
        )
    mode = execution_mode(config, force_remote=force_remote, force_local=force_local)
    if config.resume.strategy == ResumeStrategy.NONE and not checkpoint_path:
        raise click.ClickException(
            "resume.strategy is 'none'; pass --checkpoint to resume explicitly."
        )

    checkpoints_root = get_checkpoint_host_path(config, run_name)
    if checkpoints_root is None:
        checkpoints_root = f"{remote_run_dir(config, run_name)}/checkpoints"

    if mode == "local_pc":
        executor = make_executor(config)
        manifest = ssh_read_yaml(
            executor, f"{remote_run_dir(config, run_name)}/manifest.yaml"
        )
        if not manifest:
            click.echo(f"Run not found remotely: {run_name}", err=True)
            raise SystemExit(1)

        # Determine checkpoint on remote host when not explicitly provided.
        remote_checkpoint = checkpoint_path
        if not remote_checkpoint:
            click.echo(f"Looking for checkpoints in: {checkpoints_root}")
            remote_checkpoint = _discover_latest_checkpoint_remote(
                executor=executor,
                root_dir=checkpoints_root,
                run_name=run_name,
                strategy=config.resume.strategy,
                search_recursive=config.resume.search_recursive,
                checkpoint_marker=config.resume.checkpoint_marker,
                sort_pattern=config.resume.sort_pattern,
                remote_cwd=config.remote.project_dir,
            )
        else:
            completed = executor.run(
                [
                    "bash",
                    "-lc",
                    f"test -d {shlex.quote(remote_checkpoint)} && echo yes || echo no",
                ],
                remote_cwd=config.remote.project_dir,
                stream_output=False,
            )
            if (completed.stdout or "").strip() != "yes":
                click.echo(f"Checkpoint not found: {remote_checkpoint}", err=True)
                raise SystemExit(1)

        if not remote_checkpoint:
            click.echo("No valid checkpoint found to resume from.", err=True)
            click.echo(f"Searched in: {checkpoints_root}")
            click.echo("Use --checkpoint to specify one manually.")
            raise SystemExit(1)

        try:
            remote_checkpoint_container = _map_resume_container_path(
                config, run_name, remote_checkpoint
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"Resuming from remote checkpoint: {remote_checkpoint}")
        if dry_run:
            click.echo("\nWould submit new attempt with resume from this checkpoint.")
            return

        from pace.commands.submit import run_submit

        run_submit(
            run_name=run_name,
            config_path=config_path,
            force_remote=True,
            force_local=False,
            dry_run=False,
            no_submit=False,
            user_args=[],
            resume_from_override=remote_checkpoint,
            resume_from_container_override=remote_checkpoint_container,
        )
        return

    registry = RunRegistry.from_config(config)

    # Check if run exists
    if not registry.run_exists(config.project, run_name):
        click.echo(f"Run not found: {run_name}", err=True)
        raise SystemExit(1)

    # Check completion
    markers_dir = registry.markers_dir(config.project, run_name)
    done_marker = markers_dir / config.completion.marker_file
    if done_marker.exists():
        click.echo(f"Training already complete for: {run_name}")
        return

    # Find checkpoint to resume from
    if checkpoint_path:
        resume_checkpoint = Path(checkpoint_path)
        if not resume_checkpoint.exists():
            click.echo(f"Checkpoint not found: {checkpoint_path}", err=True)
            raise SystemExit(1)
    else:
        checkpoints_dir = Path(checkpoints_root)
        run_scoped_dir = checkpoints_dir / run_name
        if run_scoped_dir.exists() and run_scoped_dir.is_dir():
            checkpoints_dir = run_scoped_dir

        click.echo(f"Looking for checkpoints in: {checkpoints_dir}")

        resume_checkpoint = discover_latest_checkpoint_local(
            checkpoints_dir,
            strategy=config.resume.strategy,
            search_recursive=config.resume.search_recursive,
            checkpoint_marker=config.resume.checkpoint_marker,
            sort_pattern=config.resume.sort_pattern,
        )
        if not resume_checkpoint:
            click.echo("No valid checkpoint found to resume from.", err=True)
            click.echo(f"Searched in: {checkpoints_dir}")
            click.echo("Use --checkpoint to specify one manually.")
            raise SystemExit(1)

    try:
        resume_checkpoint_container = _map_resume_container_path(
            config, run_name, str(resume_checkpoint)
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Resuming from: {resume_checkpoint}")

    if dry_run:
        click.echo("\nWould submit new attempt with resume from this checkpoint.")
        return

    from pace.commands.submit import run_submit

    click.echo("\nSubmitting resume attempt...")
    run_submit(
        run_name=run_name,
        config_path=config_path,
        force_remote=force_remote,
        force_local=force_local,
        dry_run=dry_run,
        no_submit=False,
        user_args=[],
        resume_from_override=str(resume_checkpoint),
        resume_from_container_override=resume_checkpoint_container,
    )
