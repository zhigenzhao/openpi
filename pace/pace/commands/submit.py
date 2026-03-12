"""PACE submit command implementation."""

from __future__ import annotations

import re
import shlex
import tempfile
from pathlib import Path

import click

from pace.backends.remote import SSHRemoteError
from pace.backends.scheduler import SlurmBackend
from pace.backends.staging import BindPlanner, StagingPlanner
from pace.commands.remote import (
    execution_mode,
    local_temp_registry_config_path,
    make_executor,
    remote_attempt_dir,
    remote_run_dir,
    sync_remote_run_to_local,
    sync_local_run_to_remote,
    update_remote_attempt_submission,
)
from pace.config import load_config
from pace.config.loader import find_config
from pace.config.models import InjectionCondition, ProjectConfig
from pace.core.context import RuntimeContextBuilder
from pace.core.resume import (
    map_checkpoint_host_to_container,
    resolve_resume_output_paths,
)
from pace.core.registry import RunRegistry
from pace.core.snapshots import (
    create_all_snapshots,
    generate_snapshot_id,
    resolve_snapshot_path,
)
from pace.launch import LaunchPlanBuilder, LaunchRenderer


def _ensure_remote_dirs(
    run_config,
    context,
    executor,
) -> None:
    """Ensure persistent/scheduler host directories exist on remote host."""
    remote_dirs: set[str] = set()
    for output in run_config.persistent_outputs:
        if output.create_if_missing and output.name in context.persistent_paths_host:
            remote_dirs.add(context.persistent_paths_host[output.name])
    remote_dirs.add(context.resolve(run_config.scheduler.log_dir))

    for directory in sorted(remote_dirs):
        completed = executor.run(
            ["bash", "-lc", f"mkdir -p {shlex.quote(directory)}"],
            remote_cwd=run_config.remote.project_dir,
            stream_output=False,
        )
        if completed.returncode != 0:
            raise click.ClickException(
                "Failed to create remote directory "
                f"{directory}: {completed.stderr or completed.stdout}"
            )


def _strip_injected_args(
    candidate_args: list[str],
    config: ProjectConfig,
) -> list[str]:
    """Remove known PACE-injected arguments from inferred user args."""
    args = list(candidate_args)

    # Remove always/resume CLI injections using key-prefix heuristics for key=value args.
    templates: list[str] = list(config.resume.cli_args)
    for cli_cfg in config.injections.cli:
        if cli_cfg.when in {InjectionCondition.ALWAYS, InjectionCondition.RESUME_ONLY}:
            templates.extend(cli_cfg.args)

    for template in templates:
        if "=" in template:
            prefix = template.split("=", 1)[0] + "="
            args = [arg for arg in args if not arg.startswith(prefix)]

    return args


def _infer_user_args_from_latest_launch_plan(
    registry: RunRegistry,
    config: ProjectConfig,
    run_name: str,
    attempt_id: int,
) -> list[str] | None:
    """Infer user args from the latest launch plan when metadata is absent."""
    try:
        launch_plan = registry.load_launch_plan(config.project, run_name, attempt_id)
    except (FileNotFoundError, OSError):
        return None

    command = launch_plan.command
    base_cmd = config.launch.base_command
    if len(command) < len(base_cmd) or command[: len(base_cmd)] != base_cmd:
        return None

    inferred = command[len(base_cmd) :]
    if inferred and inferred[-1] == run_name:
        inferred = inferred[:-1]

    inferred = _strip_injected_args(inferred, config)
    if not inferred:
        return None
    return inferred


def run_submit(
    run_name: str,
    config_path: str | None = None,
    force_remote: bool = False,
    force_local: bool = False,
    dry_run: bool = False,
    no_submit: bool = False,
    user_args: list[str] | None = None,
    resume_from_override: str | None = None,
    resume_from_container_override: str | None = None,
) -> None:
    """Submit a new run.

    Args:
        run_name: Name for the run.
        config_path: Path to pace.yaml.
        dry_run: Print what would be done without executing.
        no_submit: Prepare but don't submit to SLURM.
        user_args: Additional arguments to pass to training script.
        resume_from_override: Host checkpoint directory to resume from.
        resume_from_container_override: Container-visible checkpoint directory.
    """
    user_args = list(user_args or [])

    # Find and load config
    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = find_config()
        if not cfg_path:
            click.echo("No pace.yaml found. Run 'pace init' first.", err=True)
            raise SystemExit(1)

    click.echo(f"Loading config: {cfg_path}")
    config = load_config(cfg_path)
    mode = execution_mode(config, force_remote=force_remote, force_local=force_local)

    effective_cfg_path = cfg_path
    effective_config = config
    if mode == "local_pc":
        effective_cfg_path, _ = local_temp_registry_config_path(cfg_path, config)
        effective_config = load_config(effective_cfg_path)

    # Initialize registry
    registry = RunRegistry.from_config(effective_config)
    is_resume_attempt = resume_from_override is not None

    if mode == "local_pc" and is_resume_attempt:
        # Pull remote run state before creating a new local attempt.
        executor = make_executor(config)
        local_run_dir = registry.run_dir(effective_config.project, run_name)
        if not sync_remote_run_to_local(
            executor=executor,
            remote_run_dir_path=remote_run_dir(config, run_name),
            local_run_dir=local_run_dir,
        ):
            raise click.ClickException(f"Run not found remotely: {run_name}")

    # Check if run exists or create new
    if registry.run_exists(effective_config.project, run_name):
        click.echo(f"Resuming existing run: {run_name}")
        manifest = registry.load_manifest(effective_config.project, run_name)
        resume_from = resume_from_override
        if is_resume_attempt and not user_args:
            latest_attempt = registry.get_latest_attempt(effective_config.project, run_name)
            if latest_attempt and latest_attempt.user_args:
                user_args = list(latest_attempt.user_args)
                click.echo(
                    "Reusing user args from latest attempt: "
                    f"{' '.join(user_args)}"
                )
            elif latest_attempt:
                inferred_args = _infer_user_args_from_latest_launch_plan(
                    registry=registry,
                    config=effective_config,
                    run_name=run_name,
                    attempt_id=latest_attempt.attempt_id,
                )
                if inferred_args:
                    user_args = inferred_args
                    click.echo(
                        "Inferred user args from latest launch plan: "
                        f"{' '.join(user_args)}"
                    )
                else:
                    click.echo(
                        "Latest attempt has no stored user args and inference failed; "
                        "resume will fall back to script defaults."
                    )
    else:
        if is_resume_attempt:
            raise click.ClickException(f"Cannot resume non-existent run: {run_name}")
        click.echo(f"Creating new run: {run_name}")
        manifest = registry.create_run(effective_config, run_name)
        manifest.config_path = str(effective_cfg_path)
        registry.save_manifest(manifest)
        resume_from = resume_from_override

    snapshot_id = generate_snapshot_id()

    # Create or reuse snapshots
    click.echo("Creating snapshots...")
    snapshot_paths: dict[str, str] = {}
    if is_resume_attempt and manifest.snapshots:
        snapshot_paths = {snapshot.name: snapshot.dest_path for snapshot in manifest.snapshots}
        click.echo(f"  Reusing {len(snapshot_paths)} snapshot(s) from previous attempt")
    elif dry_run:
        click.echo(f"  Snapshot ID: {snapshot_id}")
        for snap_cfg in effective_config.snapshots:
            path = resolve_snapshot_path(
                target_root=Path(snap_cfg.target_dir),
                snapshot_name=snap_cfg.name,
                snapshot_id=snapshot_id,
                avoid_collision=False,
            )
            snapshot_paths[snap_cfg.name] = str(path)
    else:
        target_overrides = None
        if mode == "local_pc":
            local_snapshot_root = Path(
                tempfile.mkdtemp(prefix=f"pace_snapshots_{run_name}_{snapshot_id}_")
            )
            target_overrides = {
                snap_cfg.name: local_snapshot_root for snap_cfg in effective_config.snapshots
            }

        snapshot_results = create_all_snapshots(
            configs=effective_config.snapshots,
            snapshot_id=snapshot_id,
            target_root_overrides=target_overrides,
            dry_run=False,
        )

        if mode == "local_pc":
            executor = make_executor(config)
            config_by_name = {c.name: c for c in effective_config.snapshots}
            for name, result in snapshot_results.items():
                snap_cfg = config_by_name[name]
                remote_snapshot_dir = Path(snap_cfg.target_dir) / result.dest_path.name
                click.echo(
                    f"  Sync snapshot '{name}': {result.dest_path} -> {remote_snapshot_dir}"
                )
                sync_local_run_to_remote(
                    executor,
                    result.dest_path,
                    str(remote_snapshot_dir),
                )
                result.manifest.dest_path = str(remote_snapshot_dir)
                snapshot_paths[name] = str(remote_snapshot_dir)
                registry.add_snapshot(effective_config.project, run_name, result.manifest)
        else:
            for name, result in snapshot_results.items():
                snapshot_paths[name] = str(result.dest_path)
                registry.add_snapshot(effective_config.project, run_name, result.manifest)

        click.echo(f"  Created {len(snapshot_paths)} snapshot(s)")

    # Create staging plan
    staging_planner = StagingPlanner(effective_config)
    staging_plan = staging_planner.create_plan()

    # Build staged paths (simplified - in practice this happens at runtime)
    staged_paths = {}
    for resource in staging_plan.copy_resources:
        staged_paths[resource.name] = resource.dest_path
    for resource in staging_plan.bind_resources:
        staged_paths[resource.name] = resource.source_path

    # Create attempt
    attempt = registry.create_attempt(
        effective_config.project,
        run_name,
        resume_from=resume_from,
        user_args=user_args,
    )
    click.echo(f"Created attempt #{attempt.attempt_id}")

    # Build runtime context
    context_builder = RuntimeContextBuilder(
        config=effective_config,
        manifest=manifest,
        run_name=run_name,
        attempt_id=attempt.attempt_id,
    )
    context_builder.with_user_args(user_args)
    context_builder.with_snapshot_paths(snapshot_paths)
    context_builder.with_staged_paths(staged_paths)

    if resume_from:
        resume_container = resume_from_container_override
        if resume_container is None:
            output_paths = resolve_resume_output_paths(effective_config, run_name)
            if output_paths is None:
                resume_container = resume_from
            else:
                try:
                    resume_container = map_checkpoint_host_to_container(
                        resume_from, output_paths
                    )
                except ValueError as exc:
                    raise click.ClickException(str(exc)) from exc
        context_builder.with_resume(resume_from, resume_container)

    context = context_builder.build()

    # Ensure persistent directories exist
    bind_planner = BindPlanner(effective_config)
    if not dry_run:
        if mode == "local_pc":
            executor = make_executor(config)
            _ensure_remote_dirs(config, context, executor)
        else:
            bind_planner.ensure_persistent_dirs(context)

    # Build launch plan
    launch_builder = LaunchPlanBuilder(effective_config)
    launch_plan = launch_builder.build(context)

    # Preview
    click.echo("\nLaunch plan:")
    click.echo(f"  Workdir: {launch_plan.workdir}")
    click.echo(f"  Command: {' '.join(launch_plan.command[:5])}...")
    click.echo(f"  Binds: {len(launch_plan.binds)}")
    click.echo(f"  Env vars: {len(launch_plan.environment)}")

    # Save launch plan and render scripts
    attempt_dir = registry.attempt_dir(effective_config.project, run_name, attempt.attempt_id)
    renderer = LaunchRenderer()

    if dry_run:
        click.echo("\n--- Compute Wrapper (dry run) ---")
        click.echo(renderer.render_compute_wrapper(launch_plan))
    else:
        launch_plan.save(attempt_dir / "launch_plan.yaml")
        renderer.save_to_attempt_dir(launch_plan, attempt_dir)
        click.echo(f"\nSaved files to: {attempt_dir}")

    # Create and submit SLURM job
    slurm = SlurmBackend()
    if mode == "local_pc":
        wrapper_path = (
            f"{remote_attempt_dir(config, run_name, attempt.attempt_id)}/compute_wrapper.sh"
        )
    else:
        wrapper_path = f"{attempt_dir}/compute_wrapper.sh"
    scheduler_log_dir = context.resolve(effective_config.scheduler.log_dir)

    job_script = slurm.create_job_script(
        plan=launch_plan,
        resources=effective_config.resources,
        job_name=f"pace-{run_name}",
        log_dir=scheduler_log_dir,
        wrapper_path=wrapper_path,
        staging_plan=staging_plan,
    )

    if dry_run:
        click.echo("\n--- SLURM Job Script (dry run) ---")
        click.echo(job_script.render())
        return

    # Save job script
    script_path = attempt_dir / "slurm_job.sh"
    script_path.write_text(job_script.render())
    script_path.chmod(0o755)

    if no_submit:
        click.echo(f"\nJob script saved: {script_path}")
        click.echo("Submit manually with: sbatch {script_path}")
        return

    if mode == "local_pc":
        executor = make_executor(config)
        local_run_dir = registry.run_dir(effective_config.project, run_name)
        remote_run_dir_path = remote_run_dir(config, run_name)
        click.echo(
            "Syncing prepared run artifacts to remote registry: "
            f"{local_run_dir} -> {remote_run_dir_path}"
        )
        try:
            sync_local_run_to_remote(executor, local_run_dir, remote_run_dir_path)
            remote_script = f"{remote_attempt_dir(config, run_name, attempt.attempt_id)}/slurm_job.sh"
            completed = executor.run(
                ["sbatch", remote_script],
                remote_cwd=config.remote.project_dir,
                stream_output=False,
            )
            if completed.returncode != 0:
                raise click.ClickException(
                    f"Remote sbatch failed: {completed.stderr or completed.stdout}"
                )
            output = (completed.stdout or "").strip()
            click.echo(output)
            match = re.search(r"Submitted batch job (\d+)", output)
            if not match:
                raise click.ClickException(f"Unexpected sbatch output: {output}")
            job_id = int(match.group(1))
            update_remote_attempt_submission(
                executor,
                f"{remote_attempt_dir(config, run_name, attempt.attempt_id)}/attempt.yaml",
                job_id,
            )
            click.echo(f"Submitted remote job: {job_id}")
            return
        except SSHRemoteError as exc:
            raise click.ClickException(str(exc)) from exc

    # Submit
    click.echo("\nSubmitting to SLURM...")
    job_id = slurm.submit(job_script, script_path=script_path)

    if job_id:
        click.echo(f"Submitted job: {job_id}")
        # Update attempt with job ID
        attempt.slurm_job_id = job_id
        attempt.status = attempt.status.SUBMITTED
        registry.save_attempt(effective_config.project, run_name, attempt)
    else:
        click.echo("Failed to submit job", err=True)
        raise SystemExit(1)
