"""Command-line interface for PACE."""

from __future__ import annotations

import click

from pace import __version__


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx: click.Context) -> None:
    """PACE - HPC Experiment Launcher.

    A tool for launching and managing Python experiments on
    HPC clusters with SLURM and Apptainer.
    """
    ctx.ensure_object(dict)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to pace.yaml config file.",
)
def init(config: str | None) -> None:
    """Initialize a new PACE project.

    Creates a starter pace.yaml configuration file.
    """
    from pace.commands.init import run_init

    run_init(config_path=config)


@main.group()
def run() -> None:
    """Manage experiment runs."""
    pass


@run.command("submit")
@click.argument("run_name")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to pace.yaml config file.",
)
@click.option(
    "--remote",
    "force_remote",
    is_flag=True,
    help="Force SSH remote execution for this command.",
)
@click.option(
    "--no-remote",
    "force_local",
    is_flag=True,
    help="Force local execution even if remote.enabled is true in config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be done without executing.",
)
@click.option(
    "--no-submit",
    is_flag=True,
    help="Prepare everything but don't submit to SLURM.",
)
@click.argument("user_args", nargs=-1)
def run_submit(
    run_name: str,
    config: str | None,
    force_remote: bool,
    force_local: bool,
    dry_run: bool,
    no_submit: bool,
    user_args: tuple[str, ...],
) -> None:
    """Submit a new run or resume an existing one.

    RUN_NAME is the name for this experiment run.
    Additional arguments after -- are passed to the training script.
    """
    from pace.commands.submit import run_submit as do_submit

    do_submit(
        run_name=run_name,
        config_path=config,
        force_remote=force_remote,
        force_local=force_local,
        dry_run=dry_run,
        no_submit=no_submit,
        user_args=list(user_args),
    )


@run.command("status")
@click.argument("run_name")
@click.option(
    "--remote",
    "force_remote",
    is_flag=True,
    help="Force SSH remote execution for this command.",
)
@click.option(
    "--no-remote",
    "force_local",
    is_flag=True,
    help="Force local execution even if remote.enabled is true in config.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to pace.yaml config file.",
)
def run_status(
    run_name: str,
    force_remote: bool,
    force_local: bool,
    config: str | None,
) -> None:
    """Show status of a run.

    Displays latest attempt, job status, checkpoints, and completion state.
    """
    from pace.commands.status import run_status as do_status

    do_status(
        run_name=run_name,
        config_path=config,
        force_remote=force_remote,
        force_local=force_local,
    )


@run.command("resume")
@click.argument("run_name")
@click.option(
    "--remote",
    "force_remote",
    is_flag=True,
    help="Force SSH remote execution for this command.",
)
@click.option(
    "--no-remote",
    "force_local",
    is_flag=True,
    help="Force local execution even if remote.enabled is true in config.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to pace.yaml config file.",
)
@click.option(
    "--checkpoint",
    type=str,
    help="Specific checkpoint to resume from.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be done without executing.",
)
def run_resume(
    run_name: str,
    force_remote: bool,
    force_local: bool,
    config: str | None,
    checkpoint: str | None,
    dry_run: bool,
) -> None:
    """Resume an existing run from its latest checkpoint.

    Submits a new attempt using the latest valid checkpoint.
    """
    from pace.commands.resume import run_resume as do_resume

    do_resume(
        run_name=run_name,
        config_path=config,
        force_remote=force_remote,
        force_local=force_local,
        checkpoint_path=checkpoint,
        dry_run=dry_run,
    )


@run.command("logs")
@click.argument("run_name")
@click.option(
    "--remote",
    "force_remote",
    is_flag=True,
    help="Force SSH remote execution for this command.",
)
@click.option(
    "--no-remote",
    "force_local",
    is_flag=True,
    help="Force local execution even if remote.enabled is true in config.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to pace.yaml config file.",
)
@click.option(
    "--attempt",
    type=int,
    help="Specific attempt number.",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    help="Follow log output (like tail -f).",
)
def run_logs(
    run_name: str,
    force_remote: bool,
    force_local: bool,
    config: str | None,
    attempt: int | None,
    follow: bool,
) -> None:
    """View logs for a run."""
    from pace.commands.logs import run_logs as do_logs

    do_logs(
        run_name=run_name,
        config_path=config,
        force_remote=force_remote,
        force_local=force_local,
        attempt_id=attempt,
        follow=follow,
    )


@run.command("list")
@click.option(
    "--remote",
    "force_remote",
    is_flag=True,
    help="Force SSH remote execution for this command.",
)
@click.option(
    "--no-remote",
    "force_local",
    is_flag=True,
    help="Force local execution even if remote.enabled is true in config.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to pace.yaml config file.",
)
def run_list(force_remote: bool, force_local: bool, config: str | None) -> None:
    """List all runs for the project."""
    from pace.commands.runs import run_list as do_list

    do_list(
        config_path=config,
        force_remote=force_remote,
        force_local=force_local,
    )


if __name__ == "__main__":
    main()
