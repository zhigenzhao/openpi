"""Tests for resume CLI argument injection in launch plan builder."""

from pace.config.models import (
    CliInjectionConfig,
    InjectionsConfig,
    LaunchConfig,
    LaunchPhase,
    ProjectConfig,
    ResumeConfig,
    RuntimeConfig,
    SchedulerConfig,
)
from pace.core.context import RuntimeContextBuilder
from pace.core.manifests import RunManifest
from pace.launch.builder import LaunchPlanBuilder


def _make_manifest() -> RunManifest:
    return RunManifest(
        project="proj",
        run_name="run1",
        config_path="pace.yaml",
        image_path="/tmp/image.sif",
        created_at="2026-03-11T00:00:00",
    )


def _make_config() -> ProjectConfig:
    return ProjectConfig(
        project="proj",
        runtime=RuntimeConfig(image="/tmp/image.sif"),
        scheduler=SchedulerConfig(type="slurm", log_dir="/tmp/{run_name}/sched"),
        launch=LaunchConfig(base_command=["python", "train.py"]),
        resume=ResumeConfig(cli_args=["runner.resume_dir={resume_path_container}"]),
        injections=InjectionsConfig(
            cli=[
                CliInjectionConfig(
                    name="append-run-name",
                    phase=LaunchPhase.FINAL,
                    args=["{run_name}"],
                )
            ]
        ),
    )


def test_resume_cli_args_are_injected_for_resume_attempts():
    """Resume CLI args should be appended in post-resume phase on resume attempts."""
    config = _make_config()
    context = (
        RuntimeContextBuilder(
            config=config,
            manifest=_make_manifest(),
            run_name="run1",
            attempt_id=2,
        )
        .with_resume(
            "/cluster/logs/run1/checkpoints/global_step_100",
            "/workspace/RLinf/logs/run1/checkpoints/global_step_100",
        )
        .build()
    )
    launch_plan = LaunchPlanBuilder(config).build(context)
    assert launch_plan.command == [
        "python",
        "train.py",
        "run1",
        "runner.resume_dir=/workspace/RLinf/logs/run1/checkpoints/global_step_100",
    ]


def test_resume_cli_args_not_injected_for_fresh_attempt():
    """Resume CLI args should not be injected for fresh attempts."""
    config = _make_config()
    context = RuntimeContextBuilder(
        config=config,
        manifest=_make_manifest(),
        run_name="run1",
        attempt_id=1,
    ).build()
    launch_plan = LaunchPlanBuilder(config).build(context)
    assert launch_plan.command == ["python", "train.py", "run1"]
