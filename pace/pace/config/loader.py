"""Configuration loader for PACE.

Loads and validates pace.yaml configuration files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from pace.config.models import (
    BindConfig,
    CliInjectionConfig,
    CompletionConfig,
    EnvInjectionConfig,
    HooksConfig,
    InjectionCondition,
    InjectionsConfig,
    LaunchConfig,
    LaunchPhase,
    PersistentOutputConfig,
    ProjectConfig,
    RemoteConfig,
    ResourcesConfig,
    SchedulerConfig,
    ResumeConfig,
    ResumeStrategy,
    RuntimeConfig,
    SharedInputConfig,
    SnapshotConfig,
    StageMode,
    UserArgsConfig,
    WandBConfig,
)


class ConfigError(Exception):
    """Error loading or validating configuration."""

    pass


def _parse_runtime(data: dict[str, Any]) -> RuntimeConfig:
    """Parse runtime section."""
    if not data:
        raise ConfigError("runtime section is required")
    return RuntimeConfig(
        image=data.get("image", ""),
        engine=data.get("engine", "apptainer"),
    )


def _parse_resources(data: dict[str, Any] | None) -> ResourcesConfig:
    """Parse resources section."""
    if not data:
        return ResourcesConfig()
    return ResourcesConfig(
        gpus=data.get("gpus", 1),
        gpu_type=data.get("gpu_type", "H100"),
        cpus=data.get("cpus", 32),
        mem_gb=data.get("mem_gb", 500),
        time=data.get("time", "2:00:00"),
        nodes=data.get("nodes", 1),
        partition=data.get("partition"),
        account=data.get("account"),
        qos=data.get("qos"),
    )


def _parse_scheduler(data: dict[str, Any] | None) -> SchedulerConfig:
    """Parse scheduler section."""
    if not data:
        raise ConfigError("scheduler section is required")
    scheduler_type = data.get("type", "slurm")
    if scheduler_type != "slurm":
        raise ConfigError("scheduler.type must be 'slurm'")
    log_dir = data.get("log_dir", "")
    if not log_dir:
        raise ConfigError("scheduler.log_dir is required")
    return SchedulerConfig(type=scheduler_type, log_dir=log_dir)


def _parse_snapshots(data: list[dict[str, Any]] | None) -> list[SnapshotConfig]:
    """Parse snapshots section."""
    if not data:
        return []
    return [
        SnapshotConfig(
            name=item["name"],
            local_dir=item["local_dir"],
            target_dir=item["target_dir"],
            exclude=item.get("exclude", []),
        )
        for item in data
    ]


def _parse_shared_inputs(data: list[dict[str, Any]] | None) -> list[SharedInputConfig]:
    """Parse shared_inputs section."""
    if not data:
        return []
    result = []
    for item in data:
        stage_mode_str = item.get("stage_mode", "bind")
        try:
            stage_mode = StageMode(stage_mode_str)
        except ValueError:
            raise ConfigError(f"Invalid stage_mode: {stage_mode_str}")
        result.append(
            SharedInputConfig(
                name=item["name"],
                host_path=item["host_path"],
                stage_mode=stage_mode,
                container_path=item.get("container_path"),
            )
        )
    return result


def _parse_persistent_outputs(
    data: list[dict[str, Any]] | None,
) -> list[PersistentOutputConfig]:
    """Parse persistent_outputs section."""
    if not data:
        return []
    return [
        PersistentOutputConfig(
            name=item["name"],
            role=item["role"],
            host_path=item["host_path"],
            container_path=item["container_path"],
            create_if_missing=item.get("create_if_missing", True),
        )
        for item in data
    ]


def _parse_binds(data: list[dict[str, Any]] | None) -> list[BindConfig]:
    """Parse binds section."""
    if not data:
        return []
    return [
        BindConfig(
            host=item["host"],
            container=item["container"],
            mode=item.get("mode", "rw"),
        )
        for item in data
    ]


def _parse_user_args(data: dict[str, Any] | None) -> UserArgsConfig:
    """Parse user_args section."""
    if not data:
        return UserArgsConfig()
    phase_str = data.get("phase", "post_base")
    try:
        phase = LaunchPhase(phase_str)
    except ValueError:
        raise ConfigError(f"Invalid user_args phase: {phase_str}")
    return UserArgsConfig(phase=phase)


def _parse_launch(data: dict[str, Any] | None) -> LaunchConfig:
    """Parse launch section."""
    if not data:
        return LaunchConfig()
    return LaunchConfig(
        workdir=data.get("workdir", "/workspace"),
        shell_init=data.get("shell_init", []),
        base_command=data.get("base_command", []),
        user_args=_parse_user_args(data.get("user_args")),
    )


def _parse_injection_condition(value: str) -> InjectionCondition:
    """Parse injection condition string."""
    try:
        return InjectionCondition(value)
    except ValueError:
        raise ConfigError(f"Invalid injection condition: {value}")


def _parse_launch_phase(value: str) -> LaunchPhase:
    """Parse launch phase string."""
    try:
        return LaunchPhase(value)
    except ValueError:
        raise ConfigError(f"Invalid launch phase: {value}")


def _parse_env_injections(
    data: list[dict[str, Any]] | None,
) -> list[EnvInjectionConfig]:
    """Parse env injection configs."""
    if not data:
        return []
    return [
        EnvInjectionConfig(
            name=item["name"],
            when=_parse_injection_condition(item.get("when", "always")),
            values=item.get("values", {}),
        )
        for item in data
    ]


def _parse_cli_injections(
    data: list[dict[str, Any]] | None,
) -> list[CliInjectionConfig]:
    """Parse CLI injection configs."""
    if not data:
        return []
    return [
        CliInjectionConfig(
            name=item["name"],
            when=_parse_injection_condition(item.get("when", "always")),
            phase=_parse_launch_phase(item.get("phase", "post_base")),
            args=item.get("args", []),
        )
        for item in data
    ]


def _parse_injections(data: dict[str, Any] | None) -> InjectionsConfig:
    """Parse injections section."""
    if not data:
        return InjectionsConfig()
    return InjectionsConfig(
        env=_parse_env_injections(data.get("env")),
        cli=_parse_cli_injections(data.get("cli")),
    )


def _parse_resume(data: dict[str, Any] | None) -> ResumeConfig:
    """Parse resume section."""
    if not data:
        return ResumeConfig()
    strategy_str = data.get("strategy", "latest_safe")
    try:
        strategy = ResumeStrategy(strategy_str)
    except ValueError:
        raise ConfigError(f"Invalid resume strategy: {strategy_str}")
    checkpoint_marker = data.get("checkpoint_marker", "CHECKPOINT_OK")
    if not isinstance(checkpoint_marker, str) or not checkpoint_marker:
        raise ConfigError("resume.checkpoint_marker must be a non-empty string")

    sort_pattern = data.get("sort_pattern")
    if sort_pattern is not None:
        if not isinstance(sort_pattern, str) or not sort_pattern:
            raise ConfigError("resume.sort_pattern must be a non-empty string when set")
        try:
            compiled = re.compile(sort_pattern)
        except re.error as exc:
            raise ConfigError(f"Invalid resume sort_pattern regex: {exc}") from exc
        if "key" not in compiled.groupindex:
            raise ConfigError("resume.sort_pattern must define a named group 'key'")

    cli_args = data.get("cli_args", [])
    if not isinstance(cli_args, list) or any(
        not isinstance(arg, str) for arg in cli_args
    ):
        raise ConfigError("resume.cli_args must be a list of strings")

    return ResumeConfig(
        enabled=data.get("enabled", True),
        checkpoint_output=data.get("checkpoint_output", "checkpoints"),
        strategy=strategy,
        search_recursive=data.get("search_recursive", True),
        checkpoint_marker=checkpoint_marker,
        sort_pattern=sort_pattern,
        cli_args=cli_args,
    )


def _parse_completion(data: dict[str, Any] | None) -> CompletionConfig:
    """Parse completion section."""
    if not data:
        return CompletionConfig()
    return CompletionConfig(
        marker_file=data.get("marker_file", "TRAINING_DONE"),
    )


def _parse_wandb(data: dict[str, Any] | None) -> WandBConfig:
    """Parse wandb section."""
    if not data:
        return WandBConfig()
    return WandBConfig(
        enabled=data.get("enabled", True),
        project=data.get("project"),
        entity=data.get("entity"),
    )


def _parse_hooks(data: dict[str, Any] | None) -> HooksConfig:
    """Parse hooks section."""
    if not data:
        return HooksConfig()
    return HooksConfig(
        pre_apptainer=data.get("pre_apptainer", []),
        pre_launch=data.get("pre_launch", []),
        post_launch=data.get("post_launch", []),
    )


def _parse_remote(data: dict[str, Any] | None) -> RemoteConfig:
    """Parse remote section."""
    if not data:
        return RemoteConfig()

    remote = RemoteConfig(
        enabled=data.get("enabled", False),
        host=data.get("host", ""),
        user=data.get("user"),
        project_dir=data.get("project_dir", ""),
        mode=data.get("mode", "auto"),
        ssh_options=data.get("ssh_options", []),
        rsync_options=data.get("rsync_options", []),
    )

    if remote.enabled:
        if not remote.host:
            raise ConfigError("remote.host is required when remote.enabled is true")
        if not remote.project_dir:
            raise ConfigError(
                "remote.project_dir is required when remote.enabled is true"
            )
        if remote.mode not in {"auto", "local", "remote"}:
            raise ConfigError("remote.mode must be one of: auto, local, remote")

    return remote


def load_config(path: str | Path) -> ProjectConfig:
    """Load project configuration from a YAML file.

    Args:
        path: Path to the pace.yaml configuration file.

    Returns:
        Parsed ProjectConfig object.

    Raises:
        ConfigError: If the configuration is invalid.
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ConfigError("Configuration file is empty")

    if "project" not in data:
        raise ConfigError("project field is required")

    if "runtime" not in data:
        raise ConfigError("runtime section is required")
    if "scheduler" not in data:
        raise ConfigError("scheduler section is required")

    return ProjectConfig(
        project=data["project"],
        runtime=_parse_runtime(data.get("runtime")),
        scheduler=_parse_scheduler(data.get("scheduler")),
        resources=_parse_resources(data.get("resources")),
        snapshots=_parse_snapshots(data.get("snapshots")),
        shared_inputs=_parse_shared_inputs(data.get("shared_inputs")),
        persistent_outputs=_parse_persistent_outputs(data.get("persistent_outputs")),
        binds=_parse_binds(data.get("binds")),
        launch=_parse_launch(data.get("launch")),
        injections=_parse_injections(data.get("injections")),
        resume=_parse_resume(data.get("resume")),
        completion=_parse_completion(data.get("completion")),
        wandb=_parse_wandb(data.get("wandb")),
        remote=_parse_remote(data.get("remote")),
        hooks=_parse_hooks(data.get("hooks")),
        registry_root=data.get("registry_root", "/cluster/pace_runs"),
    )


def find_config(start_dir: str | Path | None = None) -> Path | None:
    """Find pace.yaml in the current or parent directories.

    Args:
        start_dir: Directory to start searching from. Defaults to cwd.

    Returns:
        Path to pace.yaml if found, None otherwise.
    """
    if start_dir is None:
        start_dir = Path.cwd()
    else:
        start_dir = Path(start_dir)

    current = start_dir.resolve()
    while current != current.parent:
        config_path = current / "pace.yaml"
        if config_path.exists():
            return config_path
        current = current.parent

    return None
