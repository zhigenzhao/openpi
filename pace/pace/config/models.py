"""Configuration models for PACE.

Defines dataclasses for the pace.yaml configuration schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StageMode(str, Enum):
    """How shared inputs should be staged."""

    BIND = "bind"
    COPY_TO_TMP = "copy_to_tmp"


class InjectionCondition(str, Enum):
    """When an injection should be applied."""

    ALWAYS = "always"
    RESUME_ONLY = "resume_only"
    FIRST_ATTEMPT_ONLY = "first_attempt_only"
    NON_RESUME_ONLY = "non_resume_only"


class LaunchPhase(str, Enum):
    """Phases for CLI argument injection."""

    PRE_BASE = "pre_base"
    BASE = "base"
    POST_BASE = "post_base"
    PRE_RESUME = "pre_resume"
    POST_RESUME = "post_resume"
    FINAL = "final"


class ResumeStrategy(str, Enum):
    """Strategy for selecting checkpoint to resume from."""

    LATEST_SAFE = "latest_safe"
    LATEST = "latest"
    NONE = "none"


@dataclass
class RuntimeConfig:
    """Container runtime configuration.

    Attributes:
        image: Path to the container image (.sif file).
        engine: Container engine to use (apptainer).
    """

    image: str
    engine: str = "apptainer"


@dataclass
class SchedulerConfig:
    """Scheduler configuration.

    Attributes:
        type: Scheduler type (currently slurm).
        log_dir: Host directory template for scheduler stdout/stderr files.
    """

    type: str = "slurm"
    log_dir: str = ""


@dataclass
class ResourcesConfig:
    """Resource request configuration for SLURM.

    Attributes:
        gpus: Number of GPUs to request.
        gpu_type: Type of GPU (e.g., H100, A100).
        cpus: Number of CPUs per task.
        mem_gb: Memory in GB.
        time: Wall time limit (HH:MM:SS format).
        nodes: Number of nodes.
        partition: SLURM partition (optional).
        account: SLURM account (optional).
        qos: SLURM QOS (optional).
    """

    gpus: int = 1
    gpu_type: str = "H100"
    cpus: int = 32
    mem_gb: int = 500
    time: str = "2:00:00"
    nodes: int = 1
    partition: str | None = None
    account: str | None = None
    qos: str | None = None


@dataclass
class SnapshotConfig:
    """Snapshot source configuration.

    Attributes:
        name: Identifier for this snapshot (used in templates as {snapshot.name}).
        local_dir: Source directory path on the command machine.
        target_dir: Destination directory on host storage for snapshot materialization.
        exclude: List of patterns to exclude from snapshot.
    """

    name: str
    local_dir: str
    target_dir: str
    exclude: list[str] = field(default_factory=list)


@dataclass
class SharedInputConfig:
    """Shared input configuration.

    Attributes:
        name: Identifier for this input (used in templates as {staged.name} or {shared.name}).
        host_path: Path on the host filesystem.
        stage_mode: How to stage this input (bind or copy_to_tmp).
        container_path: Override container mount path (optional).
    """

    name: str
    host_path: str
    stage_mode: StageMode = StageMode.BIND
    container_path: str | None = None


@dataclass
class PersistentOutputConfig:
    """Persistent output directory configuration.

    Attributes:
        name: Identifier for this output (used in templates as {persistent.name}).
        role: Semantic role (logs, checkpoints, artifacts).
        host_path: Path template on the host filesystem.
        container_path: Path inside the container.
        create_if_missing: Whether to create the directory if it doesn't exist.
    """

    name: str
    role: str
    host_path: str
    container_path: str
    create_if_missing: bool = True


@dataclass
class BindConfig:
    """Container bind mount configuration.

    Attributes:
        host: Host path (can use templates like {snapshot.repo}).
        container: Container mount path.
        mode: Mount mode (ro or rw).
    """

    host: str
    container: str
    mode: str = "rw"


@dataclass
class UserArgsConfig:
    """Configuration for user argument injection.

    Attributes:
        phase: Phase at which to inject user args.
    """

    phase: LaunchPhase = LaunchPhase.POST_BASE


@dataclass
class HooksConfig:
    """Shell command hooks for different execution phases."""

    pre_apptainer: list[str] = field(default_factory=list)
    pre_launch: list[str] = field(default_factory=list)
    post_launch: list[str] = field(default_factory=list)


@dataclass
class LaunchConfig:
    """Launch recipe configuration.

    Attributes:
        workdir: Working directory inside the container.
        shell_init: Shell commands to run before the main command.
        base_command: The base command to run (list of strings with templates).
        user_args: Configuration for user argument injection.
    """

    workdir: str = "/workspace"
    shell_init: list[str] = field(default_factory=list)
    base_command: list[str] = field(default_factory=list)
    user_args: UserArgsConfig = field(default_factory=UserArgsConfig)


@dataclass
class EnvInjectionConfig:
    """Environment variable injection configuration.

    Attributes:
        name: Identifier for this injection group.
        when: Condition for when to apply this injection.
        values: Dictionary of environment variable names to values (with templates).
    """

    name: str
    when: InjectionCondition = InjectionCondition.ALWAYS
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class CliInjectionConfig:
    """CLI argument injection configuration.

    Attributes:
        name: Identifier for this injection group.
        when: Condition for when to apply this injection.
        phase: Phase at which to inject these arguments.
        args: List of arguments to inject (with templates).
    """

    name: str
    when: InjectionCondition = InjectionCondition.ALWAYS
    phase: LaunchPhase = LaunchPhase.POST_BASE
    args: list[str] = field(default_factory=list)


@dataclass
class InjectionsConfig:
    """All injection configurations.

    Attributes:
        env: List of environment variable injections.
        cli: List of CLI argument injections.
    """

    env: list[EnvInjectionConfig] = field(default_factory=list)
    cli: list[CliInjectionConfig] = field(default_factory=list)


@dataclass
class ResumeConfig:
    """Resume behavior configuration.

    Attributes:
        enabled: Whether resume is enabled.
        checkpoint_output: Name of the persistent output to use for checkpoints.
        strategy: Strategy for selecting checkpoint to resume from.
        search_recursive: Whether to search recursively for checkpoints.
        checkpoint_marker: Name of the file that indicates a valid checkpoint.
        sort_pattern: Regex pattern with a named group 'key' to extract sortable value.
                      Example: r"global_step_(?P<key>\\d+)" extracts numeric step.
                      If None, sorts alphabetically by directory name.
        cli_args: CLI arguments injected during resume attempts. These are
                  resolved as templates and appended in the post-resume phase.
    """

    enabled: bool = True
    checkpoint_output: str = "checkpoints"
    strategy: ResumeStrategy = ResumeStrategy.LATEST_SAFE
    search_recursive: bool = True
    checkpoint_marker: str = "CHECKPOINT_OK"
    sort_pattern: str | None = None
    cli_args: list[str] = field(default_factory=list)


@dataclass
class CompletionConfig:
    """Training completion configuration.

    Attributes:
        marker_file: Name of the file that indicates training is complete.
    """

    marker_file: str = "TRAINING_DONE"


@dataclass
class WandBConfig:
    """WandB integration configuration.

    Attributes:
        enabled: Whether WandB integration is enabled.
        project: WandB project name (optional).
        entity: WandB entity/team name (optional).
    """

    enabled: bool = True
    project: str | None = None
    entity: str | None = None


@dataclass
class RemoteConfig:
    """Remote orchestration configuration.

    Attributes:
        enabled: Whether to execute run commands via SSH on a remote login node.
        host: SSH host (e.g. pace-ice).
        user: Optional SSH user. If set, target becomes "user@host".
        project_dir: Remote project/workspace directory where commands run.
        mode: Execution mode selection (auto, local, remote).
        ssh_options: Extra options passed directly to ssh.
        rsync_options: Extra options passed directly to rsync.
    """

    enabled: bool = False
    host: str = ""
    user: str | None = None
    project_dir: str = ""
    mode: str = "auto"
    ssh_options: list[str] = field(default_factory=list)
    rsync_options: list[str] = field(default_factory=list)


@dataclass
class ProjectConfig:
    """Root project configuration from pace.yaml.

    Attributes:
        project: Project name identifier.
        runtime: Container runtime configuration.
        scheduler: Scheduler configuration.
        resources: Resource request configuration.
        snapshots: List of snapshot source configurations.
        shared_inputs: List of shared input configurations.
        persistent_outputs: List of persistent output configurations.
        binds: List of bind mount configurations.
        launch: Launch recipe configuration.
        injections: Environment and CLI injection configurations.
        resume: Resume behavior configuration.
        completion: Training completion configuration.
        wandb: WandB integration configuration.
        remote: Remote orchestration configuration.
        registry_root: Root directory for run registry storage.
    """

    project: str
    runtime: RuntimeConfig
    scheduler: SchedulerConfig
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    snapshots: list[SnapshotConfig] = field(default_factory=list)
    shared_inputs: list[SharedInputConfig] = field(default_factory=list)
    persistent_outputs: list[PersistentOutputConfig] = field(default_factory=list)
    binds: list[BindConfig] = field(default_factory=list)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    injections: InjectionsConfig = field(default_factory=InjectionsConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    registry_root: str = "/cluster/pace_runs"

    def get_snapshot(self, name: str) -> SnapshotConfig | None:
        """Get a snapshot config by name."""
        for snap in self.snapshots:
            if snap.name == name:
                return snap
        return None

    def get_shared_input(self, name: str) -> SharedInputConfig | None:
        """Get a shared input config by name."""
        for inp in self.shared_inputs:
            if inp.name == name:
                return inp
        return None

    def get_persistent_output(self, name: str) -> PersistentOutputConfig | None:
        """Get a persistent output config by name."""
        for out in self.persistent_outputs:
            if out.name == name:
                return out
        return None

    def get_persistent_output_by_role(self, role: str) -> PersistentOutputConfig | None:
        """Get a persistent output config by role."""
        for out in self.persistent_outputs:
            if out.role == role:
                return out
        return None
