"""Manifest models for PACE runs and attempts.

These manifests provide structured storage for run metadata,
replacing ad-hoc .meta and .args files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class AttemptStatus(str, Enum):
    """Status of an attempt."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class WandBInfo:
    """WandB run information.

    Attributes:
        run_name: Display name for the WandB run.
        run_id: Stable WandB run ID for resume.
        project: WandB project name.
        entity: WandB entity/team name.
    """

    run_name: str
    run_id: str
    project: str | None = None
    entity: str | None = None


@dataclass
class SnapshotManifest:
    """Manifest for a snapshot.

    Attributes:
        name: Snapshot name from config.
        snapshot_id: Unique snapshot identifier (timestamp-based).
        source_path: Original source path.
        dest_path: Destination path on cluster.
        created_at: When the snapshot was created.
        file_count: Number of files in snapshot.
        total_size_bytes: Total size of snapshot.
    """

    name: str
    snapshot_id: str
    source_path: str
    dest_path: str
    created_at: str
    file_count: int = 0
    total_size_bytes: int = 0


@dataclass
class RunManifest:
    """Manifest for a logical run.

    This is the root metadata file for a run, stored as manifest.yaml
    in the run directory.

    Attributes:
        project: Project name.
        run_name: Unique run name.
        config_path: Path to the pace.yaml config used.
        image_path: Path to the container image.
        created_at: When the run was created.
        wandb: WandB integration info.
        snapshots: List of snapshot manifests.
        persistent_outputs: Map of role to host path.
        latest_attempt: Number of the latest attempt.
        resume_enabled: Whether resume is enabled for this run.
    """

    project: str
    run_name: str
    config_path: str
    image_path: str
    created_at: str
    wandb: WandBInfo | None = None
    snapshots: list[SnapshotManifest] = field(default_factory=list)
    persistent_outputs: dict[str, str] = field(default_factory=dict)
    latest_attempt: int = 0
    resume_enabled: bool = True

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        data = asdict(self)
        # Convert nested dataclasses
        if self.wandb:
            data["wandb"] = asdict(self.wandb)
        data["snapshots"] = [asdict(s) for s in self.snapshots]
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> RunManifest:
        """Deserialize from YAML string."""
        data = yaml.safe_load(yaml_str)
        wandb_data = data.pop("wandb", None)
        snapshots_data = data.pop("snapshots", [])

        wandb = WandBInfo(**wandb_data) if wandb_data else None
        snapshots = [SnapshotManifest(**s) for s in snapshots_data]

        return cls(wandb=wandb, snapshots=snapshots, **data)

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Load manifest from file."""
        return cls.from_yaml(path.read_text())


@dataclass
class LaunchPlan:
    """Fully resolved launch plan for an attempt.

    This captures everything needed to reproduce the exact
    launch configuration.

    Attributes:
        workdir: Working directory inside container.
        shell_init: Shell initialization commands.
        command: Full command to execute (as list).
        environment: Environment variables to set.
        binds: List of bind mount specifications (host:container:mode).
        image_path: Path to container image.
    """

    workdir: str
    shell_init: list[str]
    command: list[str]
    environment: dict[str, str]
    binds: list[str]
    image_path: str
    pre_launch: list[str] = field(default_factory=list)
    post_launch: list[str] = field(default_factory=list)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> LaunchPlan:
        """Deserialize from YAML string."""
        data = yaml.safe_load(yaml_str)
        data.setdefault("pre_launch", [])
        data.setdefault("post_launch", [])
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save launch plan to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())

    @classmethod
    def load(cls, path: Path) -> LaunchPlan:
        """Load launch plan from file."""
        return cls.from_yaml(path.read_text())


@dataclass
class AttemptManifest:
    """Manifest for a single attempt of a run.

    Stored as attempt.yaml in the attempt directory.

    Attributes:
        attempt_id: Attempt number (1-indexed).
        run_name: Name of the parent run.
        submitted_at: When the attempt was submitted.
        status: Current status of the attempt.
        user_args: User-provided CLI arguments used for this attempt.
        slurm_job_id: SLURM job ID if submitted.
        resume_from: Path to checkpoint this attempt resumes from.
        slurm_log_path: Path to SLURM output log.
        exit_code: Exit code if completed.
        completed_at: When the attempt completed.
    """

    attempt_id: int
    run_name: str
    submitted_at: str
    status: AttemptStatus = AttemptStatus.PENDING
    user_args: list[str] = field(default_factory=list)
    slurm_job_id: int | None = None
    resume_from: str | None = None
    slurm_log_path: str | None = None
    exit_code: int | None = None
    completed_at: str | None = None

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        data = asdict(self)
        data["status"] = self.status.value
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> AttemptManifest:
        """Deserialize from YAML string."""
        data = yaml.safe_load(yaml_str)
        data["status"] = AttemptStatus(data["status"])
        if "user_args" not in data or data["user_args"] is None:
            data["user_args"] = []
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save attempt manifest to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())

    @classmethod
    def load(cls, path: Path) -> AttemptManifest:
        """Load attempt manifest from file."""
        return cls.from_yaml(path.read_text())


@dataclass
class RunState:
    """Mutable state for a run.

    Stored as state.json and updated as the run progresses.

    Attributes:
        latest_attempt: Number of the latest attempt.
        latest_checkpoint: Path to latest valid checkpoint.
        training_done: Whether training has completed.
        last_updated: When state was last updated.
    """

    latest_attempt: int = 0
    latest_checkpoint: str | None = None
    training_done: bool = False
    last_updated: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> RunState:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save state to file."""
        self.last_updated = datetime.now().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> RunState:
        """Load state from file."""
        if not path.exists():
            return cls()
        return cls.from_json(path.read_text())
