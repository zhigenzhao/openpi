"""Run registry for PACE.

Manages the directory structure and metadata for runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

from pace.config.models import ProjectConfig
from pace.core.manifests import (
    AttemptManifest,
    AttemptStatus,
    LaunchPlan,
    RunManifest,
    RunState,
    SnapshotManifest,
    WandBInfo,
)


class RunRegistry:
    """Manages run directory structure and manifests.

    Directory layout:
        {registry_root}/{project}/{run_name}/
            manifest.yaml       - Run metadata
            state.json          - Mutable run state
            snapshots/          - Snapshot directories
                snapshot_{timestamp}/
            attempts/           - Per-attempt data
                1/
                    attempt.yaml
                    launch_plan.yaml
                    slurm_job.sh
                    compute_wrapper.sh
                2/
                    ...
            logs/               - Training logs
            checkpoints/        - Training checkpoints
            artifacts/          - Other outputs
            markers/            - Completion markers
    """

    def __init__(self, registry_root: str | Path):
        """Initialize the registry.

        Args:
            registry_root: Root directory for all runs.
        """
        self.registry_root = Path(registry_root)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> RunRegistry:
        """Create registry from project config."""
        return cls(config.registry_root)

    # --- Path Methods ---

    def project_dir(self, project: str) -> Path:
        """Get the project directory."""
        return self.registry_root / project

    def run_dir(self, project: str, run_name: str) -> Path:
        """Get the run directory."""
        return self.project_dir(project) / run_name

    def manifest_path(self, project: str, run_name: str) -> Path:
        """Get path to run manifest."""
        return self.run_dir(project, run_name) / "manifest.yaml"

    def state_path(self, project: str, run_name: str) -> Path:
        """Get path to run state file."""
        return self.run_dir(project, run_name) / "state.json"

    def snapshots_dir(self, project: str, run_name: str) -> Path:
        """Get snapshots directory."""
        return self.run_dir(project, run_name) / "snapshots"

    def attempts_dir(self, project: str, run_name: str) -> Path:
        """Get attempts directory."""
        return self.run_dir(project, run_name) / "attempts"

    def attempt_dir(self, project: str, run_name: str, attempt_id: int) -> Path:
        """Get directory for a specific attempt."""
        return self.attempts_dir(project, run_name) / str(attempt_id)

    def logs_dir(self, project: str, run_name: str) -> Path:
        """Get logs directory."""
        return self.run_dir(project, run_name) / "logs"

    def checkpoints_dir(self, project: str, run_name: str) -> Path:
        """Get checkpoints directory."""
        return self.run_dir(project, run_name) / "checkpoints"

    def artifacts_dir(self, project: str, run_name: str) -> Path:
        """Get artifacts directory."""
        return self.run_dir(project, run_name) / "artifacts"

    def markers_dir(self, project: str, run_name: str) -> Path:
        """Get markers directory."""
        return self.run_dir(project, run_name) / "markers"

    # --- Run Management ---

    def run_exists(self, project: str, run_name: str) -> bool:
        """Check if a run exists."""
        return self.manifest_path(project, run_name).exists()

    def create_run(
        self,
        config: ProjectConfig,
        run_name: str,
    ) -> RunManifest:
        """Create a new run with initial manifest.

        Args:
            config: Project configuration.
            run_name: Name for the new run.

        Returns:
            The created RunManifest.

        Raises:
            ValueError: If run already exists.
        """
        project = config.project
        if self.run_exists(project, run_name):
            raise ValueError(f"Run already exists: {project}/{run_name}")

        # Create directory structure
        run_path = self.run_dir(project, run_name)
        run_path.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir(project, run_name).mkdir(exist_ok=True)
        self.attempts_dir(project, run_name).mkdir(exist_ok=True)
        self.markers_dir(project, run_name).mkdir(exist_ok=True)

        # Create persistent output directories
        persistent_outputs: dict[str, str] = {}
        for output in config.persistent_outputs:
            # Note: host_path may contain templates, but we store the template
            # The actual path is resolved at runtime
            persistent_outputs[output.role] = output.host_path

        # Create WandB info if enabled
        wandb_info = None
        if config.wandb.enabled:
            wandb_info = WandBInfo(
                run_name=run_name,
                run_id=f"pace-{project}-{run_name}",
                project=config.wandb.project,
                entity=config.wandb.entity,
            )

        # Create manifest
        manifest = RunManifest(
            project=project,
            run_name=run_name,
            config_path="",  # Will be set by caller
            image_path=config.runtime.image,
            created_at=datetime.now().isoformat(),
            wandb=wandb_info,
            snapshots=[],
            persistent_outputs=persistent_outputs,
            latest_attempt=0,
            resume_enabled=config.resume.enabled,
        )

        # Save manifest
        manifest.save(self.manifest_path(project, run_name))

        # Initialize state
        state = RunState()
        state.save(self.state_path(project, run_name))

        return manifest

    def load_manifest(self, project: str, run_name: str) -> RunManifest:
        """Load run manifest."""
        path = self.manifest_path(project, run_name)
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {project}/{run_name}")
        return RunManifest.load(path)

    def save_manifest(self, manifest: RunManifest) -> None:
        """Save run manifest."""
        path = self.manifest_path(manifest.project, manifest.run_name)
        manifest.save(path)

    def load_state(self, project: str, run_name: str) -> RunState:
        """Load run state."""
        return RunState.load(self.state_path(project, run_name))

    def save_state(self, project: str, run_name: str, state: RunState) -> None:
        """Save run state."""
        state.save(self.state_path(project, run_name))

    # --- Attempt Management ---

    def create_attempt(
        self,
        project: str,
        run_name: str,
        resume_from: str | None = None,
        user_args: list[str] | None = None,
    ) -> AttemptManifest:
        """Create a new attempt for a run.

        Args:
            project: Project name.
            run_name: Run name.
            resume_from: Path to checkpoint to resume from.
            user_args: User arguments passed to the launcher for this attempt.

        Returns:
            The created AttemptManifest.
        """
        # Load and update manifest
        manifest = self.load_manifest(project, run_name)
        attempt_id = manifest.latest_attempt + 1
        manifest.latest_attempt = attempt_id
        self.save_manifest(manifest)

        # Create attempt directory
        attempt_path = self.attempt_dir(project, run_name, attempt_id)
        attempt_path.mkdir(parents=True, exist_ok=True)

        # Create attempt manifest
        attempt = AttemptManifest(
            attempt_id=attempt_id,
            run_name=run_name,
            submitted_at=datetime.now().isoformat(),
            status=AttemptStatus.PENDING,
            user_args=list(user_args or []),
            resume_from=resume_from,
        )

        # Save attempt manifest
        attempt.save(attempt_path / "attempt.yaml")

        # Update state
        state = self.load_state(project, run_name)
        state.latest_attempt = attempt_id
        self.save_state(project, run_name, state)

        return attempt

    def load_attempt(
        self, project: str, run_name: str, attempt_id: int
    ) -> AttemptManifest:
        """Load attempt manifest."""
        path = self.attempt_dir(project, run_name, attempt_id) / "attempt.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Attempt not found: {project}/{run_name}/{attempt_id}"
            )
        return AttemptManifest.load(path)

    def save_attempt(
        self, project: str, run_name: str, attempt: AttemptManifest
    ) -> None:
        """Save attempt manifest."""
        path = self.attempt_dir(project, run_name, attempt.attempt_id) / "attempt.yaml"
        attempt.save(path)

    def save_launch_plan(
        self, project: str, run_name: str, attempt_id: int, plan: LaunchPlan
    ) -> None:
        """Save launch plan for an attempt."""
        path = self.attempt_dir(project, run_name, attempt_id) / "launch_plan.yaml"
        plan.save(path)

    def load_launch_plan(
        self, project: str, run_name: str, attempt_id: int
    ) -> LaunchPlan:
        """Load launch plan for an attempt."""
        path = self.attempt_dir(project, run_name, attempt_id) / "launch_plan.yaml"
        return LaunchPlan.load(path)

    def get_latest_attempt(
        self, project: str, run_name: str
    ) -> AttemptManifest | None:
        """Get the latest attempt for a run."""
        manifest = self.load_manifest(project, run_name)
        if manifest.latest_attempt == 0:
            return None
        return self.load_attempt(project, run_name, manifest.latest_attempt)

    # --- Snapshot Management ---

    def add_snapshot(
        self,
        project: str,
        run_name: str,
        snapshot: SnapshotManifest,
    ) -> None:
        """Add a snapshot to the run manifest."""
        manifest = self.load_manifest(project, run_name)

        # Remove existing snapshot with same name
        manifest.snapshots = [s for s in manifest.snapshots if s.name != snapshot.name]
        manifest.snapshots.append(snapshot)

        self.save_manifest(manifest)

    def get_snapshot_dest_path(
        self, project: str, run_name: str, snapshot_name: str, snapshot_id: str
    ) -> Path:
        """Get destination path for a snapshot."""
        return self.snapshots_dir(project, run_name) / f"{snapshot_name}_{snapshot_id}"

    # --- Listing ---

    def list_projects(self) -> list[str]:
        """List all projects."""
        if not self.registry_root.exists():
            return []
        return [
            d.name
            for d in self.registry_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def list_runs(self, project: str) -> list[str]:
        """List all runs in a project."""
        project_path = self.project_dir(project)
        if not project_path.exists():
            return []
        return [
            d.name
            for d in project_path.iterdir()
            if d.is_dir() and (d / "manifest.yaml").exists()
        ]

    def iter_runs(self, project: str) -> Iterator[RunManifest]:
        """Iterate over all runs in a project."""
        for run_name in self.list_runs(project):
            yield self.load_manifest(project, run_name)
