"""Staging planner for PACE.

Determines what gets copied to $TMPDIR vs bound directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pace.config.models import ProjectConfig, SharedInputConfig, StageMode


class StagingAction(str, Enum):
    """Action to take for a staged resource."""

    COPY = "copy"  # Copy to $TMPDIR
    BIND = "bind"  # Bind mount directly


@dataclass
class StagedResource:
    """A resource to be staged.

    Attributes:
        name: Resource identifier.
        source_path: Source path on cluster storage.
        stage_action: How to stage this resource.
        dest_path: Destination path (resolved at runtime for COPY).
        container_path: Mount path inside container.
    """

    name: str
    source_path: str
    stage_action: StagingAction
    dest_path: str = ""  # Set at runtime for COPY
    container_path: str = ""


@dataclass
class StagingPlan:
    """Complete staging plan for a run.

    Attributes:
        copy_resources: Resources to copy to $TMPDIR.
        bind_resources: Resources to bind directly.
        tmpdir_base: Base path for $TMPDIR staging.
    """

    copy_resources: list[StagedResource] = field(default_factory=list)
    bind_resources: list[StagedResource] = field(default_factory=list)
    tmpdir_base: str = "$TMPDIR"

    def get_resource(self, name: str) -> StagedResource | None:
        """Get a resource by name."""
        for resource in self.copy_resources + self.bind_resources:
            if resource.name == name:
                return resource
        return None

    def get_staged_path(self, name: str) -> str:
        """Get the staged path for a resource."""
        resource = self.get_resource(name)
        if resource is None:
            raise ValueError(f"Unknown resource: {name}")
        if resource.stage_action == StagingAction.COPY:
            return resource.dest_path
        return resource.source_path


class StagingPlanner:
    """Plans staging operations for shared inputs.

    Determines which inputs need to be copied to $TMPDIR and
    which can be bound directly.
    """

    def __init__(self, config: ProjectConfig):
        """Initialize the planner.

        Args:
            config: Project configuration.
        """
        self.config = config

    def create_plan(self, tmpdir_base: str = "$TMPDIR") -> StagingPlan:
        """Create a staging plan.

        Args:
            tmpdir_base: Base path for temporary staging.

        Returns:
            Complete staging plan.
        """
        plan = StagingPlan(tmpdir_base=tmpdir_base)

        for input_config in self.config.shared_inputs:
            resource = self._create_resource(input_config, tmpdir_base)
            if resource.stage_action == StagingAction.COPY:
                plan.copy_resources.append(resource)
            else:
                plan.bind_resources.append(resource)

        return plan

    def _create_resource(
        self,
        input_config: SharedInputConfig,
        tmpdir_base: str,
    ) -> StagedResource:
        """Create a staged resource from input config.

        Args:
            input_config: Shared input configuration.
            tmpdir_base: Base path for temporary staging.

        Returns:
            StagedResource configuration.
        """
        if input_config.stage_mode == StageMode.COPY_TO_TMP:
            action = StagingAction.COPY
            dest_path = f"{tmpdir_base}/{input_config.name}"
        else:
            action = StagingAction.BIND
            dest_path = input_config.host_path

        return StagedResource(
            name=input_config.name,
            source_path=input_config.host_path,
            stage_action=action,
            dest_path=dest_path,
            container_path=input_config.container_path or "",
        )

    def generate_staging_commands(self, plan: StagingPlan) -> list[str]:
        """Generate shell commands to perform staging.

        Args:
            plan: Staging plan.

        Returns:
            List of shell commands.
        """
        commands = []

        for resource in plan.copy_resources:
            # Create destination directory
            commands.append(f"mkdir -p {resource.dest_path}")
            # Copy with rsync
            source = resource.source_path
            if not source.endswith("/"):
                source += "/"
            commands.append(
                f"rsync -a --delete {source} {resource.dest_path}/"
            )

        return commands
