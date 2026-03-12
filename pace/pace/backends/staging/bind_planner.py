"""Bind planner for PACE.

Validates and plans container bind mounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pace.config.models import BindConfig, ProjectConfig
from pace.core.context import ResolvedBind, RuntimeContext


class BindConflictError(Exception):
    """Error when bind mounts conflict."""

    pass


class BindValidationError(Exception):
    """Error validating bind configuration."""

    pass


@dataclass
class BindPlan:
    """Complete bind mount plan.

    Attributes:
        binds: List of resolved bind mounts.
        bind_specs: List of bind specifications for Apptainer.
    """

    binds: list[ResolvedBind] = field(default_factory=list)

    @property
    def bind_specs(self) -> list[str]:
        """Get Apptainer bind specifications."""
        return [bind.to_bind_spec() for bind in self.binds]

    def get_bind_for_container_path(self, container_path: str) -> ResolvedBind | None:
        """Get bind for a container path."""
        for bind in self.binds:
            if bind.container_path == container_path:
                return bind
        return None


class BindPlanner:
    """Plans and validates container bind mounts.

    Ensures no conflicts in container mount paths and validates
    that required host paths exist (or can be created).
    """

    def __init__(self, config: ProjectConfig):
        """Initialize the planner.

        Args:
            config: Project configuration.
        """
        self.config = config

    def create_plan(self, context: RuntimeContext) -> BindPlan:
        """Create a bind mount plan.

        Args:
            context: Runtime context with resolved paths.

        Returns:
            BindPlan with validated mounts.

        Raises:
            BindConflictError: If container paths conflict.
            BindValidationError: If validation fails.
        """
        plan = BindPlan(binds=context.resolved_binds.copy())

        # Validate no conflicts
        self._validate_no_conflicts(plan)

        return plan

    def _validate_no_conflicts(self, plan: BindPlan) -> None:
        """Check for conflicting container mount paths.

        Two binds cannot mount to the same container path.

        Args:
            plan: Bind plan to validate.

        Raises:
            BindConflictError: If conflicts found.
        """
        container_paths: dict[str, str] = {}  # container_path -> host_path

        for bind in plan.binds:
            container = bind.container_path

            # Normalize path for comparison
            normalized = str(Path(container).resolve()) if container else container

            if normalized in container_paths:
                existing_host = container_paths[normalized]
                raise BindConflictError(
                    f"Container path conflict at {container}: "
                    f"both {existing_host} and {bind.host_path} mount to same location"
                )

            container_paths[normalized] = bind.host_path

        # Also check for nested mounts that could cause issues
        sorted_paths = sorted(container_paths.keys())
        for i, path in enumerate(sorted_paths):
            for j in range(i + 1, len(sorted_paths)):
                other_path = sorted_paths[j]
                if other_path.startswith(path + "/"):
                    # Nested mount - warn but don't fail
                    # This is actually valid in some cases
                    pass

    def validate_host_paths(
        self,
        plan: BindPlan,
        create_missing: bool = False,
    ) -> list[str]:
        """Validate that host paths exist.

        Args:
            plan: Bind plan to validate.
            create_missing: If True, create missing directories.

        Returns:
            List of missing paths that weren't created.
        """
        missing = []

        for bind in plan.binds:
            host_path = Path(bind.host_path)

            # Skip template paths that aren't resolved yet
            if "{" in str(host_path):
                continue

            # Skip $TMPDIR paths
            if str(host_path).startswith("$"):
                continue

            if not host_path.exists():
                if create_missing:
                    try:
                        host_path.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        missing.append(str(host_path))
                else:
                    missing.append(str(host_path))

        return missing

    def get_persistent_binds(self, context: RuntimeContext) -> list[ResolvedBind]:
        """Get binds for persistent output directories.

        These are the binds that should be created before the run
        to ensure outputs are saved.

        Args:
            context: Runtime context.

        Returns:
            List of persistent output binds.
        """
        persistent_binds = []

        for output in self.config.persistent_outputs:
            if output.name in context.persistent_paths_host:
                host_path = context.persistent_paths_host[output.name]
                persistent_binds.append(
                    ResolvedBind(
                        host_path=host_path,
                        container_path=output.container_path,
                        mode="rw",
                    )
                )

        return persistent_binds

    def ensure_persistent_dirs(self, context: RuntimeContext) -> None:
        """Create persistent output directories if needed.

        Args:
            context: Runtime context with resolved paths.
        """
        for output in self.config.persistent_outputs:
            if output.create_if_missing and output.name in context.persistent_paths_host:
                host_path = Path(context.persistent_paths_host[output.name])
                host_path.mkdir(parents=True, exist_ok=True)
