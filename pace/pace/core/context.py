"""Runtime context for PACE.

The RuntimeContext is the fully resolved object containing all information
needed to build the final execution plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pace.config.models import ProjectConfig
from pace.core.manifests import RunManifest, WandBInfo
from pace.core.templating import TemplateContext, TemplateEngine


@dataclass
class ResolvedBind:
    """A resolved bind mount.

    Attributes:
        host_path: Resolved host path.
        container_path: Container mount path.
        mode: Mount mode (ro or rw).
    """

    host_path: str
    container_path: str
    mode: str = "rw"

    def to_bind_spec(self) -> str:
        """Convert to Apptainer bind specification."""
        return f"{self.host_path}:{self.container_path}:{self.mode}"


@dataclass
class RuntimeContext:
    """Fully resolved runtime context for execution.

    This contains all resolved paths and values needed to build
    the launch plan and execute the job.

    Attributes:
        project: Project name.
        run_name: Run name.
        attempt_id: Attempt number.
        job_id: SLURM job ID (populated after submission).
        config: Project configuration.
        manifest: Run manifest.
        user_args: User-provided command line arguments.
        is_resume: Whether this is a resume attempt.
        resume_checkpoint_host: Resume checkpoint path on host.
        resume_checkpoint_container: Resume checkpoint path in container.
        snapshot_paths: Map of snapshot name to resolved host path.
        staged_paths: Map of staged input name to resolved path.
        persistent_paths_host: Map of output name to host path.
        persistent_paths_container: Map of output name to container path.
        resolved_binds: List of resolved bind mounts.
        environment: Resolved environment variables.
        template_engine: Template engine for further resolution.
    """

    project: str
    run_name: str
    attempt_id: int
    config: ProjectConfig
    manifest: RunManifest

    job_id: str = ""
    user_args: list[str] = field(default_factory=list)

    is_resume: bool = False
    resume_checkpoint_host: str = ""
    resume_checkpoint_container: str = ""

    snapshot_paths: dict[str, str] = field(default_factory=dict)
    staged_paths: dict[str, str] = field(default_factory=dict)
    persistent_paths_host: dict[str, str] = field(default_factory=dict)
    persistent_paths_container: dict[str, str] = field(default_factory=dict)

    resolved_binds: list[ResolvedBind] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    _template_engine: TemplateEngine | None = field(default=None, repr=False)

    @property
    def template_engine(self) -> TemplateEngine:
        """Get template engine, creating if needed."""
        if self._template_engine is None:
            self._template_engine = self._create_template_engine()
        return self._template_engine

    def _create_template_engine(self) -> TemplateEngine:
        """Create template engine with current context values."""
        wandb_values: dict[str, str] = {}
        if self.manifest.wandb:
            wandb_values = {
                "run_name": self.manifest.wandb.run_name,
                "run_id": self.manifest.wandb.run_id,
            }
            if self.manifest.wandb.project:
                wandb_values["project"] = self.manifest.wandb.project
            if self.manifest.wandb.entity:
                wandb_values["entity"] = self.manifest.wandb.entity

        context = TemplateContext(
            run_name=self.run_name,
            project=self.project,
            attempt_id=self.attempt_id,
            job_id=self.job_id,
            snapshots=self.snapshot_paths,
            staged=self.staged_paths,
            shared={
                inp.name: inp.host_path for inp in self.config.shared_inputs
            },
            persistent=self.persistent_paths_host,
            container=self.persistent_paths_container,
            wandb=wandb_values,
            resume_path_host=self.resume_checkpoint_host,
            resume_path_container=self.resume_checkpoint_container,
        )

        return TemplateEngine(context)

    def refresh_template_engine(self) -> None:
        """Refresh template engine after context changes."""
        self._template_engine = self._create_template_engine()

    def resolve(self, template: str) -> str:
        """Resolve a template string using current context."""
        return self.template_engine.resolve(template)

    def resolve_list(self, templates: list[str]) -> list[str]:
        """Resolve a list of template strings."""
        return self.template_engine.resolve_list(templates)

    def resolve_dict(self, templates: dict[str, str]) -> dict[str, str]:
        """Resolve template strings in dictionary values."""
        return self.template_engine.resolve_dict(templates)


class RuntimeContextBuilder:
    """Builder for RuntimeContext.

    Handles the multi-step process of resolving all context values.
    """

    def __init__(
        self,
        config: ProjectConfig,
        manifest: RunManifest,
        run_name: str,
        attempt_id: int,
    ):
        """Initialize the builder.

        Args:
            config: Project configuration.
            manifest: Run manifest.
            run_name: Name of the run.
            attempt_id: Attempt number.
        """
        self.config = config
        self.manifest = manifest
        self.run_name = run_name
        self.attempt_id = attempt_id

        self.user_args: list[str] = []
        self.is_resume = False
        self.resume_checkpoint_host = ""
        self.resume_checkpoint_container = ""
        self.snapshot_paths: dict[str, str] = {}
        self.staged_paths: dict[str, str] = {}

    def with_user_args(self, args: list[str]) -> RuntimeContextBuilder:
        """Set user-provided arguments."""
        self.user_args = args
        return self

    def with_resume(
        self, checkpoint_host: str, checkpoint_container: str
    ) -> RuntimeContextBuilder:
        """Set resume checkpoint paths."""
        self.is_resume = True
        self.resume_checkpoint_host = checkpoint_host
        self.resume_checkpoint_container = checkpoint_container
        return self

    def with_snapshot_paths(self, paths: dict[str, str]) -> RuntimeContextBuilder:
        """Set snapshot paths."""
        self.snapshot_paths = paths
        return self

    def with_staged_paths(self, paths: dict[str, str]) -> RuntimeContextBuilder:
        """Set staged input paths."""
        self.staged_paths = paths
        return self

    def build(self) -> RuntimeContext:
        """Build the RuntimeContext.

        Returns:
            Fully configured RuntimeContext.
        """
        # Create initial context for template resolution
        context = RuntimeContext(
            project=self.config.project,
            run_name=self.run_name,
            attempt_id=self.attempt_id,
            config=self.config,
            manifest=self.manifest,
            user_args=self.user_args,
            is_resume=self.is_resume,
            resume_checkpoint_host=self.resume_checkpoint_host,
            resume_checkpoint_container=self.resume_checkpoint_container,
            snapshot_paths=self.snapshot_paths,
            staged_paths=self.staged_paths,
        )

        # Resolve persistent output paths
        for output in self.config.persistent_outputs:
            # Resolve host path template
            host_path = context.resolve(output.host_path)
            context.persistent_paths_host[output.name] = host_path
            context.persistent_paths_container[output.name] = output.container_path

        # Refresh template engine with updated paths
        context.refresh_template_engine()

        # Resolve binds
        for bind in self.config.binds:
            host_path = context.resolve(bind.host)
            container_path = bind.container  # Container paths are not templated
            context.resolved_binds.append(
                ResolvedBind(
                    host_path=host_path,
                    container_path=container_path,
                    mode=bind.mode,
                )
            )

        return context
