"""Template engine for PACE placeholder resolution.

Handles resolution of placeholders like {run_name}, {snapshot.repo},
{persistent.logs}, etc. in configuration strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class TemplateError(Exception):
    """Error during template resolution."""

    pass


@dataclass
class TemplateContext:
    """Context for template resolution.

    This holds all the values that can be substituted into templates.

    Attributes:
        run_name: Name of the run.
        project: Project name.
        attempt_id: Current attempt number.
        job_id: SLURM job ID (if available).
        config_name: Config name (user-provided).
        robot_platform: Robot platform (user-provided, optional).
        snapshots: Map of snapshot name to host path.
        staged: Map of staged input name to path (after staging).
        shared: Map of shared input name to host path.
        persistent: Map of persistent output name to host path.
        container: Map of container path aliases (e.g., logs -> /workspace/logs).
        wandb: WandB-related values.
        resume_path_host: Resume checkpoint path on host.
        resume_path_container: Resume checkpoint path in container.
        user_vars: User-provided variables.
    """

    run_name: str = ""
    project: str = ""
    attempt_id: int = 0
    job_id: str = ""
    config_name: str = ""
    robot_platform: str = ""

    # Path mappings
    snapshots: dict[str, str] = field(default_factory=dict)
    staged: dict[str, str] = field(default_factory=dict)
    shared: dict[str, str] = field(default_factory=dict)
    persistent: dict[str, str] = field(default_factory=dict)
    container: dict[str, str] = field(default_factory=dict)

    # WandB
    wandb: dict[str, str] = field(default_factory=dict)

    # Resume paths
    resume_path_host: str = ""
    resume_path_container: str = ""

    # User-provided variables
    user_vars: dict[str, str] = field(default_factory=dict)


class TemplateEngine:
    """Engine for resolving template placeholders.

    Supports placeholders in the format {name} or {namespace.name}.
    """

    # Pattern to match {placeholder} or {namespace.placeholder}
    PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\}")

    def __init__(self, context: TemplateContext):
        """Initialize with a template context.

        Args:
            context: The template context with values.
        """
        self.context = context
        self._value_cache: dict[str, str] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        """Build the value cache from context."""
        ctx = self.context

        # Simple values
        self._value_cache["run_name"] = ctx.run_name
        self._value_cache["project"] = ctx.project
        self._value_cache["attempt_id"] = str(ctx.attempt_id)
        self._value_cache["job_id"] = ctx.job_id
        self._value_cache["config_name"] = ctx.config_name
        self._value_cache["robot_platform"] = ctx.robot_platform
        self._value_cache["resume_path_host"] = ctx.resume_path_host
        self._value_cache["resume_path_container"] = ctx.resume_path_container

        # Namespaced values
        for name, path in ctx.snapshots.items():
            self._value_cache[f"snapshot.{name}"] = path

        for name, path in ctx.staged.items():
            self._value_cache[f"staged.{name}"] = path

        for name, path in ctx.shared.items():
            self._value_cache[f"shared.{name}"] = path

        for name, path in ctx.persistent.items():
            self._value_cache[f"persistent.{name}"] = path

        for name, path in ctx.container.items():
            self._value_cache[f"container.{name}"] = path

        for name, value in ctx.wandb.items():
            self._value_cache[f"wandb.{name}"] = value

        # User variables
        for name, value in ctx.user_vars.items():
            self._value_cache[name] = value

    def resolve(self, template: str, strict: bool = True) -> str:
        """Resolve all placeholders in a template string.

        Args:
            template: String with {placeholder} patterns.
            strict: If True, raise error on missing placeholders.

        Returns:
            Resolved string with placeholders replaced.

        Raises:
            TemplateError: If strict and placeholder not found.
        """

        def replacer(match: re.Match) -> str:
            key = match.group(1)
            if key in self._value_cache:
                return self._value_cache[key]
            if strict:
                raise TemplateError(f"Unknown placeholder: {{{key}}}")
            return match.group(0)  # Leave unchanged

        return self.PLACEHOLDER_PATTERN.sub(replacer, template)

    def resolve_list(self, templates: list[str], strict: bool = True) -> list[str]:
        """Resolve placeholders in a list of strings.

        Args:
            templates: List of template strings.
            strict: If True, raise error on missing placeholders.

        Returns:
            List of resolved strings.
        """
        return [self.resolve(t, strict=strict) for t in templates]

    def resolve_dict(
        self, templates: dict[str, str], strict: bool = True
    ) -> dict[str, str]:
        """Resolve placeholders in dictionary values.

        Args:
            templates: Dictionary with template values.
            strict: If True, raise error on missing placeholders.

        Returns:
            Dictionary with resolved values.
        """
        return {k: self.resolve(v, strict=strict) for k, v in templates.items()}

    def has_placeholder(self, text: str) -> bool:
        """Check if a string contains any placeholders.

        Args:
            text: String to check.

        Returns:
            True if placeholders found.
        """
        return bool(self.PLACEHOLDER_PATTERN.search(text))

    def list_placeholders(self, text: str) -> list[str]:
        """List all placeholders in a string.

        Args:
            text: String to scan.

        Returns:
            List of placeholder names (without braces).
        """
        return self.PLACEHOLDER_PATTERN.findall(text)

    def update_context(self, **kwargs: Any) -> None:
        """Update context values and rebuild cache.

        Args:
            **kwargs: Values to update in context.
        """
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        self._build_cache()

    def add_values(self, namespace: str, values: dict[str, str]) -> None:
        """Add values to a namespace.

        Args:
            namespace: Namespace name (e.g., 'snapshot', 'staged').
            values: Dictionary of name -> value.
        """
        for name, value in values.items():
            self._value_cache[f"{namespace}.{name}"] = value

    def get_value(self, key: str) -> str | None:
        """Get a value by key.

        Args:
            key: Placeholder key (e.g., 'run_name' or 'snapshot.repo').

        Returns:
            Value if found, None otherwise.
        """
        return self._value_cache.get(key)


def validate_templates(
    templates: list[str],
    available_keys: set[str],
) -> list[str]:
    """Validate that all placeholders in templates are resolvable.

    Args:
        templates: List of template strings to validate.
        available_keys: Set of available placeholder keys.

    Returns:
        List of unknown placeholder keys.
    """
    engine = TemplateEngine(TemplateContext())
    unknown = []

    for template in templates:
        placeholders = engine.list_placeholders(template)
        for placeholder in placeholders:
            if placeholder not in available_keys:
                unknown.append(placeholder)

    return list(set(unknown))
