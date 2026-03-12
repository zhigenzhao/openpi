"""Injection logic for PACE launch plans.

Handles conditional injection of environment variables and CLI arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pace.config.models import (
    CliInjectionConfig,
    EnvInjectionConfig,
    InjectionCondition,
    LaunchPhase,
)
from pace.core.context import RuntimeContext


def should_apply_injection(
    condition: InjectionCondition,
    context: RuntimeContext,
) -> bool:
    """Determine if an injection should be applied.

    Args:
        condition: The injection condition.
        context: Runtime context.

    Returns:
        True if injection should be applied.
    """
    if condition == InjectionCondition.ALWAYS:
        return True

    if condition == InjectionCondition.RESUME_ONLY:
        return context.is_resume

    if condition == InjectionCondition.NON_RESUME_ONLY:
        return not context.is_resume

    if condition == InjectionCondition.FIRST_ATTEMPT_ONLY:
        return context.attempt_id == 1

    return False


@dataclass
class EnvInjector:
    """Handles environment variable injection.

    Collects and resolves environment variables based on conditions.
    """

    configs: list[EnvInjectionConfig] = field(default_factory=list)

    def collect_env(self, context: RuntimeContext) -> dict[str, str]:
        """Collect all applicable environment variables.

        Args:
            context: Runtime context.

        Returns:
            Dictionary of environment variable name to value.
        """
        env: dict[str, str] = {}

        for config in self.configs:
            if should_apply_injection(config.when, context):
                # Resolve template values
                resolved = context.resolve_dict(config.values)
                env.update(resolved)

        return env


@dataclass
class CliInjector:
    """Handles CLI argument injection.

    Collects and organizes CLI arguments by phase.
    """

    configs: list[CliInjectionConfig] = field(default_factory=list)

    def collect_args_by_phase(
        self, context: RuntimeContext
    ) -> dict[LaunchPhase, list[str]]:
        """Collect CLI arguments organized by phase.

        Args:
            context: Runtime context.

        Returns:
            Dictionary mapping phase to list of arguments.
        """
        args_by_phase: dict[LaunchPhase, list[str]] = {
            phase: [] for phase in LaunchPhase
        }

        for config in self.configs:
            if should_apply_injection(config.when, context):
                # Resolve template values in args
                resolved_args = context.resolve_list(config.args)
                args_by_phase[config.phase].extend(resolved_args)

        return args_by_phase

    def collect_args_for_phase(
        self, phase: LaunchPhase, context: RuntimeContext
    ) -> list[str]:
        """Collect CLI arguments for a specific phase.

        Args:
            phase: The launch phase.
            context: Runtime context.

        Returns:
            List of arguments for this phase.
        """
        args = []

        for config in self.configs:
            if config.phase == phase and should_apply_injection(config.when, context):
                resolved_args = context.resolve_list(config.args)
                args.extend(resolved_args)

        return args
