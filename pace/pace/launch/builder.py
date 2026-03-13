"""Launch plan builder for PACE.

Builds the complete launch plan by assembling command and environment
from multiple sources and phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pace.config.models import LaunchPhase, ProjectConfig
from pace.core.context import RuntimeContext
from pace.core.manifests import LaunchPlan
from pace.launch.injectors import CliInjector, EnvInjector
from pace.launch.phases import PHASE_ORDER


class LaunchPlanBuilder:
    """Builds launch plans from configuration and context.

    Assembles the final command and environment by:
    1. Starting with base command from config
    2. Injecting CLI args at each phase
    3. Inserting user args at configured phase
    4. Collecting all environment variables
    5. Resolving all templates
    """

    def __init__(self, config: ProjectConfig):
        """Initialize the builder.

        Args:
            config: Project configuration.
        """
        self.config = config
        self.env_injector = EnvInjector(configs=config.injections.env)
        self.cli_injector = CliInjector(configs=config.injections.cli)

    def build(self, context: RuntimeContext) -> LaunchPlan:
        """Build the complete launch plan.

        Args:
            context: Runtime context with resolved paths.

        Returns:
            Complete LaunchPlan ready for execution.
        """
        # Collect environment variables
        environment = self.env_injector.collect_env(context)

        # Build command in phase order
        command = self._build_command(context)

        # Resolve shell init commands
        shell_init = context.resolve_list(self.config.launch.shell_init)

        # Get bind specifications
        binds = [bind.to_bind_spec() for bind in context.resolved_binds]

        return LaunchPlan(
            workdir=self.config.launch.workdir,
            shell_init=shell_init,
            command=command,
            environment=environment,
            binds=binds,
            image_path=self.config.runtime.image,
            pre_launch=list(self.config.hooks.pre_launch),
            post_launch=list(self.config.hooks.post_launch),
        )

    def _build_command(self, context: RuntimeContext) -> list[str]:
        """Build the command by assembling phases.

        Args:
            context: Runtime context.

        Returns:
            Complete command as list of strings.
        """
        # Collect args by phase
        args_by_phase = self.cli_injector.collect_args_by_phase(context)
        if context.is_resume and self.config.resume.cli_args:
            resume_cli_args = context.resolve_list(self.config.resume.cli_args)
            # Resume protocol args (e.g. Hydra overrides) should come after
            # positional run-name arguments injected at FINAL phase.
            args_by_phase[LaunchPhase.FINAL].extend(resume_cli_args)

        # Get user args phase
        user_args_phase = self.config.launch.user_args.phase

        # Build command in phase order
        command: list[str] = []

        for phase in PHASE_ORDER:
            if phase == LaunchPhase.BASE:
                # Add base command (resolved)
                base_cmd = context.resolve_list(self.config.launch.base_command)
                command.extend(base_cmd)
            else:
                # Add injected args for this phase
                command.extend(args_by_phase.get(phase, []))

            # Insert user args at configured phase
            if phase == user_args_phase:
                command.extend(context.user_args)

        return command

    def preview(self, context: RuntimeContext) -> dict:
        """Generate a preview of the launch plan.

        Useful for dry-run and debugging.

        Args:
            context: Runtime context.

        Returns:
            Dictionary with preview information.
        """
        plan = self.build(context)

        return {
            "workdir": plan.workdir,
            "shell_init": plan.shell_init,
            "command": plan.command,
            "command_string": " ".join(plan.command),
            "environment": plan.environment,
            "binds": plan.binds,
            "image_path": plan.image_path,
        }
