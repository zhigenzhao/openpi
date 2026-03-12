"""Launch renderer for PACE.

Renders launch plans into executable shell scripts.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from pace.core.manifests import LaunchPlan


class LaunchRenderer:
    """Renders launch plans to shell scripts."""

    def render_compute_wrapper(self, plan: LaunchPlan) -> str:
        """Render the compute wrapper script.

        This is the script that runs on the compute node inside the job.

        Args:
            plan: Launch plan to render.

        Returns:
            Shell script content.
        """
        lines = [
            "#!/bin/bash",
            "set -eo pipefail",
            "",
            "# PACE compute wrapper - generated script",
            "# Do not edit - regenerate with pace run submit",
            "",
        ]

        # Export environment variables
        if plan.environment:
            lines.append("# Environment variables")
            for key, value in plan.environment.items():
                escaped_value = shlex.quote(value)
                lines.append(f"export {key}={escaped_value}")
            lines.append("")

        # Shell init commands
        if plan.shell_init:
            lines.append("# Shell initialization")
            for cmd in plan.shell_init:
                lines.append(cmd)
            lines.append("")

        # Change to working directory
        lines.append(f"# Working directory")
        lines.append(f"cd {plan.workdir}")
        lines.append("")

        # Execute command
        lines.append("# Execute command")
        command_str = " ".join(shlex.quote(arg) for arg in plan.command)
        lines.append(f"exec {command_str}")

        return "\n".join(lines) + "\n"

    def render_apptainer_command(
        self,
        plan: LaunchPlan,
        wrapper_path: str,
    ) -> list[str]:
        """Render the Apptainer execution command.

        Args:
            plan: Launch plan.
            wrapper_path: Path to compute wrapper script inside container.

        Returns:
            Command as list of strings.
        """
        cmd = [
            "apptainer",
            "exec",
            "--nv",  # Enable NVIDIA GPU support
            "--cleanenv",  # Start with clean environment
        ]

        # Add bind mounts
        for bind_spec in plan.binds:
            cmd.extend(["--bind", bind_spec])

        # Add image path
        cmd.append(plan.image_path)

        # Add wrapper script
        cmd.extend(["bash", wrapper_path])

        return cmd

    def render_env_file(self, plan: LaunchPlan) -> str:
        """Render environment variables to a sourceable file.

        Args:
            plan: Launch plan.

        Returns:
            Shell script content that can be sourced.
        """
        lines = ["# PACE environment variables", ""]
        for key, value in plan.environment.items():
            escaped_value = shlex.quote(value)
            lines.append(f"export {key}={escaped_value}")
        return "\n".join(lines) + "\n"

    def render_command_file(self, plan: LaunchPlan) -> str:
        """Render command to a file for reference.

        Args:
            plan: Launch plan.

        Returns:
            Shell script content showing the command.
        """
        lines = [
            "#!/bin/bash",
            "# PACE command - for reference only",
            "",
            "# Working directory:",
            f"# cd {plan.workdir}",
            "",
            "# Command:",
        ]
        command_str = " ".join(shlex.quote(arg) for arg in plan.command)
        lines.append(command_str)
        return "\n".join(lines) + "\n"

    def save_to_attempt_dir(
        self,
        plan: LaunchPlan,
        attempt_dir: Path,
    ) -> dict[str, Path]:
        """Save all rendered files to attempt directory.

        Args:
            plan: Launch plan.
            attempt_dir: Path to attempt directory.

        Returns:
            Dictionary mapping file type to path.
        """
        attempt_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # Save compute wrapper
        wrapper_path = attempt_dir / "compute_wrapper.sh"
        wrapper_path.write_text(self.render_compute_wrapper(plan))
        wrapper_path.chmod(0o755)
        paths["wrapper"] = wrapper_path

        # Save environment file
        env_path = attempt_dir / "env.resolved"
        env_path.write_text(self.render_env_file(plan))
        paths["env"] = env_path

        # Save command file
        cmd_path = attempt_dir / "command.resolved.sh"
        cmd_path.write_text(self.render_command_file(plan))
        cmd_path.chmod(0o755)
        paths["command"] = cmd_path

        return paths
