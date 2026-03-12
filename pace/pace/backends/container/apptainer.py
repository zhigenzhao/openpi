"""Apptainer container backend for PACE.

Handles Apptainer/Singularity container execution.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pace.core.manifests import LaunchPlan


@dataclass
class ApptainerConfig:
    """Configuration for Apptainer execution.

    Attributes:
        enable_nvidia: Enable NVIDIA GPU support (--nv).
        clean_env: Start with clean environment (--cleanenv).
        writable_tmpfs: Use writable tmpfs overlay (--writable-tmpfs).
        extra_args: Additional Apptainer arguments.
    """

    enable_nvidia: bool = True
    clean_env: bool = True
    writable_tmpfs: bool = False
    extra_args: list[str] = field(default_factory=list)


class ApptainerBackend:
    """Backend for Apptainer container execution."""

    def __init__(self, config: ApptainerConfig | None = None):
        """Initialize the backend.

        Args:
            config: Apptainer configuration.
        """
        self.config = config or ApptainerConfig()

    def build_exec_command(
        self,
        plan: LaunchPlan,
        wrapper_path: str,
    ) -> list[str]:
        """Build the Apptainer exec command.

        Args:
            plan: Launch plan with binds and image.
            wrapper_path: Path to compute wrapper script.

        Returns:
            Command as list of strings.
        """
        cmd = ["apptainer", "exec"]

        # Add flags based on config
        if self.config.enable_nvidia:
            cmd.append("--nv")

        if self.config.clean_env:
            cmd.append("--cleanenv")

        if self.config.writable_tmpfs:
            cmd.append("--writable-tmpfs")

        # Add extra args
        cmd.extend(self.config.extra_args)

        # Add bind mounts
        for bind_spec in plan.binds:
            cmd.extend(["--bind", bind_spec])

        # Add image path
        cmd.append(plan.image_path)

        # Add wrapper script execution
        cmd.extend(["bash", wrapper_path])

        return cmd

    def build_run_command(
        self,
        plan: LaunchPlan,
    ) -> list[str]:
        """Build the Apptainer run command (uses container's runscript).

        Args:
            plan: Launch plan.

        Returns:
            Command as list of strings.
        """
        cmd = ["apptainer", "run"]

        if self.config.enable_nvidia:
            cmd.append("--nv")

        if self.config.clean_env:
            cmd.append("--cleanenv")

        if self.config.writable_tmpfs:
            cmd.append("--writable-tmpfs")

        cmd.extend(self.config.extra_args)

        for bind_spec in plan.binds:
            cmd.extend(["--bind", bind_spec])

        cmd.append(plan.image_path)

        # Add the actual command
        cmd.extend(plan.command)

        return cmd

    def render_exec_script(
        self,
        plan: LaunchPlan,
        wrapper_path: str,
    ) -> str:
        """Render the Apptainer exec command as a shell script section.

        Args:
            plan: Launch plan.
            wrapper_path: Path to compute wrapper.

        Returns:
            Shell script content.
        """
        cmd = self.build_exec_command(plan, wrapper_path)
        cmd_str = " \\\n    ".join(shlex.quote(arg) for arg in cmd)

        return f"""\
# Execute Apptainer container
{cmd_str}
"""

    def validate_image(self, image_path: str) -> bool:
        """Validate that an image exists and is readable.

        Args:
            image_path: Path to container image.

        Returns:
            True if image is valid.
        """
        path = Path(image_path)
        return path.exists() and path.is_file()

    def inspect_image(self, image_path: str) -> dict[str, Any] | None:
        """Inspect an Apptainer image.

        Args:
            image_path: Path to container image.

        Returns:
            Image inspection data, or None on error.
        """
        try:
            result = subprocess.run(
                ["apptainer", "inspect", image_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Parse simple key: value format
                data = {}
                for line in result.stdout.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        data[key.strip()] = value.strip()
                return data
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
