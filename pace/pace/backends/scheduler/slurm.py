"""SLURM scheduler backend for PACE.

Handles SLURM job script generation and submission.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pace.backends.container.apptainer import ApptainerBackend
from pace.backends.staging import StagingPlan
from pace.config.models import ResourcesConfig
from pace.core.manifests import LaunchPlan


def _quote_arg_preserving_env(arg: str) -> str:
    """Quote shell arg while allowing env var expansion (e.g., $TMPDIR)."""
    if "$" not in arg:
        return shlex.quote(arg)
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass
class SlurmJobScript:
    """Represents a SLURM job script.

    Attributes:
        job_name: SLURM job name.
        output_path: Path for stdout log.
        error_path: Path for stderr log.
        resources: Resource configuration.
        pre_commands: Commands to run before staging.
        staging_commands: Commands to stage data to $TMPDIR.
        apptainer_command: The Apptainer execution command.
    """

    job_name: str
    output_path: str
    resources: ResourcesConfig
    pre_commands: list[str] = field(default_factory=list)
    staging_commands: list[str] = field(default_factory=list)
    apptainer_command: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render the complete SLURM job script.

        Returns:
            Shell script content.
        """
        lines = [
            "#!/bin/bash",
            "",
            "# PACE SLURM job script",
            "# Generated - do not edit manually",
            "",
        ]

        # SBATCH directives
        lines.append(f"#SBATCH --job-name={self.job_name}")
        lines.append(f"#SBATCH --output={self.output_path}")
        lines.append(f"#SBATCH --nodes={self.resources.nodes}")

        # GPU specification
        if self.resources.gpus > 0:
            gpu_spec = f"{self.resources.gpu_type}:{self.resources.gpus}"
            lines.append(f"#SBATCH --gres=gpu:{gpu_spec}")

        lines.append(f"#SBATCH --cpus-per-task={self.resources.cpus}")
        lines.append(f"#SBATCH --mem={self.resources.mem_gb}G")
        lines.append(f"#SBATCH --time={self.resources.time}")

        if self.resources.partition:
            lines.append(f"#SBATCH --partition={self.resources.partition}")

        if self.resources.account:
            lines.append(f"#SBATCH --account={self.resources.account}")

        if self.resources.qos:
            lines.append(f"#SBATCH --qos={self.resources.qos}")

        lines.append("")
        lines.append("# Exit on error")
        lines.append("set -euo pipefail")
        lines.append("")

        # Print job info
        lines.append("# Job information")
        lines.append('echo "Job ID: $SLURM_JOB_ID"')
        lines.append('echo "Node: $SLURMD_NODENAME"')
        lines.append('echo "Started: $(date)"')
        lines.append("")

        # Pre-commands
        if self.pre_commands:
            lines.append("# Pre-execution commands")
            lines.extend(self.pre_commands)
            lines.append("")

        # Staging commands
        if self.staging_commands:
            lines.append("# Stage data to $TMPDIR")
            lines.extend(self.staging_commands)
            lines.append("")

        # Apptainer execution
        lines.append("# Execute container")
        apptainer_str = " \\\n    ".join(
            _quote_arg_preserving_env(arg) for arg in self.apptainer_command
        )
        lines.append(apptainer_str)

        lines.append("")
        lines.append('echo "Completed: $(date)"')

        return "\n".join(lines) + "\n"


class SlurmBackend:
    """Backend for SLURM job submission."""

    def __init__(self):
        """Initialize the backend."""
        self.apptainer = ApptainerBackend()

    def create_job_script(
        self,
        plan: LaunchPlan,
        resources: ResourcesConfig,
        job_name: str,
        log_dir: str,
        wrapper_path: str,
        staging_plan: StagingPlan | None = None,
        pre_apptainer_commands: list[str] | None = None,
    ) -> SlurmJobScript:
        """Create a SLURM job script.

        Args:
            plan: Launch plan.
            resources: Resource configuration.
            job_name: SLURM job name.
            log_dir: Directory for SLURM logs.
            wrapper_path: Path to compute wrapper inside container.
            staging_plan: Optional staging plan for $TMPDIR copying.

        Returns:
            SlurmJobScript ready to render.
        """
        # Build Apptainer command
        apptainer_cmd = self.apptainer.build_exec_command(plan, wrapper_path)

        # Build staging commands
        staging_commands = []
        if staging_plan:
            from pace.backends.staging import StagingPlanner

            # Create a dummy planner just for commands
            # In practice, the commands would come from the staging planner
            for resource in staging_plan.copy_resources:
                staging_commands.append(f"mkdir -p {resource.dest_path}")
                source = resource.source_path
                if not source.endswith("/"):
                    source += "/"
                staging_commands.append(
                    f"rsync -a --delete {source} {resource.dest_path}/"
                )

        return SlurmJobScript(
            job_name=job_name,
            output_path=f"{log_dir}/slurm_%j.out",
            resources=resources,
            pre_commands=pre_apptainer_commands or [],
            staging_commands=staging_commands,
            apptainer_command=apptainer_cmd,
        )

    def submit(
        self,
        script: SlurmJobScript,
        script_path: Path | None = None,
        dry_run: bool = False,
    ) -> int | None:
        """Submit a SLURM job.

        Args:
            script: Job script to submit.
            script_path: Where to save the script (temp file if None).
            dry_run: If True, don't actually submit.

        Returns:
            SLURM job ID if submitted, None if dry run or error.
        """
        content = script.render()

        if dry_run:
            print("=== SLURM Job Script (dry run) ===")
            print(content)
            print("=== End of script ===")
            return None

        # Write script to file
        if script_path:
            script_path.write_text(content)
            script_path.chmod(0o755)
            path_to_submit = script_path
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as f:
                f.write(content)
                path_to_submit = Path(f.name)

        try:
            result = subprocess.run(
                ["sbatch", str(path_to_submit)],
                capture_output=True,
                text=True,
                check=True,
            )
            # Parse job ID from "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                return int(match.group(1))
            return None
        except subprocess.CalledProcessError as e:
            print(f"sbatch failed: {e.stderr}")
            return None
        except FileNotFoundError:
            print("sbatch command not found - are you on a SLURM cluster?")
            return None

    def get_job_state(self, job_id: int) -> str:
        """Get the state of a SLURM job.

        Args:
            job_id: SLURM job ID.

        Returns:
            Job state string (e.g., PENDING, RUNNING, COMPLETED).
        """
        try:
            # Try squeue first for running jobs
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            # Fall back to sacct for completed jobs
            result = subprocess.run(
                ["sacct", "-j", str(job_id), "-n", "-o", "State", "-P"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                states = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
                if states:
                    return states[0]

            return "UNKNOWN"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "UNKNOWN"

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a SLURM job.

        Args:
            job_id: SLURM job ID.

        Returns:
            True if cancellation was successful.
        """
        try:
            result = subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
