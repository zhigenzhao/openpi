"""Tests for scheduler backend rendering."""

from __future__ import annotations

from pace.backends.scheduler import SlurmBackend, SlurmJobScript
from pace.config.models import ResourcesConfig
from pace.core.manifests import LaunchPlan


def test_slurm_script_uses_configured_log_dir() -> None:
    """SBATCH output/error should be rendered under configured scheduler.log_dir."""
    plan = LaunchPlan(
        workdir="/workspace",
        shell_init=[],
        command=["python", "train.py"],
        environment={},
        binds=[],
        image_path="/tmp/image.sif",
    )
    backend = SlurmBackend()
    script = backend.create_job_script(
        plan=plan,
        resources=ResourcesConfig(),
        job_name="pace-test",
        log_dir="/cluster/logs/run1/scheduler",
        wrapper_path="/tmp/compute_wrapper.sh",
    )
    rendered = script.render()
    assert "#SBATCH --output=/cluster/logs/run1/scheduler/slurm_%j.out" in rendered
    assert "#SBATCH --error=/cluster/logs/run1/scheduler/slurm_%j.err" in rendered


def test_slurm_script_preserves_tmpdir_bind_expansion() -> None:
    """$TMPDIR bind sources must not be single-quoted in rendered script."""
    script = SlurmJobScript(
        job_name="pace-test",
        output_path="/tmp/out",
        error_path="/tmp/err",
        resources=ResourcesConfig(),
        apptainer_command=[
            "apptainer",
            "exec",
            "--bind",
            "$TMPDIR/pi_home:/workspace/RLinf/pi_home:rw",
            "/tmp/image.sif",
            "bash",
            "/tmp/wrapper.sh",
        ],
    )
    rendered = script.render()
    assert "'$TMPDIR/pi_home:/workspace/RLinf/pi_home:rw'" not in rendered
    assert '"$TMPDIR/pi_home:/workspace/RLinf/pi_home:rw"' in rendered
