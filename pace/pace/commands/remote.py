"""Remote execution helpers for local-PC vs cluster-side orchestration."""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path

import click
import yaml

from pace.backends.remote import SSHRemoteError, SSHRemoteExecutor
from pace.config.models import ProjectConfig

DEFAULT_RSYNC_EXCLUDES = [
    ".git/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
]


def is_cluster_side(config: ProjectConfig) -> bool:
    """Best-effort check whether current host can directly access cluster paths."""
    if not config.remote.enabled:
        return True
    return Path(config.remote.project_dir).exists()


def execution_mode(
    config: ProjectConfig,
    force_remote: bool,
    force_local: bool,
) -> str:
    """Resolve execution mode.

    Returns:
        "cluster" if command should run with direct local filesystem access.
        "local_pc" if SSH operations are needed.
    """
    if force_remote and force_local:
        raise click.ClickException("Cannot use both --remote and --no-remote")

    if force_remote:
        return "local_pc"
    if force_local:
        return "cluster"

    if not config.remote.enabled:
        return "cluster"

    if config.remote.mode == "local":
        return "cluster"
    if config.remote.mode == "remote":
        return "local_pc"

    return "cluster" if is_cluster_side(config) else "local_pc"


def make_executor(config: ProjectConfig) -> SSHRemoteExecutor:
    """Build SSH executor from config."""
    return SSHRemoteExecutor.from_config(config.remote)


def local_temp_registry_config_path(cfg_path: Path, config: ProjectConfig) -> tuple[Path, str]:
    """Create a temporary config with local writable registry_root.

    This is used in local-PC submit flow where the configured registry root is remote.
    """
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}

    tmp_registry_root = tempfile.mkdtemp(prefix="pace_registry_")
    data["registry_root"] = tmp_registry_root

    tmp_cfg = Path(tempfile.mkstemp(prefix="pace_cfg_", suffix=".yaml")[1])
    tmp_cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    return tmp_cfg, tmp_registry_root


def remote_run_dir(config: ProjectConfig, run_name: str) -> str:
    """Remote run directory for a project/run name."""
    return f"{config.registry_root}/{config.project}/{run_name}"


def remote_attempt_dir(config: ProjectConfig, run_name: str, attempt_id: int) -> str:
    """Remote attempt directory path."""
    return f"{remote_run_dir(config, run_name)}/attempts/{attempt_id}"


def remote_attempt_manifest_path(config: ProjectConfig, run_name: str, attempt_id: int) -> str:
    """Remote path for attempt manifest YAML."""
    return f"{remote_attempt_dir(config, run_name, attempt_id)}/attempt.yaml"


def resolve_template_path(config: ProjectConfig, template: str, run_name: str) -> str:
    """Resolve basic path template placeholders for command-time path operations."""
    return (
        template.replace("{run_name}", run_name)
        .replace("{project}", config.project)
        .replace("{registry_root}", config.registry_root)
    )


def scheduler_log_dir(config: ProjectConfig, run_name: str) -> str:
    """Resolve scheduler log directory path for a specific run."""
    return resolve_template_path(config, config.scheduler.log_dir, run_name)


def remote_state_path(config: ProjectConfig, run_name: str) -> str:
    """Remote path for run state.json."""
    return f"{remote_run_dir(config, run_name)}/state.json"


def remote_manifest_path(config: ProjectConfig, run_name: str) -> str:
    """Remote path for run manifest.yaml."""
    return f"{remote_run_dir(config, run_name)}/manifest.yaml"


def ssh_read_text(executor: SSHRemoteExecutor, path: str) -> str | None:
    """Read remote file content via ssh cat; returns None when missing."""
    completed = executor.run(["bash", "-lc", f"cat {path}"], remote_cwd="/", stream_output=False)
    if completed.returncode != 0:
        return None
    return completed.stdout


def ssh_read_yaml(executor: SSHRemoteExecutor, path: str) -> dict | None:
    """Read a remote YAML file."""
    text = ssh_read_text(executor, path)
    if text is None:
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def ssh_read_json(executor: SSHRemoteExecutor, path: str) -> dict | None:
    """Read a remote JSON file."""
    text = ssh_read_text(executor, path)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def sync_local_run_to_remote(
    executor: SSHRemoteExecutor,
    local_run_dir: str | Path,
    remote_run_dir_path: str,
) -> None:
    """Sync prepared local run directory to remote registry."""
    executor.sync_project(local_run_dir, remote_run_dir_path, excludes=[])


def remote_path_exists(
    executor: SSHRemoteExecutor,
    remote_path: str,
    remote_cwd: str = "/",
) -> bool:
    """Check whether a remote path exists."""
    quoted = shlex.quote(remote_path)
    completed = executor.run(
        ["bash", "-lc", f"test -e {quoted} && echo yes || echo no"],
        remote_cwd=remote_cwd,
        stream_output=False,
    )
    return completed.returncode == 0 and (completed.stdout or "").strip() == "yes"


def sync_remote_run_to_local(
    executor: SSHRemoteExecutor,
    remote_run_dir_path: str,
    local_run_dir: str | Path,
) -> bool:
    """Sync an existing remote run directory into local temporary registry.

    Returns True when remote run exists and sync was performed.
    """
    if not remote_path_exists(executor, remote_run_dir_path, remote_cwd="/"):
        return False
    executor.sync_from_remote(remote_run_dir_path, local_run_dir, excludes=[])
    return True


def update_remote_attempt_submission(
    executor: SSHRemoteExecutor,
    attempt_yaml_path: str,
    job_id: int,
) -> None:
    """Patch attempt metadata on remote host after successful submission."""
    command = (
        "python3 - <<'PY'\n"
        "import yaml\n"
        f"p = {attempt_yaml_path!r}\n"
        "with open(p) as f:\n"
        "    d = yaml.safe_load(f) or {}\n"
        f"d['slurm_job_id'] = {job_id}\n"
        "d['status'] = 'submitted'\n"
        "with open(p, 'w') as f:\n"
        "    yaml.safe_dump(d, f, sort_keys=False)\n"
        "PY"
    )
    completed = executor.run(["bash", "-lc", command], remote_cwd="/", stream_output=False)
    if completed.returncode != 0:
        raise SSHRemoteError(
            "Failed to update remote attempt metadata after submission. "
            f"stderr: {completed.stderr}"
        )
