"""Tests for remote SSH orchestration helpers and routing."""

from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from pace.backends.remote.ssh import SSHRemoteError, SSHRemoteExecutor
from pace.commands.logs import run_logs
from pace.commands.remote import execution_mode, is_cluster_side
from pace.commands.resume import run_resume
from pace.commands.runs import run_list
from pace.commands.status import run_status
from pace.commands.submit import run_submit
from pace.config.models import (
    PersistentOutputConfig,
    ProjectConfig,
    RemoteConfig,
    ResumeConfig,
    RuntimeConfig,
    SchedulerConfig,
)


def _make_config(mode: str = "auto") -> ProjectConfig:
    return ProjectConfig(
        project="proj",
        runtime=RuntimeConfig(image="/tmp/image.sif"),
        scheduler=SchedulerConfig(
            type="slurm",
            log_dir="/cluster/logs/{run_name}/scheduler",
        ),
        persistent_outputs=[
            PersistentOutputConfig(
                name="logs",
                role="logs",
                host_path="/cluster/logs",
                container_path="/workspace/RLinf/logs",
            ),
            PersistentOutputConfig(
                name="checkpoints",
                role="checkpoints",
                host_path="/cluster/logs/{run_name}/checkpoints",
                container_path="/workspace/RLinf/checkpoints",
            ),
        ],
        resume=ResumeConfig(
            checkpoint_output="checkpoints",
            checkpoint_marker="CHECKPOINT_OK",
            sort_pattern=r"global_step_(?P<key>\d+)",
            cli_args=["runner.resume_dir={resume_path_container}"],
        ),
        remote=RemoteConfig(
            enabled=True,
            host="pace-ice",
            user="alice",
            project_dir="/remote/work/RLinf",
            mode=mode,
        ),
    )


def test_execution_mode_conflict_raises():
    """Mutually exclusive force flags should fail."""
    cfg = _make_config()
    with pytest.raises(click.ClickException, match="Cannot use both"):
        execution_mode(cfg, force_remote=True, force_local=True)


def test_execution_mode_respects_config_mode():
    """Configured mode should override auto detection."""
    assert execution_mode(_make_config("remote"), False, False) == "local_pc"
    assert execution_mode(_make_config("local"), False, False) == "cluster"


def test_is_cluster_side_detection(monkeypatch):
    """Cluster-side detection is based on project_dir visibility."""
    cfg = _make_config("auto")
    monkeypatch.setattr("pace.commands.remote.Path.exists", lambda self: True)
    assert is_cluster_side(cfg) is True
    monkeypatch.setattr("pace.commands.remote.Path.exists", lambda self: False)
    assert is_cluster_side(cfg) is False


def test_executor_target_includes_user():
    """Test SSH target rendering."""
    executor = SSHRemoteExecutor(host="pace-ice", user="alice")
    assert executor.target == "alice@pace-ice"


def test_sync_project_failure_raises(monkeypatch, tmp_path: Path):
    """Test rsync errors are surfaced."""
    executor = SSHRemoteExecutor(host="pace-ice")
    monkeypatch.setattr(executor, "ensure_remote_dir", lambda _: None)

    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stderr="permission denied")

    monkeypatch.setattr("pace.backends.remote.ssh.subprocess.run", _fake_run)

    with pytest.raises(SSHRemoteError, match="rsync failed"):
        executor.sync_project(tmp_path, "/remote/work")


def test_submit_dry_run_no_remote_ops(monkeypatch):
    """Dry-run should not attempt SSH sync or remote sbatch in local-PC mode."""
    calls = {"sync": 0, "run": 0}

    monkeypatch.setattr("pace.commands.submit.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.submit.load_config", lambda _p: _make_config("remote"))

    class _FakeExec:
        def sync_project(self, *args, **kwargs):
            calls["sync"] += 1

        def run(self, *args, **kwargs):
            calls["run"] += 1
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("pace.commands.submit.make_executor", lambda _c: _FakeExec())
    monkeypatch.setattr(
        "pace.commands.submit.local_temp_registry_config_path",
        lambda _cfg, _config: (Path("pace.yaml"), "/tmp/pace_registry_test"),
    )

    # Make dry-run lightweight by faking run existence and snapshots.
    monkeypatch.setattr("pace.commands.submit.RunRegistry", _FakeRegistry)
    monkeypatch.setattr("pace.commands.submit.create_all_snapshots", lambda **kwargs: {})

    run_submit(
        run_name="run1",
        config_path=None,
        force_remote=True,
        force_local=False,
        dry_run=True,
        no_submit=False,
        user_args=["cfg", "LIBERO"],
    )

    assert calls["sync"] == 0
    assert calls["run"] == 0


class _FakeAttempt:
    def __init__(self, attempt_id: int = 1):
        self.attempt_id = attempt_id
        self.status = SimpleNamespace(SUBMITTED="submitted")


class _FakeManifest:
    def __init__(self):
        self.config_path = ""
        self.wandb = None
        self.project = "proj"
        self.run_name = "run1"


class _FakeRegistry:
    def __init__(self, *_args, **_kwargs):
        self._base = Path("/tmp/pace_registry_test/proj/run1")

    @classmethod
    def from_config(cls, _config):
        return cls()

    def run_exists(self, _project, _run_name):
        return False

    def create_run(self, _config, _run_name):
        return _FakeManifest()

    def save_manifest(self, _manifest):
        return None

    def snapshots_dir(self, _project, _run_name):
        return self._base / "snapshots"

    def add_snapshot(self, _project, _run_name, _manifest):
        return None

    def create_attempt(self, _project, _run_name, resume_from=None, user_args=None):
        return _FakeAttempt(1)

    def attempt_dir(self, _project, _run_name, _attempt_id):
        return self._base / "attempts" / "1"

    def logs_dir(self, _project, _run_name):
        return self._base / "logs"

    def run_dir(self, _project, _run_name):
        return self._base

    def save_attempt(self, *_args, **_kwargs):
        return None


def test_submit_remote_sbatch_path(monkeypatch):
    """Non-dry-run remote mode should sync run dir then call remote sbatch."""
    calls = {"sync": 0, "run_cmds": []}

    monkeypatch.setattr("pace.commands.submit.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.submit.load_config", lambda _p: _make_config("remote"))
    monkeypatch.setattr(
        "pace.commands.submit.local_temp_registry_config_path",
        lambda _cfg, _config: (Path("pace.yaml"), "/tmp/pace_registry_test"),
    )
    monkeypatch.setattr("pace.commands.submit.RunRegistry", _FakeRegistry)
    monkeypatch.setattr("pace.commands.submit.create_all_snapshots", lambda **kwargs: {})

    class _FakeExec:
        def sync_project(self, *args, **kwargs):
            calls["sync"] += 1

        def run(self, cmd, remote_cwd, stream_output=False):
            calls["run_cmds"].append((cmd, remote_cwd, stream_output))
            if cmd[0] == "sbatch":
                return SimpleNamespace(
                    returncode=0,
                    stdout="Submitted batch job 12345\n",
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("pace.commands.submit.make_executor", lambda _c: _FakeExec())

    run_submit(
        run_name="run1",
        config_path=None,
        force_remote=True,
        force_local=False,
        dry_run=False,
        no_submit=False,
        user_args=["cfg", "LIBERO"],
    )

    assert calls["sync"] == 1
    assert any(cmd[0][0] == "sbatch" for cmd in calls["run_cmds"])


def test_submit_local_pc_uses_remote_wrapper_path(monkeypatch):
    """Local-PC mode should render SLURM wrapper path using remote attempt dir."""
    captured: dict[str, str] = {}

    monkeypatch.setattr("pace.commands.submit.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.submit.load_config", lambda _p: _make_config("remote"))
    monkeypatch.setattr(
        "pace.commands.submit.local_temp_registry_config_path",
        lambda _cfg, _config: (Path("pace.yaml"), "/tmp/pace_registry_test"),
    )
    monkeypatch.setattr("pace.commands.submit.RunRegistry", _FakeRegistry)
    monkeypatch.setattr("pace.commands.submit.create_all_snapshots", lambda **kwargs: {})

    class _FakeExec:
        def sync_project(self, *args, **kwargs):
            return None

        def run(self, cmd, remote_cwd, stream_output=False):
            if cmd[0] == "sbatch":
                return SimpleNamespace(
                    returncode=0,
                    stdout="Submitted batch job 12345\n",
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    class _FakeJobScript:
        def render(self):
            return "#!/bin/bash\ntrue\n"

    class _FakeSlurmBackend:
        def create_job_script(self, **kwargs):
            captured["wrapper_path"] = kwargs["wrapper_path"]
            return _FakeJobScript()

        def submit(self, *args, **kwargs):
            return 12345

    monkeypatch.setattr("pace.commands.submit.make_executor", lambda _c: _FakeExec())
    monkeypatch.setattr("pace.commands.submit.SlurmBackend", _FakeSlurmBackend)

    run_submit(
        run_name="run1",
        config_path=None,
        force_remote=True,
        force_local=False,
        dry_run=False,
        no_submit=True,
        user_args=["cfg", "LIBERO"],
    )

    assert captured["wrapper_path"] == "/cluster/pace_runs/proj/run1/attempts/1/compute_wrapper.sh"


def test_status_local_pc_uses_direct_ssh(monkeypatch):
    """Status in local-PC mode should query remote files/commands via SSH."""
    monkeypatch.setattr("pace.commands.status.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.status.load_config", lambda _p: _make_config("remote"))
    monkeypatch.setattr(
        "pace.commands.status.ssh_read_yaml",
        lambda _e, _p: {"project": "proj", "created_at": "2026-03-11", "latest_attempt": 1},
    )
    monkeypatch.setattr(
        "pace.commands.status.ssh_read_json",
        lambda _e, _p: {"latest_checkpoint": "/remote/ckpt"},
    )

    calls = []

    class _FakeExec:
        def run(self, cmd, remote_cwd, stream_output=False):
            calls.append((cmd, remote_cwd, stream_output))
            return SimpleNamespace(returncode=0, stdout="RUNNING\n", stderr="")

    monkeypatch.setattr("pace.commands.status.make_executor", lambda _c: _FakeExec())

    run_status("run1", force_remote=True, force_local=False, config_path=None)
    assert calls  # job state + marker checks executed remotely


def test_logs_follow_local_pc_streams_tail(monkeypatch):
    """Logs follow in local-PC mode should stream remote tail command."""
    monkeypatch.setattr("pace.commands.logs.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.logs.load_config", lambda _p: _make_config("remote"))

    def _fake_yaml(_e, path):
        if path.endswith("/manifest.yaml"):
            return {"latest_attempt": 1}
        return {"slurm_job_id": 123}

    monkeypatch.setattr("pace.commands.logs.ssh_read_yaml", _fake_yaml)
    calls = []

    class _FakeExec:
        def run(self, cmd, remote_cwd, stream_output=False):
            calls.append((cmd, remote_cwd, stream_output))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("pace.commands.logs.make_executor", lambda _c: _FakeExec())

    run_logs(
        "run1",
        force_remote=True,
        force_local=False,
        config_path=None,
        attempt_id=None,
        follow=True,
    )
    assert any("tail -f" in " ".join(cmd) for cmd, _, _ in calls)


def test_list_local_pc_reads_remote_registry(monkeypatch):
    """List in local-PC mode should list remote runs via SSH and read manifests."""
    monkeypatch.setattr("pace.commands.runs.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.runs.load_config", lambda _p: _make_config("remote"))
    monkeypatch.setattr(
        "pace.commands.runs.ssh_read_yaml",
        lambda _e, _p: {"created_at": "2026-03-11", "latest_attempt": 2},
    )

    class _FakeExec:
        def run(self, cmd, remote_cwd, stream_output=False):
            return SimpleNamespace(returncode=0, stdout="run_a\nrun_b\n", stderr="")

    monkeypatch.setattr("pace.commands.runs.make_executor", lambda _c: _FakeExec())
    run_list(config_path=None, force_remote=True, force_local=False)


def test_resume_local_pc_calls_submit_with_override(monkeypatch):
    """Resume in local-PC mode should discover remote checkpoint and call submit."""
    monkeypatch.setattr("pace.commands.resume.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.resume.load_config", lambda _p: _make_config("remote"))
    monkeypatch.setattr(
        "pace.commands.resume.ssh_read_yaml",
        lambda _e, _p: {"project": "proj", "latest_attempt": 1},
    )

    class _FakeExec:
        def run(self, cmd, remote_cwd, stream_output=False):
            return SimpleNamespace(
                returncode=0,
                stdout="/cluster/logs/run1/checkpoints/global_step_10\n",
                stderr="",
            )

    monkeypatch.setattr("pace.commands.resume.make_executor", lambda _c: _FakeExec())
    captured = {}

    def _fake_submit(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("pace.commands.submit.run_submit", _fake_submit)

    run_resume(
        "run1",
        force_remote=True,
        force_local=False,
        config_path=None,
        checkpoint_path=None,
        dry_run=False,
    )

    assert captured["resume_from_override"].endswith("global_step_10")
    assert (
        captured["resume_from_container_override"]
        == "/workspace/RLinf/checkpoints/global_step_10"
    )


def test_submit_resume_reuses_cached_user_args_and_snapshots(monkeypatch):
    """Resume submit should reuse latest attempt args and existing snapshots."""
    monkeypatch.setattr("pace.commands.submit.find_config", lambda: Path("pace.yaml"))
    monkeypatch.setattr("pace.commands.submit.load_config", lambda _p: _make_config("remote"))
    monkeypatch.setattr(
        "pace.commands.submit.local_temp_registry_config_path",
        lambda _cfg, _config: (Path("pace.yaml"), "/tmp/pace_registry_test"),
    )
    monkeypatch.setattr(
        "pace.commands.submit.sync_remote_run_to_local",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "pace.commands.submit.create_all_snapshots",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected snapshot create")),
    )

    class _Manifest:
        def __init__(self):
            self.config_path = ""
            self.wandb = None
            self.snapshots = [SimpleNamespace(name="repo", dest_path="/remote/snap")]

    class _Attempt:
        def __init__(self):
            self.attempt_id = 2
            self.status = SimpleNamespace(SUBMITTED="submitted")
            self.user_args = ["cfg_name", "LIBERO"]

    captured: dict[str, object] = {}

    class _ResumeRegistry(_FakeRegistry):
        def run_exists(self, _project, _run_name):
            return True

        def load_manifest(self, _project, _run_name):
            return _Manifest()

        def get_latest_attempt(self, _project, _run_name):
            return _Attempt()

        def create_attempt(self, _project, _run_name, resume_from=None, user_args=None):
            captured["resume_from"] = resume_from
            captured["user_args"] = user_args
            return _Attempt()

    monkeypatch.setattr("pace.commands.submit.RunRegistry", _ResumeRegistry)

    class _FakeExec:
        def sync_project(self, *args, **kwargs):
            return None

        def run(self, cmd, remote_cwd, stream_output=False):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("pace.commands.submit.make_executor", lambda _c: _FakeExec())

    run_submit(
        run_name="run1",
        config_path=None,
        force_remote=True,
        force_local=False,
        dry_run=False,
        no_submit=True,
        user_args=[],
        resume_from_override="/cluster/logs/run1/checkpoints/global_step_10",
    )

    assert captured["resume_from"] == "/cluster/logs/run1/checkpoints/global_step_10"
    assert captured["user_args"] == ["cfg_name", "LIBERO"]


def test_submit_resume_infers_user_args_from_latest_launch_plan(monkeypatch):
    """Resume submit should infer old args from launch plan when attempt metadata is missing."""
    monkeypatch.setattr("pace.commands.submit.find_config", lambda: Path("pace.yaml"))
    cfg = _make_config("remote")
    cfg.launch.base_command = ["bash", "examples/embodiment/run_embodiment.sh"]
    monkeypatch.setattr("pace.commands.submit.load_config", lambda _p: cfg)
    monkeypatch.setattr(
        "pace.commands.submit.local_temp_registry_config_path",
        lambda _cfg, _config: (Path("pace.yaml"), "/tmp/pace_registry_test"),
    )
    monkeypatch.setattr(
        "pace.commands.submit.sync_remote_run_to_local",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "pace.commands.submit.create_all_snapshots",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected snapshot create")),
    )

    class _Manifest:
        def __init__(self):
            self.config_path = ""
            self.wandb = None
            self.snapshots = [SimpleNamespace(name="repo", dest_path="/remote/snap")]

    class _Attempt:
        def __init__(self):
            self.attempt_id = 2
            self.status = SimpleNamespace(SUBMITTED="submitted")
            self.user_args = []

    class _LaunchPlan:
        command = [
            "bash",
            "examples/embodiment/run_embodiment.sh",
            "libero_spatial_ppo_openpi",
            "LIBERO",
            "run1",
        ]

    captured: dict[str, object] = {}

    class _ResumeRegistry(_FakeRegistry):
        def run_exists(self, _project, _run_name):
            return True

        def load_manifest(self, _project, _run_name):
            return _Manifest()

        def get_latest_attempt(self, _project, _run_name):
            return _Attempt()

        def load_launch_plan(self, _project, _run_name, _attempt_id):
            return _LaunchPlan()

        def create_attempt(self, _project, _run_name, resume_from=None, user_args=None):
            captured["resume_from"] = resume_from
            captured["user_args"] = user_args
            return _Attempt()

    monkeypatch.setattr("pace.commands.submit.RunRegistry", _ResumeRegistry)

    class _FakeExec:
        def sync_project(self, *args, **kwargs):
            return None

        def run(self, cmd, remote_cwd, stream_output=False):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("pace.commands.submit.make_executor", lambda _c: _FakeExec())

    run_submit(
        run_name="run1",
        config_path=None,
        force_remote=True,
        force_local=False,
        dry_run=False,
        no_submit=True,
        user_args=[],
        resume_from_override="/cluster/logs/run1/checkpoints/global_step_10",
    )

    assert captured["resume_from"] == "/cluster/logs/run1/checkpoints/global_step_10"
    assert captured["user_args"] == ["libero_spatial_ppo_openpi", "LIBERO"]
