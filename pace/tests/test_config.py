"""Tests for PACE configuration loading."""

from pathlib import Path

import pytest

from pace.config import load_config
from pace.config.loader import ConfigError
from pace.config.models import (
    InjectionCondition,
    LaunchPhase,
    StageMode,
)

MINIMAL_CONFIG = """\
project: test_project

runtime:
  image: /path/to/image.sif

scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler
"""

FULL_CONFIG = """\
project: test_project

runtime:
  image: /path/to/image.sif
  engine: apptainer

resources:
  gpus: 4
  gpu_type: H100
  cpus: 32
  mem_gb: 500
  time: "2:00:00"
  nodes: 1

scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler

snapshots:
  - name: repo
    local_dir: .
    target_dir: /cluster/runs/{run_name}/snapshots
    exclude:
      - .git
      - __pycache__

shared_inputs:
  - name: datasets
    host_path: /data/datasets
    stage_mode: bind

  - name: models
    host_path: /data/models
    stage_mode: copy_to_tmp

persistent_outputs:
  - name: logs
    role: logs
    host_path: /output/{run_name}/logs
    container_path: /workspace/logs
    create_if_missing: true

binds:
  - host: "{snapshot.repo}"
    container: /workspace/repo
    mode: rw

launch:
  workdir: /workspace/repo
  shell_init:
    - source activate env
  base_command:
    - python
    - train.py
  user_args:
    phase: post_base

injections:
  env:
    - name: core
      when: always
      values:
        RUN_NAME: "{run_name}"

    - name: resume
      when: resume_only
      values:
        RESUME: "true"

  cli:
    - name: log-dir
      when: always
      phase: post_base
      args:
        - "--log-dir={container.logs}"

resume:
  enabled: true
  checkpoint_output: checkpoints
  strategy: latest_safe
  search_recursive: true
  checkpoint_marker: CHECKPOINT_SUCCESS
  sort_pattern: "global_step_(?P<key>\\\\d+)"
  cli_args:
    - "runner.resume_dir={resume_path_container}"

completion:
  marker_file: DONE

registry_root: /cluster/runs
"""

REMOTE_CONFIG = """\
project: test_project

runtime:
  image: /path/to/image.sif

scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler

remote:
  enabled: true
  host: pace-ice
  user: alice
  project_dir: /storage/project/RLinf
  mode: auto
  ssh_options:
    - -o
    - StrictHostKeyChecking=no
  rsync_options:
    - --info=progress2
"""


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_minimal_config(self, tmp_path: Path):
        """Test loading minimal valid config."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(MINIMAL_CONFIG)

        config = load_config(config_file)

        assert config.project == "test_project"
        assert config.runtime.image == "/path/to/image.sif"
        assert config.runtime.engine == "apptainer"

    def test_load_full_config(self, tmp_path: Path):
        """Test loading full config with all sections."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(FULL_CONFIG)

        config = load_config(config_file)

        # Project
        assert config.project == "test_project"

        # Resources
        assert config.resources.gpus == 4
        assert config.resources.gpu_type == "H100"
        assert config.resources.cpus == 32

        # Scheduler
        assert config.scheduler.type == "slurm"
        assert config.scheduler.log_dir == "/cluster/runs/{run_name}/scheduler"

        # Snapshots
        assert len(config.snapshots) == 1
        assert config.snapshots[0].name == "repo"
        assert config.snapshots[0].local_dir == "."
        assert config.snapshots[0].target_dir == "/cluster/runs/{run_name}/snapshots"
        assert ".git" in config.snapshots[0].exclude

        # Shared inputs
        assert len(config.shared_inputs) == 2
        assert config.shared_inputs[0].stage_mode == StageMode.BIND
        assert config.shared_inputs[1].stage_mode == StageMode.COPY_TO_TMP

        # Persistent outputs
        assert len(config.persistent_outputs) == 1
        assert config.persistent_outputs[0].role == "logs"

        # Binds
        assert len(config.binds) == 1
        assert config.binds[0].mode == "rw"

        # Launch
        assert config.launch.workdir == "/workspace/repo"
        assert len(config.launch.shell_init) == 1
        assert config.launch.user_args.phase == LaunchPhase.POST_BASE

        # Injections
        assert len(config.injections.env) == 2
        assert config.injections.env[0].when == InjectionCondition.ALWAYS
        assert config.injections.env[1].when == InjectionCondition.RESUME_ONLY

        assert len(config.injections.cli) == 1
        assert config.injections.cli[0].phase == LaunchPhase.POST_BASE

        # Resume
        assert config.resume.enabled is True
        assert config.resume.checkpoint_output == "checkpoints"
        assert config.resume.search_recursive is True
        assert config.resume.checkpoint_marker == "CHECKPOINT_SUCCESS"
        assert config.resume.sort_pattern == "global_step_(?P<key>\\d+)"
        assert config.resume.cli_args == ["runner.resume_dir={resume_path_container}"]

        # Completion
        assert config.completion.marker_file == "DONE"

        # Registry
        assert config.registry_root == "/cluster/runs"

    def test_missing_project_raises(self, tmp_path: Path):
        """Test that missing project field raises error."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text("runtime:\n  image: /path.sif\n")

        with pytest.raises(ConfigError, match="project field is required"):
            load_config(config_file)

    def test_missing_runtime_raises(self, tmp_path: Path):
        """Test that missing runtime section raises error."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text("project: test\n")

        with pytest.raises(ConfigError, match="runtime section is required"):
            load_config(config_file)

    def test_missing_scheduler_raises(self, tmp_path: Path):
        """Test that missing scheduler section raises error."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text("project: test\nruntime:\n  image: /path.sif\n")

        with pytest.raises(ConfigError, match="scheduler section is required"):
            load_config(config_file)

    def test_load_remote_config(self, tmp_path: Path):
        """Test parsing remote orchestration config."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(REMOTE_CONFIG)

        config = load_config(config_file)

        assert config.remote.enabled is True
        assert config.remote.host == "pace-ice"
        assert config.remote.user == "alice"
        assert config.remote.project_dir == "/storage/project/RLinf"
        assert config.remote.mode == "auto"
        assert config.remote.ssh_options == ["-o", "StrictHostKeyChecking=no"]
        assert config.remote.rsync_options == ["--info=progress2"]

    def test_remote_enabled_requires_host(self, tmp_path: Path):
        """Test remote.enabled requires remote.host."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(
            """\
project: test
runtime:
  image: /tmp/image.sif
scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler
remote:
  enabled: true
  project_dir: /tmp/project
"""
        )

        with pytest.raises(ConfigError, match="remote.host is required"):
            load_config(config_file)

    def test_remote_enabled_requires_project_dir(self, tmp_path: Path):
        """Test remote.enabled requires remote.project_dir."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(
            """\
project: test
runtime:
  image: /tmp/image.sif
scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler
remote:
  enabled: true
  host: pace-ice
"""
        )

        with pytest.raises(ConfigError, match="remote.project_dir is required"):
            load_config(config_file)

    def test_remote_mode_validation(self, tmp_path: Path):
        """Test remote.mode validation."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(
            """\
project: test
runtime:
  image: /tmp/image.sif
scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler
remote:
  enabled: true
  host: pace-ice
  project_dir: /tmp/project
  mode: invalid
"""
        )

        with pytest.raises(ConfigError, match="remote.mode must be one of"):
            load_config(config_file)

    def test_resume_sort_pattern_requires_key_group(self, tmp_path: Path):
        """resume.sort_pattern must define named group 'key'."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(
            """\
project: test
runtime:
  image: /tmp/image.sif
scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler
resume:
  sort_pattern: "global_step_(\\d+)"
"""
        )

        with pytest.raises(ConfigError, match="named group 'key'"):
            load_config(config_file)

    def test_resume_cli_args_must_be_string_list(self, tmp_path: Path):
        """resume.cli_args must be a list of strings."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(
            """\
project: test
runtime:
  image: /tmp/image.sif
scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler
resume:
  cli_args:
    - ok
    - 1
"""
        )

        with pytest.raises(ConfigError, match="list of strings"):
            load_config(config_file)

    def test_file_not_found(self, tmp_path: Path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestProjectConfig:
    """Tests for ProjectConfig helper methods."""

    def test_get_snapshot(self, tmp_path: Path):
        """Test get_snapshot method."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(FULL_CONFIG)
        config = load_config(config_file)

        snap = config.get_snapshot("repo")
        assert snap is not None
        assert snap.name == "repo"

        assert config.get_snapshot("nonexistent") is None

    def test_get_shared_input(self, tmp_path: Path):
        """Test get_shared_input method."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(FULL_CONFIG)
        config = load_config(config_file)

        inp = config.get_shared_input("datasets")
        assert inp is not None
        assert inp.host_path == "/data/datasets"

        assert config.get_shared_input("nonexistent") is None

    def test_get_persistent_output(self, tmp_path: Path):
        """Test get_persistent_output methods."""
        config_file = tmp_path / "pace.yaml"
        config_file.write_text(FULL_CONFIG)
        config = load_config(config_file)

        out = config.get_persistent_output("logs")
        assert out is not None
        assert out.role == "logs"

        out_by_role = config.get_persistent_output_by_role("logs")
        assert out_by_role is not None
        assert out_by_role.name == "logs"
