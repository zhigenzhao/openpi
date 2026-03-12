"""Tests for PACE run registry."""

from pathlib import Path

import pytest

from pace.config import load_config
from pace.core.registry import RunRegistry
from pace.core.manifests import AttemptStatus


MINIMAL_CONFIG = """\
project: test_project

runtime:
  image: /path/to/image.sif

scheduler:
  type: slurm
  log_dir: /cluster/runs/{{run_name}}/scheduler

registry_root: {registry_root}
"""


@pytest.fixture
def config(tmp_path: Path):
    """Create a test config."""
    registry_root = tmp_path / "registry"
    config_file = tmp_path / "pace.yaml"
    config_file.write_text(MINIMAL_CONFIG.format(registry_root=registry_root))
    return load_config(config_file)


@pytest.fixture
def registry(config):
    """Create a test registry."""
    return RunRegistry.from_config(config)


class TestRunRegistry:
    """Tests for RunRegistry."""

    def test_create_run(self, registry, config):
        """Test creating a new run."""
        manifest = registry.create_run(config, "test_run")

        assert manifest.project == "test_project"
        assert manifest.run_name == "test_run"
        assert manifest.latest_attempt == 0

        # Check directories were created
        run_dir = registry.run_dir("test_project", "test_run")
        assert run_dir.exists()
        assert registry.snapshots_dir("test_project", "test_run").exists()
        assert registry.attempts_dir("test_project", "test_run").exists()

    def test_run_exists(self, registry, config):
        """Test run_exists method."""
        assert not registry.run_exists("test_project", "test_run")

        registry.create_run(config, "test_run")

        assert registry.run_exists("test_project", "test_run")

    def test_create_duplicate_run_raises(self, registry, config):
        """Test that creating duplicate run raises error."""
        registry.create_run(config, "test_run")

        with pytest.raises(ValueError, match="Run already exists"):
            registry.create_run(config, "test_run")

    def test_load_manifest(self, registry, config):
        """Test loading run manifest."""
        registry.create_run(config, "test_run")

        manifest = registry.load_manifest("test_project", "test_run")

        assert manifest.run_name == "test_run"

    def test_load_nonexistent_manifest(self, registry):
        """Test loading nonexistent manifest raises error."""
        with pytest.raises(FileNotFoundError):
            registry.load_manifest("test_project", "nonexistent")

    def test_create_attempt(self, registry, config):
        """Test creating attempts."""
        registry.create_run(config, "test_run")

        attempt1 = registry.create_attempt("test_project", "test_run")
        assert attempt1.attempt_id == 1
        assert attempt1.status == AttemptStatus.PENDING
        assert attempt1.user_args == []

        attempt2 = registry.create_attempt("test_project", "test_run")
        assert attempt2.attempt_id == 2

        # Manifest should be updated
        manifest = registry.load_manifest("test_project", "test_run")
        assert manifest.latest_attempt == 2

    def test_create_attempt_with_resume(self, registry, config):
        """Test creating attempt with resume path."""
        registry.create_run(config, "test_run")

        attempt = registry.create_attempt(
            "test_project",
            "test_run",
            resume_from="/path/to/checkpoint",
            user_args=["cfg", "LIBERO"],
        )

        assert attempt.resume_from == "/path/to/checkpoint"
        assert attempt.user_args == ["cfg", "LIBERO"]

    def test_load_attempt(self, registry, config):
        """Test loading attempt manifest."""
        registry.create_run(config, "test_run")
        registry.create_attempt("test_project", "test_run")

        attempt = registry.load_attempt("test_project", "test_run", 1)

        assert attempt.attempt_id == 1

    def test_get_latest_attempt(self, registry, config):
        """Test getting latest attempt."""
        registry.create_run(config, "test_run")

        # No attempts yet
        assert registry.get_latest_attempt("test_project", "test_run") is None

        registry.create_attempt("test_project", "test_run")
        registry.create_attempt("test_project", "test_run")

        latest = registry.get_latest_attempt("test_project", "test_run")
        assert latest is not None
        assert latest.attempt_id == 2

    def test_list_runs(self, registry, config):
        """Test listing runs."""
        assert registry.list_runs("test_project") == []

        registry.create_run(config, "run1")
        registry.create_run(config, "run2")

        runs = registry.list_runs("test_project")
        assert sorted(runs) == ["run1", "run2"]

    def test_list_projects(self, registry, config):
        """Test listing projects."""
        assert registry.list_projects() == []

        registry.create_run(config, "test_run")

        projects = registry.list_projects()
        assert "test_project" in projects

    def test_path_methods(self, registry, config):
        """Test path accessor methods."""
        registry.create_run(config, "test_run")

        run_dir = registry.run_dir("test_project", "test_run")
        assert run_dir.exists()

        assert registry.logs_dir("test_project", "test_run") == run_dir / "logs"
        assert registry.checkpoints_dir("test_project", "test_run") == run_dir / "checkpoints"
        assert registry.attempt_dir("test_project", "test_run", 1) == run_dir / "attempts" / "1"
