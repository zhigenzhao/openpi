"""Tests for resume discovery/mapping utilities."""

from pathlib import Path

import pytest

from pace.config.models import (
    PersistentOutputConfig,
    ProjectConfig,
    ResumeConfig,
    ResumeStrategy,
    RuntimeConfig,
    SchedulerConfig,
)
from pace.core.resume import (
    ResumeOutputPaths,
    discover_latest_checkpoint_local,
    map_checkpoint_host_to_container,
    resolve_resume_output_paths,
)


def _make_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        project="proj",
        runtime=RuntimeConfig(image="/tmp/image.sif"),
        scheduler=SchedulerConfig(type="slurm", log_dir="/tmp/{run_name}/sched"),
        persistent_outputs=[
            PersistentOutputConfig(
                name="logs",
                role="logs",
                host_path=f"{tmp_path}/logs",
                container_path="/workspace/RLinf/logs",
            ),
            PersistentOutputConfig(
                name="checkpoints",
                role="checkpoints",
                host_path=f"{tmp_path}/checkpoints",
                container_path="/workspace/RLinf/checkpoints",
            ),
        ],
        resume=ResumeConfig(
            checkpoint_output="logs",
            strategy=ResumeStrategy.LATEST_SAFE,
            search_recursive=True,
            checkpoint_marker="CHECKPOINT_SUCCESS",
            sort_pattern=r"global_step_(?P<key>\d+)",
        ),
    )


def test_resolve_resume_output_paths_prefers_output_name(tmp_path: Path):
    """Resolve output roots using resume.checkpoint_output by name."""
    config = _make_config(tmp_path)
    output_paths = resolve_resume_output_paths(config, "run1")
    assert output_paths is not None
    assert output_paths.output.name == "logs"
    assert output_paths.host_root.endswith("/logs")
    assert output_paths.container_root == "/workspace/RLinf/logs"


def test_discover_latest_checkpoint_recursive_latest_safe(tmp_path: Path):
    """latest_safe should pick latest marker-backed checkpoint directory."""
    root = tmp_path / "logs" / "run1"
    ckpt_10 = root / "20260101" / "run1" / "checkpoints" / "global_step_10"
    ckpt_20 = root / "20260102" / "run1" / "checkpoints" / "global_step_20"
    ckpt_10.mkdir(parents=True)
    ckpt_20.mkdir(parents=True)
    (ckpt_10 / "CHECKPOINT_SUCCESS").write_text("ok")
    (ckpt_20 / "CHECKPOINT_SUCCESS").write_text("ok")

    latest = discover_latest_checkpoint_local(
        root_dir=root,
        strategy=ResumeStrategy.LATEST_SAFE,
        search_recursive=True,
        checkpoint_marker="CHECKPOINT_SUCCESS",
        sort_pattern=r"global_step_(?P<key>\d+)",
    )
    assert latest == ckpt_20


def test_discover_latest_checkpoint_latest_allows_missing_marker(tmp_path: Path):
    """latest should allow marker-less checkpoint dirs."""
    root = tmp_path / "logs" / "run1"
    ckpt_10 = root / "checkpoints" / "global_step_10"
    ckpt_20 = root / "checkpoints" / "global_step_20"
    ckpt_10.mkdir(parents=True)
    ckpt_20.mkdir(parents=True)
    (ckpt_10 / "CHECKPOINT_SUCCESS").write_text("ok")

    latest = discover_latest_checkpoint_local(
        root_dir=root,
        strategy=ResumeStrategy.LATEST,
        search_recursive=True,
        checkpoint_marker="CHECKPOINT_SUCCESS",
        sort_pattern=r"global_step_(?P<key>\d+)",
    )
    assert latest == ckpt_20


def test_discover_latest_checkpoint_none_strategy_returns_none(tmp_path: Path):
    """none strategy should not auto-discover checkpoints."""
    root = tmp_path / "logs" / "run1"
    ckpt_10 = root / "checkpoints" / "global_step_10"
    ckpt_10.mkdir(parents=True)
    (ckpt_10 / "CHECKPOINT_SUCCESS").write_text("ok")

    latest = discover_latest_checkpoint_local(
        root_dir=root,
        strategy=ResumeStrategy.NONE,
        search_recursive=True,
        checkpoint_marker="CHECKPOINT_SUCCESS",
        sort_pattern=r"global_step_(?P<key>\d+)",
    )
    assert latest is None


def test_map_checkpoint_host_to_container(tmp_path: Path):
    """Host checkpoint paths map to container checkpoint paths."""
    output_paths = ResumeOutputPaths(
        output=PersistentOutputConfig(
            name="logs",
            role="logs",
            host_path="/cluster/logs",
            container_path="/workspace/logs",
        ),
        host_root="/cluster/logs",
        container_root="/workspace/logs",
    )
    mapped = map_checkpoint_host_to_container(
        "/cluster/logs/run1/t1/run1/checkpoints/global_step_10",
        output_paths,
    )
    assert mapped == "/workspace/logs/run1/t1/run1/checkpoints/global_step_10"

    with pytest.raises(ValueError, match="outside configured resume output root"):
        map_checkpoint_host_to_container("/other/path/global_step_1", output_paths)
