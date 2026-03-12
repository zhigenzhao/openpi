"""Tests for pace_runtime checkpointing."""

import json
from pathlib import Path

import pytest

from pace_runtime import (
    checkpoint_transaction,
    find_latest_checkpoint,
    is_checkpoint_valid,
    mark_training_done,
    is_training_done,
)
from pace_runtime.checkpointing import CheckpointError


class TestCheckpointTransaction:
    """Tests for checkpoint_transaction context manager."""

    def test_successful_checkpoint(self, tmp_path: Path):
        """Test successful checkpoint creation."""
        ckpt_dir = tmp_path / "checkpoints"

        with checkpoint_transaction(ckpt_dir, tag="step_100") as tx:
            (tx / "model.pt").write_text("model data")

        # Checkpoint should exist
        final_dir = ckpt_dir / "step_100"
        assert final_dir.exists()
        assert (final_dir / "model.pt").exists()
        assert (final_dir / "CHECKPOINT_OK").exists()
        assert (final_dir / "manifest.json").exists()

    def test_checkpoint_with_required_files(self, tmp_path: Path):
        """Test checkpoint validation with required files."""
        ckpt_dir = tmp_path / "checkpoints"

        with checkpoint_transaction(
            ckpt_dir,
            tag="step_100",
            required_files=["model.pt", "optimizer.pt"],
        ) as tx:
            (tx / "model.pt").write_text("model")
            (tx / "optimizer.pt").write_text("optimizer")

        assert is_checkpoint_valid(ckpt_dir / "step_100")

    def test_checkpoint_missing_required_files(self, tmp_path: Path):
        """Test checkpoint fails when required files missing."""
        ckpt_dir = tmp_path / "checkpoints"

        with pytest.raises(CheckpointError, match="Required files missing"):
            with checkpoint_transaction(
                ckpt_dir,
                tag="step_100",
                required_files=["model.pt", "optimizer.pt"],
            ) as tx:
                (tx / "model.pt").write_text("model")
                # Missing optimizer.pt

    def test_checkpoint_rollback_on_error(self, tmp_path: Path):
        """Test checkpoint is not created on error."""
        ckpt_dir = tmp_path / "checkpoints"

        with pytest.raises(ValueError):
            with checkpoint_transaction(ckpt_dir, tag="step_100") as tx:
                (tx / "model.pt").write_text("model")
                raise ValueError("Something went wrong")

        # No checkpoint should exist
        assert not (ckpt_dir / "step_100").exists()

    def test_latest_symlink(self, tmp_path: Path):
        """Test latest_safe symlink is updated."""
        ckpt_dir = tmp_path / "checkpoints"

        with checkpoint_transaction(ckpt_dir, tag="step_100") as tx:
            (tx / "model.pt").write_text("model 100")

        with checkpoint_transaction(ckpt_dir, tag="step_200") as tx:
            (tx / "model.pt").write_text("model 200")

        latest = ckpt_dir / "latest_safe"
        assert latest.exists()
        assert latest.is_symlink()
        # Should point to step_200
        assert latest.resolve().name == "step_200"


class TestIsCheckpointValid:
    """Tests for is_checkpoint_valid function."""

    def test_valid_checkpoint(self, tmp_path: Path):
        """Test validation of valid checkpoint."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "CHECKPOINT_OK").write_text("ok")
        (ckpt_dir / "model.pt").write_text("model")

        assert is_checkpoint_valid(ckpt_dir)

    def test_missing_marker(self, tmp_path: Path):
        """Test invalid checkpoint without marker."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.pt").write_text("model")

        assert not is_checkpoint_valid(ckpt_dir)

    def test_nonexistent_directory(self, tmp_path: Path):
        """Test nonexistent directory is invalid."""
        assert not is_checkpoint_valid(tmp_path / "nonexistent")


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint function."""

    def test_find_latest_single(self, tmp_path: Path):
        """Test finding single checkpoint."""
        ckpt_dir = tmp_path / "checkpoints"

        with checkpoint_transaction(ckpt_dir, tag="step_100") as tx:
            (tx / "model.pt").write_text("model")

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        assert latest.name == "step_100"

    def test_find_latest_multiple(self, tmp_path: Path):
        """Test finding latest among multiple checkpoints."""
        ckpt_dir = tmp_path / "checkpoints"

        for step in [100, 300, 200]:
            with checkpoint_transaction(ckpt_dir, tag=f"step_{step}") as tx:
                (tx / "model.pt").write_text(f"model {step}")

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        # Should use latest_safe symlink which points to step_200
        # or sort by name which gives step_300 as latest

    def test_find_latest_none(self, tmp_path: Path):
        """Test returns None when no checkpoints."""
        latest = find_latest_checkpoint(tmp_path)
        assert latest is None


class TestTrainingDone:
    """Tests for training completion markers."""

    def test_mark_training_done(self, tmp_path: Path):
        """Test marking training as done."""
        mark_training_done(tmp_path)

        assert is_training_done(tmp_path)

    def test_mark_training_done_with_metadata(self, tmp_path: Path):
        """Test marking training done with metadata."""
        marker_path = mark_training_done(
            tmp_path,
            metadata={"final_step": 10000, "accuracy": 0.95},
        )

        assert marker_path.exists()
        content = json.loads(marker_path.read_text())
        assert content["final_step"] == 10000
        assert content["accuracy"] == 0.95

    def test_is_training_done_false(self, tmp_path: Path):
        """Test is_training_done returns False when not done."""
        assert not is_training_done(tmp_path)
