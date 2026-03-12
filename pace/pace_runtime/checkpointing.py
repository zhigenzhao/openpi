"""Checkpoint transaction wrapper for PACE Runtime.

Provides atomic checkpoint saving with validation markers.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pace_runtime.markers import CHECKPOINT_OK


@dataclass
class CheckpointManifest:
    """Manifest for a saved checkpoint.

    Attributes:
        tag: Checkpoint identifier (e.g., "global_step_1000").
        created_at: When the checkpoint was created.
        files: List of files in the checkpoint.
        required_files: Files that must exist for checkpoint to be valid.
        metadata: Additional user metadata.
    """

    tag: str
    created_at: str
    files: list[str] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> CheckpointManifest:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> CheckpointManifest:
        """Load manifest from file."""
        return cls.from_json(path.read_text())


@contextmanager
def checkpoint_transaction(
    checkpoint_root: str | Path,
    tag: str,
    required_files: list[str] | None = None,
    metadata: dict | None = None,
    update_latest: bool = True,
) -> Iterator[Path]:
    """Context manager for atomic checkpoint saving.

    Creates a temporary directory for writing checkpoint files.
    On successful exit, validates the checkpoint and atomically
    moves it to the final location with a completion marker.

    Args:
        checkpoint_root: Directory where checkpoints are stored.
        tag: Identifier for this checkpoint (e.g., "global_step_1000").
        required_files: List of files that must exist for checkpoint to be valid.
        metadata: Additional metadata to store in manifest.
        update_latest: Whether to update the latest_safe symlink.

    Yields:
        Path to temporary directory where files should be saved.

    Raises:
        CheckpointError: If validation fails.

    Example:
        >>> with checkpoint_transaction(ckpt_dir, "step_1000") as tx:
        ...     torch.save(model.state_dict(), f"{tx}/model.pt")
        >>> # Checkpoint is now at ckpt_dir/step_1000/
    """
    checkpoint_root = Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Create temporary directory
    tmp_name = f".tmp_{tag}_{uuid.uuid4().hex[:8]}"
    tmp_dir = checkpoint_root / tmp_name

    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Yield to let user code write files
        yield tmp_dir

        # Validate required files
        if required_files:
            missing = []
            for req_file in required_files:
                if not (tmp_dir / req_file).exists():
                    missing.append(req_file)
            if missing:
                raise CheckpointError(
                    f"Required files missing: {', '.join(missing)}"
                )

        # List all files
        files = []
        for root, _, filenames in os.walk(tmp_dir):
            for name in filenames:
                file_path = Path(root) / name
                rel_path = file_path.relative_to(tmp_dir)
                files.append(str(rel_path))

        # Write manifest
        manifest = CheckpointManifest(
            tag=tag,
            created_at=datetime.now().isoformat(),
            files=files,
            required_files=required_files or [],
            metadata=metadata or {},
        )
        manifest.save(tmp_dir / "manifest.json")

        # Write completion marker
        (tmp_dir / CHECKPOINT_OK).write_text(datetime.now().isoformat())

        # Atomic rename to final location
        final_dir = checkpoint_root / tag
        if final_dir.exists():
            # Remove old checkpoint
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)

        # Update latest symlink
        if update_latest:
            latest_link = checkpoint_root / "latest_safe"
            latest_tmp = checkpoint_root / f".latest_safe_tmp_{uuid.uuid4().hex[:8]}"
            try:
                latest_tmp.symlink_to(tag)
                latest_tmp.rename(latest_link)
            except OSError:
                # Symlink operations may fail on some filesystems
                pass

    except Exception:
        # Clean up temp directory on failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


class CheckpointError(Exception):
    """Error during checkpoint operation."""

    pass


def is_checkpoint_valid(checkpoint_dir: str | Path) -> bool:
    """Check if a checkpoint directory is valid.

    A checkpoint is valid if:
    - It has a CHECKPOINT_OK marker
    - If it has a manifest, all required files exist

    Args:
        checkpoint_dir: Path to checkpoint directory.

    Returns:
        True if checkpoint is valid.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return False

    # Must have completion marker
    if not (checkpoint_dir / CHECKPOINT_OK).exists():
        return False

    # If manifest exists, validate required files
    manifest_path = checkpoint_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = CheckpointManifest.load(manifest_path)
            for req_file in manifest.required_files:
                if not (checkpoint_dir / req_file).exists():
                    return False
        except (json.JSONDecodeError, KeyError):
            return False

    return True


def find_latest_checkpoint(checkpoint_root: str | Path) -> Path | None:
    """Find the latest valid checkpoint.

    First checks the latest_safe symlink, then falls back to
    scanning all directories.

    Args:
        checkpoint_root: Directory containing checkpoints.

    Returns:
        Path to latest valid checkpoint, or None if none found.
    """
    checkpoint_root = Path(checkpoint_root)

    if not checkpoint_root.exists():
        return None

    # Check latest symlink first
    latest_link = checkpoint_root / "latest_safe"
    if latest_link.exists() and latest_link.is_symlink():
        target = latest_link.resolve()
        if is_checkpoint_valid(target):
            return target

    # Fall back to scanning
    valid_checkpoints = []
    for entry in checkpoint_root.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            if is_checkpoint_valid(entry):
                valid_checkpoints.append(entry)

    if not valid_checkpoints:
        return None

    # Sort by name (assumes step-based naming)
    # TODO: Parse step numbers for proper numeric sorting
    valid_checkpoints.sort(key=lambda p: p.name, reverse=True)
    return valid_checkpoints[0]


def list_checkpoints(checkpoint_root: str | Path) -> list[Path]:
    """List all valid checkpoints.

    Args:
        checkpoint_root: Directory containing checkpoints.

    Returns:
        List of valid checkpoint paths, sorted by name.
    """
    checkpoint_root = Path(checkpoint_root)

    if not checkpoint_root.exists():
        return []

    valid = []
    for entry in checkpoint_root.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            if is_checkpoint_valid(entry):
                valid.append(entry)

    return sorted(valid, key=lambda p: p.name)
