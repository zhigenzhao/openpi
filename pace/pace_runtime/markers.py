"""Marker file utilities for PACE Runtime.

Provides functions for writing and checking completion markers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

# Standard marker file names
CHECKPOINT_OK = "CHECKPOINT_OK"
TRAINING_DONE = "TRAINING_DONE"
CHECKPOINT_MANIFEST = "manifest.json"


def mark_training_done(
    run_dir: str | Path,
    marker_name: str = TRAINING_DONE,
    metadata: dict | None = None,
) -> Path:
    """Mark training as complete.

    Writes a completion marker file to indicate training has finished.

    Args:
        run_dir: Run directory or markers directory.
        marker_name: Name of the marker file.
        metadata: Additional metadata to include.

    Returns:
        Path to the marker file.
    """
    run_dir = Path(run_dir)

    # If run_dir has a markers subdirectory, use it
    markers_dir = run_dir / "markers"
    if markers_dir.exists():
        marker_path = markers_dir / marker_name
    else:
        marker_path = run_dir / marker_name

    # Ensure parent exists
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    # Write marker with metadata
    content = {
        "completed_at": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
    }
    if metadata:
        content.update(metadata)

    marker_path.write_text(json.dumps(content, indent=2))
    return marker_path


def is_training_done(
    run_dir: str | Path,
    marker_name: str = TRAINING_DONE,
) -> bool:
    """Check if training is complete.

    Args:
        run_dir: Run directory or markers directory.
        marker_name: Name of the marker file to check.

    Returns:
        True if the completion marker exists.
    """
    run_dir = Path(run_dir)

    # Check both locations
    markers_dir = run_dir / "markers"
    if (markers_dir / marker_name).exists():
        return True

    if (run_dir / marker_name).exists():
        return True

    return False


def read_training_done(
    run_dir: str | Path,
    marker_name: str = TRAINING_DONE,
) -> dict | None:
    """Read training completion marker metadata.

    Args:
        run_dir: Run directory or markers directory.
        marker_name: Name of the marker file.

    Returns:
        Marker metadata dict, or None if not found.
    """
    run_dir = Path(run_dir)

    # Check both locations
    for marker_path in [
        run_dir / "markers" / marker_name,
        run_dir / marker_name,
    ]:
        if marker_path.exists():
            try:
                return json.loads(marker_path.read_text())
            except json.JSONDecodeError:
                return {"raw": marker_path.read_text()}

    return None


def mark_checkpoint_ok(
    checkpoint_dir: str | Path,
) -> Path:
    """Mark a checkpoint as valid.

    Args:
        checkpoint_dir: Checkpoint directory.

    Returns:
        Path to the marker file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    marker_path = checkpoint_dir / CHECKPOINT_OK

    content = {
        "timestamp": datetime.now().isoformat(),
    }
    marker_path.write_text(json.dumps(content, indent=2))

    return marker_path


def is_checkpoint_ok(checkpoint_dir: str | Path) -> bool:
    """Check if a checkpoint is marked as valid.

    Args:
        checkpoint_dir: Checkpoint directory.

    Returns:
        True if CHECKPOINT_OK marker exists.
    """
    return (Path(checkpoint_dir) / CHECKPOINT_OK).exists()
