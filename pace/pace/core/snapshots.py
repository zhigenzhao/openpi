"""Snapshot engine for PACE.

Handles creating snapshots of source directories for reproducible runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pace.config.models import SnapshotConfig
from pace.core.manifests import SnapshotManifest


class SnapshotError(Exception):
    """Error during snapshot creation."""

    pass


@dataclass
class SnapshotResult:
    """Result of a snapshot operation.

    Attributes:
        manifest: The snapshot manifest.
        dest_path: Path where snapshot was created.
    """

    manifest: SnapshotManifest
    dest_path: Path


def generate_snapshot_id() -> str:
    """Generate UTC snapshot ID using timestamp format YYYYMMDD_HHMMSS."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_snapshot_path(
    target_root: Path,
    snapshot_name: str,
    snapshot_id: str,
    avoid_collision: bool = True,
) -> Path:
    """Resolve snapshot path, optionally adding suffix for same-second collisions."""
    base = target_root / f"{snapshot_name}_{snapshot_id}"
    if not avoid_collision or not base.exists():
        return base

    suffix = 1
    while True:
        candidate = target_root / f"{snapshot_name}_{snapshot_id}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def create_snapshot(
    config: SnapshotConfig,
    snapshot_id: str | None = None,
    target_root_override: Path | None = None,
    use_rsync: bool = True,
    dry_run: bool = False,
) -> SnapshotResult:
    """Create a snapshot of a source directory.

    Args:
        config: Snapshot configuration.
        snapshot_id: Optional snapshot ID (generated if not provided).
        target_root_override: Optional target directory override.
        use_rsync: Whether to use rsync (faster) or shutil (portable).
        dry_run: If True, don't actually create snapshot.

    Returns:
        SnapshotResult with manifest and path.

    Raises:
        SnapshotError: If snapshot creation fails.
    """
    if snapshot_id is None:
        snapshot_id = generate_snapshot_id()

    source_path = Path(config.local_dir).resolve()
    if not source_path.exists():
        raise SnapshotError(f"Source path does not exist: {source_path}")

    target_root = (
        Path(target_root_override).resolve()
        if target_root_override
        else Path(config.target_dir).resolve()
    )
    dest_path = resolve_snapshot_path(
        target_root=target_root,
        snapshot_name=config.name,
        snapshot_id=snapshot_id,
        avoid_collision=not dry_run,
    )

    if dry_run:
        # Return manifest without creating files
        return SnapshotResult(
            manifest=SnapshotManifest(
                name=config.name,
                snapshot_id=snapshot_id,
                source_path=str(source_path),
                dest_path=str(dest_path),
                created_at=datetime.now().isoformat(),
                file_count=0,
                total_size_bytes=0,
            ),
            dest_path=dest_path,
        )

    target_root.mkdir(parents=True, exist_ok=True)

    if use_rsync and shutil.which("rsync"):
        _create_snapshot_rsync(source_path, dest_path, config.exclude)
    else:
        _create_snapshot_shutil(source_path, dest_path, config.exclude)

    # Count files and size
    file_count, total_size = _count_files_and_size(dest_path)

    manifest = SnapshotManifest(
        name=config.name,
        snapshot_id=snapshot_id,
        source_path=str(source_path),
        dest_path=str(dest_path),
        created_at=datetime.now().isoformat(),
        file_count=file_count,
        total_size_bytes=total_size,
    )

    return SnapshotResult(manifest=manifest, dest_path=dest_path)


def _create_snapshot_rsync(
    source: Path,
    dest: Path,
    excludes: list[str],
) -> None:
    """Create snapshot using rsync.

    Args:
        source: Source directory.
        dest: Destination directory.
        excludes: Patterns to exclude.
    """
    cmd = [
        "rsync",
        "-a",  # Archive mode
        "--delete",  # Delete extraneous files
    ]

    for exclude in excludes:
        cmd.extend(["--exclude", exclude])

    # Ensure trailing slash on source for rsync semantics
    source_str = str(source)
    if not source_str.endswith("/"):
        source_str += "/"

    cmd.extend([source_str, str(dest)])

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise SnapshotError(f"rsync failed: {e.stderr}")


def _create_snapshot_shutil(
    source: Path,
    dest: Path,
    excludes: list[str],
) -> None:
    """Create snapshot using shutil.

    Args:
        source: Source directory.
        dest: Destination directory.
        excludes: Patterns to exclude.
    """
    import fnmatch

    def should_exclude(path: Path, name: str) -> bool:
        """Check if a file/directory should be excluded."""
        rel_path = str(path / name)
        for pattern in excludes:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    def ignore_patterns(directory: str, names: list[str]) -> set[str]:
        """Return names to ignore in this directory."""
        dir_path = Path(directory)
        return {name for name in names if should_exclude(dir_path, name)}

    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(
        source,
        dest,
        ignore=ignore_patterns,
        symlinks=True,
    )


def _count_files_and_size(path: Path) -> tuple[int, int]:
    """Count files and total size in a directory.

    Args:
        path: Directory path.

    Returns:
        Tuple of (file_count, total_size_bytes).
    """
    file_count = 0
    total_size = 0

    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            file_count += 1
            try:
                total_size += file_path.stat().st_size
            except OSError:
                pass  # Skip files we can't stat

    return file_count, total_size


def create_all_snapshots(
    configs: list[SnapshotConfig],
    snapshot_id: str | None = None,
    target_root_overrides: dict[str, Path] | None = None,
    use_rsync: bool = True,
    dry_run: bool = False,
) -> dict[str, SnapshotResult]:
    """Create snapshots for all configured sources.

    Args:
        configs: List of snapshot configurations.
        snapshot_id: Optional shared snapshot ID.
        target_root_overrides: Optional map from snapshot name to target root override.
        use_rsync: Whether to use rsync.
        dry_run: If True, don't actually create snapshots.

    Returns:
        Dictionary mapping snapshot name to result.
    """
    if snapshot_id is None:
        snapshot_id = generate_snapshot_id()

    results = {}
    target_root_overrides = target_root_overrides or {}
    for config in configs:
        result = create_snapshot(
            config=config,
            snapshot_id=snapshot_id,
            target_root_override=target_root_overrides.get(config.name),
            use_rsync=use_rsync,
            dry_run=dry_run,
        )
        results[config.name] = result

    return results
