"""Resume path and checkpoint discovery utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pace.config.models import PersistentOutputConfig, ProjectConfig, ResumeStrategy


@dataclass(frozen=True)
class ResumeOutputPaths:
    """Resolved host/container paths for resume discovery and mapping."""

    output: PersistentOutputConfig
    host_root: str
    container_root: str


def _resolve_basic_template_path(
    config: ProjectConfig,
    template: str,
    run_name: str,
) -> str:
    """Resolve basic path template placeholders."""
    return (
        template.replace("{run_name}", run_name)
        .replace("{project}", config.project)
        .replace("{registry_root}", config.registry_root)
    )


def resolve_resume_output_paths(
    config: ProjectConfig,
    run_name: str,
) -> ResumeOutputPaths | None:
    """Resolve resume output host/container roots from config."""
    output = config.get_persistent_output(config.resume.checkpoint_output)
    if output is None:
        # Backward compatibility: allow role lookup when name lookup fails.
        output = config.get_persistent_output_by_role("checkpoints")
    if output is None:
        return None

    host_root = _resolve_basic_template_path(config, output.host_path, run_name)
    return ResumeOutputPaths(
        output=output,
        host_root=host_root,
        container_root=output.container_path,
    )


def map_checkpoint_host_to_container(
    checkpoint_host: str,
    output_paths: ResumeOutputPaths,
) -> str:
    """Map a host checkpoint directory to its container-visible path."""
    host_root = PurePosixPath(output_paths.host_root)
    checkpoint = PurePosixPath(checkpoint_host)
    try:
        rel = checkpoint.relative_to(host_root)
    except ValueError as exc:
        raise ValueError(
            "Checkpoint path is outside configured resume output root: "
            f"checkpoint={checkpoint_host}, root={output_paths.host_root}"
        ) from exc

    container_root = PurePosixPath(output_paths.container_root)
    mapped = container_root / rel
    return str(mapped)


def _iter_marker_dirs(
    root: Path,
    marker: str,
    recursive: bool,
) -> list[Path]:
    """Collect checkpoint directories that contain the marker file."""
    marker_dirs: list[Path] = []
    if recursive:
        iterator = root.rglob(marker)
    else:
        iterator = (entry / marker for entry in root.iterdir() if entry.is_dir())

    for marker_path in iterator:
        if marker_path.is_file():
            marker_dirs.append(marker_path.parent)
    return marker_dirs


def _iter_candidate_dirs(
    root: Path,
    recursive: bool,
) -> list[Path]:
    """Collect checkpoint candidate directories."""
    dirs: list[Path] = []
    if recursive:
        for entry in root.rglob("*"):
            if entry.is_dir():
                dirs.append(entry)
    else:
        for entry in root.iterdir():
            if entry.is_dir():
                dirs.append(entry)
    return dirs


def _sort_checkpoints(
    candidates: list[Path],
    sort_pattern: str | None,
) -> list[Path]:
    """Sort checkpoint candidates by regex key or lexicographic path."""
    deduped = sorted({c for c in candidates if c.name and not c.name.startswith(".")})
    if not deduped:
        return []

    if sort_pattern is None:
        return sorted(deduped, key=lambda p: p.as_posix())

    pattern = re.compile(sort_pattern)
    sortable: list[tuple[tuple[int, int | str], str, Path]] = []
    for path in deduped:
        match = pattern.search(path.name) or pattern.search(path.as_posix())
        if not match:
            continue
        key_raw = match.group("key")
        try:
            key_value: int | str = int(key_raw)
            key_type = 0
        except ValueError:
            key_value = key_raw
            key_type = 1
        sortable.append(((key_type, key_value), path.as_posix(), path))

    sortable.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in sortable]


def discover_latest_checkpoint_local(
    root_dir: Path,
    *,
    strategy: ResumeStrategy,
    search_recursive: bool,
    checkpoint_marker: str,
    sort_pattern: str | None,
) -> Path | None:
    """Discover the latest checkpoint directory under the configured root."""
    if not root_dir.exists():
        return None

    marker_dirs = _iter_marker_dirs(
        root=root_dir,
        marker=checkpoint_marker,
        recursive=search_recursive,
    )
    if strategy == ResumeStrategy.LATEST_SAFE:
        ranked = _sort_checkpoints(marker_dirs, sort_pattern=sort_pattern)
        return ranked[-1] if ranked else None

    if strategy == ResumeStrategy.LATEST:
        candidates = marker_dirs + _iter_candidate_dirs(
            root=root_dir,
            recursive=search_recursive,
        )
        ranked = _sort_checkpoints(candidates, sort_pattern=sort_pattern)
        return ranked[-1] if ranked else None

    return None
