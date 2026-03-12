"""Tests for PACE snapshot creation and naming."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from pace.config.models import SnapshotConfig
from pace.core.snapshots import (
    create_snapshot,
    generate_snapshot_id,
    resolve_snapshot_path,
)
from pace.core.templating import TemplateContext, TemplateEngine


def test_generate_snapshot_id_timestamp_format() -> None:
    """Snapshot IDs should be UTC timestamps with fixed format."""
    snapshot_id = generate_snapshot_id()
    assert re.fullmatch(r"\d{8}_\d{6}", snapshot_id)


def test_snapshot_ids_are_lexicographically_ordered(monkeypatch) -> None:
    """Later timestamps should sort after earlier timestamps."""

    class _FakeDatetime:
        values = [
            datetime(2026, 3, 11, 14, 15, 30, tzinfo=timezone.utc),
            datetime(2026, 3, 11, 14, 15, 31, tzinfo=timezone.utc),
        ]
        index = 0

        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            value = cls.values[cls.index]
            cls.index += 1
            return value

    monkeypatch.setattr("pace.core.snapshots.datetime", _FakeDatetime)
    first = generate_snapshot_id()
    second = generate_snapshot_id()
    assert first < second


def test_same_second_collision_uses_numeric_suffix(tmp_path: Path) -> None:
    """Same-second target collisions should increment suffix deterministically."""
    target_root = tmp_path / "snapshots"
    target_root.mkdir()

    first = resolve_snapshot_path(target_root, "repo", "20260311_141530")
    first.mkdir()
    second = resolve_snapshot_path(target_root, "repo", "20260311_141530")
    second.mkdir()
    third = resolve_snapshot_path(target_root, "repo", "20260311_141530")

    assert first.name == "repo_20260311_141530"
    assert second.name == "repo_20260311_141530_1"
    assert third.name == "repo_20260311_141530_2"


def test_create_snapshot_uses_target_dir_path(tmp_path: Path) -> None:
    """Snapshot output should be materialized under target_dir/name_timestamp."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "train.py").write_text("print('ok')\n")
    target = tmp_path / "target"

    cfg = SnapshotConfig(
        name="repo",
        local_dir=str(source),
        target_dir=str(target),
    )
    result = create_snapshot(cfg, snapshot_id="20260311_141530", use_rsync=False)

    assert result.dest_path == target / "repo_20260311_141530"
    assert (result.dest_path / "train.py").exists()


def test_snapshot_bind_placeholder_uses_materialized_target() -> None:
    """{snapshot.<name>} should resolve to the materialized target snapshot path."""
    ctx = TemplateContext(
        snapshots={
            "repo": "/cluster/snapshots/repo_20260311_141530",
        }
    )
    engine = TemplateEngine(ctx)
    assert (
        engine.resolve("{snapshot.repo}")
        == "/cluster/snapshots/repo_20260311_141530"
    )
