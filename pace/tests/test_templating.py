"""Tests for PACE template engine."""

import pytest

from pace.core.templating import TemplateContext, TemplateEngine, TemplateError


class TestTemplateEngine:
    """Tests for TemplateEngine."""

    def test_simple_placeholder(self):
        """Test resolving simple placeholders."""
        ctx = TemplateContext(run_name="my_run", project="test")
        engine = TemplateEngine(ctx)

        result = engine.resolve("Run: {run_name}")
        assert result == "Run: my_run"

    def test_multiple_placeholders(self):
        """Test resolving multiple placeholders."""
        ctx = TemplateContext(run_name="my_run", project="test", attempt_id=3)
        engine = TemplateEngine(ctx)

        result = engine.resolve("{project}/{run_name}/attempt_{attempt_id}")
        assert result == "test/my_run/attempt_3"

    def test_namespaced_placeholder(self):
        """Test resolving namespaced placeholders."""
        ctx = TemplateContext(
            snapshots={"repo": "/path/to/repo"},
            persistent={"logs": "/path/to/logs"},
        )
        engine = TemplateEngine(ctx)

        assert engine.resolve("{snapshot.repo}") == "/path/to/repo"
        assert engine.resolve("{persistent.logs}") == "/path/to/logs"

    def test_wandb_placeholders(self):
        """Test resolving WandB placeholders."""
        ctx = TemplateContext(
            wandb={"run_id": "abc123", "project": "my_project"}
        )
        engine = TemplateEngine(ctx)

        assert engine.resolve("{wandb.run_id}") == "abc123"
        assert engine.resolve("{wandb.project}") == "my_project"

    def test_unknown_placeholder_strict(self):
        """Test that unknown placeholder raises in strict mode."""
        ctx = TemplateContext(run_name="test")
        engine = TemplateEngine(ctx)

        with pytest.raises(TemplateError, match="Unknown placeholder"):
            engine.resolve("{unknown}")

    def test_unknown_placeholder_non_strict(self):
        """Test that unknown placeholder is preserved in non-strict mode."""
        ctx = TemplateContext(run_name="test")
        engine = TemplateEngine(ctx)

        result = engine.resolve("{unknown}", strict=False)
        assert result == "{unknown}"

    def test_resolve_list(self):
        """Test resolving a list of templates."""
        ctx = TemplateContext(run_name="test", project="proj")
        engine = TemplateEngine(ctx)

        result = engine.resolve_list(["{run_name}", "{project}", "static"])
        assert result == ["test", "proj", "static"]

    def test_resolve_dict(self):
        """Test resolving dictionary values."""
        ctx = TemplateContext(run_name="test", project="proj")
        engine = TemplateEngine(ctx)

        result = engine.resolve_dict({
            "name": "{run_name}",
            "path": "/data/{project}/{run_name}",
        })
        assert result == {
            "name": "test",
            "path": "/data/proj/test",
        }

    def test_has_placeholder(self):
        """Test placeholder detection."""
        engine = TemplateEngine(TemplateContext())

        assert engine.has_placeholder("{run_name}")
        assert engine.has_placeholder("prefix_{run_name}_suffix")
        assert not engine.has_placeholder("no placeholders")
        assert not engine.has_placeholder("curly but not {valid")

    def test_list_placeholders(self):
        """Test listing placeholders in a string."""
        engine = TemplateEngine(TemplateContext())

        placeholders = engine.list_placeholders("{run_name}/{snapshot.repo}")
        assert "run_name" in placeholders
        assert "snapshot.repo" in placeholders

    def test_add_values(self):
        """Test adding values to engine."""
        ctx = TemplateContext()
        engine = TemplateEngine(ctx)

        engine.add_values("custom", {"key1": "value1", "key2": "value2"})

        assert engine.resolve("{custom.key1}") == "value1"
        assert engine.resolve("{custom.key2}") == "value2"

    def test_get_value(self):
        """Test getting individual values."""
        ctx = TemplateContext(run_name="test", snapshots={"repo": "/path"})
        engine = TemplateEngine(ctx)

        assert engine.get_value("run_name") == "test"
        assert engine.get_value("snapshot.repo") == "/path"
        assert engine.get_value("nonexistent") is None
