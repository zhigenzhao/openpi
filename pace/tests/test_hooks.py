"""Tests for the PACE hooks feature."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from pace.backends.scheduler import SlurmBackend, SlurmJobScript
from pace.config import load_config
from pace.config.loader import ConfigError, _parse_hooks
from pace.config.models import HooksConfig, ResourcesConfig
from pace.core.manifests import LaunchPlan
from pace.launch.renderer import LaunchRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_CONFIG_TEMPLATE = """\
project: test_project
runtime:
  image: /path/to/image.sif
scheduler:
  type: slurm
  log_dir: /cluster/logs/{{run_name}}/scheduler
{extra}
"""


def _make_plan(**overrides) -> LaunchPlan:
    defaults = dict(
        workdir="/app",
        shell_init=[],
        command=["python", "train.py"],
        environment={},
        binds=[],
        image_path="/img.sif",
        pre_launch=[],
        post_launch=[],
    )
    defaults.update(overrides)
    return LaunchPlan(**defaults)


# ---------------------------------------------------------------------------
# 1. HooksConfig dataclass defaults
# ---------------------------------------------------------------------------


def test_hooks_config_defaults():
    h = HooksConfig()
    assert h.pre_apptainer == []
    assert h.pre_launch == []
    assert h.post_launch == []


# ---------------------------------------------------------------------------
# 2. YAML loading of hooks section
# ---------------------------------------------------------------------------


def test_parse_hooks_none():
    h = _parse_hooks(None)
    assert h == HooksConfig()


def test_parse_hooks_empty_dict():
    h = _parse_hooks({})
    assert h == HooksConfig()


def test_parse_hooks_all_fields():
    data = {
        "pre_apptainer": ["module load cuda/12.2"],
        "pre_launch": ["echo ready"],
        "post_launch": ["echo done"],
    }
    h = _parse_hooks(data)
    assert h.pre_apptainer == ["module load cuda/12.2"]
    assert h.pre_launch == ["echo ready"]
    assert h.post_launch == ["echo done"]


def test_load_config_no_hooks_section(tmp_path):
    """Config without hooks: section loads fine with empty HooksConfig."""
    cfg_file = tmp_path / "pace.yaml"
    cfg_file.write_text(MINIMAL_CONFIG_TEMPLATE.format(extra=""))
    config = load_config(cfg_file)
    assert config.hooks.pre_apptainer == []
    assert config.hooks.pre_launch == []
    assert config.hooks.post_launch == []


def test_load_config_with_hooks_section(tmp_path):
    """hooks: section is parsed correctly."""
    extra = textwrap.dedent("""\
        hooks:
          pre_apptainer:
            - "echo hi"
          pre_launch:
            - "source activate myenv"
          post_launch:
            - "echo bye"
    """)
    cfg_file = tmp_path / "pace.yaml"
    cfg_file.write_text(MINIMAL_CONFIG_TEMPLATE.format(extra=extra))
    config = load_config(cfg_file)
    assert config.hooks.pre_apptainer == ["echo hi"]
    assert config.hooks.pre_launch == ["source activate myenv"]
    assert config.hooks.post_launch == ["echo bye"]


# ---------------------------------------------------------------------------
# 3. LaunchPlan backward compatibility (old serialized plan has no hooks keys)
# ---------------------------------------------------------------------------


def test_launch_plan_from_yaml_no_hooks_keys():
    """Plans serialized before hooks feature must load without error."""
    data = {
        "workdir": "/app",
        "shell_init": [],
        "command": ["python", "train.py"],
        "environment": {},
        "binds": [],
        "image_path": "/img.sif",
    }
    plan = LaunchPlan.from_yaml(yaml.dump(data))
    assert plan.pre_launch == []
    assert plan.post_launch == []


def test_launch_plan_roundtrip_with_hooks():
    plan = _make_plan(pre_launch=["echo a"], post_launch=["echo b"])
    plan2 = LaunchPlan.from_yaml(plan.to_yaml())
    assert plan2.pre_launch == ["echo a"]
    assert plan2.post_launch == ["echo b"]


# ---------------------------------------------------------------------------
# 4. Renderer: no hooks → identical output to before (exec pattern)
# ---------------------------------------------------------------------------


def test_renderer_no_hooks_uses_exec():
    plan = _make_plan()
    renderer = LaunchRenderer()
    output = renderer.render_compute_wrapper(plan)
    assert "exec python train.py" in output
    assert "pre_launch" not in output
    assert "post_launch" not in output
    assert "_pace_exit" not in output


def test_renderer_no_hooks_no_extra_sections():
    """When hooks are empty, no hook section headers should appear."""
    plan = _make_plan()
    renderer = LaunchRenderer()
    output = renderer.render_compute_wrapper(plan)
    assert "Pre-launch hooks" not in output
    assert "Post-launch hooks" not in output


# ---------------------------------------------------------------------------
# 5. Renderer: pre_launch hooks appear before cd
# ---------------------------------------------------------------------------


def test_renderer_pre_launch_appears_before_workdir():
    plan = _make_plan(pre_launch=["source activate myenv", "echo ready"])
    renderer = LaunchRenderer()
    output = renderer.render_compute_wrapper(plan)
    assert "# Pre-launch hooks" in output
    assert "source activate myenv" in output
    assert "echo ready" in output
    # pre_launch must come before the cd line
    pre_idx = output.index("# Pre-launch hooks")
    cd_idx = output.index("cd /app")
    assert pre_idx < cd_idx


# ---------------------------------------------------------------------------
# 6. Renderer: post_launch hooks → exit-code capture pattern
# ---------------------------------------------------------------------------


def test_renderer_post_launch_uses_exit_capture():
    plan = _make_plan(post_launch=["echo done"])
    renderer = LaunchRenderer()
    output = renderer.render_compute_wrapper(plan)
    assert "_pace_exit=$?" in output
    assert "# Post-launch hooks" in output
    assert "echo done" in output
    assert "exit $_pace_exit" in output
    # Should NOT use exec when post_launch is set
    assert "exec python" not in output


def test_renderer_post_launch_exit_order():
    """exit $_pace_exit must come after post-launch hooks."""
    plan = _make_plan(post_launch=["cleanup.sh"])
    renderer = LaunchRenderer()
    output = renderer.render_compute_wrapper(plan)
    hook_idx = output.index("cleanup.sh")
    exit_idx = output.index("exit $_pace_exit")
    assert hook_idx < exit_idx


def test_renderer_both_hooks():
    plan = _make_plan(
        pre_launch=["echo pre"],
        post_launch=["echo post"],
    )
    renderer = LaunchRenderer()
    output = renderer.render_compute_wrapper(plan)
    assert "# Pre-launch hooks" in output
    assert "echo pre" in output
    assert "_pace_exit=$?" in output
    assert "# Post-launch hooks" in output
    assert "echo post" in output
    assert "exit $_pace_exit" in output


# ---------------------------------------------------------------------------
# 7. SLURM: merged stdout/stderr (no --error line)
# ---------------------------------------------------------------------------


def test_slurm_no_error_directive():
    plan = _make_plan()
    backend = SlurmBackend()
    script = backend.create_job_script(
        plan=plan,
        resources=ResourcesConfig(),
        job_name="pace-test",
        log_dir="/logs",
        wrapper_path="/tmp/wrapper.sh",
    )
    rendered = script.render()
    assert "#SBATCH --output=/logs/slurm_%j.out" in rendered
    assert "#SBATCH --error=" not in rendered


# ---------------------------------------------------------------------------
# 8. SLURM: pre_apptainer_commands populate pre_commands
# ---------------------------------------------------------------------------


def test_slurm_pre_apptainer_commands_rendered():
    plan = _make_plan()
    backend = SlurmBackend()
    script = backend.create_job_script(
        plan=plan,
        resources=ResourcesConfig(),
        job_name="pace-test",
        log_dir="/logs",
        wrapper_path="/tmp/wrapper.sh",
        pre_apptainer_commands=["module load cuda/12.2", "echo before container"],
    )
    rendered = script.render()
    assert "module load cuda/12.2" in rendered
    assert "echo before container" in rendered
    # pre-commands must appear before the apptainer exec line
    pre_idx = rendered.index("module load cuda/12.2")
    apt_idx = rendered.index("apptainer")
    assert pre_idx < apt_idx


def test_slurm_no_pre_apptainer_commands_no_section():
    plan = _make_plan()
    backend = SlurmBackend()
    script = backend.create_job_script(
        plan=plan,
        resources=ResourcesConfig(),
        job_name="pace-test",
        log_dir="/logs",
        wrapper_path="/tmp/wrapper.sh",
    )
    rendered = script.render()
    assert "# Pre-execution commands" not in rendered


def test_slurm_pre_apptainer_none_vs_empty_same():
    """None and [] for pre_apptainer_commands should produce the same output."""
    plan = _make_plan()
    backend = SlurmBackend()

    script_none = backend.create_job_script(
        plan=plan,
        resources=ResourcesConfig(),
        job_name="pace-test",
        log_dir="/logs",
        wrapper_path="/tmp/wrapper.sh",
        pre_apptainer_commands=None,
    )
    script_empty = backend.create_job_script(
        plan=plan,
        resources=ResourcesConfig(),
        job_name="pace-test",
        log_dir="/logs",
        wrapper_path="/tmp/wrapper.sh",
        pre_apptainer_commands=[],
    )
    assert script_none.render() == script_empty.render()
