"""PACE init command implementation."""

from __future__ import annotations

from pathlib import Path

import click


EXAMPLE_CONFIG = """\
# PACE Configuration
# See documentation for full options

project: my_project

runtime:
  image: /path/to/container.sif
  engine: apptainer

remote:
  enabled: false
  host: pace-ice
  user: null
  project_dir: /storage/ice-shared/<user>/project/RLinf
  mode: auto
  ssh_options: []
  rsync_options: []

resources:
  gpus: 4
  gpu_type: H100
  cpus: 32
  mem_gb: 500
  time: "2:00:00"
  nodes: 1

scheduler:
  type: slurm
  log_dir: /cluster/runs/{run_name}/scheduler

snapshots:
  - name: repo
    local_dir: .
    target_dir: /cluster/runs/{run_name}/snapshots
    exclude:
      - .git
      - logs
      - checkpoints
      - .venv
      - __pycache__

persistent_outputs:
  - name: logs
    role: logs
    host_path: /cluster/runs/{run_name}/logs
    container_path: /workspace/logs
    create_if_missing: true

  - name: checkpoints
    role: checkpoints
    host_path: /cluster/runs/{run_name}/checkpoints
    container_path: /workspace/checkpoints
    create_if_missing: true

binds:
  - host: "{snapshot.repo}"
    container: /workspace/repo
    mode: rw

  - host: "{persistent.logs}"
    container: /workspace/logs
    mode: rw

  - host: "{persistent.checkpoints}"
    container: /workspace/checkpoints
    mode: rw

launch:
  workdir: /workspace/repo
  base_command:
    - python
    - train.py

  user_args:
    phase: post_base

injections:
  env:
    - name: pace-vars
      when: always
      values:
        PACE_RUN_NAME: "{run_name}"
        PACE_CHECKPOINT_DIR: "{container.checkpoints}"

resume:
  enabled: true
  checkpoint_output: checkpoints
  strategy: latest_safe
  search_recursive: true
  checkpoint_marker: CHECKPOINT_OK
  sort_pattern: "global_step_(?P<key>\\d+)"
  cli_args:
    - "runner.resume_dir={resume_path_container}"

completion:
  marker_file: TRAINING_DONE

registry_root: /cluster/pace_runs
"""


def run_init(config_path: str | None = None) -> None:
    """Initialize a PACE project.

    Args:
        config_path: Optional path for config file.
    """
    if config_path:
        output_path = Path(config_path)
    else:
        output_path = Path.cwd() / "pace.yaml"

    if output_path.exists():
        click.echo(f"Config already exists: {output_path}")
        if not click.confirm("Overwrite?"):
            return

    output_path.write_text(EXAMPLE_CONFIG)
    click.echo(f"Created: {output_path}")
    click.echo("\nNext steps:")
    click.echo("  1. Edit pace.yaml with your project settings")
    click.echo("  2. Run: pace run submit <run_name>")
