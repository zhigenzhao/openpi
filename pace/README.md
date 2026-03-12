# PACE - HPC Experiment Launcher

**P**ortable **A**pptainer **C**luster **E**xecutor

A reusable tool for launching and managing Python experiments on HPC clusters with SLURM and Apptainer.

## Features

- **Declarative Configuration**: Define experiment setup in a single `pace.yaml` file
- **Automatic Snapshotting**: Create reproducible snapshots of code with rsync
- **Smart Staging**: Stage data to `$TMPDIR` for performance, or bind directly
- **Phased Launch Plan**: Flexible command and environment injection in 6 phases
- **Robust Checkpointing**: Transaction-based checkpoint saving with atomic commits
- **Auto-Resume**: Resume from the latest valid checkpoint
- **SSH Remote Mode**: Submit jobs from local machine via SSH
- **WandB Integration**: Stable run IDs for seamless resume of logging

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Configuration Guide](#configuration-guide)
- [CLI Reference](#cli-reference)
- [Runtime Library](#runtime-library)
- [Remote Execution](#remote-execution)
- [Directory Layout](#directory-layout)

## Installation

### From Source

```bash
cd pace
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Using uv (Recommended)

```bash
cd pace
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

### 1. Create a configuration file

```yaml
# pace.yaml
project: my_project

runtime:
  image: /path/to/container.sif
  engine: apptainer

scheduler:
  type: slurm
  log_dir: /cluster/logs/{run_name}/scheduler

resources:
  gpus: 4
  gpu_type: H100
  cpus: 32
  mem_gb: 500
  time: "2:00:00"

snapshots:
  - name: repo
    local_dir: /path/to/source
    target_dir: /cluster/snapshots
    exclude:
      - .git
      - __pycache__
      - .venv

persistent_outputs:
  - name: logs
    role: logs
    host_path: /cluster/logs/{run_name}/logs
    container_path: /workspace/logs
    create_if_missing: true

binds:
  - host: "{snapshot.repo}"
    container: /workspace/repo
    mode: rw

launch:
  workdir: /workspace/repo
  base_command:
    - python
    - train.py
  user_args:
    phase: post_base

registry_root: /cluster/pace_runs
```

### 2. Submit a run

```bash
# Submit a new run
pace run submit my_experiment -c pace.yaml -- --learning-rate 0.001

# Dry run (show what would be done)
pace run submit my_experiment -c pace.yaml --dry-run -- --learning-rate 0.001
```

### 3. Monitor and manage

```bash
# Check status
pace run status my_experiment -c pace.yaml

# View logs
pace run logs my_experiment -c pace.yaml --follow

# List all runs
pace run list -c pace.yaml

# Resume from latest checkpoint
pace run resume my_experiment -c pace.yaml
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                          PACE CLI                                 │
│  pace run submit | status | logs | resume | list                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│                    Core Orchestration                             │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Config  │  │ Registry │  │ Snapshots│  │ Template Engine  │  │
│  │ Loader  │  │          │  │          │  │                  │  │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│                    Launch Plan Builder                            │
│  ┌─────────────────┐  ┌───────────────┐  ┌───────────────────┐  │
│  │ Phase Ordering  │  │ Env Injectors │  │ CLI Injectors     │  │
│  │ (6 phases)      │  │ (conditional) │  │ (conditional)     │  │
│  └─────────────────┘  └───────────────┘  └───────────────────┘  │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│                    Cluster Runtime Backends                       │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ SLURM     │  │ Apptainer  │  │ Staging    │  │ SSH Remote │  │
│  │ Backend   │  │ Backend    │  │ Planner    │  │ Executor   │  │
│  └───────────┘  └────────────┘  └────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    Python Runtime Library                         │
│  ┌──────────────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ checkpoint_transaction│  │ Markers       │  │ Signal       │  │
│  │ (atomic saves)        │  │ CHECKPOINT_OK │  │ Handling     │  │
│  └──────────────────────┘  └───────────────┘  └───────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Config Loader** | Parse and validate `pace.yaml` configuration |
| **Registry** | Manage run directories, manifests, and attempts |
| **Snapshots** | Create timestamped copies of source code with rsync |
| **Template Engine** | Resolve placeholders like `{run_name}`, `{snapshot.repo}` |
| **Launch Plan Builder** | Assemble command and environment in phases |
| **SLURM Backend** | Generate and submit SLURM job scripts |
| **Apptainer Backend** | Build container execution commands |
| **SSH Remote Executor** | Enable local machine submission via SSH |
| **Runtime Library** | Atomic checkpointing for training code |

## Configuration Guide

See [Configuration Reference](#full-configuration-schema) below for the complete schema.

### Template Placeholders

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{run_name}` | Run identifier | `my_experiment` |
| `{project}` | Project name | `rlinf` |
| `{attempt_id}` | Attempt number | `1`, `2`, `3` |
| `{job_id}` | SLURM job ID | `12345678` |
| `{snapshot.NAME}` | Snapshot host path | `/cluster/snapshots/repo_20260311` |
| `{staged.NAME}` | Staged input path | `$TMPDIR/pi_home` |
| `{shared.NAME}` | Shared input path | `/cluster/datasets` |
| `{persistent.NAME}` | Persistent output host path | `/cluster/logs/my_run/checkpoints` |
| `{container.NAME}` | Persistent output container path | `/workspace/checkpoints` |
| `{resume_path_host}` | Resume checkpoint host path | `/cluster/.../step_1000` |
| `{resume_path_container}` | Resume checkpoint container path | `/workspace/.../step_1000` |
| `{wandb.run_id}` | WandB run ID | `pace-rlinf-my_run` |

### Launch Phases

Commands and arguments are assembled in this order:

```
PRE_BASE → BASE → POST_BASE → PRE_RESUME → POST_RESUME → FINAL
```

- **PRE_BASE**: Setup arguments before base command
- **BASE**: The `base_command` from config
- **POST_BASE**: Where `user_args` are inserted by default
- **PRE_RESUME**: Before resume arguments (resume only)
- **POST_RESUME**: Resume arguments (resume only)
- **FINAL**: Final arguments (e.g., run name)

### Injection Conditions

| Condition | When Applied |
|-----------|--------------|
| `always` | Every attempt |
| `resume_only` | Only when resuming from checkpoint |
| `non_resume_only` | Only on fresh runs (no resume) |
| `first_attempt_only` | Only on attempt #1 |

## CLI Reference

### `pace run submit`

Submit a new run or resume an existing one.

```bash
pace run submit <RUN_NAME> [OPTIONS] -- [USER_ARGS...]
```

| Option | Description |
|--------|-------------|
| `-c, --config PATH` | Path to pace.yaml |
| `--remote` | Force SSH remote execution |
| `--no-remote` | Force local execution |
| `--dry-run` | Show what would be done |
| `--no-submit` | Prepare artifacts without submitting |

**Examples:**

```bash
# Submit a new run
pace run submit my_run -c pace.yaml -- config_name PLATFORM

# Dry run
pace run submit my_run -c pace.yaml --dry-run -- config_name PLATFORM

# Force remote submission
pace run submit my_run -c pace.yaml --remote -- config_name PLATFORM
```

### `pace run status`

Show status of a run.

```bash
pace run status <RUN_NAME> -c pace.yaml
```

**Output includes:**
- Run metadata (project, created date)
- Latest attempt info (job ID, status)
- SLURM job state (PENDING, RUNNING, COMPLETED, etc.)
- Checkpoint count and latest checkpoint
- Training completion status

### `pace run logs`

View logs for a run.

```bash
pace run logs <RUN_NAME> -c pace.yaml [--attempt N] [--follow]
```

### `pace run resume`

Resume from latest valid checkpoint.

```bash
pace run resume <RUN_NAME> -c pace.yaml [--checkpoint PATH] [--dry-run]
```

### `pace run list`

List all runs for the project.

```bash
pace run list -c pace.yaml
```

## Runtime Library

The `pace_runtime` package provides checkpoint safety for training code.

### Atomic Checkpoint Transactions

```python
from pace_runtime.checkpointing import checkpoint_transaction
from pace_runtime.markers import mark_training_done

# Atomic checkpoint save
with checkpoint_transaction(
    checkpoint_root="/workspace/checkpoints",
    tag=f"global_step_{step}",
    required_files=["model.pt", "optimizer.pt"],  # Optional validation
) as tx_dir:
    # Save files into tx_dir
    torch.save(model.state_dict(), f"{tx_dir}/model.pt")
    torch.save(optimizer.state_dict(), f"{tx_dir}/optimizer.pt")
    torch.save(scheduler.state_dict(), f"{tx_dir}/scheduler.pt")
# On successful exit:
#   1. Writes manifest.json
#   2. Writes CHECKPOINT_OK marker
#   3. Atomically renames to final location
#   4. Updates latest_safe symlink

# Mark training complete
mark_training_done("/workspace/logs")  # Writes TRAINING_DONE marker
```

### Checkpoint Discovery

```python
from pace_runtime.checkpointing import (
    find_latest_checkpoint,
    is_checkpoint_valid,
    list_checkpoints,
)

# Find latest valid checkpoint
latest = find_latest_checkpoint("/workspace/checkpoints")
if latest:
    print(f"Resuming from: {latest}")

# Check if specific checkpoint is valid
if is_checkpoint_valid("/workspace/checkpoints/step_1000"):
    print("Checkpoint is valid")

# List all valid checkpoints
for ckpt in list_checkpoints("/workspace/checkpoints"):
    print(f"  {ckpt.name}")
```

### Signal Handling

```python
from pace_runtime.signals import GracefulShutdown

shutdown = GracefulShutdown()

for step in range(num_steps):
    if shutdown.should_exit:
        print("Received shutdown signal, saving checkpoint...")
        break
    train_step()
```

## Remote Execution

PACE supports submitting jobs from a local machine to a cluster via SSH.

### Configuration

```yaml
remote:
  enabled: true
  host: pace-ice          # SSH config host name
  project_dir: /storage/project/RLinf
  mode: auto              # auto | local | remote
```

### SSH Setup

Ensure passwordless SSH access:

```bash
# ~/.ssh/config
Host pace-ice
    HostName login-ice.pace.gatech.edu
    User yourusername
    IdentityFile ~/.ssh/id_rsa
```

### Execution Modes

| Mode | Behavior |
|------|----------|
| `auto` | Check if `project_dir` exists locally; use SSH if not |
| `local` | Always execute on current machine (login node) |
| `remote` | Always use SSH |

### What Happens During Remote Submit

1. **Local**: Create snapshot of source code
2. **Local → Remote**: Rsync snapshot to cluster
3. **Local → Remote**: Rsync run artifacts to registry
4. **Remote**: Execute `sbatch` via SSH
5. **Remote**: Update attempt manifest with job ID

## Directory Layout

### Run Registry Structure

```
{registry_root}/{project}/{run_name}/
├── manifest.yaml           # Run metadata
├── state.json              # Mutable run state
├── snapshots/              # (metadata only)
├── attempts/
│   ├── 1/
│   │   ├── attempt.yaml
│   │   ├── launch_plan.yaml
│   │   ├── slurm_job.sh
│   │   ├── compute_wrapper.sh
│   │   ├── env.resolved
│   │   └── command.resolved.sh
│   └── 2/
│       └── ...
├── logs/
├── checkpoints/
├── artifacts/
└── markers/
    └── TRAINING_DONE
```

### Checkpoint Directory Structure

```
checkpoints/
├── global_step_1000/
│   ├── model.pt
│   ├── optimizer.pt
│   ├── manifest.json
│   └── CHECKPOINT_OK       # Validity marker
├── global_step_2000/
│   └── ...
└── latest_safe -> global_step_2000
```

## Full Configuration Schema

```yaml
# Project identifier
project: rlinf

# Container runtime
runtime:
  image: /cluster/images/rlinf.sif
  engine: apptainer

# Remote SSH orchestration (optional)
remote:
  enabled: true
  host: pace-ice
  user: null
  project_dir: /storage/project/RLinf
  mode: auto  # auto | local | remote
  ssh_options: []
  rsync_options: []

# SLURM resources
resources:
  gpus: 8
  gpu_type: H100
  cpus: 64
  mem_gb: 1500
  time: "1:59:59"
  nodes: 1
  # partition: gpu
  # account: my_acct
  # qos: normal

# Scheduler configuration
scheduler:
  type: slurm
  log_dir: /storage/logs/{run_name}/scheduler

# Source code snapshots
snapshots:
  - name: repo
    local_dir: /local/path/to/source
    target_dir: /cluster/snapshots
    exclude:
      - "*.git*"
      - "logs/"
      - ".venv/"

# Shared inputs (read-only data)
shared_inputs:
  - name: pi_home
    host_path: /cluster/shared/pi_home
    stage_mode: copy_to_tmp  # or: bind
  - name: datasets
    host_path: /cluster/datasets
    stage_mode: bind

# Persistent outputs
persistent_outputs:
  - name: logs
    role: logs
    host_path: /cluster/logs/{run_name}/logs
    container_path: /workspace/logs
    create_if_missing: true
  - name: checkpoints
    role: checkpoints
    host_path: /cluster/logs/{run_name}/checkpoints
    container_path: /workspace/checkpoints
    create_if_missing: true

# Container bind mounts
binds:
  - host: "{snapshot.repo}"
    container: /workspace/RLinf
    mode: rw
  - host: "{staged.pi_home}"
    container: /workspace/pi_home
    mode: rw
  - host: "{persistent.logs}"
    container: /workspace/logs
    mode: rw

# Launch configuration
launch:
  workdir: /workspace/RLinf
  shell_init:
    - source switch_env openpi
  base_command:
    - bash
    - examples/embodiment/run_embodiment.sh
  user_args:
    phase: post_base

# Environment and CLI injections
injections:
  env:
    - name: core-env
      when: always
      values:
        PACE_RUN_NAME: "{run_name}"
        PACE_CHECKPOINT_DIR: "{container.checkpoints}"
    - name: resume-env
      when: resume_only
      values:
        PACE_RESUME_DIR: "{resume_path_container}"
  cli:
    - name: run-name
      when: always
      phase: final
      args:
        - "{run_name}"

# Resume configuration
resume:
  enabled: true
  checkpoint_output: checkpoints
  strategy: latest_safe
  search_recursive: true
  checkpoint_marker: CHECKPOINT_OK
  sort_pattern: "global_step_(?P<key>\\d+)"
  cli_args:
    - "runner.resume_dir={resume_path_container}"

# Completion detection
completion:
  marker_file: TRAINING_DONE

# Registry root for run metadata
registry_root: /cluster/pace_runs
```

## Environment Variables

PACE injects these environment variables into training jobs:

| Variable | Description |
|----------|-------------|
| `PACE_RUN_NAME` | Run identifier |
| `PACE_ATTEMPT_ID` | Current attempt number |
| `PACE_JOB_ID` | SLURM job ID |
| `PACE_LOG_DIR` | Log directory (container path) |
| `PACE_CHECKPOINT_DIR` | Checkpoint directory (container path) |
| `PACE_RESUME_DIR` | Resume checkpoint path (if resuming) |

## Troubleshooting

### "No pace.yaml found"

Specify config path with `-c path/to/pace.yaml`.

### "Run not found"

Check registry root path and run name. Use `pace run list` to see available runs.

### "sbatch command not found"

Not on a SLURM cluster. Use `remote.enabled: true` for remote submission.

### Checkpoint not resuming

Ensure `resume.checkpoint_marker` matches what your training stack writes.
If you use `pace_runtime.checkpoint_transaction`, the default marker is `CHECKPOINT_OK`:

```python
with checkpoint_transaction(ckpt_dir, tag) as tx:
    # Save checkpoint files here
    pass
```

## License

MIT
