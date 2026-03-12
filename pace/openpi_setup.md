# OpenPI Setup (PACE + HPC)

This guide shows how to prepare your HPC environment for OpenPI with PACE, wire paths into the YAML config, and run/resume training.

## 1. Copy a Template Config

Use the no-staging template as your starting point:

```bash
cp pace/examples/pace-nostage.yaml.template pace/examples/pace-nostage.yaml
```

## 2. Create Required Directories on HPC

Pick a base directory (example below uses `/path/to/your/project`) and create:

```bash
mkdir -p /path/to/your/project/{pace_logs,pace_snapshots,pace_registry,checkpoints}
mkdir -p /path/to/your/project/{openpi_home,huggingface_cache,assets,jax_cache,uv_cache}
```

Place your container image at:

```text
/path/to/your/openpi.sif
```

Folder purpose:

- `/path/to/your/project/pace_logs`
  - Scheduler stdout/stderr (`slurm_<jobid>.out/.err`).
  - Maps from `scheduler.log_dir`.
- `/path/to/your/project/pace_snapshots`
  - Immutable code snapshots created at submit time for reproducibility.
  - Maps from `snapshots[].target_dir`.
- `/path/to/your/project/pace_registry`
  - PACE run metadata (`manifest/state/attempts/markers`) used by `status`, `logs`, and `resume`.
  - Maps from `registry_root`.
- `/path/to/your/project/checkpoints`
  - Training checkpoints written by OpenPI (`<config_name>/<exp_name>/<step>`).
  - Maps from `persistent_outputs[name=checkpoints].host_path` and bind `/app/checkpoints`.
- `/path/to/your/project/openpi_home`
  - OpenPI data/model home mounted into container as `/app/openpi_home`.
  - Used by `OPENPI_DATA_HOME`.
- `/path/to/your/project/huggingface_cache`
  - HuggingFace cache directory mounted at `/app/huggingface`.
  - Used by `HF_HOME`.
- `/path/to/your/project/assets`
  - Dataset/model assets (for example normalization stats and related files).
  - Mounted at `/app/assets` if configured.
- `/path/to/your/project/jax_cache`
  - JAX compilation cache to speed up repeated runs.
  - Mounted to `/root/.cache/jax`.
- `/path/to/your/project/uv_cache`
  - `uv` package cache so installs/runs do not hit read-only system cache paths.
  - Mounted to `/app/uv_cache` and used by `UV_CACHE_DIR`.

## 3. Update `pace/examples/pace-nostage.yaml`

Fill all `# TODO` values, especially:

- `project`
- `runtime.image`
- `remote.host`
- `remote.project_dir`
- `scheduler.log_dir`
- `snapshots[].local_dir`
- `snapshots[].target_dir`
- `shared_inputs[*].host_path`
- `persistent_outputs` checkpoint path
- `registry_root`
- `WANDB_ENTITY` (and set `WANDB_API_KEY` securely)

## 4. Ensure Resume Works for OpenPI

OpenPI training resume requires `--resume` in train args.  
In `resume` section, set:

```yaml
resume:
  enabled: true
  checkpoint_output: checkpoints
  strategy: latest
  search_recursive: true
  sort_pattern: "^(?P<key>\\d+)$"
  cli_args:
    - "--resume"
```

## 5. Submit a Training Run

```bash
pace run submit my_experiment -c pace/examples/pace-nostage.yaml -- \
  pi05_libero --exp-name=my_experiment
```

## 6. Check Status

```bash
pace run status my_experiment -c pace/examples/pace-nostage.yaml
```

## 7. Resume a Run

Use PACE resume:

```bash
pace run resume my_experiment -c pace/examples/pace-nostage.yaml
```

Or explicit OpenPI-style resume (same run/config name):

```bash
pace run submit my_experiment -c pace/examples/pace-nostage.yaml -- \
  pi05_libero --exp-name=my_experiment --resume
```

## 8. Auto-Monitor + Auto-Resume

Use the helper script to continuously monitor status and trigger `pace run resume` when a run is no longer active:

```bash
python3 scripts/monitor_resume.py my_experiment -c pace/examples/pace-nostage.yaml --remote
```

Useful options:

- `--poll-seconds 60`: status polling interval.
- `--resume-cooldown-seconds 180`: minimum delay between resume attempts.
- `--max-resumes 10`: stop after N resume attempts (`-1` = unlimited).
- `--resume-on-unknown-state`: also resume when SLURM state is missing/unknown.

Example:

```bash
python3 scripts/monitor_resume.py my_experiment \
  -c pace/examples/pace-nostage.yaml \
  --remote \
  --poll-seconds 90 \
  --resume-cooldown-seconds 240 \
  --max-resumes 20
```
