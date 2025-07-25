# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPI is an open-source robotics research repository from Physical Intelligence, containing two main types of vision-language-action (VLA) models:
- **π₀ model**: Flow-based diffusion VLA model  
- **π₀-FAST model**: Autoregressive VLA based on FAST action tokenizer

The repository provides base model checkpoints pre-trained on 10k+ hours of robot data, along with fine-tuned models for specific robot platforms (ALOHA, DROID, etc.).

## Development Environment

### Installation
- Uses **uv** for Python dependency management
- Python 3.11+ required
- NVIDIA GPU required (8GB+ for inference, 22.5GB+ for LoRA fine-tuning, 70GB+ for full fine-tuning)

```bash
# Initial setup
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Essential Commands

**Linting and Testing:**
```bash
uv run ruff check                    # Lint code
uv run ruff format                   # Format code  
uv run pytest                       # Run tests
uv run pytest src/openpi/models/    # Run specific test directory
```

**Training:**
```bash
# Compute normalization statistics (required before training)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Run training with GPU memory optimization
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name> --overwrite
```

**Policy Serving:**
```bash
# Serve a trained policy checkpoint
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_path>
```

## Code Architecture

### Core Components

**Models (`src/openpi/models/`)**
- `pi0.py` - Flow-based diffusion VLA model implementation
- `pi0_fast.py` - Autoregressive VLA model implementation  
- `gemma.py`/`gemma_fast.py` - Gemma language model components
- `siglip.py` - Vision encoder
- `tokenizer.py` - Action tokenization

**Policies (`src/openpi/policies/`)**
- `policy.py` - Base policy interface and implementation
- `policy_config.py` - Policy creation and configuration
Platform-specific policies: `aloha_policy.py`, `droid_policy.py`, `libero_policy.py`, `arx_dual_arm_policy.py`, `arx_single_arm_policy.py`

**Training (`src/openpi/training/`)**
- `config.py` - Training configurations and data processing configs
- `data_loader.py` - Data loading pipeline using LeRobot datasets
- `optimizer.py` - Training optimization logic
- `checkpoints.py` - Checkpoint saving/loading
- `weight_loaders.py` - Pre-trained weight loading utilities

**Shared Utilities (`src/openpi/shared/`)**
- `download.py` - Model checkpoint downloading from GCS
- `normalize.py` - Data normalization utilities
- `image_tools.py` - Image processing utilities
- `array_typing.py` - JAX array type definitions

### Key Scripts

- `scripts/train.py` - Main training script
- `scripts/compute_norm_stats.py` - Compute normalization statistics
- `scripts/serve_policy.py` - Policy inference server

### Examples Directory Structure

Platform-specific examples with complete setups:
- `examples/aloha_real/` - Real ALOHA robot
- `examples/aloha_sim/` - ALOHA simulation  
- `examples/droid/` - DROID robot platform
- `examples/libero/` - Libero simulation environment
- `examples/arx_r5/` - ARX R5 robot arms

Each example contains data conversion scripts, environment setup, and deployment instructions.

## Configuration System

Training configs are defined in `src/openpi/training/config.py` with naming convention:
- `pi0_<platform>` - π₀ model configs
- `pi0_fast_<platform>` - π₀-FAST model configs

Available configs include: `pi0_fast_droid`, `pi0_droid`, `pi0_aloha_sim`, `pi0_fast_libero`, etc.

## Data Pipeline

Uses LeRobot dataset format. Data conversion scripts in `examples/*/convert_*_data_to_lerobot.py` show how to adapt custom datasets.

Normalization statistics must be computed before training using `scripts/compute_norm_stats.py`.

## Key Technologies

- **JAX/Flax** - Primary ML framework using Flax NNX
- **Orbax** - Checkpointing
- **LeRobot** - Robotics dataset format
- **Weights & Biases** - Experiment tracking
- **uv** - Python dependency management
- **ruff** - Code formatting and linting

## Important Notes

- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` for training to maximize GPU memory usage
- Normalization statistics are required before training and stored in `assets/` directory
- Model checkpoints are downloaded from `gs://openpi-assets` and cached in `~/.cache/openpi`
- Use `GIT_LFS_SKIP_SMUDGE=1` when installing to avoid pulling large LeRobot files
- Third-party robot SDKs are in `third_party/` (ALOHA, Libero, R5)