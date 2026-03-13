## OpenPI Docker Environment

This folder contains a dedicated OpenPI development/runtime container that uses:
- `uv` for dependency management
- a prebuilt virtual environment at `/.venv`
- editable install of this repo at `/app`

Dependency installation follows the same sequence as the project README:
`GIT_LFS_SKIP_SMUDGE=1 uv sync` and `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`.

The image is built without shipping repository source code in the final stage. You must bind-mount this repo to `/app` at runtime.

## Build

> [!WARNING]
> ⚠️ It is recommended to rebuild your uv.lock via `uv syc` or `uv lock`. If errors persist, remove the folder `.venv` first before rebuilding the lock.



From the repository root:

```bash
docker build . -t openpi-env -f docker/openpi.Dockerfile
```

Build Apptainer

```bash
apptainer build openpi.sif docker-daemon://openpi-env:latest
```

## Run

```bash
docker run --rm -it --network=host --gpus all -v $PWD:/app openpi-env
```

Optional OpenPI asset cache mount:

```bash
docker run --rm -it --network=host --gpus all \
  -v $PWD:/app \
  -v ${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets \
  -e OPENPI_DATA_HOME=/openpi_assets \
  openpi-env
```

## Docker Compose

```bash
docker compose -f docker/compose.yml run openpi
```

## Verify Editable Install

Inside the container:

```bash
python -c "import openpi; print(openpi.__file__)"
```

The printed path should resolve under `/app/src/openpi/...`, confirming the editable install points to the bind-mounted host code.

## Notes

- If `/app` is not mounted, editable imports for `openpi` will not resolve as intended.
- Source edits on the host are reflected after restarting the container; no image rebuild is needed.
