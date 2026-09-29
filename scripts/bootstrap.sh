#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pytorch}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda before continuing." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
  echo "conda environment '${CONDA_ENV}' does not exist." >&2
  echo "Create it first or select another environment with CONDA_ENV=<name>." >&2
  exit 1
fi

# Register this repository as an editable package in the shared environment.
# --no-deps is intentional: PyTorch and the other runtime dependencies stay
# managed by the conda environment instead of being copied into the project.
conda run -n "${CONDA_ENV}" python -m pip install -e . --no-deps
conda run -n "${CONDA_ENV}" python scripts/check_environment.py
