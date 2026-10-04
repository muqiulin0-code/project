#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pytorch}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"

if [[ -n "${PYTHON:-}" ]]; then
  # shellcheck disable=SC2206
  PYTHON_CMD=(${PYTHON})
elif command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
    echo "conda environment '${CONDA_ENV}' does not exist." >&2
    echo "Create it first or select another environment with CONDA_ENV=<name>." >&2
    exit 1
  fi
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
  # Keep the shared conda interpreter usable in non-login terminals where
  # `conda` is not on PATH. CONDA_ROOT can be overridden for other installs.
  PYTHON_CMD=("${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python")
fi

if ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
  echo "Python interpreter '${PYTHON_CMD[0]}' does not exist." >&2
  echo "Activate conda, set PYTHON, or override CONDA_ROOT." >&2
  exit 1
fi

# Register this repository as an editable package in the shared environment.
# --no-deps is intentional: PyTorch and the other runtime dependencies stay
# managed by the conda environment instead of being copied into the project.
"${PYTHON_CMD[@]}" -m pip install -e . --no-deps
"${PYTHON_CMD[@]}" scripts/check_environment.py
