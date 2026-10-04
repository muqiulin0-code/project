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
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
  # This keeps commands working in terminals where conda is not initialized
  # in PATH. Override CONDA_ROOT for non-default Anaconda/Miniconda installs.
  PYTHON_CMD=("${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python")
fi

if ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
  echo "Python interpreter '${PYTHON_CMD[0]}' does not exist." >&2
  echo "Activate conda, set PYTHON, or override CONDA_ROOT." >&2
  exit 1
fi

exec "${PYTHON_CMD[@]}" "$@"
