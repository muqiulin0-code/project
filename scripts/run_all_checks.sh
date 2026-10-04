#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pytorch}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
CONDA_PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif [[ -x "${CONDA_PYTHON}" ]]; then
  # Use the shared environment directly to avoid a shell hook selecting the
  # base interpreter for `conda run -n ...`.
  PYTHON_CMD=("${CONDA_PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
  # The project should still work in a fresh terminal where conda has not
  # been initialized in the shell yet.
  PYTHON_CMD=("${CONDA_PYTHON}")
fi

if ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
  echo "Python interpreter '${PYTHON_CMD[0]}' does not exist." >&2
  echo "Activate conda, set PYTHON, or override CONDA_ROOT." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" scripts/check_environment.py
"${PYTHON_CMD[@]}" -m pytest
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel
ctest --test-dir build/cpp --output-on-failure
