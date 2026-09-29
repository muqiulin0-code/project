#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pytorch}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
else
  command -v conda >/dev/null 2>&1 || {
    echo "conda was not found; install conda or set PYTHON explicitly" >&2
    exit 1
  }
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

"${PYTHON_CMD[@]}" scripts/check_environment.py
"${PYTHON_CMD[@]}" -m pytest
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel
ctest --test-dir build/cpp --output-on-failure
