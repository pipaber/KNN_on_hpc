#!/bin/bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
cd "${PROJECT_DIR}"

if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -f "../.venv/bin/activate" ]; then
  source "../.venv/bin/activate"
fi

python3 - <<'PY'
from knn import CACHE_PATH, load_dataset_with_cache

X, y = load_dataset_with_cache()
print(f"[OK] Cache listo: {CACHE_PATH}")
print(f"[OK] X.shape={X.shape} y.shape={y.shape}")
PY
