#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs .cache

if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -f "../.venv/bin/activate" ]; then
  source "../.venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON:-python3}"
N_REPS="${N_REPS:-30}"
TEST_SIZE="${TEST_SIZE:-0.3}"
N_NEIGHBORS="${N_NEIGHBORS:-5}"
