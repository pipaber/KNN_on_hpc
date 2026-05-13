#!/bin/bash
#SBATCH --job-name=knn-scale
#SBATCH --output=logs/knn_%x_%A_%a.out
#SBATCH --error=logs/knn_%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#
# Grilla completa para estudiar:
# - costo al incrementar muestras: comparar SIZES con N_JOBS y k fijos
# - efecto de vecinos: comparar K_VALUES con N_JOBS y n_samples fijos
# - escalabilidad fuerte: filtrar n_samples=300 y k fijo, variar N_JOBS
# - escalabilidad debil: usar la diagonal S_IDX=J_IDX con k fijo
#
# KNN predice comparando test contra train: O(n_train * n_test * d).
# Con test_size fijo, el costo crece aprox. O(n_samples^2 * n_features).
# Por eso, para escalabilidad debil se usa n_samples ~= n_base * sqrt(N_JOBS).
#
# N_JOBS_OPTS=(1 2 4 8 16 32) => 6 valores
# SIZES=(53 75 106 150 212 300) => 6 valores
# K_VALUES=(1 3 5 11) => 4 valores
# 6 * 6 * 4 = 144 combinaciones, indices 0..143
#SBATCH --array=0-143

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs results .cache

# Activa tu entorno si existe en el directorio del script o en el repo padre.
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -f "../.venv/bin/activate" ]; then
  source "../.venv/bin/activate"
fi

# Evitar sobresuscripcion BLAS: el paralelismo se controla solo con N_JOBS.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Procesos joblib para KNN.
N_JOBS_OPTS=(1 2 4 8 16 32)

# Dataset UCI id=45 tiene 303 filas y 13 features.
# La diagonal de esta lista aproxima n_base * sqrt(p), con n_base=53.
SIZES=(53 75 106 150 212 300)

K_VALUES=(1 3 5 11)

N_FEATURES=13
N_REPS=30
TEST_SIZE=0.3
RESULTS_DIR="results"

NUM_J=${#N_JOBS_OPTS[@]}
NUM_S=${#SIZES[@]}
NUM_K=${#K_VALUES[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$(( NUM_J * NUM_S * NUM_K ))

if (( IDX < 0 || IDX >= TOTAL )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX} fuera de rango. Use 0-$(( TOTAL - 1 ))." >&2
  exit 2
fi

J_IDX=$(( IDX % NUM_J ))
K_IDX=$(( (IDX / NUM_J) % NUM_K ))
S_IDX=$(( IDX / (NUM_J * NUM_K) ))

N_JOBS=${N_JOBS_OPTS[$J_IDX]}
N_SAMPLES=${SIZES[$S_IDX]}
N_NEIGHBORS=${K_VALUES[$K_IDX]}
STRONG_N_SAMPLES=${SIZES[$(( NUM_S - 1 ))]}

SCALE_TAG="grid"
if (( N_SAMPLES == STRONG_N_SAMPLES )); then
  SCALE_TAG="strong"
fi
if (( S_IDX == J_IDX )); then
  SCALE_TAG="${SCALE_TAG}_weak"
fi

export N_JOBS="${N_JOBS}"

JOB_ID=${SLURM_JOB_ID:-local}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_FILE="${RESULTS_DIR}/knn_${SCALE_TAG}_p${N_JOBS}_n${N_SAMPLES}_k${N_NEIGHBORS}_${JOB_ID}_${TASK_ID}.jsonl"

echo "[INFO] job_id=${JOB_ID} task_id=${TASK_ID}"
echo "[INFO] combo: N_JOBS=${N_JOBS} N_SAMPLES=${N_SAMPLES} N_FEATURES=${N_FEATURES} K=${N_NEIGHBORS}"
echo "[INFO] tag=${SCALE_TAG} output=${OUTPUT_FILE}"

srun python3 knn.py \
  --n_samples "${N_SAMPLES}" \
  --n_features "${N_FEATURES}" \
  --n_neighbors "${N_NEIGHBORS}" \
  --n_reps "${N_REPS}" \
  --test_size "${TEST_SIZE}" \
  --scaling_tag "${SCALE_TAG}" \
  --output "${OUTPUT_FILE}"
