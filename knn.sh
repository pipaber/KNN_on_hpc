#!/bin/bash
#SBATCH --job-name=knn-grid
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --array=0-53

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${PROJECT_DIR}/slurm_env.sh"

EXPERIMENTS=(samples features)
SAMPLE_FACTORS=(1 2 4 8 16)
FEATURE_FACTORS=(1 2 4 8)
P_VALUES=(1 2 4 8 16 32)

IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_P=${#P_VALUES[@]}
NUM_SAMPLE=${#SAMPLE_FACTORS[@]}
NUM_FEATURE=${#FEATURE_FACTORS[@]}
TOTAL=$(( (NUM_SAMPLE + NUM_FEATURE) * NUM_P ))

if (( IDX < 0 || IDX >= TOTAL )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX} fuera de rango. Use 0-$(( TOTAL - 1 ))." >&2
  exit 2
fi

BLOCK=$(( IDX / NUM_P ))
P_IDX=$(( IDX % NUM_P ))
export N_JOBS=${P_VALUES[$P_IDX]}

if (( BLOCK < NUM_SAMPLE )); then
  EXPERIMENT="samples"
  FACTOR=${SAMPLE_FACTORS[$BLOCK]}
else
  EXPERIMENT="features"
  FACTOR=${FEATURE_FACTORS[$(( BLOCK - NUM_SAMPLE ))]}
fi

RESULTS_DIR="results/${EXPERIMENT}"
mkdir -p "${RESULTS_DIR}"

JOB_ID=${SLURM_JOB_ID:-local}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_FILE="${RESULTS_DIR}/knn_${EXPERIMENT}_p${N_JOBS}_f${FACTOR}_${JOB_ID}_${TASK_ID}.jsonl"

echo "[INFO] experiment=${EXPERIMENT} factor=${FACTOR} n_jobs=${N_JOBS} output=${OUTPUT_FILE}"

srun "${PYTHON_BIN}" knn.py \
  --experiment "${EXPERIMENT}" \
  --factor "${FACTOR}" \
  --n_neighbors "${N_NEIGHBORS}" \
  --n_reps "${N_REPS}" \
  --test_size "${TEST_SIZE}" \
  --output "${OUTPUT_FILE}"
