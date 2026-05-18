#!/bin/bash
#SBATCH --job-name=knn-samples
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-4

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${PROJECT_DIR}/slurm_env.sh"

FACTORS=(1 2 4 8 16)
IDX=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=${#FACTORS[@]}

if (( IDX < 0 || IDX >= TOTAL )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX} fuera de rango. Use 0-$(( TOTAL - 1 ))." >&2
  exit 2
fi

FACTOR=${FACTORS[$IDX]}
export N_JOBS=1

RESULTS_DIR="results/samples"
mkdir -p "${RESULTS_DIR}"

JOB_ID=${SLURM_JOB_ID:-local}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_FILE="${RESULTS_DIR}/knn_samples_f${FACTOR}_${JOB_ID}_${TASK_ID}.jsonl"

echo "[INFO] experiment=samples factor=${FACTOR} n_jobs=${N_JOBS} output=${OUTPUT_FILE}"

srun "${PYTHON_BIN}" knn.py \
  --experiment samples \
  --factor "${FACTOR}" \
  --n_neighbors "${N_NEIGHBORS}" \
  --n_reps "${N_REPS}" \
  --test_size "${TEST_SIZE}" \
  --output "${OUTPUT_FILE}"
