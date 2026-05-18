#!/bin/bash
#SBATCH --job-name=knn-jobs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-19

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_env.sh"

FACTORS=(1 2 4 8 16)
P_VALUES=(1 2 4 8)
IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_F=${#FACTORS[@]}
NUM_P=${#P_VALUES[@]}
TOTAL=$(( NUM_F * NUM_P ))

if (( IDX < 0 || IDX >= TOTAL )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX} fuera de rango. Use 0-$(( TOTAL - 1 ))." >&2
  exit 2
fi

F_IDX=$(( IDX % NUM_F ))
P_IDX=$(( IDX / NUM_F ))

FACTOR=${FACTORS[$F_IDX]}
export N_JOBS=${P_VALUES[$P_IDX]}

RESULTS_DIR="results/jobs"
mkdir -p "${RESULTS_DIR}"

JOB_ID=${SLURM_JOB_ID:-local}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_FILE="${RESULTS_DIR}/knn_jobs_p${N_JOBS}_f${FACTOR}_${JOB_ID}_${TASK_ID}.jsonl"

echo "[INFO] experiment=jobs factor=${FACTOR} n_jobs=${N_JOBS} output=${OUTPUT_FILE}"

srun "${PYTHON_BIN}" knn.py \
  --experiment jobs \
  --factor "${FACTOR}" \
  --n_neighbors "${N_NEIGHBORS}" \
  --n_reps "${N_REPS}" \
  --test_size "${TEST_SIZE}" \
  --output "${OUTPUT_FILE}"
