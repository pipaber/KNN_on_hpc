#!/bin/bash
set -euo pipefail

cat <<'EOF'
Este proyecto ahora separa los experimentos por script SLURM:

  sbatch knn_samples.sh
  sbatch knn_features.sh
  sbatch knn_jobs.sh

Si quieres correr Python directamente:

  export N_JOBS=1
  python3 knn.py --experiment samples --factor 4 --output results/manual.jsonl
EOF
