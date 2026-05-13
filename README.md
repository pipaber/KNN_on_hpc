# KNN on HPC — Scalability Study

Scalability analysis of a **K-Nearest Neighbors (KNN)** classifier running on a High-Performance Computing (HPC) cluster managed with **SLURM**. The experiment evaluates strong scaling, weak scaling, and a full parameter grid using the [Heart Disease dataset (UCI id=45)](https://archive.ics.uci.edu/dataset/45/heart+disease).

---

## Project Structure

```
KNN_on_hpc/
├── knn.py          # KNN classifier with dataset caching and metrics
├── knn.sh          # SLURM batch script (144-job array)
├── plot_knn.sh     # Script to generate scalability plots from results
├── logs/           # SLURM stdout/stderr logs
├── results/        # Output .jsonl files (one per job)
├── plots/          # Generated PNG plots and summary CSV
└── .cache/         # Cached UCI dataset (auto-generated)
```

---

## Dataset

| Property     | Value                        |
|--------------|------------------------------|
| Source       | UCI ML Repository            |
| Dataset ID   | 45 (Heart Disease)           |
| Samples      | 303                          |
| Features     | 13                           |

The dataset is automatically downloaded on first run and cached locally under `.cache/` to avoid redundant downloads in subsequent SLURM tasks.

---

## Experiment Design

The experiment sweeps a full grid of parameters to study:

| Axis               | Values                          |
|--------------------|---------------------------------|
| **Processes (p)**  | 1, 2, 4, 8, 16, 32             |
| **Samples (N)**    | 53, 75, 106, 150, 212, 300     |
| **Neighbors (k)**  | 1, 3, 5, 11                    |
| **Features**       | 13 (fixed)                     |
| **Repetitions**    | 30 per combination             |
| **Test size**      | 30%                            |

**Total combinations:** 6 × 6 × 4 = **144 SLURM array jobs**

### Scaling Tags

Each job is automatically labeled:
- `strong` — maximum N (300) with varying p
- `weak` — diagonal combinations where `S_IDX == J_IDX` (N ≈ N_base × √p)
- `grid` — all remaining combinations

---

## Requirements

- Python ≥ 3.9
- SLURM (for HPC execution)
- Python packages:

```
numpy
scikit-learn
joblib
matplotlib
ucimlrepo
```

Install dependencies:

```
pip install numpy scikit-learn joblib matplotlib ucimlrepo
```

Or with a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn joblib matplotlib ucimlrepo
```

---

## Usage

### 1. Submit SLURM job array

```bash
sbatch knn.sh
```

This submits 144 jobs (indices 0–143). Each job writes its results to a `.jsonl` file inside `results/`.

### 2. Run locally (single combination)

```bash
export N_JOBS=4
python3 knn.py \
  --n_samples 300 \
  --n_features 13 \
  --n_neighbors 5 \
  --n_reps 10 \
  --test_size 0.3 \
  --scaling_tag manual \
  --output results/local_test.jsonl
```

### 3. Generate plots

```bash
bash plot_knn.sh [results_dir] [plots_dir] [time_metric]
```

Default values:
- `results_dir` → `./results`
- `plots_dir` → `./plots`
- `time_metric` → `pred_time_s_avg`

Available metrics: `pred_time_s_avg`, `fit_time_s_avg`, `total_time_s_avg`

Example:

```bash
bash plot_knn.sh results plots pred_time_s_avg
```

---

## Output

### Results (`.jsonl`)

Each line in a result file is a JSON object with fields including:

| Field              | Description                        |
|--------------------|------------------------------------|
| `n_jobs`           | Number of parallel workers         |
| `n_neighbors`      | K value used                       |
| `n_samples`        | Samples used                       |
| `n_features`       | Features used                      |
| `fit_time_s_avg`   | Average fit time (seconds)         |
| `pred_time_s_avg`  | Average prediction time (seconds)  |
| `accuracy_avg`     | Average accuracy                   |
| `f1_avg`           | Average F1-score (weighted)        |
| `precision_avg`    | Average precision (weighted)       |
| `recall_avg`       | Average recall (weighted)          |
| `scaling_tag`      | `strong`, `weak`, or `grid`        |

### Plots

For each value of k, three PNG plots are generated:

- `knn_time_k{k}.png` — Execution time vs. number of processes
- `knn_speedup_k{k}.png` — Speedup (S = T1/Tp) vs. number of processes
- `knn_efficiency_k{k}.png` — Efficiency (E = S/p) vs. number of processes

A summary CSV (`knn_scalability_summary.csv`) is also saved in `plots/`.

---

## Key Implementation Details

- **BLAS over-subscription prevention:** Environment variables `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, etc. are all set to `1` so that parallelism is controlled exclusively via `N_JOBS` (joblib/loky).
- **Caching:** The UCI dataset is cached using `joblib.dump` with atomic writes (`os.replace`) to avoid race conditions in parallel SLURM jobs.
- **Metrics:** Each repetition records fit time, prediction time, accuracy, F1, precision, and recall. Averages are reported.

---

## SLURM Configuration

| Parameter         | Value          |
|-------------------|----------------|
| `--cpus-per-task` | 32             |
| `--time`          | 00:30:00       |
| `--nodes`         | 1              |
| `--ntasks`        | 1              |
| `--array`         | 0–143          |

---

## License

This project is for academic/research purposes.
