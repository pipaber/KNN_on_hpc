#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:-${SCRIPT_DIR}/results}"
PLOTS_DIR="${2:-${SCRIPT_DIR}/plots}"
TIME_METRIC="${3:-pred_time_s_avg}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p "${PLOTS_DIR}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
  elif [ -f "${SCRIPT_DIR}/../.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/../.venv/bin/activate"
  fi
fi

"${PYTHON_BIN}" - "${RESULTS_DIR}" "${PLOTS_DIR}" "${TIME_METRIC}" <<'PY'
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


results_dir = Path(sys.argv[1])
plots_dir = Path(sys.argv[2])
time_metric = sys.argv[3]
valid_metrics = {"fit_time_s_avg", "pred_time_s_avg", "total_time_s_avg"}

if time_metric not in valid_metrics:
    print(f"[ERROR] Metrica invalida: {time_metric}. Use una de {sorted(valid_metrics)}", file=sys.stderr)
    sys.exit(2)


def median(values):
    return float(statistics.median(values))


def read_jsonl_results(path):
    rows = []
    for file_path in sorted(path.rglob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Ignorando JSON invalido {file_path}:{line_no}: {exc}", file=sys.stderr)
                    continue
                row["_source_file"] = str(file_path)
                rows.append(row)
    return rows


def aggregate_rows(rows, key_fields):
    grouped_times = defaultdict(list)
    metadata = {}
    for row in rows:
        try:
            key = tuple(row[field] for field in key_fields)
            grouped_times[key].append(float(row[time_metric]))
            metadata[key] = row
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[WARN] Ignorando fila incompleta ({exc}): {row.get('_source_file', '')}", file=sys.stderr)
    aggregated = []
    for key, values in grouped_times.items():
        merged = dict(metadata[key])
        merged["time_s"] = median(values)
        aggregated.append(merged)
    return aggregated


def configure_xy(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)


def configure_process_axis(ax, title, ylabel):
    configure_xy(ax, title, "p (#hilos / n_jobs)", ylabel)
    ax.set_xscale("log", base=2)


def add_overhead_box(ax, label, overhead_time, overhead_fraction):
    if not math.isfinite(overhead_time):
        return
    ax.text(
        0.02,
        0.04,
        f"{label}\nOverhead = {overhead_time:.4g}s\nOverhead/Tp = {overhead_fraction:.1%}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
    )


rows = read_jsonl_results(results_dir)
if not rows:
    print(f"[ERROR] No se encontraron archivos .jsonl en {results_dir}", file=sys.stderr)
    sys.exit(1)

rows_by_experiment = defaultdict(list)
for row in rows:
    experiment = row.get("experiment")
    if experiment in {"samples", "features", "jobs"}:
        rows_by_experiment[experiment].append(row)
    else:
        print(
            f"[WARN] Ignorando fila sin experimento valido: {row.get('_source_file', '')}",
            file=sys.stderr,
        )

summary_rows = []
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# Samples: T ~ O(n_train) with d and p fixed.
sample_points = aggregate_rows(
    rows_by_experiment.get("samples", []),
    ["experiment", "factor", "n_jobs", "n_train_used", "n_test_used", "d_used"],
)
if sample_points:
    sample_points.sort(key=lambda row: int(row["factor"]))
    base_row = next((row for row in sample_points if int(row["factor"]) == 1), sample_points[0])
    base_n = int(base_row["n_train_used"])
    base_t = float(base_row["time_s"])

    xs = [int(row["n_train_used"]) for row in sample_points]
    ys = [float(row["time_s"]) for row in sample_points]
    ideal = [base_t * (x / base_n) for x in xs]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    configure_xy(ax, "KNN - tiempo vs n_train replicado", "n_train", f"Tiempo ({time_metric}) [s]")
    ax.plot(xs, ys, marker="o", linewidth=1.8, label="medido")
    ax.plot(xs, ideal, linestyle="--", linewidth=2.0, color="black", label="teoria lineal")
    overhead_time = ys[-1] - ideal[-1]
    overhead_fraction = overhead_time / ys[-1] if ys[-1] > 0 else math.nan
    add_overhead_box(ax, f"f={int(sample_points[-1]['factor'])}", overhead_time, overhead_fraction)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "knn_samples_time.png")
    plt.close(fig)

    for row, ideal_time in zip(sample_points, ideal):
        time_s = float(row["time_s"])
        summary_rows.append(
            {
                "experiment": "samples",
                "factor": int(row["factor"]),
                "n_jobs": int(row["n_jobs"]),
                "n_train_used": int(row["n_train_used"]),
                "n_test_used": int(row["n_test_used"]),
                "d_used": int(row["d_used"]),
                "time_metric": time_metric,
                "time_s": time_s,
                "t1_s": base_t,
                "ideal_time_s": ideal_time,
                "speedup": math.nan,
                "efficiency": math.nan,
                "overhead_time_s": time_s - ideal_time,
                "overhead_fraction": (time_s - ideal_time) / time_s if time_s > 0 else math.nan,
            }
        )


# Features: T ~ O(d) with n_train and p fixed.
feature_points = aggregate_rows(
    rows_by_experiment.get("features", []),
    ["experiment", "factor", "n_jobs", "n_train_used", "n_test_used", "d_used"],
)
if feature_points:
    feature_points.sort(key=lambda row: int(row["factor"]))
    base_row = next((row for row in feature_points if int(row["factor"]) == 1), feature_points[0])
    base_d = int(base_row["d_used"])
    base_t = float(base_row["time_s"])

    xs = [int(row["d_used"]) for row in feature_points]
    ys = [float(row["time_s"]) for row in feature_points]
    ideal = [base_t * (x / base_d) for x in xs]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    configure_xy(ax, "KNN - tiempo vs atributos replicados", "d (#atributos)", f"Tiempo ({time_metric}) [s]")
    ax.plot(xs, ys, marker="o", linewidth=1.8, label="medido")
    ax.plot(xs, ideal, linestyle="--", linewidth=2.0, color="black", label="teoria lineal")
    overhead_time = ys[-1] - ideal[-1]
    overhead_fraction = overhead_time / ys[-1] if ys[-1] > 0 else math.nan
    add_overhead_box(ax, f"f={int(feature_points[-1]['factor'])}", overhead_time, overhead_fraction)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "knn_features_time.png")
    plt.close(fig)

    for row, ideal_time in zip(feature_points, ideal):
        time_s = float(row["time_s"])
        summary_rows.append(
            {
                "experiment": "features",
                "factor": int(row["factor"]),
                "n_jobs": int(row["n_jobs"]),
                "n_train_used": int(row["n_train_used"]),
                "n_test_used": int(row["n_test_used"]),
                "d_used": int(row["d_used"]),
                "time_metric": time_metric,
                "time_s": time_s,
                "t1_s": base_t,
                "ideal_time_s": ideal_time,
                "speedup": math.nan,
                "efficiency": math.nan,
                "overhead_time_s": time_s - ideal_time,
                "overhead_fraction": (time_s - ideal_time) / time_s if time_s > 0 else math.nan,
            }
        )


# Jobs: same plot family for all factors; interpretation of strong/weak is left to the curves.
job_points = aggregate_rows(
    rows_by_experiment.get("jobs", []),
    ["experiment", "factor", "n_jobs", "n_train_used", "n_test_used", "d_used"],
)
if job_points:
    factors = sorted({int(row["factor"]) for row in job_points})
    points_by_factor = {}
    for factor in factors:
        factor_rows = [row for row in job_points if int(row["factor"]) == factor]
        factor_rows.sort(key=lambda row: int(row["n_jobs"]))
        if any(int(row["n_jobs"]) == 1 for row in factor_rows):
            points_by_factor[factor] = factor_rows

    if points_by_factor:
        fig, ax = plt.subplots(figsize=(7.4, 4.9), dpi=150)
        configure_process_axis(ax, "KNN - tiempo vs n_jobs", f"Tiempo ({time_metric}) [s]")

        max_factor = max(points_by_factor)
        max_overhead = (math.nan, math.nan)
        max_label = ""

        for idx, factor in enumerate(sorted(points_by_factor)):
            factor_rows = points_by_factor[factor]
            color = colors[idx % len(colors)]
            xs = [int(row["n_jobs"]) for row in factor_rows]
            ys = [float(row["time_s"]) for row in factor_rows]
            t1 = next(float(row["time_s"]) for row in factor_rows if int(row["n_jobs"]) == 1)
            ideal_times = [t1 / p for p in xs]
            n_train = int(factor_rows[0]["n_train_used"])
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=f"f={factor} (n={n_train})")
            ax.plot(xs, ideal_times, linestyle="--", linewidth=1.2, color=color, alpha=0.75)

            if factor == max_factor:
                overhead_time = ys[-1] - ideal_times[-1]
                overhead_fraction = overhead_time / ys[-1] if ys[-1] > 0 else math.nan
                max_overhead = (overhead_time, overhead_fraction)
                max_label = f"f={factor}, p={xs[-1]}"

            for row, ideal_time in zip(factor_rows, ideal_times):
                tp = float(row["time_s"])
                speedup = t1 / tp if tp > 0 else math.nan
                efficiency = speedup / int(row["n_jobs"]) if math.isfinite(speedup) else math.nan
                summary_rows.append(
                    {
                        "experiment": "jobs",
                        "factor": int(row["factor"]),
                        "n_jobs": int(row["n_jobs"]),
                        "n_train_used": int(row["n_train_used"]),
                        "n_test_used": int(row["n_test_used"]),
                        "d_used": int(row["d_used"]),
                        "time_metric": time_metric,
                        "time_s": tp,
                        "t1_s": t1,
                        "ideal_time_s": ideal_time,
                        "speedup": speedup,
                        "efficiency": efficiency,
                        "overhead_time_s": tp - ideal_time,
                        "overhead_fraction": (tp - ideal_time) / tp if tp > 0 else math.nan,
                    }
                )

        add_overhead_box(ax, max_label, max_overhead[0], max_overhead[1])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "knn_jobs_time.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 4.9), dpi=150)
        configure_process_axis(ax, "KNN - speedup vs n_jobs", "Speedup S = T1 / Tp")

        p_union = sorted({int(row["n_jobs"]) for rows_ in points_by_factor.values() for row in rows_})
        ax.plot(p_union, p_union, linestyle="--", linewidth=2.0, color="black", label="ideal S = p")

        for idx, factor in enumerate(sorted(points_by_factor)):
            factor_rows = points_by_factor[factor]
            color = colors[idx % len(colors)]
            xs = [int(row["n_jobs"]) for row in factor_rows]
            ys = [float(row["time_s"]) for row in factor_rows]
            t1 = next(float(row["time_s"]) for row in factor_rows if int(row["n_jobs"]) == 1)
            speedups = [t1 / y if y > 0 else math.nan for y in ys]
            ax.plot(xs, speedups, marker="o", linewidth=1.8, color=color, label=f"f={factor}")

        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "knn_jobs_speedup.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 4.9), dpi=150)
        configure_process_axis(ax, "KNN - eficiencia vs n_jobs", "Eficiencia E = S / p")
        ax.plot(p_union, [1.0 for _ in p_union], linestyle="--", linewidth=2.0, color="black", label="ideal E = 1")

        for idx, factor in enumerate(sorted(points_by_factor)):
            factor_rows = points_by_factor[factor]
            color = colors[idx % len(colors)]
            xs = [int(row["n_jobs"]) for row in factor_rows]
            ys = [float(row["time_s"]) for row in factor_rows]
            t1 = next(float(row["time_s"]) for row in factor_rows if int(row["n_jobs"]) == 1)
            speedups = [t1 / y if y > 0 else math.nan for y in ys]
            efficiencies = [s / p if math.isfinite(s) else math.nan for s, p in zip(speedups, xs)]
            ax.plot(xs, efficiencies, marker="o", linewidth=1.8, color=color, label=f"f={factor}")

        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "knn_jobs_efficiency.png")
        plt.close(fig)


summary_csv = plots_dir / "knn_scalability_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "experiment",
        "factor",
        "n_jobs",
        "n_train_used",
        "n_test_used",
        "d_used",
        "time_metric",
        "time_s",
        "t1_s",
        "ideal_time_s",
        "speedup",
        "efficiency",
        "overhead_time_s",
        "overhead_fraction",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow(row)

print(f"[OK] Filas leidas: {len(rows)}")
print(f"[OK] Experimentos detectados: {sorted(rows_by_experiment)}")
print(f"[OK] Resumen CSV: {summary_csv}")
print(f"[OK] Plots escritos en: {plots_dir}")
print(f"[INFO] Metrica de tiempo usada: {time_metric}")
PY
