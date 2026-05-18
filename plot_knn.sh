#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:-${SCRIPT_DIR}/results}"
PLOTS_DIR="${2:-${SCRIPT_DIR}/plots}"
TIME_METRIC="${3:-total_time_s_avg}"
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


def configure_panel(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("p (#hilos / n_jobs)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)


def panel_legend(ax, ncol=2):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=ncol,
        fontsize=8,
        frameon=True,
    )


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
    if experiment in {"samples", "features"}:
        rows_by_experiment[experiment].append(row)
    else:
        print(f"[WARN] Ignorando fila sin experimento valido: {row.get('_source_file', '')}", file=sys.stderr)

panel_specs = {
    "samples": {
        "title": "Muestras replicadas",
        "label": lambda row: f"f={int(row['factor'])} (n={int(row['n_train_used'])})",
    },
    "features": {
        "title": "Atributos replicados",
        "label": lambda row: f"f={int(row['factor'])} (d={int(row['d_used'])})",
    },
}

aggregated = {}
for experiment in ("samples", "features"):
    aggregated[experiment] = aggregate_rows(
        rows_by_experiment.get(experiment, []),
        ["experiment", "factor", "n_jobs", "n_train_used", "n_test_used", "d_used"],
    )

summary_rows = []
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Time figure
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), dpi=150)
has_any = False
for ax, experiment in zip(axes, ("samples", "features")):
    points = aggregated[experiment]
    configure_panel(ax, panel_specs[experiment]["title"], f"Tiempo ({time_metric}) [s]")
    if not points:
        ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes, ha="center", va="center")
        continue

    has_any = True
    factors = sorted({int(row["factor"]) for row in points})
    max_factor = max(factors)
    max_overhead = (math.nan, math.nan)
    max_label = ""

    for idx, factor in enumerate(factors):
        factor_rows = [row for row in points if int(row["factor"]) == factor]
        factor_rows.sort(key=lambda row: int(row["n_jobs"]))
        if not any(int(row["n_jobs"]) == 1 for row in factor_rows):
            continue

        color = colors[idx % len(colors)]
        xs = [int(row["n_jobs"]) for row in factor_rows]
        ys = [float(row["time_s"]) for row in factor_rows]
        t1 = next(float(row["time_s"]) for row in factor_rows if int(row["n_jobs"]) == 1)
        ideal_times = [t1 / p for p in xs]

        ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=panel_specs[experiment]["label"](factor_rows[0]))
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
                    "experiment": experiment,
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
    panel_legend(ax, ncol=2)

fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
if has_any:
    fig.savefig(plots_dir / "knn_time.png")
plt.close(fig)

# Speedup figure
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), dpi=150)
has_any = False
for ax, experiment in zip(axes, ("samples", "features")):
    points = aggregated[experiment]
    configure_panel(ax, panel_specs[experiment]["title"], "Speedup S = T1 / Tp")
    if not points:
        ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes, ha="center", va="center")
        continue

    has_any = True
    p_union = sorted({int(row["n_jobs"]) for row in points})
    ax.plot(p_union, p_union, linestyle="--", linewidth=2.0, color="black", label="ideal S = p")

    factors = sorted({int(row["factor"]) for row in points})
    for idx, factor in enumerate(factors):
        factor_rows = [row for row in points if int(row["factor"]) == factor]
        factor_rows.sort(key=lambda row: int(row["n_jobs"]))
        if not any(int(row["n_jobs"]) == 1 for row in factor_rows):
            continue

        color = colors[idx % len(colors)]
        xs = [int(row["n_jobs"]) for row in factor_rows]
        ys = [float(row["time_s"]) for row in factor_rows]
        t1 = next(float(row["time_s"]) for row in factor_rows if int(row["n_jobs"]) == 1)
        speedups = [t1 / y if y > 0 else math.nan for y in ys]
        ax.plot(xs, speedups, marker="o", linewidth=1.8, color=color, label=panel_specs[experiment]["label"](factor_rows[0]))

    panel_legend(ax, ncol=2)

fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
if has_any:
    fig.savefig(plots_dir / "knn_speedup.png")
plt.close(fig)

# Efficiency figure
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), dpi=150)
has_any = False
for ax, experiment in zip(axes, ("samples", "features")):
    points = aggregated[experiment]
    configure_panel(ax, panel_specs[experiment]["title"], "Eficiencia E = S / p")
    if not points:
        ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes, ha="center", va="center")
        continue

    has_any = True
    p_union = sorted({int(row["n_jobs"]) for row in points})
    ax.plot(p_union, [1.0 for _ in p_union], linestyle="--", linewidth=2.0, color="black", label="ideal E = 1")

    factors = sorted({int(row["factor"]) for row in points})
    for idx, factor in enumerate(factors):
        factor_rows = [row for row in points if int(row["factor"]) == factor]
        factor_rows.sort(key=lambda row: int(row["n_jobs"]))
        if not any(int(row["n_jobs"]) == 1 for row in factor_rows):
            continue

        color = colors[idx % len(colors)]
        xs = [int(row["n_jobs"]) for row in factor_rows]
        ys = [float(row["time_s"]) for row in factor_rows]
        t1 = next(float(row["time_s"]) for row in factor_rows if int(row["n_jobs"]) == 1)
        speedups = [t1 / y if y > 0 else math.nan for y in ys]
        efficiencies = [s / p if math.isfinite(s) else math.nan for s, p in zip(speedups, xs)]
        ax.plot(xs, efficiencies, marker="o", linewidth=1.8, color=color, label=panel_specs[experiment]["label"](factor_rows[0]))

    ax.set_ylim(bottom=0)
    panel_legend(ax, ncol=2)

fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
if has_any:
    fig.savefig(plots_dir / "knn_efficiency.png")
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
