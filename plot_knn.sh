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


def get_time_value(row):
    if time_metric == "total_time_s_avg":
        return float(row["fit_time_s_avg"]) + float(row["pred_time_s_avg"])
    return float(row[time_metric])


def median(values):
    return float(statistics.median(values))


def configure_axis(ax, ylabel, title):
    ax.set_title(title)
    ax.set_xlabel("p (#procesos)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)


def add_overhead_box(ax, summary_rows, k):
    rows = [r for r in summary_rows if r["n_neighbors"] == k and math.isfinite(r["overhead_time_s"])]
    if not rows:
        return

    max_p = max(r["n_jobs"] for r in rows)
    max_p_rows = [r for r in rows if r["n_jobs"] == max_p]
    if not max_p_rows:
        return

    ov_values = [r["overhead_time_s"] for r in max_p_rows if math.isfinite(r["overhead_time_s"])]
    ov_frac_values = [r["overhead_fraction"] for r in max_p_rows if math.isfinite(r["overhead_fraction"])]
    if not ov_values or not ov_frac_values:
        return

    ov = median(ov_values)
    ov_frac = median(ov_frac_values)
    text = (
        f"Overhead @ p={max_p}\n"
        f"median Tp-T1/p = {ov:.4g}s\n"
        f"median overhead/Tp = {ov_frac:.1%}"
    )
    ax.text(
        0.02,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
    )


rows = read_jsonl_results(results_dir)
if not rows:
    print(f"[ERROR] No se encontraron archivos .jsonl en {results_dir}", file=sys.stderr)
    sys.exit(1)

grouped_times = defaultdict(list)
skipped = 0

for row in rows:
    try:
        key = (int(row["n_neighbors"]), int(row["n_samples"]), int(row["n_jobs"]))
        grouped_times[key].append(get_time_value(row))
    except (KeyError, TypeError, ValueError) as exc:
        skipped += 1
        print(f"[WARN] Ignorando fila incompleta ({exc}): {row.get('_source_file', '')}", file=sys.stderr)

if not grouped_times:
    print("[ERROR] No hay filas validas para graficar.", file=sys.stderr)
    sys.exit(1)

time_by_key = {key: median(values) for key, values in grouped_times.items()}
k_values = sorted({key[0] for key in time_by_key})
n_values = sorted({key[1] for key in time_by_key})
p_values = sorted({key[2] for key in time_by_key})

summary_rows = []

for k in k_values:
    for n in n_values:
        t1 = time_by_key.get((k, n, 1))
        for p in p_values:
            tp = time_by_key.get((k, n, p))
            if tp is None:
                continue

            if t1 is None:
                speedup = math.nan
                efficiency = math.nan
                ideal_time = math.nan
                overhead_time = math.nan
                overhead_work = math.nan
                overhead_fraction = math.nan
            else:
                speedup = t1 / tp if tp > 0 else math.nan
                efficiency = speedup / p if math.isfinite(speedup) else math.nan
                ideal_time = t1 / p
                overhead_time = tp - ideal_time
                overhead_work = p * tp - t1
                overhead_fraction = overhead_time / tp if tp > 0 else math.nan

            summary_rows.append(
                {
                    "n_neighbors": k,
                    "n_samples": n,
                    "n_jobs": p,
                    "time_metric": time_metric,
                    "time_s": tp,
                    "t1_s": t1 if t1 is not None else math.nan,
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "ideal_time_1_over_p_s": ideal_time,
                    "overhead_time_s": overhead_time,
                    "overhead_work_s": overhead_work,
                    "overhead_fraction": overhead_fraction,
                }
            )

summary_csv = plots_dir / "knn_scalability_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "n_neighbors",
        "n_samples",
        "n_jobs",
        "time_metric",
        "time_s",
        "t1_s",
        "speedup",
        "efficiency",
        "ideal_time_1_over_p_s",
        "overhead_time_s",
        "overhead_work_s",
        "overhead_fraction",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow(row)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for k in k_values:
    k_n_values = sorted({n for kk, n, _ in time_by_key if kk == k})
    k_p_values = sorted({p for kk, _, p in time_by_key if kk == k})
    if not k_p_values:
        continue

    p_min = min(k_p_values)
    p_max = max(k_p_values)
    p_ref = [p for p in k_p_values if p > 0]

    # Execution time plot.
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    configure_axis(ax, f"Tiempo ({time_metric}) [s]", f"KNN - tiempo vs procesos (k={k})")
    ax.set_yscale("log")

    largest_n_with_t1 = None
    for idx, n in enumerate(k_n_values):
        points = [(p, time_by_key[(k, n, p)]) for p in k_p_values if (k, n, p) in time_by_key]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=f"N={n}", color=colors[idx % len(colors)])
        if (k, n, 1) in time_by_key:
            largest_n_with_t1 = n

    if largest_n_with_t1 is not None:
        t1_ref = time_by_key[(k, largest_n_with_t1, 1)]
        ax.plot(
            p_ref,
            [t1_ref / p for p in p_ref],
            linestyle="--",
            linewidth=2.0,
            color="black",
            label=f"ideal T1/p (N={largest_n_with_t1})",
        )

    add_overhead_box(ax, summary_rows, k)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"knn_time_k{k}.png")
    plt.close(fig)

    # Speedup plot.
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    configure_axis(ax, "Speedup S=T1/Tp", f"KNN - speedup vs procesos (k={k})")
    ax.set_yscale("log")

    for idx, n in enumerate(k_n_values):
        t1 = time_by_key.get((k, n, 1))
        if t1 is None:
            continue
        points = [(p, t1 / time_by_key[(k, n, p)]) for p in k_p_values if (k, n, p) in time_by_key]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=f"N={n}", color=colors[idx % len(colors)])

    ax.plot(p_ref, p_ref, linestyle="--", linewidth=2.0, color="black", label="ideal S=p")
    add_overhead_box(ax, summary_rows, k)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"knn_speedup_k{k}.png")
    plt.close(fig)

    # Efficiency plot.
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    configure_axis(ax, "Eficiencia E=S/p", f"KNN - eficiencia vs procesos (k={k})")
    ax.set_yscale("log")

    for idx, n in enumerate(k_n_values):
        t1 = time_by_key.get((k, n, 1))
        if t1 is None:
            continue
        points = [
            (p, (t1 / time_by_key[(k, n, p)]) / p)
            for p in k_p_values
            if (k, n, p) in time_by_key
        ]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=f"N={n}", color=colors[idx % len(colors)])

    ax.plot(p_ref, [1.0 for _ in p_ref], linestyle="--", linewidth=2.0, color="black", label="ideal E=1")
    ax.set_ylim(bottom=max(1e-3, ax.get_ylim()[0]), top=max(1.2, ax.get_ylim()[1]))
    add_overhead_box(ax, summary_rows, k)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"knn_efficiency_k{k}.png")
    plt.close(fig)

print(f"[OK] Filas leidas: {len(rows)}")
print(f"[OK] Filas ignoradas: {skipped}")
print(f"[OK] Resumen CSV: {summary_csv}")
print(f"[OK] Plots escritos en: {plots_dir}")
print(f"[INFO] Metrica de tiempo usada: {time_metric}")
PY
