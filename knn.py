from pathlib import Path
from joblib import dump, load, parallel_backend
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Evitar sobre-suscripcion BLAS: el paralelismo se controla con N_JOBS/joblib.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DATASET_ID = 45
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / f"ucimlrepo_{DATASET_ID}.joblib"
EXPERIMENTS = ("samples", "features", "jobs")


def approx_mem_bytes(*arrays):
    return int(sum(np.asarray(arr).nbytes for arr in arrays))


def load_dataset_with_cache(dataset_id=DATASET_ID, cache_path=CACHE_PATH):
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            cached = load(cache_path)
            print(f"[INFO] Dataset {dataset_id} cargado desde cache: {cache_path}", file=sys.stderr)
            return np.asarray(cached["X"]), np.asarray(cached["y"]).ravel()
        except Exception as exc:
            print(
                f"[WARN] No se pudo leer el cache {cache_path} ({exc}). "
                "Se descargará nuevamente.",
                file=sys.stderr,
            )

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=dataset_id)
    X = np.asarray(dataset.data.features)
    y = dataset.data.targets.to_numpy().ravel()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f"{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp"

    try:
        dump({"X": X, "y": y}, tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"[INFO] Dataset {dataset_id} descargado y guardado en cache: {cache_path}", file=sys.stderr)
    return X, y


def parse_args():
    parser = argparse.ArgumentParser(description="KNN scalability experiments")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True, help="Experiment to run")
    parser.add_argument("--factor", type=int, default=1, help="Replication factor f")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--n_reps", type=int, default=30, help="Number of repetitions")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test size fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results.jsonl", help="Output JSONL file")
    return parser.parse_args()


def split_dataset(X, y, test_size, random_state):
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        print("[WARN] No se pudo usar stratify=y; se realiza split sin estratificacion.", file=sys.stderr)
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )


def build_experiment_data(experiment, factor, X_train, X_test, y_train, y_test):
    if factor < 1:
        raise ValueError(f"El factor debe ser >= 1. Recibido: {factor}")

    if experiment in {"samples", "jobs"}:
        X_train_exp = np.tile(X_train, (factor, 1))
        y_train_exp = np.tile(y_train, factor)
        X_test_exp = X_test
        y_test_exp = y_test
        applied_factor = factor
    elif experiment == "features":
        X_train_exp = np.tile(X_train, (1, factor))
        X_test_exp = np.tile(X_test, (1, factor))
        y_train_exp = y_train
        y_test_exp = y_test
        applied_factor = factor
    else:
        raise ValueError(f"Experimento desconocido: {experiment}")

    return X_train_exp, X_test_exp, y_train_exp, y_test_exp, applied_factor


def main():
    args = parse_args()
    n_jobs = int(os.environ.get("N_JOBS", 1))

    X, y = load_dataset_with_cache()
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    base_n_train = int(X_train.shape[0])
    base_n_test = int(X_test.shape[0])
    base_d = int(X_train.shape[1])

    X_train_exp, X_test_exp, y_train_exp, y_test_exp, applied_factor = build_experiment_data(
        args.experiment,
        args.factor,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    n_train_used = int(X_train_exp.shape[0])
    n_test_used = int(X_test_exp.shape[0])
    d_used = int(X_train_exp.shape[1])

    mem_bytes = approx_mem_bytes(X_train_exp, X_test_exp, y_train_exp, y_test_exp)
    print(
        f"[INFO] experiment={args.experiment} factor={applied_factor} n_jobs={n_jobs} "
        f"base_train={base_n_train} base_test={base_n_test} base_d={base_d}",
        file=sys.stderr,
    )
    print(
        f"[INFO] transformed train={n_train_used} test={n_test_used} d={d_used} "
        f"mem~{mem_bytes / 1e6:.2f} MB",
        file=sys.stderr,
    )

    fitting_times = []
    prediction_times = []
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []

    for _ in range(args.n_reps):
        knn = KNeighborsClassifier(
            algorithm="brute",
            n_neighbors=args.n_neighbors,
            n_jobs=n_jobs,
        )

        t_fit_start = time.perf_counter()
        with parallel_backend("loky", n_jobs=n_jobs):
            knn.fit(X_train_exp, y_train_exp)
        fit_time = time.perf_counter() - t_fit_start
        fitting_times.append(fit_time)

        t_pred_start = time.perf_counter()
        with parallel_backend("loky", n_jobs=n_jobs):
            y_pred = knn.predict(X_test_exp)
        pred_time = time.perf_counter() - t_pred_start
        prediction_times.append(pred_time)

        f1_scores.append(f1_score(y_test_exp, y_pred, average="weighted", zero_division=0))
        accuracies.append(accuracy_score(y_test_exp, y_pred))
        precisions.append(precision_score(y_test_exp, y_pred, average="weighted", zero_division=0))
        recalls.append(recall_score(y_test_exp, y_pred, average="weighted", zero_division=0))

    fit_avg = float(np.mean(fitting_times))
    pred_avg = float(np.mean(prediction_times))
    total_avg = fit_avg + pred_avg

    results = {
        "dataset_id": DATASET_ID,
        "experiment": args.experiment,
        "factor": applied_factor,
        "n_jobs": n_jobs,
        "n_neighbors": args.n_neighbors,
        "reps": args.n_reps,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "base_n_train": base_n_train,
        "base_n_test": base_n_test,
        "base_d": base_d,
        "n_train_used": n_train_used,
        "n_test_used": n_test_used,
        "d_used": d_used,
        "fit_time_s_avg": round(fit_avg, 6),
        "pred_time_s_avg": round(pred_avg, 6),
        "total_time_s_avg": round(total_avg, 6),
        "f1_avg": round(float(np.mean(f1_scores)), 6),
        "accuracy_avg": round(float(np.mean(accuracies)), 6),
        "precision_avg": round(float(np.mean(precisions)), 6),
        "recall_avg": round(float(np.mean(recalls)), 6),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")

    print(
        f"[OK] experiment={args.experiment} factor={applied_factor} n_jobs={n_jobs} "
        f"train={n_train_used} test={n_test_used} d={d_used} "
        f"fit_avg={fit_avg:.4f}s pred_avg={pred_avg:.4f}s total_avg={total_avg:.4f}s"
    )


if __name__ == "__main__":
    main()
