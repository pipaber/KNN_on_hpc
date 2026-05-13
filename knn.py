from pathlib import Path
from joblib import parallel_backend, dump, load
import os, time, sys, argparse, json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from ucimlrepo import fetch_ucirepo

# ===== Evitar sobre-suscripción BLAS =====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DATASET_ID = 45
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / f"ucimlrepo_{DATASET_ID}.joblib"

def approx_mem_bytes(n_samples, n_features, dtype_bytes=8):
    # X: n_samples * n_features, y: n_samples (int64 ~8B), factor ~2.5 por copias/overhead
    return int(2.5 * (n_samples * n_features * dtype_bytes + n_samples * 8))

def load_dataset_with_cache(dataset_id=DATASET_ID, cache_path=CACHE_PATH):
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            cached = load(cache_path)
            print(f"[INFO] Dataset {dataset_id} cargado desde cache: {cache_path}", file=sys.stderr)
            return cached["X"], np.asarray(cached["y"]).ravel()
        except Exception as exc:
            print(
                f"[WARN] No se pudo leer el cache {cache_path} ({exc}). "
                "Se descargará nuevamente.",
                file=sys.stderr,
            )

    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
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
    parser = argparse.ArgumentParser(description="KNN classifier")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=100, help="Number of features")
    parser.add_argument("--n_reps", type=int, default=10, help="Number of iterations")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size")
    parser.add_argument("--scaling_tag", type=str, default="", help="Scaling experiment label")
    parser.add_argument("--output", type=str, default="results.jsonl", help="Output file")

    return parser.parse_args()

def main():
    args = parse_args()

    # Paralelismo solo desde N_JOBS
    n_jobs = int(os.environ.get("N_JOBS", 1))

    # Load dataset
    X, y = load_dataset_with_cache()

    total_n_samples, total_n_features = X.shape
    effective_n_samples = min(args.n_samples, total_n_samples)
    effective_n_features = min(args.n_features, total_n_features)

    if effective_n_samples != args.n_samples or effective_n_features != args.n_features:
        print(
            "[WARN] Dataset id=45 no tiene el tamaño solicitado. "
            f"Usando n_samples={effective_n_samples} y n_features={effective_n_features} "
            f"(disponibles: {total_n_samples}, {total_n_features}).",
            file=sys.stderr,
        )

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(total_n_samples, size=effective_n_samples, replace=False)
    X = X.iloc[sample_idx, :effective_n_features]
    y = y[sample_idx]

    memB = approx_mem_bytes(effective_n_samples, effective_n_features)
    print(f"[INFO] Aproximación de uso de memoria (dataset) ~ {memB/1e9:.2f} GB", file=sys.stderr)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # Model
    knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, n_jobs=n_jobs)

    fitting_times, prediction_times = [], []
    f1_scores, accuracies, precisions, recalls = [], [], [], []

    for _ in range(args.n_reps):
        # Training
        t_0 = time.perf_counter()

        with parallel_backend("loky", n_jobs=n_jobs):
            knn.fit(X_train, y_train)

        fit_t = time.perf_counter() - t_0
        fitting_times.append(fit_t)

        # Prediction

        t_1 = time.perf_counter()
        y_pred = knn.predict(X_test)
        pred_t = time.perf_counter() - t_1
        prediction_times.append(pred_t)

        # Metrics
        f1_scores.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))

    fitting_avg = float(np.mean(fitting_times))
    prediction_avg = float(np.mean(prediction_times))
    f1_avg = float(np.mean(f1_scores))
    accuracy_avg = float(np.mean(accuracies))
    precision_avg = float(np.mean(precisions))
    recall_avg = float(np.mean(recalls))

    # Save Results to .jsonl format to output

    results = {
        "dataset_id": DATASET_ID,
        "n_jobs": n_jobs,
        "n_neighbors": args.n_neighbors,
        "n_samples": effective_n_samples,
        "n_features": effective_n_features,
        "reps": args.n_reps,
        "requested_n_samples": args.n_samples,
        "requested_n_features": args.n_features,
        "n_samples_used": effective_n_samples,
        "n_features_used": effective_n_features,
        "test_size": args.test_size,
        "scaling_tag": args.scaling_tag,
        "fit_time_s_avg": round(fitting_avg, 4),
        "pred_time_s_avg": round(prediction_avg, 4),
        "f1_avg": round(f1_avg, 4),
        "accuracy_avg": round(accuracy_avg, 4),
        "precision_avg": round(precision_avg, 4),
        "recall_avg": round(recall_avg, 4),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }

    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")

    print(
        f"[OK] n_jobs={n_jobs} ns={effective_n_samples} nf={effective_n_features} "
        f"fit_avg={fitting_avg:.3f}s pred_avg={prediction_avg:.3f}s "
        f"acc_avg={accuracy_avg:.4f} f1_avg={f1_avg:.4f} "
        f"precision_avg={precision_avg:.4f} recall_avg={recall_avg:.4f}"
    )

if __name__ == "__main__":
    main()
