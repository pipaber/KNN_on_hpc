# KNN on HPC

Estudio del costo computacional de **KNN** sobre el dataset **Heart Disease** de UCI (`id=45`), con foco en tiempo de ejecución, speedup, eficiencia y overhead.

## Estructura

```text
KNN_on_hpc/
├── knn.py
├── knn.sh
├── prepare_cache.sh
├── slurm_env.sh
├── plot_knn.sh
├── data/
├── logs/
├── results/
├── plots/
└── .cache/
```

## Diseño del experimento

Primero se hace un split fijo `70/30` del dataset completo:

- entrenamiento: `212` filas
- prueba: `91` filas
- atributos base: `13`

La grilla SLURM combina dos familias de experimentos:

- `samples`: replica solo `X_train` e `y_train` con factores `1, 2, 4, 8, 16`
- `features`: replica columnas de `X_train` y `X_test` con factores `1, 2, 4, 8`

Ambos casos se ejecutan con:

- `n_jobs = 1, 2, 4, 8, 16, 32`

El objetivo final es producir solo 3 figuras:

- `time`
- `speedup`
- `efficiency`

Cada figura tiene dos paneles:

- panel `samples`
- panel `features`

## Dependencias

Con `uv`:

```bash
uv sync
```

## Dataset

El repo incluye el archivo versionado `data/processed.cleveland.data`, por lo que los compute nodes no necesitan internet para leer el dataset.

Opcionalmente, puede precargar el cache desde el login node:

```bash
bash prepare_cache.sh
```

## Ejecución en HPC

Enviar toda la grilla con un solo script:

```bash
sbatch knn.sh
```

Variables opcionales:

```bash
sbatch --export=ALL,N_REPS=10,N_NEIGHBORS=5 knn.sh
```

La grilla total tiene:

- `5 x 6 = 30` combinaciones para `samples`
- `4 x 6 = 24` combinaciones para `features`
- total `54` tareas en el array

Los resultados quedan separados en:

- `results/samples`
- `results/features`

## Ejecución local

Ejemplo para `samples`:

```bash
export N_JOBS=8
python3 knn.py \
  --experiment samples \
  --factor 4 \
  --n_neighbors 5 \
  --n_reps 5 \
  --test_size 0.3 \
  --output results/local_samples.jsonl
```

Ejemplo para `features`:

```bash
export N_JOBS=8
python3 knn.py \
  --experiment features \
  --factor 4 \
  --n_neighbors 5 \
  --n_reps 5 \
  --test_size 0.3 \
  --output results/local_features.jsonl
```

## Plots

Generar gráficas a partir de `results/`:

```bash
bash plot_knn.sh [results_dir] [plots_dir] [time_metric]
```

Valores por defecto:

- `results_dir`: `./results`
- `plots_dir`: `./plots`
- `time_metric`: `pred_time_s_avg`

Métricas disponibles:

- `fit_time_s_avg`
- `pred_time_s_avg`
- `total_time_s_avg`

Archivos generados:

- `knn_time.png`
- `knn_speedup.png`
- `knn_efficiency.png`
- `knn_scalability_summary.csv`

## Salida JSONL

Cada línea contiene:

- `experiment`
- `factor`
- `n_jobs`
- `base_n_train`, `base_n_test`, `base_d`
- `n_train_used`, `n_test_used`, `d_used`
- `fit_time_s_avg`, `pred_time_s_avg`, `total_time_s_avg`
- `accuracy_avg`, `f1_avg`, `precision_avg`, `recall_avg`

## Notas

- El dataset se cachea en `.cache/`.
- `knn.py` prioriza `data/processed.cleveland.data` antes de intentar cualquier descarga.
- Si los nodos de cómputo no tienen acceso a internet, el archivo versionado en `data/` es suficiente; `bash prepare_cache.sh` solo adelanta la creación del cache local.
- `parallel_backend("loky", n_jobs=n_jobs)` se mantiene para las corridas paralelas.
- `algorithm="brute"` se fija en sklearn para comparar con el costo teórico de KNN por fuerza bruta.
- Se fijan las variables BLAS a `1` para evitar sobre-suscripción.
