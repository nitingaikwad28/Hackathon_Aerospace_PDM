# Aerospace Predictive Maintenance - v2 (Production-Oriented)

A more production-ready evolution of the original hackathon MVP: a 5x larger
simulated fleet, 5 sensor channels instead of 3, engineered rolling-window
features, real ensemble ML algorithms (Isolation Forest + Random Forest,
implemented from scratch in numpy since this sandbox has no internet access to
install scikit-learn), a proper train/validation/test split, a model registry
with saved artifacts + model cards, and a train/serve separation
(`train.py` vs `predict_service.py`).

**The original `src/` MVP is untouched.** This is a parallel `src_v2/` project;
`README.md` and `outputs/code_documentation.pdf` still describe the original
version exactly as before. See `outputs/code_documentation_v2.pdf` for full
documentation of this version.

## What changed vs. v1

| Aspect | v1 (`src/`) | v2 (`src_v2/`) |
|---|---|---|
| Fleet size | 8 aircraft, 32 parts, ~3,000 rows | 40 aircraft, 160 parts, ~16,300 rows |
| Sensor channels | 3 (vibration, temperature, pressure) | 5 (+ oil debris, RPM deviation) |
| Features | Raw instantaneous readings | Raw + rolling mean/std/min/max/slope (31 features) |
| Anomaly detection | Per-unit z-score threshold | Isolation Forest per component (from scratch) |
| RUL estimation | Linear regression per component | Random Forest Regressor per component (from scratch) |
| Data split | Train/test by unit | Train/validation/test by unit (val tunes alert thresholds) |
| Model lifecycle | Retrained every run inside `main.py` | `train.py` (offline) saves artifacts; `predict_service.py` (serving) loads and scores only |
| Model persistence | None | `artifacts/*.pkl` + `artifacts/model_card.json` (version, hyperparameters, metrics, timestamp) |
| Tests | None | `tests/test_models.py` (unittest, 7 passing tests) |
| MRO integration | Fire-and-forget mock send | Idempotency key + simulated retry/backoff |

## Benchmark comparison (representative runs)

| Metric | v1 | v2 |
|---|---|---|
| RUL MAE (overall) | ~13.5 cycles | ~4.1 cycles |
| RUL MAE (landing gear, weakest component) | ~21 cycles | ~5 cycles |
| Anomaly detection algorithm | Z-score (statistics) | Isolation Forest (ML ensemble) |
| Mean detection lead time | ~150 cycles | ~196 cycles |

RUL accuracy improves substantially across every component type. Anomaly-detection
precision/recall are measured on a much smaller absolute number of held-out test
anomalies in v2 (a 24-unit test split vs. scoring the whole v1 fleet), so treat
those specific figures as indicative rather than final - see the caveats in
`code_documentation_v2.pdf`.

## Running v2

```bash
pip install -r requirements_v2.txt      # numpy + pandas only
cd src_v2
python train.py              # offline: builds data, trains + saves models, benchmarks them
python predict_service.py    # serving: loads saved models, scores new telemetry, builds dashboard
python -m unittest tests.test_models -v   # run the unit tests
```

`train.py` writes to `data_v2/`, `artifacts/`, and `outputs_v2/model_metrics_v2.json`.
`predict_service.py` (no retraining, ~3x faster than train.py) writes
`outputs_v2/dashboard_v2.html`, `fleet_status_v2.csv`, and `work_orders_v2.json`.

## Why still no scikit-learn?

This sandbox environment has no outbound internet access for `pip install`, so
`scikit-learn`, `joblib`, and similar packages cannot be installed. Isolation
Forest and Random Forest were therefore reimplemented from scratch in numpy,
matching the published algorithms and scikit-learn's own semantics (same
subsampling/bagging strategy, same path-length / variance-reduction split logic).
If deployed somewhere with normal internet access, swapping in
`sklearn.ensemble.IsolationForest` / `RandomForestRegressor` is a drop-in
replacement behind the same `fit`/`predict`/`score` interface used here.
