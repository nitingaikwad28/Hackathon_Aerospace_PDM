# Aerospace Predictive Maintenance - v2 (scikit-learn Production Architecture)

A production-oriented evolution of the original hackathon MVP, now built on real
**scikit-learn** ensemble algorithms: `IsolationForest` for anomaly detection and
`RandomForestRegressor` for Remaining Useful Life (RUL). An earlier pass at v2 used
from-scratch numpy implementations of these algorithms (no internet access was
available to install scikit-learn at the time); that code has been **deleted** now
that scikit-learn is available, so only this version remains.

This repository now contains only the v2 pipeline; the original from-scratch v1
MVP has been removed. See `outputs_v2/code_documentation_v2.pdf` for full
documentation of this version.

## Verified end-to-end

This pipeline has been run end-to-end with scikit-learn 1.9 installed: training,
serving, and the unit test suite all complete successfully.

```bash
pip install -r requirements_v2.txt
cd src_v2
python train.py
python predict_service.py
python -m unittest tests.test_models -v
```

Latest verified run: test-set anomaly detection recall=0.667, early-life false
alarm rate=0.082; RUL regression overall MAE=4.91, RMSE=9.42. All 5 unit tests
pass. See `outputs_v2/model_metrics_v2.json` for the full benchmark report.

## What's in src_v2/ now

| File | Role |
|---|---|
| `config.py` | Central hyperparameters (now scikit-learn constructor arguments) and paths |
| `data_simulator_v2.py` | 40-aircraft, 160-part, 5-sensor synthetic fleet + train/val/test unit split (unchanged from before) |
| `features.py` | 31 rolling-window engineered features (unchanged from before) |
| `anomaly_detection_v2.py` | Trains/scores one `sklearn.ensemble.IsolationForest` per component |
| `rul_model_v2.py` | Trains/predicts with one `sklearn.ensemble.RandomForestRegressor` per component |
| `evaluation_v2.py` | Detection lead time, false alarm rate, per-component RUL accuracy (unchanged) |
| `model_registry.py` | Saves/loads models via `joblib` + a JSON model card |
| `dashboard_v2.py` | Fleet health dashboard + model-performance panel (unchanged) |
| `mro_integration_v2.py` | Mock MRO adapter with idempotency key + retry/backoff (unchanged) |
| `train.py` | Offline training entry point |
| `predict_service.py` | Production serving entry point (loads saved models, no retraining) |
| `tests/test_models.py` | Unit tests for the scikit-learn-backed wrapper functions |

## What changed vs. the from-scratch v2

| Aspect | From-scratch v2 (deleted) | This version |
|---|---|---|
| Anomaly detection | Custom `_IsolationTree` / `IsolationForest` classes in numpy | `sklearn.ensemble.IsolationForest` |
| RUL estimation | Custom `_DecisionTreeRegressor` / `RandomForestRegressor` classes in numpy | `sklearn.ensemble.RandomForestRegressor` |
| Isolation Forest size | 100 trees | 200 trees |
| Random Forest size | 25 trees, max_depth=6 | 300 trees, max_depth=12 |
| Model persistence | Raw `pickle`, `.pkl` files | `joblib`, `.joblib` files |
| Feature importance | Custom split-count proxy | Built-in `model.feature_importances_` |

## Why scikit-learn

Battle-tested and optimized (Cython/C, parallelized with `n_jobs=-1`), which makes
much larger ensembles (200-300 trees vs. 25-100) practical; full ecosystem
compatibility (joblib, hyperparameter search, model inspection tools) with no
extra code; and significantly less custom code to maintain long-term. See
`outputs_v2/code_documentation_v2.pdf` for the full reasoning and a complete
file-by-file reference.
