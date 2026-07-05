# train.py
# OFFLINE TRAINING entry point - the "production" pattern of separating slow,
# occasional model training from fast, frequent model serving (predict_service.py).
# Run this whenever you have new historical data to retrain on; it saves everything
# predict_service.py needs into artifacts/, and never has to run again until the
# next retraining cycle. Requires scikit-learn + joblib (see requirements_v2.txt).

import json
import time

from data_simulator_v2 import simulate_fleet, split_units
from features import add_rolling_features, feature_column_names
from anomaly_detection_v2 import train_anomaly_models, compute_thresholds, score_anomalies
from rul_model_v2 import train_rul_models, predict_rul, evaluate as evaluate_rul_overall
from evaluation_v2 import evaluate_anomaly_detection, evaluate_rul_by_component
from model_registry import save_models
from config import (ARTIFACT_DIR, DATA_DIR, OUTPUT_DIR, COMPONENTS,
                     IFOREST_N_ESTIMATORS, IFOREST_MAX_SAMPLES, IFOREST_CONTAMINATION,
                     RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF)


def main():
    t0 = time.time()
    print("=" * 60)
    print("TRAIN.PY - offline training pipeline (v2)")
    print("=" * 60)

    print("\n[1/5] Generating training data (larger simulated fleet)...")
    fleet_df = simulate_fleet()
    splits = split_units(fleet_df)
    fleet_df = fleet_df.merge(splits, on="unit_id", how="left")
    fleet_df.to_csv(f"{DATA_DIR}/fleet_telemetry_v2.csv", index=False)
    print(f"   -> {fleet_df['unit_id'].nunique()} units, {len(fleet_df)} rows, "
          f"split: {splits['split'].value_counts().to_dict()}")

    print("\n[2/5] Engineering rolling-window features...")
    featured_df = add_rolling_features(fleet_df)
    train_df = featured_df[featured_df["split"] == "train"]
    val_df = featured_df[featured_df["split"] == "val"]
    test_df = featured_df[featured_df["split"] == "test"].copy()
    print(f"   -> {len(feature_column_names())} feature columns; "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)} rows")

    print("\n[3/5] Training Isolation Forest anomaly models (one per component)...")
    anomaly_models = train_anomaly_models(train_df)
    thresholds = compute_thresholds(anomaly_models, val_df)
    scored_test = score_anomalies(anomaly_models, thresholds, test_df)
    anomaly_metrics = evaluate_anomaly_detection(scored_test)
    print(f"   -> thresholds (tuned on validation split): {thresholds}")
    print(f"   -> test-set recall={anomaly_metrics['recall']}, "
          f"early-life false alarm rate={anomaly_metrics['early_life_false_alarm_rate']}")

    print("\n[4/5] Training Random Forest RUL models (one per component)...")
    rul_models = train_rul_models(train_df)
    test_df["predicted_RUL"] = predict_rul(rul_models, test_df)
    rul_overall = evaluate_rul_overall(test_df["RUL"].values, test_df["predicted_RUL"].values)
    rul_by_component = evaluate_rul_by_component(test_df)
    print(f"   -> test-set overall MAE={rul_overall['MAE']}, RMSE={rul_overall['RMSE']}")
    for component, m in rul_by_component.items():
        print(f"      {component}: MAE={m['MAE']} RMSE={m['RMSE']} (n={m['n_rows']})")

    print("\n[5/5] Saving model artifacts + model card + benchmark report...")
    save_models(anomaly_models, "iforest", ARTIFACT_DIR,
                hyperparameters={"n_estimators": IFOREST_N_ESTIMATORS, "max_samples": IFOREST_MAX_SAMPLES,
                                  "contamination": IFOREST_CONTAMINATION},
                metrics=anomaly_metrics, extra={"thresholds": thresholds})
    save_models(rul_models, "rf", ARTIFACT_DIR,
                hyperparameters={"n_estimators": RF_N_ESTIMATORS, "max_depth": RF_MAX_DEPTH,
                                  "min_samples_split": RF_MIN_SAMPLES_SPLIT, "min_samples_leaf": RF_MIN_SAMPLES_LEAF},
                metrics={"overall": rul_overall, "by_component": rul_by_component})

    all_metrics = {
        "rul_overall": rul_overall,
        "rul_by_component": rul_by_component,
        "anomaly_detection": anomaly_metrics,
        "anomaly_thresholds": thresholds,
        "dataset": {"n_units": int(fleet_df["unit_id"].nunique()), "n_rows": int(len(fleet_df)),
                    "split_counts": splits["split"].value_counts().to_dict()},
    }
    with open(f"{OUTPUT_DIR}/model_metrics_v2.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts saved to {ARTIFACT_DIR}/, "
          f"benchmark report saved to {OUTPUT_DIR}/model_metrics_v2.json")


if __name__ == "__main__":
    main()
