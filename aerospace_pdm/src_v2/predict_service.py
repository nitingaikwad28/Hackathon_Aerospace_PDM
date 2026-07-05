# predict_service.py
# PRODUCTION SERVING entry point - loads the artifacts train.py already produced
# (no retraining here) and scores a fresh batch of incoming telemetry, the way a
# real deployment would run on a schedule (e.g. every night, or every time a new
# flight's data lands). This train/serve split is a core production ML pattern:
# training is slow and occasional; serving is fast and frequent, and must not
# depend on retraining every time.

import time

from data_simulator_v2 import simulate_fleet, split_units
from features import add_rolling_features
from anomaly_detection_v2 import score_anomalies
from rul_model_v2 import predict_rul
from dashboard_v2 import build_dashboard, save_dashboard_html
from mro_integration_v2 import create_work_orders, send_to_mro_system
from model_registry import load_models, load_model_card
from config import ARTIFACT_DIR, OUTPUT_DIR, COMPONENTS


def load_new_telemetry_batch(seed=99):
    # Simulates "new" telemetry arriving after deployment (a different random seed
    # than the one train.py used), standing in for a fresh batch of live sensor data.
    fleet_df = simulate_fleet(seed=seed)
    splits = split_units(fleet_df, seed=seed)          # only used here for bookkeeping/consistency
    fleet_df = fleet_df.merge(splits, on="unit_id", how="left")
    return fleet_df


def main():
    t0 = time.time()
    print("=" * 60)
    print("PREDICT_SERVICE.PY - production serving pipeline (v2)")
    print("=" * 60)

    print("\n[1/5] Loading trained model artifacts (no retraining)...")
    anomaly_models = load_models("iforest", COMPONENTS, ARTIFACT_DIR)
    rul_models = load_models("rf", COMPONENTS, ARTIFACT_DIR)
    card = load_model_card(ARTIFACT_DIR)
    thresholds = card.get("iforest", {}).get("extra", {}).get("thresholds", {})
    print(f"   -> loaded {len(anomaly_models)} anomaly models + {len(rul_models)} RUL models, "
          f"trained_at={card.get('rf', {}).get('trained_at_utc', 'unknown')}")

    print("\n[2/5] Fetching new telemetry batch...")
    fleet_df = load_new_telemetry_batch()
    print(f"   -> {fleet_df['unit_id'].nunique()} units, {len(fleet_df)} sensor readings")

    print("\n[3/5] Engineering features + scoring...")
    featured_df = add_rolling_features(fleet_df)
    scored_df = score_anomalies(anomaly_models, thresholds, featured_df)
    scored_df["predicted_RUL"] = predict_rul(rul_models, scored_df)
    print(f"   -> flagged {scored_df['is_anomaly'].sum()} anomalous readings")

    print("\n[4/5] Building fleet dashboard...")
    import json
    with open(f"{OUTPUT_DIR}/model_metrics_v2.json") as f:
        metrics_summary = json.load(f)
    dashboard_df = build_dashboard(scored_df)
    save_dashboard_html(dashboard_df, f"{OUTPUT_DIR}/dashboard_v2.html", metrics_summary=metrics_summary)
    dashboard_df.to_csv(f"{OUTPUT_DIR}/fleet_status_v2.csv", index=False)
    print(f"   -> fleet status: {dashboard_df['health_status'].value_counts().to_dict()}")

    print("\n[5/5] Creating + sending maintenance work orders...")
    work_orders = create_work_orders(dashboard_df)
    sent, failed = send_to_mro_system(work_orders)

    print("\n" + "=" * 60)
    print("TOP PRIORITY MAINTENANCE ACTIONS (most urgent first)")
    print("=" * 60)
    print(dashboard_df.head(10).to_string(index=False))
    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
