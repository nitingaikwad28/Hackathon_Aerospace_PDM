# main.py
# This is the ONE FILE to run for the full hackathon demo.
# It runs every step of the predictive maintenance pipeline, in order, and prints
# a clear summary so judges can see the whole system working end-to-end.

import json                                                                    # to save the metrics report

from data_simulator import simulate_fleet                                     # step 1: fake sensor data
from anomaly_detection import detect_anomalies                                 # step 2: flag anomalies
from rul_model import split_train_test, train_linear_model, predict_rul, FEATURE_COLUMNS, evaluate, TARGET_COLUMN  # step 3
from evaluation import evaluate_anomaly_detection, evaluate_rul_by_component    # step 3b: benchmark both models
from dashboard import build_dashboard, save_dashboard_html                     # step 4: fleet dashboard
from mro_integration import create_work_orders, send_to_mro_system             # step 5: MRO integration


def run_pipeline():
    print("=" * 60)
    print("AEROSPACE PREDICTIVE MAINTENANCE - HACKATHON MVP DEMO")
    print("=" * 60)

    # STEP 1: Simulate multi-channel sensor/telemetry data for the whole fleet
    print("\n[1/6] Simulating fleet sensor data (vibration, temperature, pressure)...")
    fleet_df = simulate_fleet()                                                # build the fake fleet dataset
    fleet_df.to_csv("../data/fleet_telemetry.csv", index=False)                # save raw telemetry to disk
    print(f"   -> {fleet_df['unit_id'].nunique()} components across "
          f"{fleet_df['aircraft_id'].nunique()} aircraft, {len(fleet_df)} sensor readings.")

    # STEP 2: Detect anomalies (deviations from each part's own healthy baseline)
    print("\n[2/6] Running anomaly detection (statistical deviation from healthy baseline)...")
    scored_df = detect_anomalies(fleet_df)                                     # attach anomaly scores/flags
    n_anomalies = scored_df["is_anomaly"].sum()                                # count flagged readings
    print(f"   -> Flagged {n_anomalies} anomalous readings out of {len(scored_df)}.")

    # STEP 3: Train and evaluate the Remaining Useful Life (RUL) model
    print("\n[3/6] Training Remaining Useful Life (RUL) model...")
    train_df, test_df = split_train_test(fleet_df)                             # split fleet by unit (no leakage)
    weights_by_component = train_linear_model(train_df)                        # train one small model per component
    test_predictions = predict_rul(weights_by_component, test_df)              # predict on held-out test parts
    metrics = evaluate(test_df[TARGET_COLUMN].values, test_predictions)        # overall MAE / RMSE on test set
    print(f"   -> Overall model accuracy on unseen parts: MAE={metrics['MAE']} cycles, RMSE={metrics['RMSE']} cycles.")

    # STEP 4: Benchmark both models with the metrics that matter to a maintenance planner
    print("\n[4/6] Benchmarking model quality (lead time, false alarm rate, per-component RUL accuracy)...")
    anomaly_metrics = evaluate_anomaly_detection(scored_df)                    # precision/recall/false-alarm/lead-time
    rul_by_component = evaluate_rul_by_component(test_df, test_predictions)    # MAE/RMSE per component type
    print(f"   -> Anomaly recall: {anomaly_metrics['recall']*100:.0f}% of real anomalies caught, "
          f"early-life false alarm rate: {anomaly_metrics['early_life_false_alarm_rate']*100:.1f}%.")
    print(f"   -> Mean detection lead time: {anomaly_metrics['mean_detection_lead_time_cycles']} cycles "
          f"before true failure.")
    for component, comp_metrics in rul_by_component.items():
        print(f"   -> RUL accuracy [{component}]: MAE={comp_metrics['MAE']}, RMSE={comp_metrics['RMSE']} "
              f"(n={comp_metrics['n_test_rows']} test rows)")

    all_metrics = {
        "rul_overall": metrics,
        "rul_by_component": rul_by_component,
        "anomaly_detection": anomaly_metrics,
    }
    with open("../outputs/model_metrics.json", "w") as f:                      # save full benchmark report
        json.dump(all_metrics, f, indent=2)
    print("   -> Saved outputs/model_metrics.json")

    # STEP 5: Build the fleet-level maintenance dashboard (current status of every part)
    print("\n[5/6] Building fleet maintenance dashboard...")
    dashboard_df = build_dashboard(scored_df, weights_by_component, predict_rul, FEATURE_COLUMNS)  # summary table
    save_dashboard_html(dashboard_df, "../outputs/dashboard.html", metrics_summary=all_metrics)  # w/ metrics panel
    status_counts = dashboard_df["health_status"].value_counts().to_dict()     # count per status
    print(f"   -> Fleet status: {status_counts}")
    print(f"   -> Saved outputs/dashboard.html")

    # STEP 6: Create work orders for anything Critical/Warning and "send" to the MRO system
    print("\n[6/6] Creating maintenance work orders and sending to MRO system...")
    work_orders = create_work_orders(dashboard_df)                              # build work orders
    send_to_mro_system(work_orders)                                              # "send" them (saved as JSON here)

    # Final summary table, most urgent parts first
    print("\n" + "=" * 60)
    print("TOP PRIORITY MAINTENANCE ACTIONS (most urgent first)")
    print("=" * 60)
    print(dashboard_df.head(10).to_string(index=False))                         # show the 10 most urgent parts

    dashboard_df.to_csv("../outputs/fleet_status.csv", index=False)             # save full dashboard table as CSV
    print("\nSaved outputs/fleet_status.csv")
    print("\nDone. Open outputs/dashboard.html in a browser to view the fleet dashboard.")


if __name__ == "__main__":
    run_pipeline()                                # run everything when this file is executed directly
