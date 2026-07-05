# mro_integration.py
# This file simulates sending maintenance work orders to an existing MRO
# (Maintenance, Repair & Overhaul) or asset-management system, like SAP PM or IBM Maximo.
# In a real deployment, "send_work_order" would call that system's real API instead of
# just printing/saving to a file - but the rest of our pipeline would stay exactly the same.

import json                                 # to save work orders as a .json file
from datetime import datetime               # to timestamp each work order


def build_work_order(row):
    # Turns one dashboard row (a part that needs attention) into a structured work order.
    return {
        "work_order_id": f"WO-{row['unit_id']}-{row['cycle']}",   # unique id for this work order
        "aircraft_id": row["aircraft_id"],                          # which aircraft this part is on
        "component": row["component"],                              # which type of part (engine, brakes, etc.)
        "unit_id": row["unit_id"],                                   # unique id of the specific part
        "predicted_RUL_cycles": round(float(row["predicted_RUL"]), 1),  # how many cycles of life we think are left
        "anomaly_detected": bool(row["is_anomaly"]),                 # whether a live sensor anomaly was seen
        "priority": row["health_status"],                            # Critical / Warning
        "created_at": datetime.now().isoformat(timespec="seconds"),   # when this work order was generated
        "recommended_action": (
            "Immediate inspection and maintenance" if row["health_status"] == "Critical"
            else "Schedule maintenance during next planned downtime"
        ),
    }


def create_work_orders(dashboard_df):
    # Only create work orders for parts that actually need attention (skip "Healthy" parts).
    needs_attention = dashboard_df[dashboard_df["health_status"].isin(["Critical", "Warning"])]
    work_orders = [build_work_order(row) for _, row in needs_attention.iterrows()]  # build one order per row
    return work_orders                                                              # list of work order dicts


def send_to_mro_system(work_orders, path="../outputs/work_orders.json"):
    # "Sends" the work orders to the MRO system. Here we just write them to a JSON file,
    # which stands in for a real API call (e.g. requests.post(mro_api_url, json=work_order)).
    with open(path, "w") as f:                       # open the output file for writing
        json.dump(work_orders, f, indent=2)           # save the work orders as readable JSON
    print(f"Sent {len(work_orders)} work orders to MRO system -> {path}")


if __name__ == "__main__":
    # quick manual test: build a dashboard, then generate + "send" work orders for it
    from data_simulator import simulate_fleet
    from anomaly_detection import detect_anomalies
    from rul_model import split_train_test, train_linear_model, predict_rul, FEATURE_COLUMNS
    from dashboard import build_dashboard

    fleet = simulate_fleet()                                     # 1. fake sensor data
    scored = detect_anomalies(fleet)                              # 2. flag anomalies
    train_df, _ = split_train_test(fleet)                          # 3. training data
    weights = train_linear_model(train_df)                         # 4. train RUL model
    dash = build_dashboard(scored, weights, predict_rul, FEATURE_COLUMNS)  # 5. build dashboard table

    work_orders = create_work_orders(dash)                         # 6. turn urgent rows into work orders
    send_to_mro_system(work_orders)                                 # 7. "send" them to the MRO system
    print(json.dumps(work_orders[:2], indent=2))                    # show the first 2 as an example
