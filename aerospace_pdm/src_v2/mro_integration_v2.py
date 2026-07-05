# mro_integration_v2.py
# Same mock MRO adapter idea as v1, with two production-hardening additions:
#   1. An idempotency key on every work order, so re-sending the same work order
#      (e.g. after a retry) never creates a duplicate in a real MRO system.
#   2. A simulated retry-with-backoff wrapper around the "send" call, since real
#      network calls to an external system occasionally fail transiently and a
#      production integration must not just give up on the first error.

import json
import time
import random
from datetime import datetime, timezone


def build_work_order(row):
    return {
        "work_order_id": f"WO-{row['unit_id']}-{row['cycle']}",        # also used as the idempotency key
        "aircraft_id": row["aircraft_id"],
        "component": row["component"],
        "unit_id": row["unit_id"],
        "predicted_RUL_cycles": round(float(row["predicted_RUL"]), 1),
        "anomaly_score": round(float(row["anomaly_score"]), 3),
        "anomaly_detected": bool(row["is_anomaly"]),
        "priority": row["health_status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recommended_action": (
            "Immediate inspection and maintenance" if row["health_status"] == "Critical"
            else "Schedule maintenance during next planned downtime"
        ),
    }


def create_work_orders(dashboard_df):
    needs_attention = dashboard_df[dashboard_df["health_status"].isin(["Critical", "Warning"])]
    return [build_work_order(row) for _, row in needs_attention.iterrows()]


def _simulated_transient_failure(rng, failure_rate=0.0):
    # In a real deployment this would be a network/API call that can fail; here we
    # simulate an occasional transient failure so the retry logic below has
    # something real to demonstrate. failure_rate=0.0 by default (deterministic demo).
    return rng.random() < failure_rate


def send_to_mro_system(work_orders, path="../outputs_v2/work_orders_v2.json",
                        max_retries=3, failure_rate=0.0, seed=0):
    # "Sends" the work orders with retry + exponential backoff, the way a real
    # integration would guard against a flaky network call to the MRO system's API.
    rng = random.Random(seed)
    sent, failed = [], []
    for wo in work_orders:
        attempt = 0
        while attempt <= max_retries:
            if not _simulated_transient_failure(rng, failure_rate):
                sent.append(wo)
                break
            attempt += 1
            time.sleep(0)                      # would be time.sleep(backoff_seconds) in a real client
        else:
            failed.append(wo)                   # exhausted retries - would go to a dead-letter queue in production

    with open(path, "w") as f:
        json.dump(sent, f, indent=2)
    print(f"Sent {len(sent)} work orders to MRO system -> {path} ({len(failed)} failed after retries)")
    return sent, failed
