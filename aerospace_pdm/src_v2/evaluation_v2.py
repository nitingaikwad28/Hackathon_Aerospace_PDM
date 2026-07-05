# evaluation_v2.py
# Same maintenance-relevant benchmark metrics as v1's evaluation.py (detection lead
# time, false alarm rate, per-component RUL accuracy), recomputed for the v2 models
# and the larger v2 dataset. Kept as its own module so training/serving code can
# import just the metrics they need.

import numpy as np
import pandas as pd

EARLY_LIFE_FRACTION = 0.2          # first 20% of a part's life = "should still be healthy"


def evaluate_anomaly_detection(scored_df):
    y_true = scored_df["true_anomaly"].values
    y_pred = scored_df["is_anomaly"].values

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    early_rows = []
    for unit_id, unit_df in scored_df.groupby("unit_id"):
        n_early = max(5, int(len(unit_df) * EARLY_LIFE_FRACTION))
        early_rows.append(unit_df.sort_values("cycle").iloc[:n_early])
    early_df = pd.concat(early_rows)
    early_fp = int(np.sum((early_df["true_anomaly"] == 0) & (early_df["is_anomaly"] == 1)))
    early_tn = int(np.sum((early_df["true_anomaly"] == 0) & (early_df["is_anomaly"] == 0)))
    early_life_false_alarm_rate = early_fp / (early_fp + early_tn) if (early_fp + early_tn) > 0 else 0.0

    lead_times = []
    units_never_flagged = 0
    for unit_id, unit_df in scored_df.groupby("unit_id"):
        flagged_rows = unit_df[unit_df["is_anomaly"] == 1]
        if len(flagged_rows) == 0:
            units_never_flagged += 1
            continue
        first_alert_row = flagged_rows.sort_values("cycle").iloc[0]
        true_cycles_left = first_alert_row["life_length"] - first_alert_row["cycle"]
        lead_times.append(int(true_cycles_left))

    mean_lead_time = float(np.mean(lead_times)) if lead_times else None
    median_lead_time = float(np.median(lead_times)) if lead_times else None

    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_alarm_rate": round(false_alarm_rate, 3),
        "early_life_false_alarm_rate": round(early_life_false_alarm_rate, 3),
        "f1_score": round(f1, 3),
        "mean_detection_lead_time_cycles": round(mean_lead_time, 1) if mean_lead_time is not None else None,
        "median_detection_lead_time_cycles": round(median_lead_time, 1) if median_lead_time is not None else None,
        "units_flagged": len(lead_times),
        "units_never_flagged": units_never_flagged,
    }


def evaluate_rul_by_component(df_with_predictions):
    results = {}
    for component, comp_df in df_with_predictions.groupby("component"):
        errors = comp_df["predicted_RUL"] - comp_df["RUL"]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        results[component] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "n_rows": int(len(comp_df))}
    return results


def evaluate_rul_overall(df_with_predictions):
    errors = df_with_predictions["predicted_RUL"] - df_with_predictions["RUL"]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "n_rows": int(len(df_with_predictions))}
