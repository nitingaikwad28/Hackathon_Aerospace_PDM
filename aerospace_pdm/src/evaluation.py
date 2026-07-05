# evaluation.py
# This file measures HOW GOOD our anomaly detector and RUL model actually are,
# using metrics that matter to a maintenance planner: how often we cry wolf
# (false alarm rate), how early we warn them before failure (detection lead time),
# and how far off our life predictions are (MAE/RMSE), broken down per component type.

import numpy as np                          # for math
import pandas as pd                         # for tables

EARLY_LIFE_FRACTION = 0.2                    # first 20% of a part's simulated life = "should still be healthy"
                                              # used as a fair test bed for a PURE false-alarm rate, since our
                                              # simulator's wear grows continuously (no flat "100% healthy" plateau)


def evaluate_anomaly_detection(scored_df):
    # scored_df = fleet data that already has is_anomaly / anomaly_score / true_anomaly columns

    y_true = scored_df["true_anomaly"].values           # ground-truth: 1 = a real injected anomaly happened
    y_pred = scored_df["is_anomaly"].values              # our detector's guess: 1 = flagged as anomaly

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))       # correctly caught a real anomaly
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))       # flagged something that was NOT a real anomaly
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))       # missed a real anomaly
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))       # correctly stayed quiet on normal data

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0                 # of our alarms, % that were real
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0                    # of real anomalies, % we caught
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0          # of normal readings, % we wrongly flagged
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # A SECOND, stricter false-alarm check: only look at the EARLY portion of each part's life,
    # where wear is minimal and readings SHOULD look healthy. Flags here are much closer to "pure"
    # false alarms (noise mistaken for a problem), since late-life flags are often genuine early
    # degradation warnings rather than detector mistakes - see the caveat printed with these results.
    early_rows = []
    for unit_id, unit_df in scored_df.groupby("unit_id"):
        n_early = max(5, int(len(unit_df) * EARLY_LIFE_FRACTION))
        early_rows.append(unit_df.sort_values("cycle").iloc[:n_early])
    early_df = pd.concat(early_rows)
    early_fp = int(np.sum((early_df["true_anomaly"] == 0) & (early_df["is_anomaly"] == 1)))
    early_tn = int(np.sum((early_df["true_anomaly"] == 0) & (early_df["is_anomaly"] == 0)))
    early_life_false_alarm_rate = early_fp / (early_fp + early_tn) if (early_fp + early_tn) > 0 else 0.0

    # DETECTION LEAD TIME: for every part that got at least one alert, how many cycles
    # BEFORE its true end-of-life did we raise the FIRST alert? Bigger number = more advance
    # warning the maintenance team gets. We use the part's TRUE uncapped lifespan
    # (life_length - cycle) here rather than the capped RUL column, because RUL is
    # deliberately clipped at RUL_CAP (130) for model training and would flatten out this
    # metric for any part still far from failure.
    lead_times = []                                        # collect one lead time per alerted unit
    units_never_flagged = 0                                # count parts we never warned about at all
    for unit_id, unit_df in scored_df.groupby("unit_id"):   # go part by part
        flagged_rows = unit_df[unit_df["is_anomaly"] == 1]  # rows where we raised an alert for this part
        if len(flagged_rows) == 0:                          # this part was never flagged
            units_never_flagged += 1
            continue
        first_alert_row = flagged_rows.sort_values("cycle").iloc[0]  # the EARLIEST alert for this part
        true_cycles_left = first_alert_row["life_length"] - first_alert_row["cycle"]  # uncapped remaining life
        lead_times.append(int(true_cycles_left))             # cycles of advance warning before true failure

    mean_lead_time = float(np.mean(lead_times)) if lead_times else None    # average warning across the fleet
    median_lead_time = float(np.median(lead_times)) if lead_times else None

    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 3),                    # e.g. 0.4 = 40% of our alarms were real
        "recall": round(recall, 3),                          # e.g. 0.8 = we caught 80% of real anomalies
        "false_alarm_rate": round(false_alarm_rate, 3),       # over the WHOLE life (includes early-wear flags)
        "early_life_false_alarm_rate": round(early_life_false_alarm_rate, 3),  # "pure noise" false alarm rate
        "f1_score": round(f1, 3),
        "mean_detection_lead_time_cycles": round(mean_lead_time, 1) if mean_lead_time is not None else None,
        "median_detection_lead_time_cycles": round(median_lead_time, 1) if median_lead_time is not None else None,
        "units_flagged": len(lead_times),                     # how many parts got at least 1 alert
        "units_never_flagged": units_never_flagged,           # how many parts got zero alerts
    }


def evaluate_rul_by_component(test_df, predictions):
    # Breaks down RUL accuracy separately for each component type, since some component
    # types (e.g. engine) may be easier or harder to predict than others.
    results = {}                                              # dict: component -> {MAE, RMSE, n_rows}
    test_df = test_df.copy()
    test_df["predicted_RUL"] = predictions                     # attach predictions row-by-row

    for component, comp_df in test_df.groupby("component"):     # loop over engine / landing_gear / brakes / bogie
        errors = comp_df["predicted_RUL"] - comp_df["RUL"]       # predicted minus actual
        mae = float(np.mean(np.abs(errors)))                     # average absolute miss, in cycles
        rmse = float(np.sqrt(np.mean(errors ** 2)))               # penalizes big misses more than small ones
        results[component] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "n_test_rows": int(len(comp_df))}
    return results


if __name__ == "__main__":
    # quick manual test: run the full pipeline and print the evaluation metrics
    from data_simulator import simulate_fleet
    from anomaly_detection import detect_anomalies
    from rul_model import split_train_test, train_linear_model, predict_rul

    fleet = simulate_fleet()                                    # 1. fake sensor data
    scored = detect_anomalies(fleet)                             # 2. flag anomalies
    train_df, test_df = split_train_test(fleet)                   # 3. split by unit
    weights = train_linear_model(train_df)                        # 4. train RUL model
    test_predictions = predict_rul(weights, test_df)               # 5. predict on held-out parts

    anomaly_metrics = evaluate_anomaly_detection(scored)            # anomaly detector quality
    rul_metrics = evaluate_rul_by_component(test_df, test_predictions)  # RUL accuracy per component

    print("Anomaly detection metrics:", anomaly_metrics)
    print("RUL accuracy by component:", rul_metrics)
