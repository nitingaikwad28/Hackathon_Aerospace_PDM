# anomaly_detection.py
# This file looks at sensor readings and flags any reading that looks "unusual"
# compared to that part's own normal/healthy behavior. This is a simple statistics
# method (z-score), not a fancy black-box model, so it's easy to explain in a demo.

import numpy as np                       # for math
import pandas as pd                      # for tables

HEALTHY_FRACTION = 0.3                   # we treat the first 30% of a part's life as "known healthy"
MIN_HEALTHY_ROWS = 10                    # but always use AT LEAST this many rows, so std-dev isn't computed
                                          # from a handful of noisy samples (which would cause false alarms)
Z_SCORE_LIMIT = 4.0                      # if a reading is more than 4 standard deviations away, flag it
SENSOR_COLUMNS = ["vibration", "temperature", "pressure"]   # the 3 sensors we check


def get_healthy_baseline(unit_df):
    # unit_df = all rows (cycles) for ONE part
    n_healthy_rows = max(MIN_HEALTHY_ROWS, int(len(unit_df) * HEALTHY_FRACTION))  # at least MIN_HEALTHY_ROWS rows
    n_healthy_rows = min(n_healthy_rows, len(unit_df))          # but never more rows than the part actually has
    healthy_rows = unit_df.iloc[:n_healthy_rows]                 # grab just those early, healthy rows

    means = healthy_rows[SENSOR_COLUMNS].mean()                   # average sensor value when healthy
    stds = healthy_rows[SENSOR_COLUMNS].std().replace(0, 1e-6)    # spread of sensor value when healthy (avoid /0)
    stds = stds.fillna(1e-6)                                      # a single-row baseline has no std -> avoid NaN
    return means, stds                                            # give back both


def score_unit(unit_df, means, stds):
    # compute how many "standard deviations away from normal" each reading is (per sensor)
    z_scores = (unit_df[SENSOR_COLUMNS] - means) / stds          # z-score formula: (value - mean) / std
    max_abs_z = z_scores.abs().max(axis=1)                       # worst (biggest) z-score across the 3 sensors
    return max_abs_z                                              # one anomaly score per cycle


def detect_anomalies(fleet_df):
    # Runs anomaly detection separately for every unique part (unit_id),
    # because every part has its own healthy baseline.
    fleet_df = fleet_df.sort_values(["unit_id", "cycle"]).copy()  # make sure cycles are in order per unit
    all_scores = []                                                # collect anomaly scores for every row

    for unit_id, unit_df in fleet_df.groupby("unit_id"):           # loop through each part one at a time
        means, stds = get_healthy_baseline(unit_df)                # learn what "normal" looks like for this part
        scores = score_unit(unit_df, means, stds)                  # score every cycle of this part
        all_scores.append(scores)                                  # save the scores

    fleet_df["anomaly_score"] = pd.concat(all_scores)               # attach scores back to the big table
    fleet_df["is_anomaly"] = (fleet_df["anomaly_score"] > Z_SCORE_LIMIT).astype(int)  # flag if over the limit
    return fleet_df                                                  # return table with new anomaly columns


if __name__ == "__main__":
    # quick manual test: run this file directly to check anomaly detection works
    from data_simulator import simulate_fleet                       # reuse our fake data generator
    fleet = simulate_fleet()                                        # make fake fleet data
    scored = detect_anomalies(fleet)                                # run anomaly detection on it
    n_flagged = scored["is_anomaly"].sum()                          # count how many cycles got flagged
    print(f"Flagged {n_flagged} anomalous readings out of {len(scored)} total readings.")
    print(scored[scored["is_anomaly"] == 1].head(10))                # show a few flagged examples
