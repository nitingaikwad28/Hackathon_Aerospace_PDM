# data_simulator_v2.py
# v2 of the fleet telemetry simulator: a bigger fleet (40 aircraft instead of 8) and
# 5 sensor channels instead of 3 (adds oil_debris and rpm_deviation), so the models
# below have meaningfully more data and signal to learn from. Also produces an
# explicit, saved train/validation/test SPLIT BY UNIT, which is what a production
# training pipeline needs (never split by row - that would leak a part's future
# into its own training data).

import numpy as np                          # math / random sampling
import pandas as pd                         # tables

from config import (COMPONENTS, N_AIRCRAFT, MIN_LIFE, MAX_LIFE, RUL_CAP,
                     SENSOR_COLUMNS, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)

# Each component's normal ("healthy") sensor readings and noise level, for all 5 channels.
SENSOR_BASELINE = {
    "engine":       {"vibration": 0.30, "temperature": 550, "pressure": 40,   "oil_debris": 2.0, "rpm_deviation": 0.0},
    "landing_gear": {"vibration": 0.15, "temperature": 60,  "pressure": 3000, "oil_debris": 1.0, "rpm_deviation": 0.0},
    "brakes":       {"vibration": 0.20, "temperature": 120, "pressure": 1500, "oil_debris": 1.5, "rpm_deviation": 0.0},
    "bogie":        {"vibration": 0.25, "temperature": 80,  "pressure": 200,  "oil_debris": 1.2, "rpm_deviation": 0.0},
}
SENSOR_NOISE = {
    "engine":       {"vibration": 0.02, "temperature": 5, "pressure": 1,  "oil_debris": 0.30, "rpm_deviation": 0.08},
    "landing_gear": {"vibration": 0.01, "temperature": 3, "pressure": 50, "oil_debris": 0.20, "rpm_deviation": 0.05},
    "brakes":       {"vibration": 0.015,"temperature": 8, "pressure": 30, "oil_debris": 0.25, "rpm_deviation": 0.06},
    "bogie":        {"vibration": 0.02, "temperature": 4, "pressure": 10, "oil_debris": 0.20, "rpm_deviation": 0.06},
}
# how much each sensor drifts by the time a part is fully worn out (multiplied by the
# 0->1 wear curve below); oil_debris and rpm_deviation both RISE with wear (more metal
# shavings in the oil, more rotational instability), same as vibration/temperature.
DEGRADE_MAGNITUDE_FRACTION = {
    "vibration": 1.5, "temperature": 0.25, "pressure": -0.2, "oil_debris": 3.0, "rpm_deviation": 12.0,
}


def make_one_unit_data(rng, aircraft_id, component, unit_id):
    # rng = random number generator (so every run can be reproducible)
    total_life = int(rng.integers(MIN_LIFE, MAX_LIFE))          # total cycles this part will last before failing
    # "current_age" = how far into its life this part is RIGHT NOW (a real fleet snapshot in time).
    current_age = int(rng.triangular(1, total_life * 0.3, total_life))
    current_age = min(current_age, total_life)                   # never go past the part's total life

    cycles = np.arange(1, current_age + 1)                        # cycle numbers simulated: 1, 2, ... current_age
    wear = (cycles / total_life) ** 2                              # wear grows slowly then speeds up near failure

    base = SENSOR_BASELINE[component]                              # normal readings for this component
    noise = SENSOR_NOISE[component]                                # noise level for this component
    data = {}                                                       # will hold one array per sensor column

    for sensor in SENSOR_COLUMNS:                                  # loop over all 5 sensor channels
        trend_fraction = DEGRADE_MAGNITUDE_FRACTION[sensor]          # how strongly this sensor drifts with wear
        trend = base[sensor] * trend_fraction * wear                 # wear-driven drift for this sensor
        data[sensor] = base[sensor] + trend + rng.normal(0, noise[sensor], size=len(cycles))  # + measurement noise

    true_anomaly = np.zeros(len(cycles), dtype=int)                  # ground truth: 1 = injected transient fault

    has_anomaly = rng.random() < 0.2                                  # 20% chance this unit gets a random spike
    anomaly_cycle = rng.integers(5, current_age) if (has_anomaly and current_age > 10) else -1
    if anomaly_cycle > 0:
        spike_sensor = rng.choice(SENSOR_COLUMNS)                     # pick one sensor to spike
        idx = anomaly_cycle - 1                                        # array index for that cycle
        data[spike_sensor][idx] += 3 * abs(base[spike_sensor] if base[spike_sensor] != 0 else 1) * np.sign(rng.normal())
        true_anomaly[idx] = 1

    rul = np.clip(total_life - cycles, 0, RUL_CAP)                     # capped RUL - what the model trains on

    rows = {
        "aircraft_id": aircraft_id, "unit_id": unit_id, "component": component, "cycle": cycles,
        **data,
        "RUL": rul, "true_anomaly": true_anomaly, "life_length": total_life,
    }
    return pd.DataFrame(rows)


def simulate_fleet(seed=42):
    # Builds the FULL fleet: every aircraft x every component, each at its own point in life.
    rng = np.random.default_rng(seed)
    all_data = []
    unit_counter = 0
    for a in range(1, N_AIRCRAFT + 1):
        aircraft_id = f"AC-{a:03d}"
        for component in COMPONENTS:
            unit_counter += 1
            unit_id = f"{aircraft_id}-{component}-{unit_counter}"
            all_data.append(make_one_unit_data(rng, aircraft_id, component, unit_id))
    return pd.concat(all_data, ignore_index=True)


def split_units(fleet_df, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    # PRODUCTION PRACTICE: split by unit_id (never by row), and keep a 3-way split -
    # train (fit the model), validation (tune hyperparameters / pick thresholds),
    # test (final, untouched accuracy check). Returns a DataFrame: unit_id -> split.
    unit_ids = fleet_df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unit_ids)

    n_val = int(len(unit_ids) * val_fraction)
    n_test = int(len(unit_ids) * test_fraction)
    val_ids = set(unit_ids[:n_val])
    test_ids = set(unit_ids[n_val:n_val + n_test])
    # everything else is training data
    split_map = []
    for uid in unit_ids:
        if uid in val_ids:
            split_map.append((uid, "val"))
        elif uid in test_ids:
            split_map.append((uid, "test"))
        else:
            split_map.append((uid, "train"))
    return pd.DataFrame(split_map, columns=["unit_id", "split"])


if __name__ == "__main__":
    # quick manual test / dataset generation entry point
    fleet = simulate_fleet()
    splits = split_units(fleet)
    fleet = fleet.merge(splits, on="unit_id", how="left")               # attach split label to every row
    fleet.to_csv("../data_v2/fleet_telemetry_v2.csv", index=False)
    print(f"Created {fleet['unit_id'].nunique()} units ({fleet['aircraft_id'].nunique()} aircraft), "
          f"{len(fleet)} sensor readings, {len(SENSOR_COLUMNS)} sensor channels.")
    print(splits["split"].value_counts())
