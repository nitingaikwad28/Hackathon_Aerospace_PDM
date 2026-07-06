# data_simulator_v2.py
# v2 of the fleet telemetry simulator: a bigger fleet (40 aircraft instead of 8) and
# 5 sensor channels instead of 3, so the models below have meaningfully more data and
# signal to learn from. Also produces an explicit, saved train/validation/test SPLIT
# BY UNIT, which is what a production training pipeline needs (never split by row -
# that would leak a part's future into its own training data).
#
# Sensor channels, baseline values, and degradation behavior are grounded in publicly
# documented aerospace references rather than picked arbitrarily:
#   - Real transport-category aircraft record ~88 parameters per FAA AC 20-141B /
#     ARINC 717, including engine EGT, vibration, oil pressure and debris, and
#     rotational speed (N1/N2) deviation - the 5 channels here are a simplified
#     stand-in for that same parameter family, with real units attached.
#   - Vibration analysis is well documented as the EARLIEST and most sensitive
#     precursor of mechanical wear (bearing degradation shows up in vibration weeks
#     before temperature or performance shift) - see WEAR_EXPONENT below, which
#     encodes this by making vibration/oil-debris rise earlier in a part's life than
#     temperature does.
#   - Baseline magnitudes reference commonly cited public figures: ISO 10816-family
#     vibration severity zones (mm/s RMS), ~3000 psi as the standard commercial
#     aircraft hydraulic system pressure, ~180-220 psi large-aircraft tire pressure,
#     and typical turbofan cruise EGT/oil-pressure ranges. These are engineering-
#     plausible reference points, not exact OEM specifications for any one aircraft
#     type, and are NOT sourced from any accident investigation data - real flight-
#     recorder readouts from crash investigations are not published as raw datasets,
#     and reproducing a real tragedy's telemetry would not be appropriate here.

import numpy as np                          # math / random sampling
import pandas as pd                         # tables

from config import (COMPONENTS, N_AIRCRAFT, MIN_LIFE, MAX_LIFE, RUL_CAP,
                     SENSOR_COLUMNS, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)

# Each component's normal ("healthy") sensor readings, by real aerospace channel:
#   vibration_mm_s                  - ISO 10816-style vibration velocity, mm/s RMS
#   temperature_c                   - EGT for engine; brake/bearing temperature for the rest
#   pressure_psi                    - oil pressure (engine); strut/hydraulic/tire pressure (rest)
#   oil_debris_ppm                  - magnetic chip-detector debris index, parts per million
#   rotational_speed_deviation_pct  - deviation from expected rotational speed (generalizes
#                                     engine N1/N2 trend monitoring to any rotating part)
SENSOR_BASELINE = {
    "engine":       {"vibration_mm_s": 2.5, "temperature_c": 620, "pressure_psi": 45,   "oil_debris_ppm": 2.0, "rotational_speed_deviation_pct": 0.0},
    "landing_gear": {"vibration_mm_s": 1.2, "temperature_c": 55,  "pressure_psi": 2200, "oil_debris_ppm": 1.0, "rotational_speed_deviation_pct": 0.0},
    "brakes":       {"vibration_mm_s": 1.6, "temperature_c": 140, "pressure_psi": 3000, "oil_debris_ppm": 1.5, "rotational_speed_deviation_pct": 0.0},
    "bogie":        {"vibration_mm_s": 2.0, "temperature_c": 90,  "pressure_psi": 210,  "oil_debris_ppm": 1.2, "rotational_speed_deviation_pct": 0.0},
}
SENSOR_NOISE = {
    "engine":       {"vibration_mm_s": 0.30, "temperature_c": 8, "pressure_psi": 2,  "oil_debris_ppm": 0.30, "rotational_speed_deviation_pct": 0.15},
    "landing_gear": {"vibration_mm_s": 0.15, "temperature_c": 3, "pressure_psi": 60, "oil_debris_ppm": 0.20, "rotational_speed_deviation_pct": 0.10},
    "brakes":       {"vibration_mm_s": 0.20, "temperature_c": 10,"pressure_psi": 80, "oil_debris_ppm": 0.25, "rotational_speed_deviation_pct": 0.12},
    "bogie":        {"vibration_mm_s": 0.25, "temperature_c": 5, "pressure_psi": 12, "oil_debris_ppm": 0.20, "rotational_speed_deviation_pct": 0.12},
}
# how much each sensor drifts by the time a part is fully worn out (multiplied by the
# 0->1 wear curve below); oil_debris_ppm and rotational_speed_deviation_pct both RISE
# with wear (more metal shavings in the oil, more rotational instability), same as
# vibration/temperature. pressure_psi FALLS with wear (seal/leak-driven pressure loss).
DEGRADE_MAGNITUDE_FRACTION = {
    "vibration_mm_s": 1.5, "temperature_c": 0.25, "pressure_psi": -0.2,
    "oil_debris_ppm": 3.0, "rotational_speed_deviation_pct": 12.0,
}
# WEAR EXPONENT: controls HOW EARLY in a part's life each sensor starts visibly
# deviating from baseline (wear_fraction ** exponent - a lower exponent rises earlier
# and more gradually; a higher exponent stays flat longer, then rises sharply near
# end-of-life). Values reflect the documented real-world finding that vibration and
# oil-debris (chip-detector) monitoring catch developing wear earlier than
# temperature or performance-based measures do:
#   vibration_mm_s (1.3) and oil_debris_ppm (1.4)   -> earliest, most sensitive precursors
#   rotational_speed_deviation_pct (1.8)             -> mid-life symptom
#   pressure_psi (2.0)                               -> fairly steady leak-driven decline
#   temperature_c (2.5)                              -> latest-rising, lagging indicator
WEAR_EXPONENT = {
    "vibration_mm_s": 1.3, "oil_debris_ppm": 1.4, "rotational_speed_deviation_pct": 1.8,
    "pressure_psi": 2.0, "temperature_c": 2.5,
}

CYCLES_PER_DAY_RANGE = (2.0, 5.0)   # typical short/medium-haul commercial utilization


def make_one_unit_data(rng, aircraft_id, component, unit_id, now=None):
    # rng = random number generator (so every run can be reproducible)
    total_life = int(rng.integers(MIN_LIFE, MAX_LIFE))          # total cycles this part will last before failing
    # "current_age" = how far into its life this part is RIGHT NOW (a real fleet snapshot in time).
    current_age = int(rng.triangular(1, total_life * 0.3, total_life))
    current_age = min(current_age, total_life)                   # never go past the part's total life

    cycles = np.arange(1, current_age + 1)                        # cycle numbers simulated: 1, 2, ... current_age

    base = SENSOR_BASELINE[component]                              # normal readings for this component
    noise = SENSOR_NOISE[component]                                # noise level for this component
    data = {}                                                       # will hold one array per sensor column

    for sensor in SENSOR_COLUMNS:                                  # loop over all 5 sensor channels
        wear = (cycles / total_life) ** WEAR_EXPONENT[sensor]         # sensor-specific wear-onset curve
        trend_fraction = DEGRADE_MAGNITUDE_FRACTION[sensor]           # how strongly this sensor drifts with wear
        trend = base[sensor] * trend_fraction * wear                  # wear-driven drift for this sensor
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

    # ---- realistic per-cycle UTC timestamps (live-fleet feel, not an abstract counter) ----
    # Each row is modeled as a stabilized-cruise EHM snapshot (real engine health monitoring
    # data is conventionally sampled once per flight at stable cruise, so readings are
    # comparable cycle-to-cycle). The part's most recent cycle lands close to "now"; earlier
    # cycles step back in time at this unit's simulated flight cadence, with realistic jitter
    # for day-to-day schedule variance.
    cycles_per_day = rng.uniform(*CYCLES_PER_DAY_RANGE)
    last_cycle_time = (now or pd.Timestamp.now(tz="UTC")) - pd.Timedelta(hours=rng.uniform(0, 12))
    cycles_ago = current_age - cycles                                   # 0 for the most recent cycle
    jitter_hours = rng.normal(0, 2.5, size=len(cycles))
    timestamps = last_cycle_time - pd.to_timedelta(cycles_ago / cycles_per_day, unit="D") \
        + pd.to_timedelta(jitter_hours, unit="h")

    rows = {
        "aircraft_id": aircraft_id, "unit_id": unit_id, "component": component, "cycle": cycles,
        "timestamp_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **data,
        "RUL": rul, "true_anomaly": true_anomaly, "life_length": total_life,
    }
    return pd.DataFrame(rows)


def simulate_fleet(seed=42):
    # Builds the FULL fleet: every aircraft x every component, each at its own point in life.
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.now(tz="UTC")             # one shared "now" so every unit's recency is consistent
    all_data = []
    unit_counter = 0
    for a in range(1, N_AIRCRAFT + 1):
        aircraft_id = f"AC-{a:03d}"
        for component in COMPONENTS:
            unit_counter += 1
            unit_id = f"{aircraft_id}-{component}-{unit_counter}"
            all_data.append(make_one_unit_data(rng, aircraft_id, component, unit_id, now=now))
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
