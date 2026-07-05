# features.py
# Turns raw instantaneous sensor readings into a richer feature set for the models,
# using trailing rolling-window statistics. This is standard practice in real PdM
# pipelines (e.g. NASA C-MAPSS RUL literature): a single instantaneous reading is
# noisy, but its recent MEAN, SPREAD, RANGE and TREND (slope) over the last few
# cycles are much more informative and stable signals of developing wear.

import numpy as np                          # math
import pandas as pd                         # tables

from config import SENSOR_COLUMNS, ROLLING_WINDOW


def _slope(values):
    # simple trailing-window slope: fit a straight line (y = a*x + b) through the window
    # and return "a" (the rate of change per cycle) - a rising slope is an early wear signal
    # even before the absolute sensor value itself looks unusual.
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n)
    x_mean, y_mean = x.mean(), values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def add_rolling_features(df, window=ROLLING_WINDOW):
    # df must be sorted by unit_id, cycle. Adds, for every sensor column:
    #   <sensor>_mean, <sensor>_std, <sensor>_min, <sensor>_max, <sensor>_slope
    # all computed over the trailing `window` cycles (inclusive of the current row).
    df = df.sort_values(["unit_id", "cycle"]).copy()
    grouped = df.groupby("unit_id", group_keys=False)

    for sensor in SENSOR_COLUMNS:
        roll = grouped[sensor].rolling(window, min_periods=1)
        df[f"{sensor}_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{sensor}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        df[f"{sensor}_min"] = roll.min().reset_index(level=0, drop=True)
        df[f"{sensor}_max"] = roll.max().reset_index(level=0, drop=True)
        df[f"{sensor}_slope"] = (
            grouped[sensor]
            .apply(lambda s: s.rolling(window, min_periods=1).apply(_slope, raw=True))
            .reset_index(level=0, drop=True)
        )
    return df


def feature_column_names():
    # the full list of feature column names, in a fixed, stable order - both
    # training and serving code must use exactly this order. Includes the RAW
    # instantaneous sensor reading alongside its rolling-window statistics: the raw
    # value keeps single-cycle spikes fully visible (a 5-cycle rolling mean/std can
    # partially dilute a one-point spike), while the rolling stats capture the
    # slower, accelerating wear trend that a single reading cannot show on its own.
    cols = ["cycle"]
    for sensor in SENSOR_COLUMNS:
        cols += [sensor, f"{sensor}_mean", f"{sensor}_std", f"{sensor}_min", f"{sensor}_max", f"{sensor}_slope"]
    return cols


if __name__ == "__main__":
    # quick manual test: confirm the engineered features look sane
    from data_simulator_v2 import simulate_fleet
    fleet = simulate_fleet()
    featured = add_rolling_features(fleet)
    cols = feature_column_names()
    print(f"Engineered {len(cols)} feature columns from {len(SENSOR_COLUMNS)} raw sensors.")
    print(featured[["unit_id", "cycle"] + cols[:6]].head(8))
