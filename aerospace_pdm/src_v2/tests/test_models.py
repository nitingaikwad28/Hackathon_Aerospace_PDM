# test_models.py
# Basic sanity/unit tests for the v2 pipeline, using only Python's built-in
# `unittest` (no pytest dependency needed). These tests exercise the real
# scikit-learn-backed wrapper functions in anomaly_detection_v2.py and
# rul_model_v2.py, so scikit-learn must be installed to run them:
#
#   python -m unittest tests.test_models -v      (run from src_v2/)

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import _slope, add_rolling_features, feature_column_names
from anomaly_detection_v2 import train_anomaly_models, compute_thresholds, score_anomalies
from rul_model_v2 import train_rul_models, predict_rul, evaluate


def _toy_featured_df(n_units=6, n_cycles=40, seed=0):
    # Builds a small, already-"featured" DataFrame (skips the simulator/feature
    # engineering step) so these tests run in well under a second and don't
    # depend on data_simulator_v2's randomness.
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        unit_id = f"unit-{u}"
        life = n_cycles
        for cyc in range(1, life + 1):
            wear = (cyc / life) ** 2
            vibration = 0.3 + wear * 0.4 + rng.normal(0, 0.02)
            rul = max(0, min(130, life - cyc))
            rows.append({"unit_id": unit_id, "component": "engine", "cycle": cyc,
                         "vibration": vibration, "RUL": rul,
                         "true_anomaly": 0, "life_length": life})
    df = pd.DataFrame(rows)
    # minimal engineered features (mirrors what feature_column_names expects,
    # trimmed to just the columns these tests actually use)
    df["vibration_mean"] = df.groupby("unit_id")["vibration"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["vibration_std"] = df.groupby("unit_id")["vibration"].transform(lambda s: s.rolling(5, min_periods=1).std()).fillna(0)
    df["vibration_min"] = df.groupby("unit_id")["vibration"].transform(lambda s: s.rolling(5, min_periods=1).min())
    df["vibration_max"] = df.groupby("unit_id")["vibration"].transform(lambda s: s.rolling(5, min_periods=1).max())
    df["vibration_slope"] = 0.0
    return df


class TestFeatureEngineering(unittest.TestCase):
    def test_slope_of_rising_series_is_positive(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertGreater(_slope(values), 0)

    def test_slope_of_flat_series_is_zero(self):
        values = np.array([3.0, 3.0, 3.0, 3.0])
        self.assertAlmostEqual(_slope(values), 0.0, places=6)

    def test_feature_column_count_matches_dataframe(self):
        df = pd.DataFrame({
            "unit_id": ["u1"] * 10,
            "cycle": range(1, 11),
            "vibration_mm_s": np.linspace(0.1, 0.2, 10),
            "temperature_c": np.linspace(50, 60, 10),
            "pressure_psi": np.linspace(100, 90, 10),
            "oil_debris_ppm": np.linspace(1, 2, 10),
            "rotational_speed_deviation_pct": np.linspace(0, 1, 10),
        })
        featured = add_rolling_features(df)
        cols = feature_column_names()
        for c in cols:
            self.assertIn(c, featured.columns)


class TestAnomalyDetectionV2(unittest.TestCase):
    def test_train_score_pipeline_produces_valid_flags(self):
        df = _toy_featured_df()
        # monkeypatch feature_column_names is not needed here since this test
        # only exercises the parts of the pipeline that use a fixed small
        # feature subset present in _toy_featured_df; see integration note below.
        import anomaly_detection_v2 as adv2
        original_cols = adv2.feature_column_names
        adv2.feature_column_names = lambda: ["cycle", "vibration", "vibration_mean",
                                              "vibration_std", "vibration_min", "vibration_max"]
        try:
            models = train_anomaly_models(df)
            thresholds = compute_thresholds(models, df)
            scored = score_anomalies(models, thresholds, df)
            self.assertIn("is_anomaly", scored.columns)
            self.assertTrue(set(scored["is_anomaly"].unique()).issubset({0, 1}))
        finally:
            adv2.feature_column_names = original_cols


class TestRulModelV2(unittest.TestCase):
    def test_train_predict_reduces_error_vs_naive_baseline(self):
        df = _toy_featured_df()
        import rul_model_v2 as rmv2
        original_cols = rmv2.feature_column_names
        rmv2.feature_column_names = lambda: ["cycle", "vibration", "vibration_mean",
                                              "vibration_std", "vibration_min", "vibration_max"]
        try:
            train_df = df[df["cycle"] <= 30]
            test_df = df[df["cycle"] > 30].copy()
            models = train_rul_models(train_df)
            test_df["predicted_RUL"] = predict_rul(models, test_df)

            model_mae = evaluate(test_df["RUL"].values, test_df["predicted_RUL"].values)["MAE"]
            naive_mae = float(np.mean(np.abs(test_df["RUL"].values - train_df["RUL"].mean())))
            self.assertLess(model_mae, naive_mae)
        finally:
            rmv2.feature_column_names = original_cols


if __name__ == "__main__":
    unittest.main()
