# test_models.py
# Basic sanity/unit tests for the from-scratch model code, using only Python's
# built-in `unittest` (no pytest dependency needed - keeps this runnable anywhere).
# Run with:  python -m unittest tests/test_models.py  (from src_v2/)

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from isolation_forest import IsolationForest
from random_forest import RandomForestRegressor
from features import _slope, add_rolling_features, feature_column_names


class TestIsolationForest(unittest.TestCase):
    def test_outliers_score_higher_than_inliers(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1, size=(200, 3))
        outliers = rng.uniform(8, 10, size=(8, 3))
        X = np.vstack([normal, outliers])

        forest = IsolationForest(n_trees=60, subsample_size=128, random_state=1).fit(X)
        scores = forest.score(X)

        self.assertGreater(scores[200:].mean(), scores[:200].mean())

    def test_scores_are_bounded(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(100, 2))
        forest = IsolationForest(n_trees=20, subsample_size=64, random_state=1).fit(X)
        scores = forest.score(X)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1.01))     # allow tiny floating point slack above 1.0


class TestRandomForestRegressor(unittest.TestCase):
    def test_fits_a_simple_linear_relationship(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-5, 5, size=(300, 3))
        y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, size=300)

        rf = RandomForestRegressor(n_trees=15, max_depth=5, random_state=1).fit(X[:250], y[:250])
        preds = rf.predict(X[250:])
        mae = np.mean(np.abs(preds - y[250:]))
        self.assertLess(mae, 2.0)                   # should learn the relationship reasonably well

    def test_important_feature_ranked_above_irrelevant_one(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-5, 5, size=(300, 3))
        y = 5 * X[:, 0] + rng.normal(0, 0.1, size=300)          # only column 0 matters

        rf = RandomForestRegressor(n_trees=15, max_depth=5, random_state=1).fit(X, y)
        importances = rf.feature_importances(3)
        self.assertEqual(np.argmax(importances), 0)


class TestFeatureEngineering(unittest.TestCase):
    def test_slope_of_rising_series_is_positive(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertGreater(_slope(values), 0)

    def test_slope_of_flat_series_is_zero(self):
        values = np.array([3.0, 3.0, 3.0, 3.0])
        self.assertAlmostEqual(_slope(values), 0.0, places=6)

    def test_feature_column_count_matches_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({
            "unit_id": ["u1"] * 10,
            "cycle": range(1, 11),
            "vibration": np.linspace(0.1, 0.2, 10),
            "temperature": np.linspace(50, 60, 10),
            "pressure": np.linspace(100, 90, 10),
            "oil_debris": np.linspace(1, 2, 10),
            "rpm_deviation": np.linspace(0, 1, 10),
        })
        featured = add_rolling_features(df)
        cols = feature_column_names()
        for c in cols:
            self.assertIn(c, featured.columns)


if __name__ == "__main__":
    unittest.main()
