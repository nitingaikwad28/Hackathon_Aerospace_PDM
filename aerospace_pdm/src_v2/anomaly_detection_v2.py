# anomaly_detection_v2.py
# Production-style anomaly detection: one scikit-learn IsolationForest per
# component type, trained on engineered rolling-window features (not just raw
# instantaneous readings). The alert threshold per component is TUNED ON A HELD-
# OUT VALIDATION SET (not hand-picked, and not sklearn's own contamination-based
# cutoff), which is standard practice: you decide "how suspicious is suspicious
# enough" using data the model wasn't fit on.

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import feature_column_names
from config import (IFOREST_N_ESTIMATORS, IFOREST_MAX_SAMPLES, IFOREST_MAX_FEATURES,
                     IFOREST_CONTAMINATION, IFOREST_N_JOBS, IFOREST_RANDOM_SEED)


def _anomaly_score(model, X):
    # scikit-learn's IsolationForest.score_samples returns LOWER values for more
    # abnormal points (the opposite of the original paper's convention). We negate
    # it so that, consistently with the rest of this project, HIGHER = more
    # anomalous - this is the only place that sign flip needs to happen.
    return -model.score_samples(X)


def train_anomaly_models(featured_train_df):
    # Trains one IsolationForest per component type, using ONLY rows from the
    # training split. Returns dict {component: fitted IsolationForest}.
    cols = feature_column_names()
    models = {}
    for component, comp_df in featured_train_df.groupby("component"):
        X = comp_df[cols].values
        model = IsolationForest(
            n_estimators=IFOREST_N_ESTIMATORS,
            max_samples=min(IFOREST_MAX_SAMPLES, len(X)),
            max_features=IFOREST_MAX_FEATURES,
            contamination=IFOREST_CONTAMINATION,
            n_jobs=IFOREST_N_JOBS,
            random_state=IFOREST_RANDOM_SEED,
        )
        model.fit(X)
        models[component] = model
    return models


def compute_thresholds(models, featured_val_df, contamination=IFOREST_CONTAMINATION):
    # Uses the VALIDATION split (never seen during training) to pick each
    # component's alert threshold: the score above which we call a reading
    # anomalous, set so that roughly `contamination` fraction of validation
    # readings would be flagged. This mirrors how a real deployment calibrates
    # alert sensitivity against a representative "recent normal operations"
    # sample, rather than trusting sklearn's own internal contamination cutoff
    # (which is fit on the training data, not a held-out set).
    cols = feature_column_names()
    thresholds = {}
    for component, model in models.items():
        comp_df = featured_val_df[featured_val_df["component"] == component]
        if len(comp_df) == 0:                     # fall back if no validation rows exist for this type
            thresholds[component] = 0.0
            continue
        scores = _anomaly_score(model, comp_df[cols].values)
        thresholds[component] = float(np.quantile(scores, 1 - contamination))
    return thresholds


def score_anomalies(models, thresholds, featured_df):
    # Scores every row using the RIGHT component's model + threshold.
    cols = feature_column_names()
    df = featured_df.copy()
    scores = np.zeros(len(df))
    for component, model in models.items():
        mask = (df["component"] == component).values
        if mask.sum() == 0:
            continue
        scores[mask] = _anomaly_score(model, df.loc[mask, cols].values)
    df["anomaly_score"] = scores
    df["anomaly_threshold"] = df["component"].map(thresholds).astype(float)
    df["is_anomaly"] = (df["anomaly_score"] > df["anomaly_threshold"]).astype(int)
    return df


if __name__ == "__main__":
    # quick manual test (run this on a machine with scikit-learn installed)
    from data_simulator_v2 import simulate_fleet, split_units
    from features import add_rolling_features

    fleet = simulate_fleet()
    splits = split_units(fleet)
    fleet = fleet.merge(splits, on="unit_id")
    featured = add_rolling_features(fleet)

    train_df = featured[featured["split"] == "train"]
    val_df = featured[featured["split"] == "val"]

    models = train_anomaly_models(train_df)
    thresholds = compute_thresholds(models, val_df)
    scored = score_anomalies(models, thresholds, featured)

    print("Thresholds per component:", thresholds)
    print(f"Flagged {scored['is_anomaly'].sum()} of {len(scored)} readings.")
