# rul_model_v2.py
# Production-style Remaining Useful Life (RUL) estimation: one scikit-learn
# RandomForestRegressor per component type, trained on engineered rolling-window
# features instead of raw instantaneous sensor values. Random Forests handle the
# very different sensor scales across component types natively (tree splits
# don't care about feature scale), are far more expressive than linear
# regression at capturing the accelerating, non-linear wear curve, and give a
# feature importance readout for free.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from features import feature_column_names
from config import (RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF,
                     RF_MAX_FEATURES, RF_N_JOBS, RF_RANDOM_SEED)

TARGET_COLUMN = "RUL"


def train_rul_models(featured_train_df):
    # Trains one Random Forest per component type, using ONLY the training split.
    # Returns dict {component: fitted RandomForestRegressor}.
    cols = feature_column_names()
    models = {}
    for component, comp_df in featured_train_df.groupby("component"):
        X = comp_df[cols].values
        y = comp_df[TARGET_COLUMN].values
        model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            n_jobs=RF_N_JOBS,
            random_state=RF_RANDOM_SEED,
        )
        model.fit(X, y)
        models[component] = model
    return models


def predict_rul(models, df):
    # Predicts RUL row-by-row, using the RIGHT model for each row's component type.
    cols = feature_column_names()
    predictions = np.zeros(len(df))
    for component, model in models.items():
        mask = (df["component"] == component).values
        if mask.sum() == 0:
            continue
        predictions[mask] = model.predict(df.loc[mask, cols].values)
    return np.clip(predictions, 0, None)          # RUL can never be negative


def get_feature_importances(models):
    # scikit-learn exposes this as the `feature_importances_` attribute directly
    # (no method call needed) once a model is fitted.
    cols = feature_column_names()
    return {component: dict(zip(cols, model.feature_importances_)) for component, model in models.items()}


def evaluate(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}


def evaluate_by_component(df_with_predictions):
    # df_with_predictions must have component, RUL (actual), predicted_RUL columns
    results = {}
    for component, comp_df in df_with_predictions.groupby("component"):
        m = evaluate(comp_df[TARGET_COLUMN].values, comp_df["predicted_RUL"].values)
        m["n_rows"] = int(len(comp_df))
        results[component] = m
    return results


if __name__ == "__main__":
    # quick manual test (run this on a machine with scikit-learn installed):
    # train on train split, evaluate on held-out test split
    from data_simulator_v2 import simulate_fleet, split_units
    from features import add_rolling_features

    fleet = simulate_fleet()
    splits = split_units(fleet)
    fleet = fleet.merge(splits, on="unit_id")
    featured = add_rolling_features(fleet)

    train_df = featured[featured["split"] == "train"]
    test_df = featured[featured["split"] == "test"].copy()

    models = train_rul_models(train_df)
    test_df["predicted_RUL"] = predict_rul(models, test_df)

    overall = evaluate(test_df[TARGET_COLUMN].values, test_df["predicted_RUL"].values)
    by_component = evaluate_by_component(test_df)
    print("Overall test performance:", overall)
    print("Per-component test performance:", by_component)
