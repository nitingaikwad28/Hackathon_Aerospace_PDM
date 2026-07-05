# rul_model.py
# This file predicts "Remaining Useful Life" (RUL) - how many more cycles
# a part can safely run before it needs maintenance. We use a simple linear
# regression model (built with plain numpy math, no external ML library needed).
#
# IMPORTANT: engine / landing_gear / brakes / bogie sensors live on very different
# scales (e.g. engine temperature ~550 vs landing_gear temperature ~60), so we train
# ONE SMALL MODEL PER COMPONENT TYPE instead of one big mixed model. This keeps the
# math simple while giving each component type its own accurate model.

import numpy as np                        # for math and linear algebra
import pandas as pd                       # for tables

FEATURE_COLUMNS = ["cycle", "vibration", "temperature", "pressure"]  # inputs the model will use
TARGET_COLUMN = "RUL"                     # what we are trying to predict


def split_train_test(fleet_df, test_fraction=0.25, seed=1):
    # Split by unit_id (not by row!) so the same part never appears in BOTH train and test.
    unit_ids = fleet_df["unit_id"].unique()                    # list of all unique part ids
    rng = np.random.default_rng(seed)                          # random generator for reproducibility
    rng.shuffle(unit_ids)                                      # shuffle the part ids randomly

    n_test = int(len(unit_ids) * test_fraction)                # how many parts go into the test set
    test_ids = set(unit_ids[:n_test])                          # first chunk = test parts
    train_ids = set(unit_ids[n_test:])                         # rest = training parts

    train_df = fleet_df[fleet_df["unit_id"].isin(train_ids)]   # rows belonging to training parts
    test_df = fleet_df[fleet_df["unit_id"].isin(test_ids)]     # rows belonging to test parts
    return train_df, test_df                                   # return both halves


def add_bias_column(X):
    # Linear regression needs a constant "1" column so the line doesn't have to pass through zero.
    ones = np.ones((X.shape[0], 1))                            # column of 1s, one per row
    return np.hstack([ones, X])                                 # stick the 1s column in front of the features


def train_linear_model(train_df):
    # Trains ONE small linear model PER component type, and returns them all in a dictionary.
    weights_by_component = {}                                   # dict: component name -> learned weights
    for component, comp_df in train_df.groupby("component"):     # loop over engine / landing_gear / brakes / bogie
        X = comp_df[FEATURE_COLUMNS].values                      # feature matrix for just this component type
        y = comp_df[TARGET_COLUMN].values                        # true RUL values for just this component type
        X_with_bias = add_bias_column(X)                          # add the constant term

        # solve the classic linear regression equation using least squares (closed-form, no training loop)
        weights, _, _, _ = np.linalg.lstsq(X_with_bias, y, rcond=None)
        weights_by_component[component] = weights                 # save this component's own little model
    return weights_by_component                                   # give back all 4 models together


def predict_rul(weights_by_component, df):
    # Predicts RUL row-by-row, using the RIGHT small model for each row's component type.
    predictions = np.zeros(len(df))                                # start with an empty predictions array
    for component, weights in weights_by_component.items():         # loop over each component's trained model
        mask = (df["component"] == component).values                # find rows that belong to this component
        if mask.sum() == 0:                                          # skip if this component isn't in df at all
            continue
        X = df.loc[mask, FEATURE_COLUMNS].values                     # features for just those rows
        X_with_bias = add_bias_column(X)                              # add the constant term, same as training
        predictions[mask] = X_with_bias @ weights                     # matrix multiply = weighted sum = prediction
    predictions = np.clip(predictions, 0, None)                       # RUL can never be negative, so clip at 0
    return predictions                                                 # return predicted RUL for every row


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))                       # Mean Absolute Error: average prediction miss
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))               # Root Mean Squared Error: penalizes big misses more
    return {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2)}


if __name__ == "__main__":
    # quick manual test: run this file directly to train + check the RUL model
    from data_simulator import simulate_fleet                     # reuse our fake data generator
    fleet = simulate_fleet()                                       # build fake fleet data
    train_df, test_df = split_train_test(fleet)                     # split into train/test by unit

    weights_by_component = train_linear_model(train_df)              # train one small model per component
    test_predictions = predict_rul(weights_by_component, test_df)    # predict RUL for the held-out test parts

    metrics = evaluate(test_df[TARGET_COLUMN].values, test_predictions)  # compare predictions vs truth
    print("RUL model test performance:", metrics)                   # e.g. {'MAE': 8.5, 'RMSE': 11.2}
    for component, weights in weights_by_component.items():          # show each component's learned coefficients
        print(f"  {component}: {weights}")
