# model_registry.py
# A tiny "model registry": saves/loads trained scikit-learn model artifacts to
# disk using joblib (the standard, scikit-learn-recommended serialization tool -
# more efficient than raw pickle for objects holding large numpy arrays), along
# with a JSON "model card" recording what was trained, when, with what
# hyperparameters, and how well it scored. This is the production practice of
# separating TRAINING (slow, done occasionally, offline) from SERVING (fast, done
# every time new telemetry arrives) - train.py writes artifacts here,
# predict_service.py reads them.

import os                                    # file paths
import json                                  # model card format
import joblib                                # scikit-learn's recommended model serialization tool
from datetime import datetime, timezone


def save_models(models_by_component, name, artifact_dir, hyperparameters=None, metrics=None, extra=None):
    # models_by_component: dict {component_name: fitted scikit-learn estimator}
    # Writes one .joblib file per component, plus/updates a shared model_card.json.
    os.makedirs(artifact_dir, exist_ok=True)
    for component, model in models_by_component.items():
        path = os.path.join(artifact_dir, f"{name}_{component}.joblib")
        joblib.dump(model, path)

    card_path = os.path.join(artifact_dir, "model_card.json")
    card = {}
    if os.path.exists(card_path):                      # keep any other model's entry already in the card
        with open(card_path) as f:
            card = json.load(f)

    card[name] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "components": list(models_by_component.keys()),
        "library": "scikit-learn",
        "sklearn_estimator": type(next(iter(models_by_component.values()))).__name__,
        "hyperparameters": hyperparameters or {},
        "metrics": metrics or {},
        "extra": extra or {},
    }
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Saved {len(models_by_component)} '{name}' model(s) + model card -> {artifact_dir}/")


def load_models(name, components, artifact_dir):
    # Loads back the per-component models saved by save_models().
    models = {}
    for component in components:
        path = os.path.join(artifact_dir, f"{name}_{component}.joblib")
        models[component] = joblib.load(path)
    return models


def load_model_card(artifact_dir):
    card_path = os.path.join(artifact_dir, "model_card.json")
    if not os.path.exists(card_path):
        return {}
    with open(card_path) as f:
        return json.load(f)
