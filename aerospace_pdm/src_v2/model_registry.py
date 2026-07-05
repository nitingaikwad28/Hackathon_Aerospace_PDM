# model_registry.py
# A tiny "model registry": saves/loads trained model artifacts to disk, along with
# a JSON "model card" recording what was trained, when, with what hyperparameters,
# and how well it scored. This is the production practice of separating TRAINING
# (slow, done occasionally, offline) from SERVING (fast, done every time new
# telemetry arrives) - train.py writes artifacts here, predict_service.py reads them.

import os                                    # file paths
import json                                  # model card format
import pickle                                # serializes our custom Python model objects
from datetime import datetime, timezone


def save_models(models_by_component, name, artifact_dir, hyperparameters=None, metrics=None, extra=None):
    # models_by_component: dict {component_name: fitted model object}
    # Writes one .pkl file per component, plus/updates a shared model_card.json.
    os.makedirs(artifact_dir, exist_ok=True)
    for component, model in models_by_component.items():
        path = os.path.join(artifact_dir, f"{name}_{component}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

    card_path = os.path.join(artifact_dir, "model_card.json")
    card = {}
    if os.path.exists(card_path):                      # keep any other model's entry already in the card
        with open(card_path) as f:
            card = json.load(f)

    card[name] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "components": list(models_by_component.keys()),
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
        path = os.path.join(artifact_dir, f"{name}_{component}.pkl")
        with open(path, "rb") as f:
            models[component] = pickle.load(f)
    return models


def load_model_card(artifact_dir):
    card_path = os.path.join(artifact_dir, "model_card.json")
    if not os.path.exists(card_path):
        return {}
    with open(card_path) as f:
        return json.load(f)
