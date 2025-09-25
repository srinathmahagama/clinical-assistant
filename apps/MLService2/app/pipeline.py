from pathlib import Path
import pandas as pd, numpy as np, json
import joblib

APP_ROOT   = Path(__file__).resolve().parent      # /app (inside container)
MODELS_DIR = APP_ROOT / "models"

def _load_model_and_metadata():
    # Prefer XGB, else LR; clear error if neither found
    xgb = MODELS_DIR / "severity_xgb.joblib"
    lr  = MODELS_DIR / "severity_lr.joblib"
    model_path = xgb if xgb.exists() else lr
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found in {MODELS_DIR}. "
            "Run training (train_severity.py) to create a model, then rebuild the container."
        )

    model = joblib.load(model_path)
    features = pd.read_csv(MODELS_DIR / "feature_order.csv", header=None)[0].tolist()

    with open(MODELS_DIR / "label_classes.json") as f:
        label_list = json.load(f)

    # LR exposes encoded classes_; XGB doesn’t
    if hasattr(model, "classes_"):
        classes = [label_list[i] for i in model.classes_]
    else:
        classes = label_list

    return model, features, classes

MODEL, FEATURES, CLASSES = _load_model_and_metadata()

def _empty_row():
    return {c: 0 for c in FEATURES}

def predict_one(payload: dict):
    row = _empty_row()

    numeric_keys = {"age","hr","heart_rate","rr","resp_rate","sbp","dbp",
                    "spo2","o2sat","temp","temperature"}
    for k in numeric_keys:
        if (k in FEATURES) and (k in payload) and payload[k] is not None:
            row[k] = payload[k]

    # allow direct cc_* flags
    for c in FEATURES:
        if c.startswith("cc_") and (c in payload):
            row[c] = 1 if payload[c] else 0

    X = pd.DataFrame([row])[FEATURES]
    probs = MODEL.predict_proba(X)[0]
    label = CLASSES[int(np.argmax(probs))]
    return label, dict(zip(CLASSES, map(float, probs)))
