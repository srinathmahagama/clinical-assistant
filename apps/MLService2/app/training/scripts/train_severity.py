# path: apps/MLService2/app/training/scripts/train_severity.py
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
import joblib

APP_ROOT = Path(__file__).resolve().parents[2]  # .../MLService2/app
PROC     = APP_ROOT / "training" / "data" / "processed"
MODELS   = APP_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Load processed data
# -----------------------------
X = pd.read_parquet(PROC / "X.parquet")
y = pd.read_parquet(PROC / "y.parquet")["severity"].astype(str)

# -----------------------------
# 2) Keep only valid labels
# -----------------------------
valid_labels = {"Mild", "Moderate", "Severe"}
mask = y.isin(valid_labels)
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

# -----------------------------
# 3) Encode labels
# -----------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)
label_list = le.classes_.tolist()
print("[train] label order:", label_list)

# -----------------------------
# 4) Train/Val/Test split
#    80% train, 20% test; then 10% of train -> val
# -----------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
tr_idx, va_idx = next(sss.split(X_tr, y_tr))

# IMPORTANT: copy() to avoid SettingWithCopyWarning
X_tr_sub, X_va = X_tr.iloc[tr_idx].copy(), X_tr.iloc[va_idx].copy()
y_tr_sub, y_va = y_tr[tr_idx], y_tr[va_idx]
X_tr, X_te     = X_tr.copy(), X_te.copy()

# -----------------------------
# 5) Decide which columns to scale (continuous only)
# -----------------------------
continuous_candidates = [
    "age", "hr", "heart_rate", "rr", "resp_rate", "sbp", "dbp",
    "spo2", "o2sat", "temp", "temperature",
    "shock_index", "pulse_pressure"
]
scale_cols = [c for c in continuous_candidates if c in X.columns]

scaler = None
if scale_cols:
    scaler = StandardScaler()
    X_tr_sub[scale_cols] = scaler.fit_transform(X_tr_sub[scale_cols])
    X_va[scale_cols]     = scaler.transform(X_va[scale_cols])
    X_tr[scale_cols]     = scaler.transform(X_tr[scale_cols])
    X_te[scale_cols]     = scaler.transform(X_te[scale_cols])

# -----------------------------
# 6) Logistic Regression baseline (class-balanced)
# -----------------------------
lr = LogisticRegression(max_iter=4000, class_weight="balanced")
lr.fit(X_tr, y_tr)
lr_pred = lr.predict(X_te)
lr_acc  = accuracy_score(y_te, lr_pred)
lr_f1   = f1_score(y_te, lr_pred, average="macro")
print("\n[LR] accuracy:", lr_acc, "  macro-F1:", lr_f1)
print(classification_report(le.inverse_transform(y_te),
                            le.inverse_transform(lr_pred)))

# -----------------------------
# 7) XGBoost (load tuned params if available) + class-balanced sample weights
# -----------------------------
best_params_path = PROC / "best_xgb_params.json"
if best_params_path.exists():
    with open(best_params_path) as f:
        xgb_params = json.load(f)
else:
    xgb_params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        reg_alpha=0.0, min_child_weight=1, gamma=0.0,
        objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", nthread=-1, verbosity=0,
    )

# Ensure essentials even if missing in tuned JSON
xgb_params.setdefault("objective", "multi:softprob")
xgb_params.setdefault("eval_metric", "mlogloss")
xgb_params.setdefault("tree_method", "hist")
xgb_params.setdefault("nthread", -1)
xgb_params.setdefault("verbosity", 0)

xgb = XGBClassifier(**xgb_params)

# Balanced sample weights on train-sub
weights = compute_class_weight(class_weight="balanced",
                               classes=np.unique(y_tr_sub), y=y_tr_sub)
class_weight_map = {i: w for i, w in enumerate(weights)}
w_tr = np.array([class_weight_map[i] for i in y_tr_sub], dtype=np.float32)

# Fit (no callbacks to stay version-compatible)
xgb.fit(X_tr_sub, y_tr_sub, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)

xgb_pred = xgb.predict(X_te)
xgb_acc  = accuracy_score(y_te, xgb_pred)
xgb_f1   = f1_score(y_te, xgb_pred, average="macro")
print("\n[XGB] accuracy:", xgb_acc, "  macro-F1:", xgb_f1)
print(classification_report(le.inverse_transform(y_te),
                            le.inverse_transform(xgb_pred)))

# -----------------------------
# 8) Pick best & save artifacts
# -----------------------------
if (xgb_f1 > lr_f1) or (xgb_f1 == lr_f1 and xgb_acc >= lr_acc):
    best_model, name, best_acc, best_f1 = (xgb, "severity_xgb.joblib", xgb_acc, xgb_f1)
else:
    best_model, name, best_acc, best_f1 = (lr, "severity_lr.joblib", lr_acc, lr_f1)

joblib.dump(best_model, MODELS / name)

# Save scaler (and columns) so inference mirrors training
if scaler is not None:
    joblib.dump({"scaler": scaler, "columns": scale_cols}, MODELS / "scaler.joblib")

# Save feature order & label classes for pipeline
X.columns.to_series().to_csv(MODELS / "feature_order.csv", index=False)
with open(MODELS / "label_classes.json", "w") as f:
    json.dump(label_list, f)

# Save a small metrics snapshot (optional)
with open(PROC / "training_metrics.json", "w") as f:
    json.dump(
        {
            "lr_acc": float(lr_acc), "lr_f1": float(lr_f1),
            "xgb_acc": float(xgb_acc), "xgb_f1": float(xgb_f1),
            "chosen": name
        },
        f, indent=2
    )

print(f"\n[train] Saved model: {name}  (acc={best_acc:.3f}, macro-F1={best_f1:.3f})")
print("        Saved feature_order.csv, label_classes.json, scaler.joblib (if used) to app/models")
