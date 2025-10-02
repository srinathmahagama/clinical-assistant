# path: apps/MLService2/app/training/scripts/train_severity.py
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

APP_ROOT = Path(__file__).resolve().parents[2]  # .../MLService2/app
PROC     = APP_ROOT / "training" / "data" / "processed"
MODELS   = APP_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# 1) Load processed data
X = pd.read_parquet(PROC / "X.parquet").astype(np.float32)
y = pd.read_parquet(PROC / "y.parquet")["severity"]

# 2) Clean labels
mask = y.notna() & y.astype(str).str.len().gt(0)
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].astype(str).reset_index(drop=True)

vc = y.value_counts()
print("[train] class counts:\n", vc)
if vc.size < 2:
    print("[train] ERROR: Need at least two classes"); sys.exit(1)

# 3) Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
label_list = le.classes_.tolist()
print("[train] label order:", label_list)

# 4) Train/Val/Test split (80/20; then 10% of train => val)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
tr_idx, va_idx = next(sss.split(X_tr, y_tr))
X_tr_sub, X_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
y_tr_sub, y_va = y_tr[tr_idx], y_tr[va_idx]

# 5) Logistic Regression baseline
lr = LogisticRegression(max_iter=4000, class_weight="balanced")
lr.fit(X_tr, y_tr)
lr_pred = lr.predict(X_te)
lr_acc  = accuracy_score(y_te, lr_pred)
lr_f1   = f1_score(y_te, lr_pred, average="macro")
print("\n[LR] accuracy:", lr_acc, "  macro-F1:", lr_f1)
print(classification_report(le.inverse_transform(y_te),
                            le.inverse_transform(lr_pred)))

# 6) XGBoost (load tuned params if available) + class-balanced sample weights
best_params_path = PROC / "best_xgb_params.json"
if best_params_path.exists():
    with open(best_params_path) as f:
        xgb_params = json.load(f)
else:
    xgb_params = dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        reg_alpha=0.0, min_child_weight=1.0,
        objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", nthread=-1, verbosity=0,
    )

xgb = XGBClassifier(**xgb_params)

binc = np.bincount(y_tr_sub)
wmap = {i: (1.0 / cnt) for i, cnt in enumerate(binc)}
w_tr = np.array([wmap[i] for i in y_tr_sub], dtype=np.float32)

xgb.fit(X_tr_sub, y_tr_sub, sample_weight=w_tr)
xgb_pred = xgb.predict(X_te)
xgb_acc  = accuracy_score(y_te, xgb_pred)
xgb_f1   = f1_score(y_te, xgb_pred, average="macro")
print("\n[XGB] accuracy:", xgb_acc, "  macro-F1:", xgb_f1)
print(classification_report(le.inverse_transform(y_te),
                            le.inverse_transform(xgb_pred)))

# 7) Pick best & save artifacts
best_model, name, best_acc, best_f1 = (xgb, "severity_xgb.joblib", xgb_acc, xgb_f1)
if (lr_f1 > xgb_f1) or (lr_f1 == xgb_f1 and lr_acc >= xgb_acc):
    best_model, name, best_acc, best_f1 = (lr, "severity_lr.joblib", lr_acc, lr_f1)

joblib.dump(best_model, MODELS / name)
X.columns.to_series().to_csv(MODELS / "feature_order.csv", index=False)
with open(MODELS / "label_classes.json", "w") as f:
    json.dump(label_list, f)

# Save quick metrics snapshot (optional)
with open(PROC / "training_metrics.json", "w") as f:
    json.dump({"lr_acc": float(lr_acc), "lr_f1": float(lr_f1),
               "xgb_acc": float(xgb_acc), "xgb_f1": float(xgb_f1),
               "chosen": name}, f, indent=2)

print(f"\n[train] Saved model: {name}  (acc={best_acc:.3f}, macro-F1={best_f1:.3f})")
print("        Saved feature_order.csv and label_classes.json to app/models")
