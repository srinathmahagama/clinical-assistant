# path: apps/MLService2/app/training/scripts/train_severity.py
from pathlib import Path
import json, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

APP_ROOT = Path(__file__).resolve().parents[2]  # .../MLService2/app
PROC     = APP_ROOT / "training" / "data" / "processed"
MODELS   = APP_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# 1) Load processed data
X = pd.read_parquet(PROC / "X.parquet")
y = pd.read_parquet(PROC / "y.parquet")["severity"]

# 2) Clean labels
mask = y.notna() & y.astype(str).str.len().gt(0)
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].astype(str).reset_index(drop=True)

vc = y.value_counts()
print("[train] class counts:\n", vc)
if vc.size < 2:
    print("[train] ERROR: Need at least two classes")
    sys.exit(1)

# 3) Encode labels for models
le = LabelEncoder()
y_enc = le.fit_transform(y)
label_list = le.classes_.tolist()
print("[train] label order:", label_list)

# 4) Train/val split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 5) Logistic Regression (baseline)
lr = LogisticRegression(max_iter=4000, class_weight="balanced")
lr.fit(X_tr, y_tr)
lr_pred = lr.predict(X_te)
lr_acc  = accuracy_score(y_te, lr_pred)
print("\n[LR] accuracy:", lr_acc)
print(classification_report(le.inverse_transform(y_te),
                            le.inverse_transform(lr_pred)))

# 6) XGBoost
xgb = XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    objective="multi:softprob", eval_metric="mlogloss"
)
xgb.fit(X_tr, y_tr)
xgb_pred = xgb.predict(X_te)
xgb_acc  = accuracy_score(y_te, xgb_pred)
print("\n[XGB] accuracy:", xgb_acc)
print(classification_report(le.inverse_transform(y_te),
                            le.inverse_transform(xgb_pred)))

# 7) Save best model + artifacts to app/models
best_model, name, best_acc = (xgb, "severity_xgb.joblib", xgb_acc)
if lr_acc >= xgb_acc:
    best_model, name, best_acc = (lr, "severity_lr.joblib", lr_acc)

joblib.dump(best_model, MODELS / name)
X.columns.to_series().to_csv(MODELS / "feature_order.csv", index=False)
with open(MODELS / "label_classes.json", "w") as f:
    json.dump(label_list, f)

print(f"[train] Saved model: {name}  (acc={best_acc:.3f})")
print("        Saved feature_order.csv and label_classes.json to app/models")
