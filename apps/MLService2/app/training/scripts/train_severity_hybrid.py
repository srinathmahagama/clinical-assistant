# path: apps/MLService2/app/training/scripts/train_severity_hybrid.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

# NOTE: save artifacts into processed dir (per your structure)
APP_ROOT = Path(__file__).resolve().parents[2]         # .../apps/MLService2/app
PROC     = APP_ROOT / "training" / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

print("[hybrid] Loading processed data...")
X = pd.read_parquet(PROC / "X.parquet")
y = pd.read_parquet(PROC / "y.parquet")["severity"].astype(str)

# Clean labels
valid = y.isin({"Mild", "Moderate", "Severe"})
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("[hybrid] Classes:", le.classes_.tolist())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Scale continuous columns (must match prepare_data continuous_candidates)
continuous_candidates = [
    "age","hr","rr","sbp","dbp","spo2","temp",
    "shock_index","pulse_pressure","mean_arterial_pressure","spo2_deficit"
]
scale_cols = [c for c in continuous_candidates if c in X.columns]

scaler = None
if scale_cols:
    scaler = StandardScaler()
    X_train.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test.loc[:, scale_cols]  = scaler.transform(X_test[scale_cols])

# Base model: Logistic Regression
print("\n[hybrid] Training Logistic Regression base model...")
lr = LogisticRegression(
    max_iter=5000,
    C=10,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=-1
)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average="macro")
print(f"[LR] accuracy={lr_acc:.4f}, macro-F1={lr_f1:.4f}")
print(classification_report(le.inverse_transform(y_test),
                            le.inverse_transform(lr_pred)))

# Meta features (LR probabilities)
print("\n[hybrid] Generating meta features (LR probabilities)...")
lr_train_probs = lr.predict_proba(X_train)
lr_test_probs  = lr.predict_proba(X_test)

lr_prob_cols = [f"lr_prob_{cls}" for cls in le.classes_]
X_train_meta = pd.concat([X_train.reset_index(drop=True),
                          pd.DataFrame(lr_train_probs, columns=lr_prob_cols)], axis=1)
X_test_meta  = pd.concat([X_test.reset_index(drop=True),
                          pd.DataFrame(lr_test_probs,  columns=lr_prob_cols)],  axis=1)

# Clean colnames (parity with inference)
def clean_colnames(df):
    df.columns = (
        df.columns
        .str.replace(r"[\[\]\(\)<>\s]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
    )
    return df

X_train_meta = clean_colnames(X_train_meta)
X_test_meta  = clean_colnames(X_test_meta)

# Meta model: XGBoost
print("[hybrid] Training XGBoost meta model...")
weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_map = {i: w for i, w in enumerate(weights)}
sample_weights = np.array([class_weight_map[i] for i in y_train], dtype=np.float32)

xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    min_child_weight=1,
    gamma=0.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    nthread=-1,
    verbosity=0,
    random_state=42,
)
xgb.fit(X_train_meta, y_train, sample_weight=sample_weights)

# Evaluate hybrid
xgb_pred = xgb.predict(X_test_meta)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average="macro")
print("\n[XGB Hybrid] accuracy={:.4f}, macro-F1={:.4f}".format(xgb_acc, xgb_f1))
print(classification_report(le.inverse_transform(y_test),
                            le.inverse_transform(xgb_pred)))

# Save artifacts to processed dir
joblib.dump({"lr": lr, "xgb": xgb, "scaler": scaler}, PROC / "severity_hybrid.joblib")

# Persist labels
with open(PROC / "label_classes.json", "w") as f:
    json.dump(le.classes_.tolist(), f)

# Persist meta feature order & serving contract
meta_feature_order = X_train_meta.columns.tolist()
persist = {
    "label_classes": le.classes_.tolist(),
    "scale_cols": [c for c in scale_cols if c in X.columns],
    "lr_prob_cols": lr_prob_cols,
    "meta_feature_order": meta_feature_order,
    "clean_names": True
}
with open(PROC / "severity_hybrid_meta.json", "w") as f:
    json.dump(persist, f, indent=2)

# Overwrite feature_order.csv with META columns (base + lr_prob_*)
pd.Series(meta_feature_order).to_csv(PROC / "feature_order.csv", index=False)

# Training metrics
metrics = {
    "lr_acc": float(lr_acc), "lr_f1": float(lr_f1),
    "xgb_acc": float(xgb_acc), "xgb_f1": float(xgb_f1),
    "improvement": round(xgb_acc - lr_acc, 4)
}
with open(PROC / "training_metrics_hybrid.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[hybrid] Saved hybrid model to {PROC} (acc={xgb_acc:.3f}, macro-F1={xgb_f1:.3f})")
