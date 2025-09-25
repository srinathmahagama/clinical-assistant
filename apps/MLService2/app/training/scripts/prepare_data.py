# path: apps/MLService2/app/training/scripts/prepare_data.py
import pyreadr
import pandas as pd
from pathlib import Path
import json

# Resolve /app (the "app" folder inside MLService2)
APP_ROOT = Path(__file__).resolve().parents[2]     # .../MLService2/app
DATA_RAW = APP_ROOT / "training" / "data" / "raw"
DATA_OUT = APP_ROOT / "training" / "data" / "processed"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# 1) Load the RData (first data.frame)
res = pyreadr.read_r(DATA_RAW / "5v_cleandf.rdata")
df = next(iter(res.values()))

# 2) Map ESI -> Mild/Moderate/Severe
def map_severity(x):
    xi = int(str(x).strip())
    if xi in (1, 2):
        return "Severe"
    if xi == 3:
        return "Moderate"
    return "Mild"  # 4 or 5

df["severity"] = df["esi"].apply(map_severity)

# 3) Select features
cc_cols = [c for c in df.columns if str(c).startswith("cc_")]
vital_aliases = {
    "hr", "heart_rate", "rr", "resp_rate", "sbp", "dbp",
    "spo2", "o2sat", "temp", "temperature"
}
vital_cols = [c for c in df.columns if str(c).lower() in vital_aliases]
base_cols  = [c for c in ["age"] if c in df.columns]

feat_cols = base_cols + vital_cols + cc_cols
X = df[feat_cols].copy()
y = df["severity"].copy()

# 4) Types/cleaning
numeric_cols = base_cols + vital_cols
for c in numeric_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

for c in cc_cols:
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[c] = X[c].fillna(0).astype(int)

# 5) Impute numeric with median + optional winsorize
if numeric_cols:
    X[numeric_cols] = X[numeric_cols].apply(
        lambda s: s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s
    )

def winsorize(s, lo=0.005, hi=0.995):
    if not pd.api.types.is_numeric_dtype(s):
        return s
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)

for c in numeric_cols:
    X[c] = winsorize(X[c])

# 6) Save processed parquet + metadata
X_out = DATA_OUT / "X.parquet"
y_out = DATA_OUT / "y.parquet"
X.to_parquet(X_out, index=False)
y.to_frame("severity").to_parquet(y_out, index=False)
print(f"[prepare_data] Saved: {X_out}, {y_out}")

meta = {
    "feature_order": feat_cols,
    "numeric_cols": numeric_cols,
    "cc_cols_count": len(cc_cols),
    "class_counts": y.value_counts().to_dict()
}
with open(DATA_OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("[prepare_data] Saved metadata:", DATA_OUT / "meta.json")
