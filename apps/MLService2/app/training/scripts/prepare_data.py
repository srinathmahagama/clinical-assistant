# path: apps/MLService2/app/training/scripts/prepare_data.py
import pyreadr
import pandas as pd
import numpy as np
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

# 3) Select base features
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

# ---- Robust one-hot for low-cardinality categoricals ----
extra_cats = [
    "gender", "arrivalmode", "arrivalhour_bin", "previousdispo",
    "insurance_status", "employstatus", "maritalstatus"
]
cats = [c for c in extra_cats if c in df.columns]

for c in cats:
    s = df[c].astype("string").fillna("__missing__")
    top = s.value_counts(dropna=False).nlargest(30).index
    s = s.where(s.isin(top), "__other__")
    tmp = pd.get_dummies(s, prefix=c, dummy_na=False)
    X = pd.concat([X, tmp], axis=1)
    feat_cols.extend(list(tmp.columns))
# ---------------------------------------------------------

# 4) Numeric cleaning
numeric_cols = base_cols + vital_cols
for c in numeric_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

for c in cc_cols:
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[c] = X[c].fillna(0).astype(int)

# 5) Impute numeric + winsorize
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

# 6) Engineered clinical features (consolidated to avoid fragmentation)
eng = {}

if {"hr","sbp"}.issubset(X.columns):
    si = (X["hr"] / X["sbp"]).replace([np.inf, -np.inf], np.nan)
    eng["shock_index"] = pd.to_numeric(si, errors="coerce")

if {"sbp","dbp"}.issubset(X.columns):
    pp = X["sbp"] - X["dbp"]
    eng["pulse_pressure"] = pd.to_numeric(pp, errors="coerce")

def mk_flag(col, cond, name):
    if col in X.columns:
        eng[name] = cond(X[col]).astype("int8")

mk_flag("spo2", lambda s: s < 92,   "hypoxia_flag")
mk_flag("rr",   lambda s: s > 22,   "tachypnea_flag")
mk_flag("hr",   lambda s: s > 110,  "tachycardia_flag")
mk_flag("sbp",  lambda s: s < 100,  "hypotension_flag")
mk_flag("temp", lambda s: s >= 38,  "fever_flag")
mk_flag("age",  lambda s: s >= 75,  "age75_flag")

if eng:
    eng_df = pd.DataFrame(eng, index=X.index)
    # impute numeric engineered columns
    for c in eng_df.columns:
        if pd.api.types.is_numeric_dtype(eng_df[c]):
            eng_df[c] = pd.to_numeric(eng_df[c], errors="coerce").fillna(eng_df[c].median())
    X = pd.concat([X, eng_df], axis=1)

eng_cols = [c for c in ["shock_index","pulse_pressure","hypoxia_flag","tachypnea_flag",
                        "tachycardia_flag","hypotension_flag","fever_flag","age75_flag"]
            if c in X.columns]
feat_cols += [c for c in eng_cols if c not in feat_cols]

# 7) Cast to float32 for speed/memory
X = X.astype(np.float32)

# 8) Save processed parquet + metadata
X_out = DATA_OUT / "X.parquet"
y_out = DATA_OUT / "y.parquet"
X.to_parquet(X_out, index=False)
y.to_frame("severity").to_parquet(y_out, index=False)
print(f"[prepare_data] Saved: {X_out}, {y_out}")

meta = {
    "feature_order": list(X.columns),
    "numeric_cols": numeric_cols,
    "cc_cols_count": len(cc_cols),
    "engineered_cols": eng_cols,
    "class_counts": y.value_counts().to_dict()
}
with open(DATA_OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("[prepare_data] Saved metadata:", DATA_OUT / "meta.json")
