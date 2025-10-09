# path: apps/MLService2/app/training/scripts/prepare_data.py
import pyreadr
import pandas as pd
import numpy as np
import re
from pathlib import Path
import json

# Resolve /app (the "app" folder inside MLService2)
APP_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = APP_ROOT / "training" / "data" / "raw"
DATA_OUT = APP_ROOT / "training" / "data" / "processed"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# 1) Load RData
res = pyreadr.read_r(DATA_RAW / "5v_cleandf.rdata")
df = next(iter(res.values()))

# 2) Map ESI -> Severity
def map_severity_safe(x):
    try:
        xi = int(str(x).strip())
    except Exception:
        return np.nan
    if xi in (1, 2):
        return "Severe"
    if xi == 3:
        return "Moderate"
    if xi in (4, 5):
        return "Mild"
    return np.nan

df["severity"] = df["esi"].apply(map_severity_safe)
df = df[df["severity"].notna()].reset_index(drop=True)

# 3) --- Robust Vital Alias Detection ---
def find_first(df, regexes):
    for rx in regexes:
        for c in df.columns:
            if re.search(rx, str(c), flags=re.I):
                return str(c)
    return None

vital_map = {
    "hr":   [r"triage[_]?vital[_]?hr", r"\bheart[_ ]?rate\b", r"\bpulse\b"],
    "rr":   [r"triage[_]?vital[_]?rr", r"\bresp\b", r"respiratory[_ ]?rate"],
    "sbp":  [r"triage[_]?vital[_]?sbp", r"\bsystolic\b"],
    "dbp":  [r"triage[_]?vital[_]?dbp", r"\bdiastolic\b"],
    "spo2": [r"triage[_]?vital[_]?spo2", r"o2sat", r"oxygen", r"oximetry", r"sp02"],
    "temp": [r"triage[_]?vital[_]?temp", r"temperature", r"temp[_ ]?[cf]"],
}

resolved = {}
for key, pats in vital_map.items():
    col = find_first(df, pats)
    if col:
        resolved[key] = col

print("[prepare_data] resolved vital columns:", resolved)

# create canonical numeric columns
for canon, src in resolved.items():
    df[canon] = pd.to_numeric(df[src], errors="coerce")

vital_cols = list(resolved.keys())
base_cols = [c for c in ["age"] if c in df.columns]

# 4) Chief complaint and categorical setup
cc_cols = [c for c in df.columns if str(c).startswith("cc_")]
extra_cats = [
    "gender", "arrivalmode", "arrivalhour_bin", "previousdispo",
    "insurance_status", "employstatus", "maritalstatus"
]
cats = [c for c in extra_cats if c in df.columns]

X_blocks = []
feat_cols = []

# Base numeric
base_numeric = df[base_cols + vital_cols].copy()
for c in base_numeric.columns:
    base_numeric[c] = pd.to_numeric(base_numeric[c], errors="coerce")
X_blocks.append(base_numeric)
feat_cols.extend(list(base_numeric.columns))

# CC one-hots
if cc_cols:
    cc_block = df[cc_cols].copy()
    for c in cc_cols:
        cc_block[c] = pd.to_numeric(cc_block[c], errors="coerce").fillna(0).astype(np.int8)
    X_blocks.append(cc_block)
    feat_cols.extend(cc_cols)

# Categoricals one-hot
cat_blocks = []
for c in cats:
    s = df[c].astype("string").fillna("__missing__")
    top = s.value_counts(dropna=False).nlargest(30).index
    s = s.where(s.isin(top), "__other__")
    tmp = pd.get_dummies(s, prefix=c, dummy_na=False)
    cat_blocks.append(tmp)
    feat_cols.extend(list(tmp.columns))
if cat_blocks:
    X_blocks.append(pd.concat(cat_blocks, axis=1))

# 5) Impute + Winsorize numeric
def winsorize(s, lo=0.005, hi=0.995):
    if not pd.api.types.is_numeric_dtype(s): return s
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)

if not base_numeric.empty:
    for c in base_numeric.columns:
        base_numeric[c] = base_numeric[c].fillna(base_numeric[c].median())
        base_numeric[c] = winsorize(base_numeric[c])

# 6) Engineered clinical features
eng = {}
if {"hr","sbp"}.issubset(base_numeric.columns):
    si = (base_numeric["hr"] / base_numeric["sbp"]).replace([np.inf, -np.inf], np.nan)
    eng["shock_index"] = si
if {"sbp","dbp"}.issubset(base_numeric.columns):
    eng["pulse_pressure"] = base_numeric["sbp"] - base_numeric["dbp"]
    eng["mean_arterial_pressure"] = (2 * base_numeric["dbp"] + base_numeric["sbp"]) / 3
if "spo2" in base_numeric.columns:
    eng["spo2_deficit"] = (100 - base_numeric["spo2"]).clip(lower=0)

def mk_flag_from(s, cond):
    return cond(s).astype("int8")

if "sbp" in base_numeric.columns:
    eng["hypotension_flag"] = mk_flag_from(base_numeric["sbp"], lambda s: s < 100)
if "hr" in base_numeric.columns:
    eng["tachycardia_flag"] = mk_flag_from(base_numeric["hr"], lambda s: s > 110)
if "rr" in base_numeric.columns:
    eng["tachypnea_flag"] = mk_flag_from(base_numeric["rr"], lambda s: s > 22)
if "spo2" in base_numeric.columns:
    eng["hypoxia_flag"] = mk_flag_from(base_numeric["spo2"], lambda s: s < 92)
if "temp" in base_numeric.columns:
    eng["fever_flag"] = mk_flag_from(base_numeric["temp"], lambda s: s >= 38)
if "age" in base_numeric.columns:
    eng["age75_flag"] = mk_flag_from(base_numeric["age"], lambda s: s >= 75)

eng_df = pd.DataFrame(eng, index=df.index)
for c in eng_df.columns:
    eng_df[c] = pd.to_numeric(eng_df[c], errors="coerce").fillna(eng_df[c].median())

if not eng_df.empty:
    X_blocks.append(eng_df)
    feat_cols.extend(eng_df.columns.tolist())

# 7) Binned versions (optional)
bin_blocks = []
if "age" in base_numeric.columns:
    age_bins = pd.cut(base_numeric["age"], bins=[0,18,30,45,60,75,200], right=False)
    bin_blocks.append(pd.get_dummies(age_bins, prefix="age_bin"))
if "temp" in base_numeric.columns:
    temp_bins = pd.cut(base_numeric["temp"], bins=[30,36,37.5,38.5,41], right=False)
    bin_blocks.append(pd.get_dummies(temp_bins, prefix="temp_bin"))
if "hr" in base_numeric.columns:
    hr_bins = pd.cut(base_numeric["hr"], bins=[0,60,100,120,200], right=False)
    bin_blocks.append(pd.get_dummies(hr_bins, prefix="hr_bin"))
if "sbp" in base_numeric.columns:
    sbp_bins = pd.cut(base_numeric["sbp"], bins=[0,90,110,140,260], right=False)
    bin_blocks.append(pd.get_dummies(sbp_bins, prefix="sbp_bin"))
if bin_blocks:
    bins_df = pd.concat(bin_blocks, axis=1)
    X_blocks.append(bins_df)
    feat_cols.extend(bins_df.columns.tolist())

# 8) Combine all
X = pd.concat(X_blocks, axis=1)
X = X.astype(np.float32)
y = df["severity"].copy()

# 9) Save processed data
X_out = DATA_OUT / "X.parquet"
y_out = DATA_OUT / "y.parquet"
X.to_parquet(X_out, index=False)
y.to_frame("severity").to_parquet(y_out, index=False)
print(f"[prepare_data] Saved: {X_out}, {y_out}")

meta = {
    "feature_order": list(X.columns),
    "resolved_vitals": resolved,
    "base_numeric": base_cols + vital_cols,
    "engineered_cols": list(eng_df.columns),
    "bins_added": bool(bin_blocks),
    "continuous_candidates": [
        "age","hr","rr","sbp","dbp","spo2","temp",
        "shock_index","pulse_pressure","mean_arterial_pressure","spo2_deficit"
    ],
    "class_counts": y.value_counts().to_dict(),
}
with open(DATA_OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("[prepare_data] Saved metadata:", DATA_OUT / "meta.json")
