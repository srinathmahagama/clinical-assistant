# path: apps/MLService2/app/pipeline.py
from pathlib import Path
from typing import Dict, Tuple, Any, List
import json, math, re
import numpy as np
import pandas as pd
import joblib

APP_ROOT = Path(__file__).resolve().parent  # .../apps/MLService2/app
# Artifacts live in processed dir as per your request
ART_DIR  = APP_ROOT / "training" / "data" / "processed"

# ------------------------ helpers ------------------------

def _normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns to match model training names (undo bracket sanitization)."""
    rename_map = {}
    for c in df.columns:
        # if it looks like 'age_bin_0,_18_' -> 'age_bin_[0, 18)'
        if "_bin_" in c and "," in c and "_" in c:
            fixed = c.replace("_", " ").replace(" ,", ",").strip()
            if not fixed.startswith("age_bin_["):
                fixed = fixed.replace("age bin", "age_bin_[")
            if not fixed.endswith(")"):
                fixed += ")"
            rename_map[c] = fixed
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _clean_colnames_inplace(df: pd.DataFrame) -> None:
    df.columns = (
        df.columns
        .str.replace(r"[\[\]\(\)<>\s]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
    )

def _mk_flag(v: Any, cond) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return 1.0 if cond(float(v)) else 0.0
    except Exception:
        return 0.0

def _to_float_or_none(v: Any) -> float:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)
    return np.asarray(model.predict_proba(X))

def _interval_label(left: float, right: float, right_closed: bool = False) -> str:
    lb = "[" if not right_closed else "("
    rb = ")" if not right_closed else "]"
    return f"{lb}{left}, {right}{rb}"

def _one_hot_bin(prefix: str, value: float, edges: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return out
    for i in range(len(edges) - 1):
        l, r = edges[i], edges[i + 1]
        if value >= l and value < r:
            out[f"{prefix}_{_interval_label(l, r, right_closed=False)}"] = 1.0
            return out
    return out

# ------------------------ load artifacts ------------------------

def _load_artifacts():
    labels_path = ART_DIR / "label_classes.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")

    with open(labels_path) as f:
        label_classes = json.load(f)

    # Hybrid preferred
    hybrid_path = ART_DIR / "severity_hybrid.joblib"
    hybrid_meta = ART_DIR / "severity_hybrid_meta.json"
    if hybrid_path.exists() and hybrid_meta.exists():
        bundle = joblib.load(hybrid_path)  # {"lr":..., "xgb":..., "scaler":...}
        meta = json.loads(hybrid_meta.read_text())
        meta_feature_order = meta.get("meta_feature_order", [])
        scale_cols = meta.get("scale_cols", [])
        lr_prob_cols = meta.get("lr_prob_cols", [])
        clean_names = bool(meta.get("clean_names", True))

        # Fallback to CSV if needed
        if not meta_feature_order:
            fo_csv = ART_DIR / "feature_order.csv"
            if fo_csv.exists():
                meta_feature_order = pd.read_csv(fo_csv, header=None)[0].tolist()

        return "hybrid", {
            "lr": bundle.get("lr"),
            "xgb": bundle.get("xgb"),
            "scaler": bundle.get("scaler"),
            "label_classes": label_classes,
            "meta_feature_order": meta_feature_order,
            "scale_cols": scale_cols,
            "lr_prob_cols": lr_prob_cols,
            "clean_names": clean_names,
        }

    # Single model fallback
    lr_path = ART_DIR / "severity_lr.joblib"
    xgb_path = ART_DIR / "severity_xgb.joblib"
    fo_csv = ART_DIR / "feature_order.csv"
    if not fo_csv.exists():
        raise FileNotFoundError(f"Missing {fo_csv}")
    features = pd.read_csv(fo_csv, header=None)[0].tolist()

    if lr_path.exists():
        model = joblib.load(lr_path)
        # optional external scaler
        scaler_bundle = None
        scaler_path = ART_DIR / "scaler.joblib"
        if scaler_path.exists():
            scaler_bundle = joblib.load(scaler_path)
        if hasattr(model, "classes_"):
            classes = [label_classes[i] for i in model.classes_]
        else:
            classes = label_classes
        return "single", {
            "model": model,
            "model_name": "LogisticRegression",
            "features": features,
            "classes": classes,
            "scaler_bundle": scaler_bundle,
        }

    if xgb_path.exists():
        model = joblib.load(xgb_path)
        classes = label_classes
        return "single", {
            "model": model,
            "model_name": "XGBoost",
            "features": features,
            "classes": classes,
            "scaler_bundle": None,
        }

    raise FileNotFoundError("No trained model found in processed dir.")

MODE, ARTS = _load_artifacts()

# ------------------------ feature assembly ------------------------

def _start_row(columns: List[str]) -> Dict[str, float]:
    return {c: 0.0 for c in columns}

def _fill_numeric(row: Dict[str, float], payload: Dict[str, Any], columns: List[str]) -> None:
    alias_map = {
        "age": ["age"],
        "hr": ["hr", "heart_rate"],
        "rr": ["rr", "resp_rate"],
        "sbp": ["sbp"],
        "dbp": ["dbp"],
        "spo2": ["spo2", "o2sat"],
        "temp": ["temp", "temperature"],
    }
    for canon, aliases in alias_map.items():
        if canon not in columns:
            continue
        val = None
        for a in aliases:
            if a in payload and payload.get(a) is not None:
                val = _to_float_or_none(payload[a])
                break
        if val is not None:
            row[canon] = val

def _apply_engineering(row: Dict[str, float], columns: List[str]) -> None:
    age  = row.get("age", None)
    hr   = row.get("hr", None)
    rr   = row.get("rr", None)
    sbp  = row.get("sbp", None)
    dbp  = row.get("dbp", None)
    spo2 = row.get("spo2", None)
    temp = row.get("temp", None)

    # engineered continuous
    if "shock_index" in columns and _to_float_or_none(hr) is not None and _to_float_or_none(sbp) is not None and sbp not in (0, None):
        row["shock_index"] = float(hr) / float(sbp)
    if "pulse_pressure" in columns and _to_float_or_none(sbp) is not None and _to_float_or_none(dbp) is not None:
        row["pulse_pressure"] = float(sbp) - float(dbp)
    if "mean_arterial_pressure" in columns and _to_float_or_none(dbp) is not None and _to_float_or_none(sbp) is not None:
        row["mean_arterial_pressure"] = (2.0 * float(dbp) + float(sbp)) / 3.0
    if "spo2_deficit" in columns and _to_float_or_none(spo2) is not None:
        row["spo2_deficit"] = max(0.0, 100.0 - float(spo2))

    # flags
    if "hypotension_flag" in columns and _to_float_or_none(sbp) is not None:
        row["hypotension_flag"] = _mk_flag(sbp, lambda v: v < 100.0)
    if "tachycardia_flag" in columns and _to_float_or_none(hr) is not None:
        row["tachycardia_flag"] = _mk_flag(hr, lambda v: v > 110.0)
    if "tachypnea_flag" in columns and _to_float_or_none(rr) is not None:
        row["tachypnea_flag"] = _mk_flag(rr, lambda v: v > 22.0)
    if "hypoxia_flag" in columns and _to_float_or_none(spo2) is not None:
        row["hypoxia_flag"] = _mk_flag(spo2, lambda v: v < 92.0)
    if "fever_flag" in columns and _to_float_or_none(temp) is not None:
        row["fever_flag"] = _mk_flag(temp, lambda v: v >= 38.0)
    if "age75_flag" in columns and _to_float_or_none(age) is not None:
        row["age75_flag"] = _mk_flag(age, lambda v: v >= 75.0)

    # bins (right=False)
    if _to_float_or_none(age) is not None:
        for k, v in _one_hot_bin("age_bin", float(age), [0,18,30,45,60,75,200]).items():
            if k in columns: row[k] = v
    if _to_float_or_none(temp) is not None:
        for k, v in _one_hot_bin("temp_bin", float(temp), [30,36,37.5,38.5,41]).items():
            if k in columns: row[k] = v
    if _to_float_or_none(hr) is not None:
        for k, v in _one_hot_bin("hr_bin", float(hr), [0,60,100,120,200]).items():
            if k in columns: row[k] = v
    if _to_float_or_none(sbp) is not None:
        for k, v in _one_hot_bin("sbp_bin", float(sbp), [0,90,110,140,260]).items():
            if k in columns: row[k] = v

def _apply_cc_flags(row: Dict[str, float], payload: Dict[str, Any], columns: List[str]) -> None:
    for c in columns:
        if c.startswith("cc_") and c in payload:
            val = payload[c]
            is_on = False
            if isinstance(val, (int, float)):
                is_on = (float(val) != 0.0)
            elif isinstance(val, str):
                is_on = val.strip().lower() in {"1", "true", "yes", "y"}
            elif isinstance(val, bool):
                is_on = val
            row[c] = 1.0 if is_on else 0.0

def _apply_scaler(df: pd.DataFrame, scaler_bundle: Dict[str, Any]) -> None:
    if not scaler_bundle:
        return
    cols = scaler_bundle.get("columns", [])
    scaler = scaler_bundle.get("scaler", None)
    if scaler is None or not cols:
        return
    cols_to_scale = [c for c in cols if c in df.columns]
    if cols_to_scale:
        df.loc[:, cols_to_scale] = scaler.transform(df[cols_to_scale])

# ------------------------ inference paths ------------------------

def _predict_single(payload: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    model         = ARTS["model"]
    model_name    = ARTS["model_name"]
    features      = ARTS["features"]
    classes       = ARTS["classes"]
    scaler_bundle = ARTS.get("scaler_bundle", None)

    row = _start_row(features)
    _fill_numeric(row, payload, features)
    _apply_cc_flags(row, payload, features)
    _apply_engineering(row, features)

    X = pd.DataFrame([row])[features]
    if model_name != "XGBoost" and scaler_bundle is not None:
        _apply_scaler(X, scaler_bundle)

    probs = _predict_proba(model, X)[0]
    label = classes[int(np.argmax(probs))]
    return label, dict(zip(classes, map(float, probs)))

def _predict_hybrid(payload: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    lr         = ARTS["lr"]
    xgb        = ARTS["xgb"]
    scaler     = ARTS.get("scaler", None)
    classes    = ARTS["label_classes"]
    meta_cols  = ARTS["meta_feature_order"]
    lr_prob_cols = ARTS.get("lr_prob_cols", [])
    scale_cols = set(ARTS.get("scale_cols", []))
    clean_names = bool(ARTS.get("clean_names", True))

    # --- IMPORTANT: build LR base features using the exact names from training ---
    lr_expected_cols = list(getattr(lr, "feature_names_in_", []))
    if lr_expected_cols:
        base_cols = lr_expected_cols
    else:
        # Fallback: meta base cols (rare)
        base_cols = [c for c in meta_cols if c not in set(lr_prob_cols)]

    # Build one row and compute engineered features with those names
    row = _start_row(base_cols)
    _fill_numeric(row, payload, base_cols)
    _apply_cc_flags(row, payload, base_cols)
    _apply_engineering(row, base_cols)

    base_df = pd.DataFrame([row])
    # Align columns exactly to LR
    base_df = base_df.reindex(columns=base_cols, fill_value=0.0)

    if scaler is not None and scale_cols:
        cols_to_scale = [c for c in base_df.columns if c in scale_cols]
        if cols_to_scale:
            base_df.loc[:, cols_to_scale] = scaler.transform(base_df[cols_to_scale])

    lr_probs = _predict_proba(lr, base_df)[0]
    prob_df = pd.DataFrame([lr_probs], columns=lr_prob_cols)

    meta_df = pd.concat([base_df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)

    if clean_names:
        _clean_colnames_inplace(meta_df)

    for c in meta_cols:
        if c not in meta_df.columns:
            meta_df[c] = 0.0
    meta_df = meta_df[meta_cols]

    probs = _predict_proba(xgb, meta_df)[0]
    label = classes[int(np.argmax(probs))]
    return label, dict(zip(classes, map(float, probs)))

# ------------------------ public ------------------------

def predict_one(payload: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    if MODE == "hybrid":
        return _predict_hybrid(payload)
    return _predict_single(payload)
