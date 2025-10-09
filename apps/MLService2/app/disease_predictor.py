# path: apps/MLService2/app/disease_predictor.py
from typing import Dict, Any
import math

def _to_float(x):
    try:
        if x is None: return None
        f = float(x)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except Exception:
        return None

def predict_diseases(feats: Dict[str, Any]) -> Dict[str, float]:
    """
    Lightweight heuristic disease identifier.
    Returns a dict of {disease_name: score 0..1}.
    Replace with an ML model later if desired.
    """
    age = _to_float(feats.get("age"))
    hr  = _to_float(feats.get("hr"))
    rr  = _to_float(feats.get("rr"))
    sbp = _to_float(feats.get("sbp"))
    spo2 = _to_float(feats.get("spo2"))
    temp = _to_float(feats.get("temp"))

    cc = lambda k: bool(feats.get(k, 0))

    scores: Dict[str, float] = {}

    # Acute Coronary Syndrome (ACS) heuristic
    acs = 0.0
    if cc("cc_chestpain"):
        acs += 0.5
    if age is not None and age >= 40:
        acs += 0.15
    if hr is not None and hr >= 110:
        acs += 0.1
    if sbp is not None and sbp < 100:
        acs += 0.15
    scores["Acute Coronary Syndrome"] = min(1.0, round(acs, 2))

    # Pneumonia heuristic
    pna = 0.0
    if cc("cc_cough"):
        pna += 0.35
    if cc("cc_dyspnea") or (rr is not None and rr > 22):
        pna += 0.2
    if temp is not None and temp >= 38.0:
        pna += 0.2
    if spo2 is not None and spo2 < 92:
        pna += 0.25
    scores["Pneumonia"] = min(1.0, round(pna, 2))

    # Sepsis heuristic (qSOFA-inspired)
    sepsis = 0.0
    if temp is not None and temp >= 38.0:
        sepsis += 0.2
    if rr is not None and rr > 22:
        sepsis += 0.2
    if sbp is not None and sbp <= 100:
        sepsis += 0.25
    if hr is not None and hr >= 110:
        sepsis += 0.15
    if cc("cc_cough") or cc("cc_sorethroat") or cc("cc_diarrhea") or cc("cc_vomiting"):
        sepsis += 0.1
    scores["Sepsis"] = min(1.0, round(sepsis, 2))

    # Dehydration heuristic
    dehyd = 0.0
    if hr is not None and hr >= 110:
        dehyd += 0.25
    if sbp is not None and sbp < 100:
        dehyd += 0.25
    if temp is not None and temp >= 38.0:
        dehyd += 0.15
    if cc("cc_diarrhea") or cc("cc_vomiting"):
        dehyd += 0.25
    scores["Dehydration"] = min(1.0, round(dehyd, 2))

    # Headache syndrome heuristic
    head = 0.0
    if cc("cc_headache"):
        head += 0.5
    if cc("cc_dizziness") or cc("cc_syncope"):
        head += 0.2
    scores["Headache syndrome"] = min(1.0, round(head, 2))

    return scores

def predict_health_problems(feats: Dict[str, Any]) -> Dict[str, float]:
    """
    Simple clinical problem flags (0..1).
    """
    hr  = _to_float(feats.get("hr"))
    rr  = _to_float(feats.get("rr"))
    sbp = _to_float(feats.get("sbp"))
    spo2 = _to_float(feats.get("spo2"))
    temp = _to_float(feats.get("temp"))

    problems: Dict[str, float] = {}

    # Shock risk
    shock = 0.0
    if sbp is not None and sbp < 90:
        shock += 0.6
    if hr is not None and sbp is not None and sbp > 0 and (hr / sbp) > 1.0:
        shock += 0.3
    problems["Shock risk"] = min(1.0, round(shock, 2))

    # Hypoxia
    hypox = 0.0
    if spo2 is not None and spo2 < 92:
        hypox = 0.8
    problems["Hypoxia"] = min(1.0, round(hypox, 2))

    # Hyperthermia
    hyper = 0.0
    if temp is not None and temp >= 38.5:
        hyper = 0.6
    problems["Hyperthermia"] = min(1.0, round(hyper, 2))

    # Respiratory distress
    resp = 0.0
    if rr is not None and rr > 22:
        resp += 0.6
    problems["Respiratory distress"] = min(1.0, round(resp, 2))

    return problems
