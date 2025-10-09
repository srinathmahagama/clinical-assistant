# path: apps/MLService2/app/integrator.py
from typing import Dict, Any, Tuple
import re
import httpx

# NLP service URL (change if needed)
NLP_URL = "http://127.0.0.1:8000/analyze"

# Map English symptoms to cc_* flags (extend as needed)
SYMPTOM_TO_CC = {
    # Cardio/resp
    "chest pain": "cc_chestpain",
    "chest tightness": "cc_chestpain",
    "pressure": "cc_chestpain",
    "palpitations": "cc_palpitations",
    "shortness of breath": "cc_dyspnea",
    "breathless": "cc_dyspnea",
    "dyspnea": "cc_dyspnea",
    "cough": "cc_cough",
    "sore throat": "cc_sorethroat",
    "wheeze": "cc_wheeze",

    # Infection/general
    "fever": "cc_fever",
    "chills": "cc_fever",
    "tired": "cc_fatigue",
    "fatigue": "cc_fatigue",
    "weakness": "cc_weakness",
    "pain": "cc_pain_general",

    # Neuro
    "headache": "cc_headache",
    "dizziness": "cc_dizziness",
    "syncope": "cc_syncope",

    # GI/GU
    "abdominal pain": "cc_abdopain",
    "stomach pain": "cc_abdopain",
    "vomiting": "cc_vomiting",
    "nausea": "cc_nausea",
    "diarrhea": "cc_diarrhea",
    "dysuria": "cc_dysuria",
}

# Simple numeric extraction from free text if present (optional)
NUM_PATTERNS = [
    (r"(?:temp(?:erature)?|fever)\s*(\d{2,3}(?:\.\d)?)", "temp"),
    (r"(?:hr|heart\s*rate|pulse)\s*(\d{2,3})", "hr"),
    (r"(?:rr|resp(?:iratory)?\s*rate)\s*(\d{1,2})", "rr"),
    (r"(?:sbp|systolic)\s*(\d{2,3})", "sbp"),
    (r"(?:dbp|diastolic)\s*(\d{2,3})", "dbp"),
    (r"(?:spo2|o2|oxygen|sat(?:uration)?)\s*(\d{2,3})", "spo2"),
    (r"(?:age)\s*(\d{1,3})", "age"),
]

def call_nlp(text: str, language: str) -> Dict[str, Any]:
    """
    Calls the NLP translator service. Expects JSON response that includes
    English terms (e.g., entities[*].english_translation or an English summary).
    """
    with httpx.Client(timeout=10.0) as client:
        r = client.post(NLP_URL, json={"text": text, "language": language})
        r.raise_for_status()
        return r.json()

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_numbers_freeform(txt: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    low = _norm(txt)
    for pat, key in NUM_PATTERNS:
        m = re.search(pat, low)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    if "fever" in low and "temp" not in out:
        out.setdefault("cc_fever", 1)
    return out

def nlp_to_ml_payload(nlp_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Convert NLP result -> (numeric, cc_flags).
    We prioritize english translations provided by NLP.
    """
    numeric: Dict[str, Any] = {}
    cc_flags: Dict[str, int] = {}

    # Try to read structured symptoms with English translation
    symptoms = nlp_json.get("clinical_summary", {}).get("symptoms", [])
    for s in symptoms:
        eng = _norm(s.get("translation") or s.get("english_translation") or s.get("word", ""))
        if eng in SYMPTOM_TO_CC:
            cc_flags[SYMPTOM_TO_CC[eng]] = 1
        else:
            for k, cc in SYMPTOM_TO_CC.items():
                if k in eng:
                    cc_flags[cc] = 1

    # Fall back to english interpretation string
    english_text = nlp_json.get("english_interpretation") or ""
    numeric.update(extract_numbers_freeform(english_text))
    numeric.update(extract_numbers_freeform(nlp_json.get("text", "")))

    # If entities array is present, also scan english_translation fields
    for ent in nlp_json.get("entities", []):
        eng = _norm(ent.get("english_translation") or ent.get("word", ""))
        for k, cc in SYMPTOM_TO_CC.items():
            if k in eng:
                cc_flags[cc] = 1

    return numeric, cc_flags
