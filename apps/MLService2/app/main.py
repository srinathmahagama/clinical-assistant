# path: apps/MLService2/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from .pipeline import predict_one
from .integrator import call_nlp, nlp_to_ml_payload
from .disease_predictor import predict_diseases, predict_health_problems

app = FastAPI(title="MLService2", version="0.3.0")

# ---------- Schemas ----------

class PredictRequest(BaseModel):
    # Optional raw inputs (others like cc_* flags can be passed as extra fields)
    chief_complaint: Optional[str] = None
    age: Optional[float] = None
    hr: Optional[float] = None
    rr: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    spo2: Optional[float] = None
    temp: Optional[float] = None

    # Pydantic v2: allow extra keys (e.g., cc_chestpain, cc_cough, etc.)
    model_config = ConfigDict(extra="allow")

class PredictResponse(BaseModel):
    severity_label: str
    severity_probs: Dict[str, float]

class TriageRequest(BaseModel):
    # Input intended for the NLP translator component
    text: str
    language: str
    # Optional vitals if available from the UI
    age: Optional[float] = None
    hr: Optional[float] = None
    rr: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    spo2: Optional[float] = None
    temp: Optional[float] = None

class TriageResponse(BaseModel):
    nlp: Dict[str, Any]
    ml_payload: Dict[str, Any]
    severity_label: str
    severity_probs: Dict[str, float]
    diagnosis: Dict[str, float]
    health_problems: Dict[str, float]

# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    label, probs = predict_one(payload.model_dump(exclude_none=True))
    return PredictResponse(severity_label=label, severity_probs=probs)

@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    """
    End-to-end flow:
      1) Calls NLP translator: POST http://127.0.0.1:8000/analyze
      2) Converts English terms -> ML payload (cc_* flags, inferred vitals if found)
      3) Merges any explicit vitals from request (request values take precedence)
      4) Predicts severity
      5) Identifies disease(s) & health problems via rules (can be swapped to ML later)
    """
    # 1) NLP call
    try:
        nlp_json = call_nlp(req.text, req.language)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NLP service error: {e}")

    # 2) Map NLP -> ML payload (flags + inferred vitals if present)
    numeric_from_nlp, cc_flags = nlp_to_ml_payload(nlp_json)

    # 3) Merge priorities: explicit request vitals > inferred numbers > none
    ml_payload: Dict[str, Any] = {**cc_flags}
    for k in ("age", "hr", "rr", "sbp", "dbp", "spo2", "temp"):
        v = getattr(req, k)
        if v is not None:
            ml_payload[k] = v
        elif k in numeric_from_nlp:
            ml_payload[k] = numeric_from_nlp[k]

    # 4) Severity prediction (hybrid-aware in pipeline)
    severity_label, severity_probs = predict_one(ml_payload)

    # 5) Disease + health problem heuristics (lightweight)
    diagnosis = predict_diseases(ml_payload)
    health_problems = predict_health_problems(ml_payload)

    return TriageResponse(
        nlp=nlp_json,
        ml_payload=ml_payload,
        severity_label=severity_label,
        severity_probs=severity_probs,
        diagnosis=diagnosis,
        health_problems=health_problems,
    )
