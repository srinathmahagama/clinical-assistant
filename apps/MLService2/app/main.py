from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict

app = FastAPI(title="MLService2", version="0.1.0")

class PredictRequest(BaseModel):
    chief_complaint: str
    hr: Optional[float] = None
    rr: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    spo2: Optional[float] = None
    temp: Optional[float] = None
    age: Optional[int] = None
    sex: Optional[str] = None  # "M"/"F"

class PredictResponse(BaseModel):
    severity_label: str
    severity_probs: Dict[str, float]
    diagnosis_label: Optional[str] = None
    diagnosis_probs: Optional[Dict[str, float]] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Stub: replace with real pipeline (TF-IDF/ClinicalBERT + XGBoost).
    """
    # dummy output
    return PredictResponse(
        severity_label="Moderate",
        severity_probs={"Mild": 0.2, "Moderate": 0.6, "Severe": 0.2},
        diagnosis_label="Pneumonia",
        diagnosis_probs={"Pneumonia": 0.55, "MI": 0.15, "Headache": 0.30},
    )
