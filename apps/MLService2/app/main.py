# path: apps/MLService2/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from .pipeline import predict_one
from .integrator import call_nlp, nlp_to_ml_payload
from .disease_predictor import predict_diseases, predict_health_problems

app = FastAPI(title="MLService2", version="0.3.1")

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

# ---- BYPASS NLP: direct analysis from ML-ready payload ----

class AnalyzeDirectRequest(PredictRequest):
    """Same fields as PredictRequest + any cc_* flags; bypasses NLP."""
    model_config = ConfigDict(extra="allow")

class AnalyzeDirectResponse(BaseModel):
    ml_payload: Dict[str, Any]
    severity_label: str
    severity_probs: Dict[str, float]
    diagnosis: Dict[str, float]
    health_problems: Dict[str, float]
    # NEW FIELDS for recommendations
    top_disease: str
    recommendations: List[str]

# ---------- Recommendation Dictionary ----------

DISEASE_RECOMMENDATIONS = {
    "Common Cold": [
        "Rest and drink plenty of fluids",
        "You can take over-the-counter pain relievers like paracetamol",
        "Use a humidifier to ease congestion",
        "Get plenty of sleep to help your body fight the virus"
    ],
    "Influenza": [
        "Rest and stay hydrated",
        "Take antiviral medications if prescribed early",
        "Use over-the-counter fever reducers like ibuprofen",
        "Stay home to avoid spreading the virus to others"
    ],
    "Migraine": [
        "Rest in a quiet, dark room",
        "Apply a cold compress to your forehead",
        "Take prescribed migraine medication as directed",
        "Avoid triggers like bright lights and loud noises"
    ],
    "Hypertension": [
        "Reduce sodium intake in your diet",
        "Exercise regularly for at least 30 minutes daily",
        "Monitor your blood pressure regularly",
        "Limit alcohol consumption and avoid smoking"
    ],
    "Headache syndrome": [
        "Practice relaxation techniques like deep breathing",
        "Ensure adequate hydration throughout the day",
        "Maintain regular sleep patterns",
        "Consider over-the-counter pain relief if needed"
    ],
    "Acute Coronary Syndrome": [
        "Seek immediate medical attention - this is an emergency",
        "Chew aspirin if recommended by healthcare provider",
        "Stay calm and avoid physical exertion",
        "Call emergency services immediately"
    ],
    "Pneumonia": [
        "Get plenty of rest to help your body recover",
        "Stay hydrated and drink warm fluids",
        "Take all prescribed antibiotics as directed",
        "Use a humidifier to help with breathing"
    ],
    "Sepsis": [
        "THIS IS A MEDICAL EMERGENCY - seek immediate care",
        "Do not delay treatment - sepsis requires urgent attention",
        "Go to the nearest emergency department",
        "Inform medical staff of all your symptoms"
    ],
    "Dehydration": [
        "Drink oral rehydration solutions or water",
        "Avoid caffeine and alcohol which can worsen dehydration",
        "Rest in a cool environment",
        "Seek medical help if unable to keep fluids down"
    ],
    "General": [
        "Rest and monitor your symptoms",
        "Stay hydrated and maintain a healthy diet",
        "Consult a healthcare professional if symptoms worsen",
        "Keep track of any changes in your condition"
    ]
}

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
      1) Calls NLP translator (NLP_URL/analyze)
      2) Converts English terms -> ML payload (cc_* flags, inferred vitals if found)
      3) Merges any explicit vitals from request (request values take precedence)
      4) Predicts severity
      5) Identifies disease(s) & health problems (heuristics for now)
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

@app.post("/triage_direct", response_model=AnalyzeDirectResponse)
def triage_direct(req: AnalyzeDirectRequest):
    """
    Bypass NLP: directly analyze an ML-ready payload (vitals + cc_* flags).
    Enhanced to include recommendations.
    """
    ml_payload = req.model_dump(exclude_none=True)

    # Severity via model
    severity_label, severity_probs = predict_one(ml_payload)

    # Rule-based disease & health-problem identification
    diagnosis = predict_diseases(ml_payload)
    health_problems = predict_health_problems(ml_payload)
    
    # Get top disease for recommendations
    top_disease = max(diagnosis.items(), key=lambda x: x[1])[0] if diagnosis else "General"
    
    # Get recommendations for top disease (limit to 3)
    top_recommendations = DISEASE_RECOMMENDATIONS.get(top_disease, DISEASE_RECOMMENDATIONS["General"])
    recommendations = top_recommendations[:3]

    return AnalyzeDirectResponse(
        ml_payload=ml_payload,
        severity_label=severity_label,
        severity_probs=severity_probs,
        diagnosis=diagnosis,
        health_problems=health_problems,
        top_disease=top_disease,
        recommendations=recommendations
    )

@app.get("/recommendations/{disease_name}")
def get_recommendations(disease_name: str):
    """Get recommendations for a specific disease"""
    recommendations = DISEASE_RECOMMENDATIONS.get(disease_name, [])
    
    if not recommendations:
        return {
            "disease": disease_name,
            "recommendations": DISEASE_RECOMMENDATIONS["General"],
            "message": "General health recommendations"
        }
    
    return {
        "disease": disease_name,
        "recommendations": recommendations,
        "message": f"Recommendations for {disease_name}"
    }

@app.get("/recommendations")
def get_all_recommendations():
    """Get all available disease recommendations"""
    return {
        "available_diseases": list(DISEASE_RECOMMENDATIONS.keys()),
        "recommendations": DISEASE_RECOMMENDATIONS
    }