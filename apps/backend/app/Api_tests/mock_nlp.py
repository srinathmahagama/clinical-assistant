from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# This simulates the NLP input the frontend sends
class NERRequest(BaseModel):
    selected_symptoms: str
    other_symptoms: str

# Mock NLP endpoint
@app.post("/test-nlp-service")
def nlp_service(request: NERRequest):
    combined = request.selected_symptoms
    if request.other_symptoms:
        combined += ", " + request.other_symptoms
    return {
        "translation": f"My mock translation for symptoms: {combined}",
        "clinical_translation": f"Patient reports: {combined}",
        "confidence": 0.95
    }

# Mock backend endpoint
@app.get("/test-backend")
def test_backend():
    return {"message": "Mock backend OK"}
