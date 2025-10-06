# backend/integration.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

router = APIRouter()

# Services (adjust URLs if needed)
NLP_URL = "http://127.0.0.1:8000/analyze"
ML_URL = "http://0.0.0.0:8101/predict"

class TextRequest(BaseModel):
    text: str
    language: str

@router.post("/integrate")
async def integrate_services(data: TextRequest):
    try:
        # STEP 1 → Send text to NLP service
        async with httpx.AsyncClient() as client:
            nlp_response = await client.post(
                NLP_URL,
                json={"text": data.text, "language": data.language},
                headers={"Content-Type": "application/json"}
            )

        if nlp_response.status_code != 200:
            raise HTTPException(status_code=nlp_response.status_code, detail="NLP service failed")

        nlp_json = nlp_response.json()

        # STEP 2 → Extract symptoms for ML
        symptoms = {}
        for symptom in nlp_json["clinical_summary"]["symptoms"]:
            word = symptom["word"].lower()
            symptoms[word] = 1  # Mark as present

        # STEP 3 → Send extracted symptoms to ML model
        async with httpx.AsyncClient() as client:
            ml_response = await client.post(
                ML_URL,
                json={"symptoms": symptoms},
                headers={"Content-Type": "application/json"}
            )

        if ml_response.status_code != 200:
            raise HTTPException(status_code=ml_response.status_code, detail="ML service failed")

        ml_json = ml_response.json()

        # STEP 4 → Combine both results
        return {
            "nlp_result": nlp_json,
            "ml_result": ml_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
