from app.schemas.symptom_input import SymptomInput
from app.services.predictor import make_prediction
from app.utils.semantic_mapper import interpretation_to_symptoms
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="ML Service 1")

# Allow requests from your Next.js frontend
origins = [
    "http://localhost:3000",  # Next.js dev
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/hit-ML-service-1")
async def testMLService1():
    msg = "ML Service 1 tested sucessfully ..."
    return msg


class PredictInput(BaseModel):
    input_data: str

@app.post("/predict")
def predict(input: PredictInput):
    semanticMappings = interpretation_to_symptoms(input.input_data)
    return make_prediction(semanticMappings)