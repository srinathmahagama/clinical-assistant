from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os

# Service URLs - using Docker service names
NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL", "http://nlpservice:8100")
ML_SERVICE1_URL = os.getenv("ML_SERVICE1_URL", "http://mlservice1:8101")
ML_SERVICE2_URL = os.getenv("ML_SERVICE2_URL", "http://mlservice2:8102")

app = FastAPI(title="Clinical Assistant API Gateway")

# Allow requests from your frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://frontend:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

@app.post("/analyze-clinical-text")
async def analyze_clinical_text(text_in: TextIn):
    """
    Main integration endpoint:
    1. Receives text from frontend
    2. Sends to NLP service for analysis
    3. NLP service automatically calls ML service
    4. Returns combined result to frontend
    """
    try:
        print(f"📥 Received text from frontend: {text_in.text}")
        
        # Send to NLP service (which will call ML service internally)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NLP_SERVICE_URL}/analyze",
                json={"text": text_in.text, "language": "noongar"},
                timeout=60.0
            )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"NLP service error: {response.text}"
            )
        
        result = response.json()
        print(f"✅ Analysis completed successfully")
        
        return {
            "success": True,
            "input_text": text_in.text,
            "analysis": result,
            "status": "analysis_complete"
        }
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Service timeout")
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/test-backend")
async def testAPI():
    return {"message": "Backend tested successfully"}

@app.get("/test-nlp-service")
async def testNlpService():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{NLP_SERVICE_URL}/health")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="NLP service error")
    return resp.json()

@app.get("/test-ml-service1")
async def testMLService1():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ML_SERVICE1_URL}/hit-ML-service-1")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="ML Service 1 error")
    return resp.json()

# Health check endpoint for frontend
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "API Gateway",
        "endpoints": {
            "analyze": "/analyze-clinical-text",
            "test_nlp": "/test-nlp-service", 
            "test_ml1": "/test-ml-service1"
        }
    }