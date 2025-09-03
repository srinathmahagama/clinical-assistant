from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx, os

NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL","http://nlpService:8100")
app = FastAPI(title="API Gateway")

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


class TextIn(BaseModel):
    text: str

@app.post("/process_text")
async def process_text(text_in: TextIn):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{ML_URL}/process_text", json={"text": text_in.text})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="ML service error")
    return resp.json()

@app.get("/test-backend")
async def testAPI():
    msg = "backend tested sucessfully ..."
    return msg

@app.get("/test-nlp-service")
async def testNlpService():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{NLP_SERVICE_URL}/hit-NLP-service")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="NLP service error")
    return resp.json()