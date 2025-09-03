from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx, os

# ML_URL = os.getenv("ML_SERVICE_URL","http://ml:8100")
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



@app.get("/hit-NLP-service")
async def testNLPService():
    msg = "NLP Service tested sucessfully ..."
    return msg