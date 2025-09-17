from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx, os


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