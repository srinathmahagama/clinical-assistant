from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .production_processor import NoongarClinicalProcessor

app = FastAPI()
processor = NoongarClinicalProcessor()

class AnalyzeRequest(BaseModel):
    text: str
    language: str

@app.post("/analyze")
def analyze_text(request: AnalyzeRequest):
    try:
        result = processor.process(request.text)
        return result
    except Exception as e:
        return {"detail": f"Processing error: {str(e)}"}

@app.post("/analyze-batch")
def analyze_batch(requests: List[AnalyzeRequest]):
    results = []
    for req in requests:
        try:
            res = processor.process(req.text)
            results.append(res)
        except Exception as e:
            results.append({"text": req.text, "error": str(e)})
    return results
