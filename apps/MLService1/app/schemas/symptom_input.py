from pydantic import BaseModel
from typing import Dict

class SymptomInput(BaseModel):
    symptoms: Dict[str, int]  # {"itching": 1, "skin_rash": 0, ...}
