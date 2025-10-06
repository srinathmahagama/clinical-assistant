import pandas as pd
from typing import Dict
from app.schemas.symptom_input import SymptomInput
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util



df = pd.read_csv("./data/Testing.csv")
exclude_cols = {"prognosis", "class", "target"}
symptom_vocab = [col for col in df.columns if col.lower() not in exclude_cols]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
symptom_embeddings = model.encode(symptom_vocab, convert_to_tensor=True)

# Mapping function
def interpretation_to_symptoms(interpretation: str, threshold: float = 0.45) -> SymptomInput:
    """
    Map an english_interpretation string to SymptomInput schema
    using semantic similarity against 132 symptoms.
    """
     # Encode interpretation
    interp_embedding = model.encode(interpretation, convert_to_tensor=True)

    # Compute cosine similarity
    cos_scores = util.cos_sim(interp_embedding, symptom_embeddings)[0]

    # Only include symptoms above threshold
    present_symptoms = {
        symptom_vocab[idx]: 1
        for idx, score in enumerate(cos_scores)
        if score.item() >= threshold
    }

    return SymptomInput(symptoms=present_symptoms)
