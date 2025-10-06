# main.py (in nlpService root directory)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import os
import httpx

app = FastAPI(
    title="Noongar Clinical NER API",
    description="API for analyzing Noongar clinical text using NER",
    version="1.0.0"
)

class AnalyzeRequest(BaseModel):
    text: str
    language: str = "noongar"

class EntityResponse(BaseModel):
    word: str
    entity: str
    confidence: float
    start: int
    end: int
    english_translation: str = ""

class AnalyzeResponse(BaseModel):
    text: str
    entities: List[EntityResponse]
    clinical_summary: dict
    english_interpretation: str
    english_translation: str  # NEW FIELD
    entity_count: int
    success: bool

class NoongarClinicalProcessor:
    def __init__(self):
        self.ner_pipeline = None
        
        # Noongar dictionary for entity analysis
        self.noongar_dictionary = {
            "ngaitj": {"entity": "POSSESSIVE", "translation": "my"},
            "kadak": {"entity": "NEGATION", "translation": "no"},
            "boola": {"entity": "QUALITY", "translation": "very"},
            "kwop": {"entity": "QUALITY", "translation": "well"},
            "koort": {"entity": "BODY_PART", "translation": "heart"},
            "miyal": {"entity": "BODY_PART", "translation": "eye"},
            "kaat": {"entity": "BODY_PART", "translation": "head"},
            "korbol": {"entity": "BODY_PART", "translation": "stomach"},
            "kalyakal": {"entity": "SYMPTOM", "translation": "tired"},
            "wara": {"entity": "SYMPTOM", "translation": "sick"},
            "yoowart": {"entity": "SYMPTOM", "translation": "fever"},
            "moorditj": {"entity": "SYMPTOM", "translation": "severe"},
            "ngoorndiny": {"entity": "BODY_PART", "translation": "ear"},
            "woort": {"entity": "BODY_PART", "translation": "throat"},
            "nyidiny": {"entity": "SYMPTOM", "translation": "cold"}
        }
        
        # Try to load the model
        self.load_model()

    def load_model(self):
        """Try to load the NER model, but use dictionary as fallback"""
        try:
            from transformers import pipeline
            
            # FIXED: Go up one level from app/ to find models/
            current_dir = Path(__file__).parent
            MODEL_PATH = current_dir.parent / "models" / "noongar-clinical-ner-model-finetuned"
            
            print(f"🔍 Looking for model at: {MODEL_PATH}")
            print(f"🔍 Path exists: {MODEL_PATH.exists()}")
            
            if not MODEL_PATH.exists():
                print(f"❌ Model path not found: {MODEL_PATH}")
                # List what's actually in the models directory
                models_dir = current_dir.parent / "models"
                if models_dir.exists():
                    print(f"📁 Contents of models directory: {list(models_dir.iterdir())}")
                else:
                    print(f"❌ Models directory doesn't exist: {models_dir}")
                return
                
            print(f"🔄 Loading model from: {MODEL_PATH}")
            
            self.ner_pipeline = pipeline(
                "ner",
                model=str(MODEL_PATH),
                tokenizer=str(MODEL_PATH),
                aggregation_strategy="simple",
                device=-1
            )
            
            print("✅ NER Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading NER model: {e}")
            print("🔄 Using dictionary-based analysis only")

    def generate_english_translation(self, text: str) -> str:
        """Generate a clean English-only translation of the Noongar text"""
        if not text.strip():
            return ""
            
        words = text.split()
        english_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.noongar_dictionary:
                english_words.append(self.noongar_dictionary[word_lower]["translation"])
            else:
                # Keep unknown words as-is
                english_words.append(word)
        
        # Join into a proper English sentence
        english_sentence = " ".join(english_words)
        
        # Basic sentence capitalization
        if english_sentence:
            english_sentence = english_sentence[0].upper() + english_sentence[1:]
            
        return english_sentence

    def process(self, text: str) -> Dict[str, Any]:
        """Process Noongar text using dictionary analysis"""
        print(f"🔍 Processing: '{text}'")
        
        # Use dictionary-based analysis (simpler and more reliable)
        entities = self.dictionary_based_analysis(text)
        clinical_summary = self.create_clinical_summary(entities)
        english_interpretation = self.generate_english_interpretation(entities)
        english_translation = self.generate_english_translation(text)  # NEW: Add English translation
        
        return {
            'text': text,
            'entities': entities,
            'clinical_summary': clinical_summary,
            'english_interpretation': english_interpretation,
            'english_translation': english_translation,  # NEW FIELD
            'entity_count': len(entities),
            'success': True,
            'method_used': 'dictionary'
        }

    def dictionary_based_analysis(self, text: str) -> List[Dict]:
        """Extract entities using dictionary lookup"""
        words = text.split()
        entities = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.noongar_dictionary:
                entity_info = self.noongar_dictionary[word_lower]
                start_pos = text.find(word)
                
                entities.append({
                    'word': word,
                    'entity': entity_info['entity'],
                    'confidence': 0.95,
                    'start': start_pos,
                    'end': start_pos + len(word),
                    'english_translation': entity_info['translation']
                })
        
        return entities

    def create_clinical_summary(self, entities: List[Dict]) -> Dict[str, Any]:
        """Create clinical summary from entities"""
        body_parts = []
        symptoms = []
        qualities = []
        negations = []
        possessives = []
        
        for entity in entities:
            entity_data = {
                'word': entity['word'],
                'translation': entity.get('english_translation', '')
            }
            
            if entity['entity'] == 'BODY_PART':
                body_parts.append(entity_data)
            elif entity['entity'] == 'SYMPTOM':
                symptoms.append(entity_data)
            elif entity['entity'] == 'QUALITY':
                qualities.append(entity_data)
            elif entity['entity'] == 'NEGATION':
                negations.append(entity_data)
            elif entity['entity'] == 'POSSESSIVE':
                possessives.append(entity_data)
        
        return {
            'body_parts': body_parts,
            'symptoms': symptoms,
            'qualifiers': qualities,
            'negations': negations,
            'possessives': possessives,
            'has_negation': len(negations) > 0,
            'symptom_count': len(symptoms),
            'body_part_count': len(body_parts)
        }

    def generate_english_interpretation(self, entities: List[Dict]) -> str:
        """Generate English interpretation"""
        if not entities:
            return "No clinical entities detected"
        
        parts = []
        for entity in entities:
            translation = entity.get('english_translation', '')
            if translation:
                parts.append(f"{entity['word']} ({translation})")
            else:
                parts.append(entity['word'])
        
        return "Patient describes: " + ", ".join(parts)

# Initialize processor
processor = NoongarClinicalProcessor()

@app.get("/")
def read_root():
    return {
        "message": "Noongar Clinical NER API", 
        "status": "running",
        "model_loaded": processor.ner_pipeline is not None,
        "endpoints": ["/analyze", "/analyze-batch", "/docs", "/health"]
    }
    
ML_SERVICE_URL = "http://localhost:8101/predict"  # where MLService1 is running

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze Noongar clinical text"""
    try:
        # Step 1: Local text analysis
        result = processor.process(request.text)

        # Step 2: Call MLService1
        async with httpx.AsyncClient() as client:
            nlpInterprestation = result['english_interpretation']
            mlService1Response = await client.post(
                ML_SERVICE_URL,
                json=nlpInterprestation 
            )

        if mlService1Response.status_code != 200:
            raise HTTPException(status_code=mlService1Response.status_code,
                                detail=f"MLService1 error: {mlService1Response.text}")

        ml_result = mlService1Response.json()

        # Step 3: Merge both results
        return AnalyzeResponse(
            text_analysis=result,
            ml_prediction=ml_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze-batch")
def analyze_batch(requests: List[AnalyzeRequest]):
    """Analyze multiple Noongar texts in batch"""
    results = []
    for req in requests:
        try:
            result = processor.process(req.text)
            results.append(result)
        except Exception as e:
            results.append({
                "text": req.text,
                "error": str(e),
                "success": False
            })
    return {"results": results}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Noongar Clinical NER API",
        "model_loaded": processor.ner_pipeline is not None,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="127.0.0.1", port=8000)