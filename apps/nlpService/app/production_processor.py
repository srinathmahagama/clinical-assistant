#!/usr/bin/env python3
"""
Noongar clinical processor combining ML + dictionary with safe serialization
"""

from transformers import pipeline, AutoTokenizer
from pathlib import Path

BASE_DIR = Path("C:/Users/imtha/Documents/Semester 3/TRP/new proj/clinical-assistant")
MODEL_PATH = BASE_DIR / "apps" / "nlpService" / "models" / "noongar-clinical-ner-model-finetuned"

class NoongarClinicalProcessor:
    def __init__(self):
        self.ner_pipeline = None
        self.tokenizer = None
        self.load_model()

        self.clinical_dict = {
            # Body parts
            "koort": {"type": "BODY_PART", "english": "heart"},
            "miyal": {"type": "BODY_PART", "english": "eye"},
            "kaat": {"type": "BODY_PART", "english": "head"},
            "korbol": {"type": "BODY_PART", "english": "stomach"},
            "woort": {"type": "BODY_PART", "english": "throat"},
            "ngoorndiny": {"type": "BODY_PART", "english": "ear"},
            "mooly": {"type": "BODY_PART", "english": "nose"},
            "waarngk": {"type": "BODY_PART", "english": "mouth"},
            "djen": {"type": "BODY_PART", "english": "foot"},

            # Symptoms
            "kalyakal": {"type": "SYMPTOM", "english": "tired"},
            "wara": {"type": "SYMPTOM", "english": "sick"},
            "moorn": {"type": "SYMPTOM", "english": "unwell"},
            "yoowart": {"type": "SYMPTOM", "english": "fever"},
            "nyidiny": {"type": "SYMPTOM", "english": "cold"},
            "moorditj": {"type": "SYMPTOM", "english": "severe"},

            # Qualities
            "boola": {"type": "QUALITY", "english": "very"},
            "kwop": {"type": "QUALITY", "english": "well"},

            # Negations
            "kadak": {"type": "NEGATION", "english": "no"},
        }

    def load_model(self):
        """Load the fine-tuned model"""
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=str(MODEL_PATH),
                tokenizer=str(MODEL_PATH),
                aggregation_strategy="simple",
                device=0
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
            print("✅ Noongar clinical model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def process(self, text: str):
        """Process Noongar text and extract clinical information"""
        if not self.ner_pipeline:
            return self._fallback_processing(text)

        try:
            ml_results = self.ner_pipeline(text)
            grouped = self._group_entities_by_words(ml_results, text)
            validated = self._validate_with_dictionary(grouped)
            summary = self._create_clinical_summary(validated)
            return {
                "original_text": text,
                "entities": validated,
                "clinical_summary": summary,
                "processing_method": "ml_model_with_validation",
                "confidence": "high"
            }
        except Exception as e:
            print(f"⚠️ ML processing failed, using fallback: {e}")
            return self._fallback_processing(text)

    def _group_entities_by_words(self, entities, text):
        words = text.split()
        grouped = []
        for word in words:
            word_entities = [e for e in entities if e.get("word") and e["word"] in word]
            if word_entities:
                main_entity = max(word_entities, key=lambda x: x.get("score", 0))
                grouped.append({
                    "word": word,
                    "entity_group": main_entity.get("entity_group", "UNKNOWN"),
                    "confidence": float(main_entity.get("score", 0.0)),
                    "source": "ml_model"
                })
            else:
                grouped.append({
                    "word": word,
                    "entity_group": "O",
                    "confidence": 0.0,
                    "source": "unknown"
                })
        return grouped

    def _validate_with_dictionary(self, entities):
        validated = []
        for entity in entities:
            word_lower = entity['word'].lower()
            dict_info = self.clinical_dict.get(word_lower)
            if dict_info:
                entity['english'] = dict_info['english']
                if entity['entity_group'] == dict_info['type']:
                    entity['confidence'] = float(min(1.0, entity['confidence'] + 0.05))
                    entity['validated'] = True
                elif entity['confidence'] < 0.8:
                    entity['entity_group'] = dict_info['type']
                    entity['source'] = "dictionary_override"
            else:
                entity['english'] = ""
            validated.append(entity)
        return validated

    def _fallback_processing(self, text: str):
        findings = []
        for word in text.split():
            clean_word = word.lower()
            if clean_word in self.clinical_dict:
                info = self.clinical_dict[clean_word]
                findings.append({
                    "word": word,
                    "entity_group": info["type"],
                    "english": info["english"],
                    "confidence": 0.8,
                    "source": "dictionary_only"
                })
            else:
                findings.append({
                    "word": word,
                    "entity_group": "O",
                    "english": "",
                    "confidence": 0.0,
                    "source": "unknown"
                })
        summary = self._create_clinical_summary(findings)
        return {
            "original_text": text,
            "entities": findings,
            "clinical_summary": summary,
            "processing_method": "dictionary_only",
            "confidence": "medium"
        }

    def _create_clinical_summary(self, entities):
        parts = []
        for e in entities:
            if e['entity_group'] in ['BODY_PART', 'SYMPTOM']:
                parts.append(f"{e['word']} ({e['entity_group']})")
        return ", ".join(parts) if parts else "No relevant clinical information found."
