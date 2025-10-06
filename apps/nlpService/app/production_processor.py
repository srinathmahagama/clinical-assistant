# production_processor.py
from transformers import pipeline, AutoTokenizer
from pathlib import Path
from typing import Dict, List, Any
import os

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

    def generate_english_translation(self, text: str, entities: List[Dict]) -> str:
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
        
        # Use dictionary-based analysis
        entities = self.dictionary_based_analysis(text)
        clinical_summary = self.create_clinical_summary(entities)
        english_interpretation = self.generate_english_interpretation(entities)
        english_translation = self.generate_english_translation(text, entities)
        
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
    def __init__(self):
        self.ner_pipeline = None
        self.tokenizer = None
        self.load_model()
        
        # Noongar dictionary for entity correction
        self.noongar_dictionary = {
            "ngaitj": "POSSESSIVE",
            "kadak": "NEGATION",
            "boola": "QUALITY", 
            "kwop": "QUALITY",
            "koort": "BODY_PART",
            "miyal": "BODY_PART",
            "kaat": "BODY_PART",
            "korbol": "BODY_PART",
            "kalyakal": "SYMPTOM",
            "wara": "SYMPTOM",
            "yoowart": "SYMPTOM",
            "moorditj": "SYMPTOM"
        }
        
        # English translations
        self.english_translations = {
            "ngaitj": "my",
            "kadak": "no",
            "boola": "very",
            "kwop": "well",
            "koort": "heart",
            "miyal": "eye", 
            "kaat": "head",
            "korbol": "stomach",
            "kalyakal": "tired",
            "wara": "sick",
            "yoowart": "fever",
            "moorditj": "severe"
        }

    def load_model(self):
        """Load the NER model and tokenizer"""
        try:
            BASE_DIR = Path(__file__).parent
            MODEL_PATH = BASE_DIR / "models" / "noongar-clinical-ner-model-finetuned"
            
            model_path_str = str(MODEL_PATH)
            
            if not os.path.exists(model_path_str):
                raise FileNotFoundError(f"Model not found at: {model_path_str}")
            
            print(f"Loading Noongar NER model from: {model_path_str}")
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model_path_str,
                tokenizer=model_path_str,
                aggregation_strategy="simple",
                device=-1
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_str)
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def correct_entity(self, word: str, predicted_entity: str) -> str:
        """Correct entity predictions using dictionary knowledge"""
        word_lower = word.lower()
        
        # Use dictionary mapping if available
        if word_lower in self.noongar_dictionary:
            return self.noongar_dictionary[word_lower]
        
        return predicted_entity

    def group_entities_by_words(self, entities: List[Dict], text: str) -> List[Dict]:
        """Group token-level entities into word-level entities"""
        if not entities:
            return []
        
        words = text.split()
        grouped_entities = []
        used_indices = set()
        
        for word in words:
            word_start = text.find(word)
            word_end = word_start + len(word)
            
            # Find entities that belong to this word
            word_entities = []
            for i, entity in enumerate(entities):
                if i in used_indices:
                    continue
                    
                entity_start = entity.get('start', 0)
                entity_end = entity.get('end', 0)
                
                # Check if entity is within word boundaries
                if (entity_start >= word_start and entity_end <= word_end):
                    word_entities.append(entity)
                    used_indices.add(i)
            
            if word_entities:
                # Use entity with highest confidence
                best_entity = max(word_entities, key=lambda x: x.get('score', 0))
                
                # Correct entity if needed
                corrected_entity = self.correct_entity(word, best_entity.get('entity_group', 'UNKNOWN'))
                
                grouped_entities.append({
                    'word': word,
                    'entity': corrected_entity,
                    'confidence': best_entity.get('score', 0.0),
                    'start': word_start,
                    'end': word_end,
                    'english_translation': self.english_translations.get(word.lower(), '')
                })
            else:
                # No entity detected for this word
                corrected_entity = self.correct_entity(word, 'O')
                if corrected_entity != 'O':
                    grouped_entities.append({
                        'word': word,
                        'entity': corrected_entity,
                        'confidence': 0.9,  # High confidence for dictionary
                        'start': word_start,
                        'end': word_end,
                        'english_translation': self.english_translations.get(word.lower(), ''),
                        'source': 'dictionary'
                    })
        
        return grouped_entities

    def create_clinical_summary(self, entities: List[Dict]) -> Dict[str, Any]:
        """Create a structured clinical summary from entities"""
        body_parts = []
        symptoms = []
        qualities = []
        negations = []
        possessives = []
        
        for entity in entities:
            if entity['entity'] == 'BODY_PART':
                body_parts.append({
                    'word': entity['word'],
                    'translation': entity.get('english_translation', '')
                })
            elif entity['entity'] == 'SYMPTOM':
                symptoms.append({
                    'word': entity['word'],
                    'translation': entity.get('english_translation', ''),
                    'confidence': entity['confidence']
                })
            elif entity['entity'] == 'QUALITY':
                qualities.append({
                    'word': entity['word'],
                    'translation': entity.get('english_translation', '')
                })
            elif entity['entity'] == 'NEGATION':
                negations.append({
                    'word': entity['word'],
                    'translation': entity.get('english_translation', '')
                })
            elif entity['entity'] == 'POSSESSIVE':
                possessives.append({
                    'word': entity['word'],
                    'translation': entity.get('english_translation', '')
                })
        
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

    def process(self, text: str) -> Dict[str, Any]:
        """Main processing function for Noongar clinical text"""
        if not self.ner_pipeline:
            raise Exception("NER model not loaded")
        
        try:
            # Get raw NER results
            raw_entities = self.ner_pipeline(text)
            
            # Group entities by words
            grouped_entities = self.group_entities_by_words(raw_entities, text)
            
            # Filter out non-entity results (keep only detected entities)
            detected_entities = [e for e in grouped_entities if e['entity'] != 'O']
            
            # Create clinical summary
            clinical_summary = self.create_clinical_summary(detected_entities)
            
            # Generate English interpretation
            english_interpretation = self.generate_english_interpretation(detected_entities)
            
            return {
                'text': text,
                'entities': detected_entities,
                'clinical_summary': clinical_summary,
                'english_interpretation': english_interpretation,
                'entity_count': len(detected_entities),
                'success': True
            }
            
        except Exception as e:
            return {
                'text': text,
                'error': str(e),
                'success': False
            }

    def generate_english_interpretation(self, entities: List[Dict]) -> str:
        """Generate an English interpretation of the Noongar text"""
        if not entities:
            return "No clinical entities detected"
        
        parts = []
        for entity in entities:
            word = entity['word']
            entity_type = entity['entity']
            translation = entity.get('english_translation', '')
            
            if translation:
                parts.append(f"{word} ({translation})")
            else:
                parts.append(word)
        
        return "Patient describes: " + ", ".join(parts)