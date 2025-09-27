#!/usr/bin/env python3
"""
Test models with PROPER tokenization for Noongar language - FIXED VERSION
"""

from transformers import pipeline, AutoTokenizer
from pathlib import Path
import re

# ABSOLUTE PATHS
BASE_DIR = Path("C:/Users/imtha/Documents/Semester 3/TRP/new proj/clinical-assistant")
MODEL_PATH = BASE_DIR / "apps" / "nlpService" / "models" / "noongar-clinical-ner-model-finetuned"

print("🧪 NOONGAR CLINICAL NER WITH PROPER TOKENIZATION")
print("=" * 60)

def create_noongar_friendly_pipeline():
    """Create a pipeline that handles Noongar words properly"""
    try:
        # Load tokenizer and model normally
        ner_pipeline = pipeline(
            "ner",
            model=str(MODEL_PATH),
            tokenizer=str(MODEL_PATH),
            aggregation_strategy="simple"
        )
        
        # Load tokenizer separately for analysis
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        
        return ner_pipeline, tokenizer
        
    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        return None, None

def debug_entity_structure(ner_pipeline, sentence):
    """Debug the actual structure of entities returned"""
    print(f"\n🔧 DEBUGGING ENTITY STRUCTURE FOR: {sentence}")
    results = ner_pipeline(sentence)
    
    if results:
        print("📋 RAW ENTITY STRUCTURE:")
        for i, entity in enumerate(results):
            print(f"   Entity {i+1}:")
            for key, value in entity.items():
                print(f"      {key}: {value}")
    else:
        print("   No entities detected")
    
    return results

def test_with_proper_grouping(ner_pipeline, tokenizer):
    """Test with proper word grouping for Noongar"""
    
    test_sentences = [
        "Ngaitj koort kalyakal",
        "Ngaitj miyal yoowart kadak", 
        "Ngaitj kaat wara",
        "Ngaitj korbol boola kalyakal"
    ]
    
    print("🔍 TESTING WITH PROPER GROUPING:")
    print("=" * 50)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. 📝 Sentence: {sentence}")
        
        # First, see how the tokenizer splits it
        tokens = tokenizer.tokenize(sentence)
        print(f"   Tokenized: {tokens}")
        
        # Get NER results
        results = ner_pipeline(sentence)
        
        if results:
            print("   🔍 Entities detected:")
            
            # Group entities by word boundaries
            grouped_entities = group_entities_by_words(results, sentence)
            
            for entity in grouped_entities:
                # Handle different entity structures
                entity_group = entity.get('entity_group') or entity.get('entity') or 'UNKNOWN'
                word = entity.get('word', '')
                score = entity.get('score', entity.get('confidence', 0.0))
                
                print(f"      '{word}' -> {entity_group} ({score:.3f})")
        else:
            print("   ⚠️  No entities detected")

def group_entities_by_words(entities, original_sentence):
    """Group entities by word boundaries in the original sentence"""
    if not entities:
        return []
    
    # Get all words from the original sentence
    words = original_sentence.split()
    grouped = []
    
    for word in words:
        # Find entities that belong to this word
        word_entities = []
        for entity in entities:
            entity_word = entity.get('word', '')
            # Check if this entity is part of the current word
            if entity_word and entity_word in word:
                word_entities.append(entity)
        
        if word_entities:
            # Group entities for this word
            main_entity = max(word_entities, key=lambda x: x.get('score', 0))
            grouped.append({
                'word': word,
                'entity_group': main_entity.get('entity_group', 'UNKNOWN'),
                'confidence': main_entity.get('score', 0.0),
                'sub_entities': word_entities
            })
        else:
            # No entities detected for this word
            grouped.append({
                'word': word,
                'entity_group': 'O',
                'confidence': 0.0,
                'sub_entities': []
            })
    
    return grouped

def simple_word_based_analysis():
    """Simple word-based analysis as fallback"""
    print("\n🔧 SIMPLE WORD-BASED ANALYSIS:")
    print("=" * 50)
    
    # Noongar word dictionaries based on your training data
    body_parts = {
        "koort": "heart", "miyal": "eye", "kaat": "head", "korbol": "stomach",
        "woort": "throat", "ngoorndiny": "ear", "mooly": "nose", 
        "waarngk": "mouth", "djen": "foot"
    }
    
    symptoms = {
        "kalyakal": "tired", "wara": "sick", "moorn": "unwell", 
        "yoowart": "fever", "nyidiny": "cold", "moorditj": "severe"
    }
    
    qualities = {"boola": "very", "kwop": "well"}
    negations = {"kadak": "no"}
    
    test_cases = [
        "Ngaitj koort kalyakal",
        "Ngaitj miyal yoowart kadak", 
        "Ngaitj kaat wara",
        "Ngaitj korbol boola kalyakal",
        "Ngaitj ngoorndiny moorditj"
    ]
    
    for sentence in test_cases:
        print(f"\n📝 Analyzing: {sentence}")
        words = sentence.split()
        
        clinical_findings = []
        for word in words:
            if word in body_parts:
                clinical_findings.append(f"{word} ({body_parts[word]}) -> BODY_PART")
            elif word in symptoms:
                clinical_findings.append(f"{word} ({symptoms[word]}) -> SYMPTOM")
            elif word in qualities:
                clinical_findings.append(f"{word} ({qualities[word]}) -> QUALITY")
            elif word in negations:
                clinical_findings.append(f"{word} ({negations[word]}) -> NEGATION")
            elif word == "Ngaitj":
                clinical_findings.append(f"{word} -> POSSESSIVE (my)")
        
        if clinical_findings:
            for finding in clinical_findings:
                print(f"   🔹 {finding}")
        else:
            print("   ⚠️  No recognizable clinical terms")

def compare_approaches(ner_pipeline, tokenizer):
    """Compare different approaches"""
    print("\n🔍 COMPARING APPROACHES:")
    print("=" * 50)
    
    test_sentence = "Ngaitj koort kalyakal"
    print(f"📝 Test sentence: {test_sentence}")
    
    # Approach 1: Raw model output
    print("\n1. 🧠 RAW MODEL OUTPUT:")
    results = ner_pipeline(test_sentence)
    if results:
        for entity in results:
            word = entity.get('word', '')
            entity_type = entity.get('entity_group', 'UNKNOWN')
            score = entity.get('score', 0.0)
            print(f"   '{word}' -> {entity_type} ({score:.3f})")
    
    # Approach 2: Word-based grouping
    print("\n2. 🔧 WORD-BASED GROUPING:")
    grouped = group_entities_by_words(results, test_sentence)
    for entity in grouped:
        if entity['entity_group'] != 'O':
            print(f"   '{entity['word']}' -> {entity['entity_group']} ({entity['confidence']:.3f})")
    
    # Approach 3: Dictionary lookup
    print("\n3. 📚 DICTIONARY LOOKUP:")
    words = test_sentence.split()
    for word in words:
        if word in ["koort", "miyal", "kaat"]:
            print(f"   {word} -> BODY_PART")
        elif word in ["kalyakal", "wara", "yoowart"]:
            print(f"   {word} -> SYMPTOM")
        elif word == "Ngaitj":
            print(f"   {word} -> POSSESSIVE")

if __name__ == "__main__":
    # Create the pipeline
    ner_pipeline, tokenizer = create_noongar_friendly_pipeline()
    
    if ner_pipeline and tokenizer:
        # First debug the entity structure
        debug_entity_structure(ner_pipeline, "Ngaitj koort kalyakal")
        
        # Then test with proper grouping
        test_with_proper_grouping(ner_pipeline, tokenizer)
        
        # Compare approaches
        compare_approaches(ner_pipeline, tokenizer)
    else:
        print("❌ Could not create pipeline")
    
    # Always show the simple word-based analysis
    simple_word_based_analysis()
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS:")
    print("   1. Use word-based grouping for cleaner results")
    print("   2. Combine with dictionary lookup for accuracy")
    print("   3. Consider training a custom Noongar tokenizer long-term")