import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from difflib import get_close_matches
import numpy as np
import torch

# Simulated dataset for exact matches (replace with your full dataset)
DATASET = [
    {"id": 1, "text": "Ngaitj koort kalyakal", "translation": "My heart is tired", 
     "entities": [{"start": 7, "end": 12, "label": "BODY_PART"}, {"start": 13, "end": 21, "label": "SYMPTOM"}]},
    {"id": 16, "text": "Ngaitj mooly yoowart kadak", "translation": "My nose is without fever", 
     "entities": [{"start": 7, "end": 12, "label": "BODY_PART"}, {"start": 13, "end": 20, "label": "SYMPTOM"}, {"start": 21, "end": 26, "label": "NEGATION"}]},
    # Add your full dataset here or load from file
]

# Vocabulary mapping from dataset
ENTITY_MAPPING = {
    "BODY_PART": {
        "koort": "heart", "mooly": "nose", "ngoorndiny": "ear", "djen": "foot",
        "waarngk": "mouth", "woort": "throat", "kaat": "head", "korbol": "stomach", "miyal": "eye"
    },
    "SYMPTOM": {
        "kalyakal": "tired", "yoowart": "fever", "wara": "bad/sick", "moorditj": "strong/severe",
        "moorn": "sick/unwell", "nyidiny": "cold"
    },
    "NEGATION": {"kadak": "without", "kwop yoowart": "no fever"},
    "QUALITY": {"kwop": "good/well", "boola": "very", "boola boola": "always"}
}

def load_ner_model(model_path="apps/nlpService/models/noongar-clinical-ner-model-final"):
    """Load the NER model and tokenizer, with error handling."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
        return ner_pipeline, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load NER model from {model_path}: {str(e)}")

def find_exact_match(noongar_text):
    """Check for exact match in dataset."""
    for entry in DATASET:
        if entry["text"].lower() == noongar_text.lower():
            return entry["translation"], 1.0, "Exact match found in dataset."
    return None, 0.0, "No exact match found."

def vocabulary_fallback(noongar_text):
    """Fallback translation using vocabulary lookup."""
    words = noongar_text.lower().split()
    body_part, symptom, negation, quality = None, None, None, None
    for word in words:
        if word in ENTITY_MAPPING["BODY_PART"]:
            body_part = ENTITY_MAPPING["BODY_PART"][word]
        elif word in ENTITY_MAPPING["SYMPTOM"]:
            symptom = ENTITY_MAPPING["SYMPTOM"][word]
        elif word in ENTITY_MAPPING["NEGATION"]:
            negation = ENTITY_MAPPING["NEGATION"][word]
        elif word in ENTITY_MAPPING["QUALITY"]:
            quality = ENTITY_MAPPING["QUALITY"][word]
        else:
            matches = get_close_matches(word, 
                list(ENTITY_MAPPING["BODY_PART"].keys()) + 
                list(ENTITY_MAPPING["SYMPTOM"].keys()) + 
                list(ENTITY_MAPPING["NEGATION"].keys()) + 
                list(ENTITY_MAPPING["QUALITY"].keys()), n=1, cutoff=0.8)
            if matches:
                match = matches[0]
                if match in ENTITY_MAPPING["BODY_PART"]:
                    body_part = ENTITY_MAPPING["BODY_PART"][match]
                elif match in ENTITY_MAPPING["SYMPTOM"]:
                    symptom = ENTITY_MAPPING["SYMPTOM"][match]
                elif match in ENTITY_MAPPING["NEGATION"]:
                    negation = ENTITY_MAPPING["NEGATION"][match]
                elif match in ENTITY_MAPPING["QUALITY"]:
                    quality = ENTITY_MAPPING["QUALITY"][match]

    translation_parts = []
    if body_part:
        translation_parts.append(body_part)
    if symptom:
        translation_parts.append(f"is {symptom}")
    if quality:
        translation_parts.append(quality)
    if negation:
        translation_parts.append(negation)

    if translation_parts:
        translation = f"My {' '.join(translation_parts)}."
        clinical_translation = f"Patient reports {' '.join(translation_parts[1:] if len(translation_parts) > 1 else translation_parts)}."
        if negation:
            clinical_translation = f"Patient reports no {translation_parts[1].replace('is ', '') if len(translation_parts) > 1 else translation_parts[0]} in the {body_part}."
        elif body_part and len(translation_parts) > 1:
            clinical_translation = f"Patient reports {translation_parts[1].replace('is ', '')} {' '.join(translation_parts[2:])} in the {body_part}."
        return translation, clinical_translation, 0.8, "Translation inferred from vocabulary fallback."
    return "Unable to translate.", "Unable to translate; please clarify input.", 0.0, "No recognized entities in vocabulary."

def merge_subwords(ner_results, tokenizer, input_text):
    """Merge subword tokens into whole words with aggregated labels."""
    try:
        tokens = tokenizer(input_text, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True)
        offset_mapping = tokens["offset_mapping"][0].tolist()
        input_ids = tokens["input_ids"][0].tolist()
        words = input_text.split()
        merged_entities = []
        current_word = ""
        current_label = None
        current_scores = []
        current_start = None
        word_idx = 0

        # Debug tokenizer output
        print(f"Debug: Tokens for '{input_text}': {tokenizer.convert_ids_to_tokens(input_ids)}")
        print(f"Debug: Offset mapping: {offset_mapping}")
        print(f"Debug: NER results: {ner_results}")

        # Track word boundaries
        word_boundaries = [(0, 0)]
        pos = 0
        for word in words:
            start = input_text.find(word, pos)
            end = start + len(word)
            word_boundaries.append((start, end))
            pos = end + 1

        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # Skip special tokens
                continue
            token = tokenizer.convert_ids_to_tokens(input_ids[i])
            if token in ["<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"]:
                continue
            token_text = token.lstrip("▁").lstrip("##")
            # Handle both 'entity' and 'entity_group' keys
            entity_key = "entity_group" if "entity_group" in ner_results[0] else "entity"
            score = float(next((r["score"] for r in ner_results if r["start"] == start and r["end"] == end), 0.0))
            label = next((r[entity_key] for r in ner_results if r["start"] == start and r["end"] == end), "O")
            # Strip 'B-' prefix from label
            label = label.replace("B-", "") if label.startswith("B-") else label

            if not current_word:
                current_word = token_text
                current_start = start
                current_label = label
                current_scores = [score]
            else:
                current_word += token_text
                current_scores.append(score)
                # Use the label with the highest score
                if score > max(current_scores[:-1], default=0.0):
                    current_label = label

            # Check if current_word matches a whole word
            if word_idx < len(words) and input_text[current_start:current_start + len(current_word)].lower() == words[word_idx].lower():
                if current_label != "O":
                    merged_entities.append({
                        "word": current_word,
                        "label": current_label,
                        "score": float(np.mean(current_scores)) if current_scores else 0.0
                    })
                current_word = ""
                current_label = None
                current_scores = []
                current_start = None
                word_idx += 1

        # Handle remaining word
        if current_word and current_label != "O" and word_idx < len(words):
            if input_text[current_start:current_start + len(current_word)].lower() == words[word_idx].lower():
                merged_entities.append({
                    "word": current_word,
                    "label": current_label,
                    "score": float(np.mean(current_scores)) if current_scores else 0.0
                })

        return merged_entities
    except Exception as e:
        print(f"Debug: Subword merging error: {str(e)}")
        return []

def translate_with_ner(noongar_text, ner_pipeline, tokenizer):
    """Translate Noongar text to English using NER model and dataset."""
    output = {
        "input": noongar_text,
        "entities_detected": [],
        "translation": "",
        "clinical_translation": "",
        "confidence": 0.0,
        "notes": ""
    }

    # Validate input
    if not noongar_text.strip():
        output["notes"] = "Empty input provided."
        output["translation"] = "Unable to translate."
        output["clinical_translation"] = "Unable to translate; please clarify input."
        return output

    # Check for exact dataset match first
    exact_translation, exact_confidence, exact_notes = find_exact_match(noongar_text)
    if exact_translation:
        output["translation"] = exact_translation
        output["clinical_translation"] = f"Patient reports {exact_translation.replace('My ', '').replace('.', '')}."
        output["confidence"] = float(exact_confidence)
        output["notes"] = exact_notes
        return output

    # Run NER on input
    try:
        ner_results = ner_pipeline(noongar_text)
        if not ner_results:
            output["notes"] = "No entities detected by NER pipeline."
            output["translation"], output["clinical_translation"], output["confidence"], output["notes"] = vocabulary_fallback(noongar_text)
            return output

        # Merge subword tokens
        detected_entities = merge_subwords(ner_results, tokenizer, noongar_text)
        if not detected_entities:
            output["notes"] = "No valid entities after subword merging."
            output["translation"], output["clinical_translation"], output["confidence"], output["notes"] = vocabulary_fallback(noongar_text)
            return output

        avg_confidence = float(np.mean([ent["score"] for ent in detected_entities if ent["score"] > 0.0])) if any(ent["score"] > 0.0 for ent in detected_entities) else 0.0
        output["entities_detected"] = detected_entities
        output["confidence"] = avg_confidence if not np.isnan(avg_confidence) else 0.0
    except Exception as e:
        print(f"Debug: NER pipeline error: {str(e)}")
        output["notes"] = f"NER pipeline failed: {str(e)}"
        output["translation"], output["clinical_translation"], output["confidence"], output["notes"] = vocabulary_fallback(noongar_text)
        return output

    # Map entities to English
    body_part, symptom, negation, quality = None, None, None, None
    for ent in detected_entities:
        word = ent["word"].lower()
        label = ent["label"]
        # Direct mapping
        if label == "BODY_PART" and word in ENTITY_MAPPING["BODY_PART"]:
            body_part = ENTITY_MAPPING["BODY_PART"][word]
        elif label == "SYMPTOM" and word in ENTITY_MAPPING["SYMPTOM"]:
            symptom = ENTITY_MAPPING["SYMPTOM"][word]
        elif label == "NEGATION" and word in ENTITY_MAPPING["NEGATION"]:
            negation = ENTITY_MAPPING["NEGATION"][word]
        elif label == "QUALITY" and word in ENTITY_MAPPING["QUALITY"]:
            quality = ENTITY_MAPPING["QUALITY"][word]
        # Fuzzy matching for close terms
        elif label == "BODY_PART":
            matches = get_close_matches(word, ENTITY_MAPPING["BODY_PART"].keys(), n=1, cutoff=0.8)
            if matches:
                body_part = ENTITY_MAPPING["BODY_PART"][matches[0]]
        elif label == "SYMPTOM":
            matches = get_close_matches(word, ENTITY_MAPPING["SYMPTOM"].keys(), n=1, cutoff=0.8)
            if matches:
                symptom = ENTITY_MAPPING["SYMPTOM"][matches[0]]
        elif label == "NEGATION":
            matches = get_close_matches(word, ENTITY_MAPPING["NEGATION"].keys(), n=1, cutoff=0.8)
            if matches:
                negation = ENTITY_MAPPING["NEGATION"][matches[0]]
        elif label == "QUALITY":
            matches = get_close_matches(word, ENTITY_MAPPING["QUALITY"].keys(), n=1, cutoff=0.8)
            if matches:
                quality = ENTITY_MAPPING["QUALITY"][matches[0]]

    # Construct translation
    translation_parts = []
    if body_part:
        translation_parts.append(body_part)
    if symptom:
        translation_parts.append(f"is {symptom}")
    if quality:
        translation_parts.append(quality)
    if negation:
        translation_parts.append(negation)

    if translation_parts:
        output["translation"] = f"My {' '.join(translation_parts)}."
        clinical_translation = f"Patient reports {' '.join(translation_parts[1:] if len(translation_parts) > 1 else translation_parts)}."
        if negation:
            output["clinical_translation"] = f"Patient reports no {translation_parts[1].replace('is ', '') if len(translation_parts) > 1 else translation_parts[0]} in the {body_part}."
        elif body_part and len(translation_parts) > 1:
            output["clinical_translation"] = f"Patient reports {translation_parts[1].replace('is ', '')} {' '.join(translation_parts[2:])} in the {body_part}."
        else:
            output["clinical_translation"] = clinical_translation
        output["notes"] = "Translation inferred from NER and vocabulary."
        if avg_confidence < 0.7:
            output["notes"] += " Low confidence; verify with native speaker."
    else:
        output["translation"] = "Unable to identify key entities."
        output["clinical_translation"] = "Unable to translate; please clarify input."
        output["notes"] = "No recognized entities detected."

    return output

def main():
    # Load NER model and tokenizer
    try:
        ner_pipeline, tokenizer = load_ner_model()
        print("NER model loaded successfully on CUDA.")
    except ValueError as e:
        print(json.dumps({"error": str(e)}, indent=2))
        return

    # Example inputs
    sample_inputs = [
        "Ngaitj koort kalyakal boola",
        "Ngaitj mooly yoowart kadak",
        "Ngaitj djen wara boola"
    ]

    # Process each input
    results = []
    for sample_input in sample_inputs:
        result = translate_with_ner(sample_input, ner_pipeline, tokenizer)
        results.append(result)

    # Save all results to JSON
    output_file = "ner_translation_output.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": f"Failed to save JSON output: {str(e)}"}, indent=2))

if __name__ == "__main__":
    main()