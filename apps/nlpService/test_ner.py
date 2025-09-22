# test_ner.py
# Author: Mohammed Munabeer Imthath Saabir
# Student ID: 105282609

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os

# Relative path to your model folder (forward slashes are safe on Windows)
MODEL_PATH = "apps/nlpService/models/noongar-clinical-ner-model-final"

# Load tokenizer and model explicitly from local files
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Create NER pipeline
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# Example text
text = "Ngaitj koort kalyakal"

# Run NER
entities = ner(text)

# Print results
print("Input text:", text)
print("Extracted entities:", entities)
