from transformers import pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "noongar-clinical-ner-model-finetuned"

# Load NER pipeline
ner_pipeline = pipeline("ner", model=str(MODEL_PATH), tokenizer=str(MODEL_PATH), aggregation_strategy="simple")

test_sentences = [
    ("Ngaitj djena", "My foot"),
    ("Ngaitj moorditj", "I am strong/healthy"),
    ("Ngaitj kaat", "My head"),
]

print("🧪 BASIC TRANSLATION TEST")
print("=" * 40)

for i, (noongar, english) in enumerate(test_sentences, 1):
    print(f"\n{i}. 📝 Noongar: {noongar} | Expected English: {english}")
    results = ner_pipeline(noongar)
    print("   🔍 Detected entities:", results)
