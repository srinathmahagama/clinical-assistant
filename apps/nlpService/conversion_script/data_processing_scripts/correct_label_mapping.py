# correct_label_mapping.py
from datasets import load_from_disk, DatasetDict, Sequence, ClassLabel
import json

# Load your dataset
dataset = load_from_disk("./noongar_hf_dataset")

# Your actual label mapping based on the patterns
LABEL_MAPPING = {
    0: "O",           # Outside (ngaitj)
    1: "B-SYMPTOM",   # Beginning of Symptom
    2: "I-SYMPTOM",   # Inside Symptom
    3: "B-BODY_PART", # Beginning of Body Part
    4: "I-BODY_PART", # Inside Body Part
    5: "B-NEGATION",  # Beginning of Negation
    6: "I-NEGATION",  # Inside Negation
    7: "B-QUALITY",   # Beginning of Quality
    8: "I-QUALITY",   # Inside Quality
}

# Create reverse mappings
label2id = {v: k for k, v in LABEL_MAPPING.items()}
id2label = LABEL_MAPPING

print("✅ Label Mapping:")
for id, label in id2label.items():
    print(f"  {id}: {label}")

# Update dataset with correct label features
new_features = dataset["train"].features.copy()
new_features["ner_tags"] = Sequence(feature=ClassLabel(names=list(label2id.keys())))

# Create corrected dataset
corrected_dataset = DatasetDict({
    "train": dataset["train"].cast(new_features),
    "test": dataset["test"].cast(new_features)
})

# Save corrected dataset
corrected_dataset.save_to_disk("./noongar_hf_dataset_corrected")
print("✅ Corrected dataset saved to './noongar_hf_dataset_corrected'")

# Save label mappings for training
with open("label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
print("✅ Label mappings saved to 'label_mappings.json'")

# Verify the correction
print("\n✅ Verification:")
test_example = corrected_dataset["train"][0]
print(f"Tokens: {test_example['tokens']}")
print(f"NER Tags: {test_example['ner_tags']}")
print(f"Mapped Labels: {[id2label[tag] for tag in test_example['ner_tags']]}")