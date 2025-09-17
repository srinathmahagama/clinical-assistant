import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np

# Path to your dataset
DATA_PATH = r"C:\Users\imtha\Documents\Semester 3\TRP\clinical-assitant\clinical-assistant\apps\nlpService\data_generating_script\noongar_clinical_dataset_5000.json"

# Pretrained tokenizer
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define label set
LABELS = ["O", "B-SYMPTOM", "I-SYMPTOM",
          "B-BODY_PART", "I-BODY_PART",
          "B-NEGATION", "I-NEGATION",
          "B-QUALITY", "I-QUALITY"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

# --- Load and preprocess data ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare data in IOB format
examples = []
for item in data:
    text = item["text"]
    tokens = text.split()  # Simple tokenization
    labels = ["O"] * len(tokens)
    
    # Convert character offsets to token positions
    for entity in item["entities"]:
        entity_text = text[entity["start"]:entity["end"]]
        entity_tokens = entity_text.split()
        
        # Find where this entity appears in the tokenized text
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                labels[i] = f"B-{entity['label']}"
                for j in range(1, len(entity_tokens)):
                    labels[i+j] = f"I-{entity['label']}"
                break
    
    examples.append({
        "tokens": tokens,
        "ner_tags": [label2id[label] for label in labels]
    })

# Convert to HuggingFace dataset
def convert_to_features(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True,
        return_tensors=None
    )
    
    # Align labels with tokenizer output
    labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(tags[word_idx])
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized

# Create dataset
dataset = Dataset.from_list(examples)
tokenized_dataset = dataset.map(convert_to_features, batched=True)

# Split and save
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

dataset_dict.save_to_disk("./noongar_hf_dataset")
print("✅ Dataset saved successfully!")