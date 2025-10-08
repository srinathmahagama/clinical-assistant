
# verification_script.py
from datasets import load_from_disk
import numpy as np

# Load your dataset
dataset = load_from_disk("./noongar_hf_dataset")

# Check what labels are actually used
all_ner_tags = []
for split in ["train", "test"]:
    for example in dataset[split]:
        all_ner_tags.extend(example["ner_tags"])

unique_labels = sorted(set(all_ner_tags))
print("Unique NER tag values found:", unique_labels)

# Check first few examples to understand the mapping
print("\n=== First 5 Examples ===")
for i in range(5):
    example = dataset["train"][i]
    print(f"\nExample {i+1}:")
    print(f"Tokens: {example['tokens']}")
    print(f"NER Tags: {example['ner_tags']}")
    
    # Try to infer the mapping
    for token, tag in zip(example['tokens'], example['ner_tags']):
        print(f"  {token} -> {tag}")

# Check if we can find patterns
print("\n=== Common Patterns ===")
common_patterns = {}
for i in range(min(20, len(dataset["train"]))):
    example = dataset["train"][i]
    for token, tag in zip(example['tokens'], example['ner_tags']):
        if tag not in common_patterns:
            common_patterns[tag] = []
        common_patterns[tag].append(token)

for tag, tokens in common_patterns.items():
    print(f"Tag {tag}: {tokens[:5]}...")  # Show first 5 tokens per tag