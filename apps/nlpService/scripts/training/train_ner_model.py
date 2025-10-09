#!/usr/bin/env python3
"""
Train the Noongar Clinical NER Model
"""

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification, 
        TrainingArguments, 
        Trainer,
        DataCollatorForTokenClassification
    )
    from datasets import load_from_disk
    import json
    import numpy as np
    from seqeval.metrics import classification_report, f1_score
    from pathlib import Path
    import torch
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install torch torchvision torchaudio transformers datasets seqeval accelerate")
    exit(1)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # apps/nlpService/
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "noongar_hf_dataset_corrected"
LABEL_MAPPINGS_PATH = DATA_DIR / "label_mappings.json"
MODEL_CHECKPOINTS_PATH = BASE_DIR / "models" / "noongar-clinical-ner-model"
FINAL_MODEL_PATH = BASE_DIR / "models" / "noongar-clinical-ner-model-final"

print("BASE_DIR:", BASE_DIR)
print("PROCESSED_DATA_PATH:", PROCESSED_DATA_PATH)
print("PROCESSED_DATA_PATH exists:", PROCESSED_DATA_PATH.exists())

# Model constants
MODEL_NAME = "xlm-roberta-base"
BATCH_SIZE = 16
MAX_LENGTH = 128

def train_model():
    """Train the Noongar clinical NER model"""
    
    # Load dataset
    print("📦 Loading dataset...")
    try:
        dataset = load_from_disk(str(PROCESSED_DATA_PATH))
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load label mappings
    try:
        with open(LABEL_MAPPINGS_PATH, "r") as f:
            label_mappings = json.load(f)
            label2id = label_mappings["label2id"]
            id2label = {int(k): v for k, v in label_mappings["id2label"].items()}
        print("✅ Label mappings loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading label mappings: {e}")
        return

    print("📊 Dataset Info:")
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"Number of labels: {len(label2id)}")

    # Initialize tokenizer and model
    print("🔄 Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    print("✅ Model and tokenizer loaded successfully!")

    # Tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],  # make sure your dataset has "tokens" field
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # ignore special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # ignore subword tokens
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("📦 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    print("✅ Dataset tokenized successfully!")

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Compute metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "f1": f1_score(true_labels, true_predictions),
            "precision": classification_report(true_labels, true_predictions, output_dict=True)["weighted avg"]["precision"],
            "recall": classification_report(true_labels, true_predictions, output_dict=True)["weighted avg"]["recall"],
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_CHECKPOINTS_PATH),
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(BASE_DIR / "logs"),
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training completed successfully!")

    # Save model
    FINAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_PATH))
    print(f"✅ Model saved to '{FINAL_MODEL_PATH}'")

    # Evaluate final model
    print("📈 Final evaluation:")
    eval_results = trainer.evaluate()
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall: {eval_results['eval_recall']:.4f}")

if __name__ == "__main__":
    train_model()
