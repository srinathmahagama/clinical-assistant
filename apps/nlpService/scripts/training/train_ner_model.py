#!/usr/bin/env python3
"""
Train the Noongar Clinical NER Model
"""

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

# Check if PyTorch is available
print("PyTorch available:", torch.__version__ if torch else "No")

# Define paths - FIXED PATHS
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
    
    # Load corrected dataset and label mappings
    print("📦 Loading dataset...")
    try:
        dataset = load_from_disk(str(PROCESSED_DATA_PATH))
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Please make sure the dataset exists at:", PROCESSED_DATA_PATH)
        return
    
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("You may need to install PyTorch: pip install torch torchvision torchaudio")
        return

    # Data collator for dynamic padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )

    # Compute metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
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
        num_train_epochs=3,  # Reduced for testing
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(BASE_DIR / "logs"),
        logging_steps=50,
        report_to="none",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # Save the final model
    try:
        MODEL_CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
        FINAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(FINAL_MODEL_PATH))
        print(f"✅ Model saved to '{FINAL_MODEL_PATH}'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    # Evaluate final model
    print("📈 Final evaluation:")
    try:
        eval_results = trainer.evaluate()
        print(f"F1 Score: {eval_results['eval_f1']:.4f}")
        print(f"Precision: {eval_results['eval_precision']:.4f}")
        print(f"Recall: {eval_results['eval_recall']:.4f}")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    train_model()