#!/usr/bin/env python3
"""
Fine-tune the Noongar Clinical NER Model on existing trained model
Refined for your directory structure
"""

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification, 
        TrainingArguments, 
        Trainer,
        DataCollatorForTokenClassification
    )
    from datasets import Dataset, load_from_disk
    import json
    import numpy as np
    from seqeval.metrics import classification_report, f1_score
    from pathlib import Path
    import torch
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install transformers datasets seqeval")
    exit(1)

# Define paths - ADJUSTED FOR YOUR STRUCTURE
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # apps/nlpService/
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "noongar_hf_dataset_corrected"
LABEL_MAPPINGS_PATH = DATA_DIR / "label_mappings.json"
FINE_TUNING_DATA_PATH = DATA_DIR / "noongar_fine_tuning_dataset.json"

# Model paths - USING YOUR EXISTING MODELS
EXISTING_MODEL_PATH = BASE_DIR / "models" / "noongar-clinical-ner-model-final"
FINE_TUNED_MODEL_PATH = BASE_DIR / "models" / "noongar-clinical-ner-model-finetuned"

print("=== Directory Structure ===")
print(f"BASE_DIR: {BASE_DIR}")
print(f"EXISTING_MODEL_PATH: {EXISTING_MODEL_PATH}")
print(f"EXISTING_MODEL_PATH exists: {EXISTING_MODEL_PATH.exists()}")
print(f"PROCESSED_DATA_PATH exists: {PROCESSED_DATA_PATH.exists()}")

# Fine-tuning constants
BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 1e-5

def load_fine_tuning_data():
    """Load the new fine-tuning dataset"""
    try:
        with open(FINE_TUNING_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        training_data = data['noongar_fine_tuning_dataset']['training_data']
        
        # Convert to tokens and labels format compatible with your existing dataset
        examples = []
        for item in training_data:
            # Simple tokenization (split by space)
            tokens = item['noongar_text'].split()
            # Create dummy labels (you'll need to map these properly)
            ner_tags = [0] * len(tokens)  # Default to O tag
            
            examples.append({
                'tokens': tokens,
                'ner_tags': ner_tags,
                'clinical_english': item['clinical_english'],
                'symptom_type': item['symptom_type']
            })
        
        return Dataset.from_list(examples)
        
    except Exception as e:
        print(f"❌ Error loading fine-tuning data: {e}")
        return None

def fine_tune_model():
    """Fine-tune the existing Noongar clinical NER model"""
    
    # Load existing dataset for consistency
    print("📦 Loading existing dataset...")
    try:
        dataset = load_from_disk(str(PROCESSED_DATA_PATH))
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load fine-tuning data
    print("📦 Loading fine-tuning data...")
    fine_tuning_data = load_fine_tuning_data()
    if fine_tuning_data is None:
        print("❌ No fine-tuning data found, using existing dataset only")
        fine_tuning_data = dataset['train']
    else:
        print(f"✅ Fine-tuning data loaded: {len(fine_tuning_data)} examples")
    
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
    print(f"Fine-tuning examples: {len(fine_tuning_data)}")
    print(f"Number of labels: {len(label2id)}")

    # Load EXISTING model
    print("🔄 Loading existing trained model...")
    try:
        if EXISTING_MODEL_PATH.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(EXISTING_MODEL_PATH))
            model = AutoModelForTokenClassification.from_pretrained(
                str(EXISTING_MODEL_PATH),
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id
            )
            print("✅ Existing model loaded successfully!")
        else:
            raise FileNotFoundError("Existing model not found")
    except Exception as e:
        print(f"❌ Error loading existing model: {e}")
        print("Falling back to base model...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForTokenClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )

    # Freeze early layers to preserve existing knowledge
    print("🔒 Freezing base layers...")
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if "classifier" not in name:  # Only train classifier layer initially
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True
    
    print(f"📊 Frozen {frozen_params}/{total_params} layers")
    print("📋 Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

    # Tokenize function (same as your original)
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
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
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("📦 Tokenizing datasets...")
    # Tokenize fine-tuning data
    tokenized_fine_tuning = fine_tuning_data.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=fine_tuning_data.column_names if hasattr(fine_tuning_data, 'column_names') else []
    )
    
    # Tokenize existing dataset for evaluation
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    print("✅ Datasets tokenized successfully!")

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Compute metrics (same as your original)
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

    # FINE-TUNING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=str(FINE_TUNED_MODEL_PATH / "checkpoints"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,  # Fewer epochs for fine-tuning
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(BASE_DIR / "logs"),
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=1,
        warmup_steps=50,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_fine_tuning,
        eval_dataset=tokenized_dataset["test"],  # Use original test set for evaluation
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("🚀 Starting FINE-TUNING...")
    print("📈 Strategy: Freeze base layers, train only classifier")
    
    # Train the model
    trainer.train()
    
    print("✅ Fine-tuning completed successfully!")

    # Save final model
    FINE_TUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINE_TUNED_MODEL_PATH))
    tokenizer.save_pretrained(str(FINE_TUNED_MODEL_PATH))
    print(f"✅ Fine-tuned model saved to '{FINE_TUNED_MODEL_PATH}'")

    # Evaluate final model
    print("📈 Final evaluation:")
    eval_results = trainer.evaluate()
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall: {eval_results['eval_recall']:.4f}")

    print(f"\n🎯 Fine-tuning complete! Model saved to: {FINE_TUNED_MODEL_PATH}")

if __name__ == "__main__":
    fine_tune_model()