"""
Meal NER fine-tuning script for Google Colab.
Base model: cointegrated/rubert-tiny2
Task: token classification (NER) for meal planning voice commands (13 labels).

Usage:
  1. Upload train.json, val.json, test.json to Colab (or mount Google Drive) into TRAINING_DATA_DIR,
     or pass the folder as the first CLI argument: python finetune.py /path/to/data
  2. Run all cells top to bottom.
  3. Fine-tuned model is saved to /content/meal-ner-model (and optionally Google Drive).

Training data format:
  [
    {
      "text": "добавь курицу 150 грамм на ужин",
      "intent": "add_dish",
      "tokens": ["добавь", "курицу", "150", "грамм", "на", "ужин"],
      "labels": ["O", "B-FOOD_NAME", "B-QUANTITY", "B-UNIT", "O", "B-MEAL_TITLE"]
    },
    ...
  ]
"""

# ── 0. Install dependencies (run once in Colab) ───────────────────────────────
# !pip install transformers datasets seqeval evaluate accelerate -q

import json
import random
import sys
import numpy as np
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import evaluate

# ── 1. Config ─────────────────────────────────────────────────────────────────

MODEL_NAME = "cointegrated/rubert-tiny2"
TRAINING_DATA_DIR = "/content/training_data"   # folder with train.json, val.json, test.json (override: argv[1])
OUTPUT_DIR = "/content/meal-ner-model"
DRIVE_OUTPUT = "/content/drive/MyDrive/meal-ner-model"  # set None to skip Drive save

MAX_LENGTH = 128
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
SEED = 42

# ── 2. Label map ──────────────────────────────────────────────────────────────
# INTENT is sentence-level (top-level field in JSON) — not labeled at token level.
# The ORDER is the label-id order stored in the model config and read by the CoreML
# conversion / Swift decoder. Do not reorder.

LABELS = [
    "O",
    "B-MEAL_TITLE",  "I-MEAL_TITLE",
    "B-FOOD_NAME",   "I-FOOD_NAME",
    "B-QUANTITY",    "I-QUANTITY",
    "B-UNIT",        "I-UNIT",
    "B-DATE",        "I-DATE",
    "B-TIME",        "I-TIME",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ── 3. Load training data ─────────────────────────────────────────────────────

def load_split(data_dir: str, name: str) -> list[dict]:
    with open(f"{data_dir}/{name}.json", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} {name} samples from {data_dir}")
    return samples

# ── 4. Tokenize with label alignment ─────────────────────────────────────────
# RuBERT uses WordPiece — words split into subwords.
# Only the first subword of each word gets the real label; others get -100 (ignored in loss).

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(samples: list[dict]) -> Dataset:
    all_input_ids, all_attention_masks, all_labels = [], [], []

    for sample in samples:
        word_tokens = sample["tokens"]
        word_labels = [LABEL2ID.get(l, 0) for l in sample["labels"]]

        encoding = tokenizer(
            word_tokens,
            is_split_into_words=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )

        label_ids = []
        previous_word_idx = None
        for word_idx in encoding.word_ids():
            if word_idx is None:
                label_ids.append(-100)          # [CLS], [SEP], [PAD]
            elif word_idx != previous_word_idx:
                label_ids.append(word_labels[word_idx])  # first subword → real label
            else:
                label_ids.append(-100)          # continuation subword → ignore
            previous_word_idx = word_idx

        all_input_ids.append(encoding["input_ids"])
        all_attention_masks.append(encoding["attention_mask"])
        all_labels.append(label_ids)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })

# ── 5. Metrics ────────────────────────────────────────────────────────────────

seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(predictions, labels):
        true_labels.append([ID2LABEL[l] for l in label_row if l != -100])
        true_preds.append([ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100])

    result = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": result["overall_precision"],
        "recall": result["overall_recall"],
        "f1": result["overall_f1"],
        "accuracy": result["overall_accuracy"],
    }

# ── 6. Train ──────────────────────────────────────────────────────────────────

def print_test_report(trainer, test_samples):
    """Entity-level precision/recall/F1 per entity type (seqeval) on test.json."""
    from seqeval.metrics import classification_report
    test_dataset = tokenize_and_align(test_samples)
    output = trainer.predict(test_dataset)
    predictions = np.argmax(output.predictions, axis=2)
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(predictions, output.label_ids):
        true_labels.append([ID2LABEL[l] for l in label_row if l != -100])
        true_preds.append([ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100])
    print("\nTest set (entity level, seqeval):")
    print(classification_report(true_labels, true_preds, digits=4))


def train():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else TRAINING_DATA_DIR
    train_samples = load_split(data_dir, "train")
    val_samples = load_split(data_dir, "val")
    test_samples = load_split(data_dir, "test")

    random.Random(SEED).shuffle(train_samples)
    train_dataset = tokenize_and_align(train_samples)
    eval_dataset  = tokenize_and_align(val_samples)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print_test_report(trainer, test_samples)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # Optionally save to Google Drive
    if DRIVE_OUTPUT:
        import shutil
        from google.colab import drive
        drive.mount("/content/drive")
        shutil.copytree(OUTPUT_DIR, DRIVE_OUTPUT, dirs_exist_ok=True)
        print(f"Model copied to Google Drive: {DRIVE_OUTPUT}")

if __name__ == "__main__":
    train()
