import os
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/WELFake_Dataset.csv"
MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256
SAMPLE_SIZE = 20000

OUTPUT_DIR = "models/truthlens-bert"

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TruthLens AI - BERT Fake News Detection")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original dataset: {len(df)} articles")

# Remove missing values
df = df.dropna(subset=["title", "text", "label"])

# Combine title + article
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

# Keep only required columns
df = df[["content", "label"]]

# ============================================================
# SAMPLE DATA
# ============================================================

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["label"],
        random_state=42
    )

print(f"Using {len(df)} articles for training.")

print("\nClass distribution:")
print(df["label"].value_counts())

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("\nDataset split:")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df.reset_index(drop=True)
)

val_dataset = Dataset.from_pandas(
    val_df.reset_index(drop=True)
)

test_dataset = Dataset.from_pandas(
    test_df.reset_index(drop=True)
)

# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["content"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

print("Tokenizing...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

# Rename label correctly for Trainer
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# Remove unnecessary columns
columns_to_remove = ["content"]

if "__index_level_0__" in train_dataset.column_names:
    columns_to_remove.append("__index_level_0__")

train_dataset = train_dataset.remove_columns(
    [c for c in columns_to_remove if c in train_dataset.column_names]
)

val_dataset = val_dataset.remove_columns(
    [c for c in columns_to_remove if c in val_dataset.column_names]
)

test_dataset = test_dataset.remove_columns(
    [c for c in columns_to_remove if c in test_dataset.column_names]
)

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

# ============================================================
# MODEL
# ============================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "FAKE",
        1: "REAL"
    },
    label2id={
        "FAKE": 0,
        "REAL": 1
    }
)

model.to(device)

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    logging_steps=50,

    fp16=torch.cuda.is_available(),

    report_to="none",

    save_total_limit=2
)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING BERT TRAINING")
print("=" * 60)

trainer.train()

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

results = trainer.evaluate(test_dataset)

print("\nTest Results:")

for key, value in results.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")

    else:
        print(f"{key}: {value}")

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved to:")
print(OUTPUT_DIR)

print("\nTraining completed successfully!")