import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/WELFake_Dataset.csv"
MODEL_PATH = "models/truthlens-bert"
OUTPUT_DIR = "outputs"

SAMPLE_SIZE = 20000
TEST_SIZE = 2000
BATCH_SIZE = 16
MAX_LENGTH = 256


os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRUTHLENS AI - MODEL EVALUATION")
print("=" * 70)
print(f"Device: {device}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["title", "text", "label"]
).copy()

df["content"] = (
    df["title"].astype(str)
    + " "
    + df["text"].astype(str)
)

print(f"Total available articles: {len(df)}")


# ============================================================
# SAMPLE DATA
# ============================================================

if len(df) > SAMPLE_SIZE:

    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["label"],
        random_state=42
    )

print(f"Evaluation pool: {len(df)}")


# ============================================================
# CREATE TRAIN / TEST SPLIT
# ============================================================

_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=42
)

test_df = test_df.reset_index(drop=True)

print(f"Test samples: {len(test_df)}")

print("\nTest label distribution:")
print(test_df["label"].value_counts())


# ============================================================
# DATASET CLASS
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


class NewsDataset(Dataset):

    def __init__(self, dataframe):

        self.texts = dataframe["content"].tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoding = tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }


test_dataset = NewsDataset(test_df)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# INFERENCE
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []


print("\nRunning evaluation...")

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of REAL class (label 1)
        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n")
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=["FAKE", "REAL"],
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("CONFUSION MATRIX")
print("=" * 70)
print(cm)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "BERT",
    "dataset": "WELFake",
    "evaluation_samples": len(test_df),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist()
}


with open(
    os.path.join(OUTPUT_DIR, "metrics.json"),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("TruthLens AI - Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["FAKE", "REAL"]
)

plt.yticks(
    [0, 1],
    ["FAKE", "REAL"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=200
)

plt.close()


print("\n")
print("=" * 70)
print("FILES GENERATED")
print("=" * 70)

print("outputs/metrics.json")
print("outputs/classification_report.txt")
print("outputs/confusion_matrix.png")

print("\nEvaluation complete.")