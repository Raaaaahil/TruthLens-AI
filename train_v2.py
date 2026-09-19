"""
TruthLens AI V2 — Multi-Dataset BERT Training

Datasets:
    1. WELFake (local CSV)
    2. ISOT cleaned/fine-tuned dataset (Hugging Face)

Label convention:
    0 = FAKE
    1 = REAL

Important:
    - V1 model is NOT touched.
    - V2 is saved separately in models/truthlens-bert-v2
    - WELFake has its own validation/test split.
    - ISOT validation is kept completely external to training.
    - TrainingArguments are built dynamically so this script works
      with the installed Transformers API instead of depending on
      version-specific arguments.
"""

import os
import re
import random
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_MODEL = "bert-base-uncased"

WELFAKE_PATH = "data/WELFake_Dataset.csv"

ISOT_DATASET = "Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022"

OUTPUT_DIR = "models/truthlens-bert-v2"

MAX_LENGTH = 256

TRAIN_BATCH = 8
EVAL_BATCH = 8

GRAD_ACCUM = 2

EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

# ============================================================
# REPRODUCIBILITY
# ============================================================

set_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS AI V2 TRAINING")
print("=" * 70)

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GB:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        ),
    )

print("Base model:", BASE_MODEL)
print("Datasets: WELFake + ISOT")
print("Labels: 0=FAKE, 1=REAL")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 1. LOAD WELFAKE
# ============================================================

print("\n[1/6] Loading WELFake...")

if not os.path.exists(WELFAKE_PATH):
    raise FileNotFoundError(
        f"WELFake dataset not found at: {WELFAKE_PATH}"
    )

wf = pd.read_csv(WELFAKE_PATH)

required_columns = {"title", "text", "label"}

missing_columns = required_columns - set(wf.columns)

if missing_columns:
    raise ValueError(
        f"WELFake is missing required columns: {missing_columns}"
    )

wf["title"] = wf["title"].fillna("").map(clean_text)

wf["text"] = wf["text"].fillna("").map(clean_text)

wf["label"] = pd.to_numeric(
    wf["label"],
    errors="coerce",
)

wf = wf.dropna(subset=["label"])

wf["label"] = wf["label"].astype(int)

wf = wf[wf["label"].isin([0, 1])].copy()


# Combine title and article text.
wf["text_all"] = (
    wf["title"] + ". " + wf["text"]
).str.strip()

wf = wf[
    wf["text_all"].str.len() >= 80
].copy()

wf = wf[
    ["text_all", "label"]
].rename(
    columns={
        "text_all": "text"
    }
)

wf = wf.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)


# ============================================================
# WELFAKE SPLIT
# ============================================================

wf_train, wf_temp = train_test_split(
    wf,
    test_size=0.20,
    stratify=wf["label"],
    random_state=SEED,
)

wf_val, wf_test = train_test_split(
    wf_temp,
    test_size=0.50,
    stratify=wf_temp["label"],
    random_state=SEED,
)

print("WELFake:", len(wf))
print("  train:", len(wf_train))
print("  validation:", len(wf_val))
print("  test:", len(wf_test))


# ============================================================
# 2. LOAD ISOT
# ============================================================

print("\n[2/6] Loading ISOT...")

isot = load_dataset(ISOT_DATASET)

print(isot)


def convert_isot(split_name):
    df = isot[split_name].to_pandas()

    if "text" not in df.columns:
        raise ValueError(
            f"ISOT split '{split_name}' does not contain 'text'."
        )

    if "target" not in df.columns:
        raise ValueError(
            f"ISOT split '{split_name}' does not contain 'target'."
        )

    df["text"] = (
        df["text"]
        .fillna("")
        .map(clean_text)
    )

    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["target"]
    )

    df["target"] = df["target"].astype(int)

    df = df[
        df["target"].isin([0, 1])
    ].copy()

    # ISOT convention:
    #   0 = true
    #   1 = false
    #
    # TruthLens convention:
    #   0 = fake
    #   1 = real
    #
    # Therefore:
    #   TruthLens label = 1 - ISOT target

    df["label"] = 1 - df["target"]

    df = df[
        ["text", "label"]
    ]

    df = df[
        df["text"].str.len() >= 80
    ]

    df = df.drop_duplicates(
        subset=["text"]
    ).reset_index(drop=True)

    return df


isot_train = convert_isot("train")

isot_val = convert_isot("validation")

print("ISOT train:", len(isot_train))

print(
    "ISOT external validation:",
    len(isot_val),
)


# ============================================================
# REMOVE CROSS-DATASET DUPLICATES
# ============================================================

wf_train_texts = set(
    wf_train["text"]
    .str.lower()
)

isot_train = isot_train[
    ~isot_train["text"]
    .str.lower()
    .isin(wf_train_texts)
].copy()


# ============================================================
# 3. BUILD TRAINING DATA
# ============================================================

print("\n[3/6] Building combined training set...")


train_df = pd.concat(
    [
        wf_train,
        isot_train,
    ],
    ignore_index=True,
)


train_df = (
    train_df
    .drop_duplicates(
        subset=["text"]
    )
    .sample(
        frac=1.0,
        random_state=SEED,
    )
    .reset_index(drop=True)
)


val_df = (
    wf_val
    .drop_duplicates(
        subset=["text"]
    )
    .reset_index(drop=True)
)


test_df = (
    wf_test
    .drop_duplicates(
        subset=["text"]
    )
    .reset_index(drop=True)
)


print("Combined train:", len(train_df))

print(
    "WELFake validation:",
    len(val_df),
)

print(
    "WELFake test:",
    len(test_df),
)

print(
    "ISOT external test:",
    len(isot_val),
)


print("\nTrain labels:")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

print("0=FAKE, 1=REAL")


# ============================================================
# DATASET BALANCE
# ============================================================

fake_count = int(
    (train_df["label"] == 0).sum()
)

real_count = int(
    (train_df["label"] == 1).sum()
)

print("\nTraining class balance:")

print(
    f"FAKE: {fake_count:,}"
)

print(
    f"REAL: {real_count:,}"
)

print(
    "REAL/FAKE ratio:",
    round(real_count / max(fake_count, 1), 3),
)


# ============================================================
# 4. TOKENIZATION + MODEL
# ============================================================

print("\n[4/6] Loading tokenizer/model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


def to_hf(df):
    return Dataset.from_pandas(
        df[
            [
                "text",
                "label",
            ]
        ].reset_index(drop=True),
        preserve_index=False,
    )


train_ds = to_hf(train_df)

val_ds = to_hf(val_df)

test_ds = to_hf(test_df)

isot_ds = to_hf(isot_val)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_tok = train_ds.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

val_tok = val_ds.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

test_tok = test_ds.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

isot_tok = isot_ds.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)


model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    id2label={
        0: "FAKE",
        1: "REAL",
    },
    label2id={
        "FAKE": 0,
        "REAL": 1,
    },
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    logits_tensor = torch.tensor(
        logits
    )

    probs = torch.softmax(
        logits_tensor,
        dim=-1,
    ).numpy()

    predictions = probs.argmax(
        axis=1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    try:
        auc = roc_auc_score(
            labels,
            probs[:, 1],
        )
    except Exception:
        auc = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


# ============================================================
# DATA COLLATOR
# ============================================================

collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# TRANSFORMERS COMPATIBILITY
# ============================================================

"""
Your installed Transformers version has already shown that some
TrainingArguments parameters differ from older/newer releases.

Instead of making you manually change the script every time,
we inspect the installed API and only pass arguments that it
actually supports.
"""

training_signature = inspect.signature(
    TrainingArguments.__init__
)

supported_training_args = set(
    training_signature.parameters.keys()
)


def add_argument(
    dictionary,
    name,
    value,
):
    if name in supported_training_args:
        dictionary[name] = value
    else:
        print(
            f"[Compatibility] Skipping unsupported "
            f"TrainingArguments option: {name}"
        )


training_kwargs = {}

add_argument(
    training_kwargs,
    "output_dir",
    OUTPUT_DIR,
)

add_argument(
    training_kwargs,
    "num_train_epochs",
    EPOCHS,
)

add_argument(
    training_kwargs,
    "learning_rate",
    LEARNING_RATE,
)

add_argument(
    training_kwargs,
    "per_device_train_batch_size",
    TRAIN_BATCH,
)

add_argument(
    training_kwargs,
    "per_device_eval_batch_size",
    EVAL_BATCH,
)

add_argument(
    training_kwargs,
    "gradient_accumulation_steps",
    GRAD_ACCUM,
)

add_argument(
    training_kwargs,
    "weight_decay",
    WEIGHT_DECAY,
)

# Use warmup_steps rather than warmup_ratio.
# 5% warmup of the approximate total update steps.
approx_steps_per_epoch = max(
    1,
    len(train_df)
    // (
        TRAIN_BATCH
        * GRAD_ACCUM
    ),
)

approx_total_steps = (
    approx_steps_per_epoch
    * EPOCHS
)

warmup_steps = max(
    100,
    int(
        approx_total_steps
        * 0.05
    ),
)

add_argument(
    training_kwargs,
    "warmup_steps",
    warmup_steps,
)

add_argument(
    training_kwargs,
    "lr_scheduler_type",
    "cosine",
)

add_argument(
    training_kwargs,
    "fp16",
    torch.cuda.is_available(),
)

add_argument(
    training_kwargs,
    "logging_steps",
    100,
)

# Current Transformers uses eval_strategy.
add_argument(
    training_kwargs,
    "eval_strategy",
    "epoch",
)

# Older Transformers may use evaluation_strategy.
if (
    "eval_strategy"
    not in supported_training_args
    and "evaluation_strategy"
    in supported_training_args
):
    add_argument(
        training_kwargs,
        "evaluation_strategy",
        "epoch",
    )

add_argument(
    training_kwargs,
    "save_strategy",
    "epoch",
)

add_argument(
    training_kwargs,
    "save_total_limit",
    2,
)

add_argument(
    training_kwargs,
    "load_best_model_at_end",
    True,
)

add_argument(
    training_kwargs,
    "metric_for_best_model",
    "f1",
)

add_argument(
    training_kwargs,
    "greater_is_better",
    True,
)

add_argument(
    training_kwargs,
    "report_to",
    "none",
)

add_argument(
    training_kwargs,
    "dataloader_num_workers",
    0,
)

add_argument(
    training_kwargs,
    "seed",
    SEED,
)

# Helps prevent GPU memory accumulation during evaluation.
add_argument(
    training_kwargs,
    "eval_accumulation_steps",
    16,
)


print("\n[Compatibility] Supported TrainingArguments detected.")

print(
    "[Compatibility] Warmup steps:",
    warmup_steps,
)

print(
    "[Compatibility] DataLoader workers: 0 (Windows-safe)"
)

print(
    "[Compatibility] Creating TrainingArguments..."
)

args = TrainingArguments(
    **training_kwargs
)


# ============================================================
# TRAINER COMPATIBILITY
# ============================================================

trainer_signature = inspect.signature(
    Trainer.__init__
)

supported_trainer_args = set(
    trainer_signature.parameters.keys()
)

trainer_kwargs = {
    "model": model,
    "args": args,
    "train_dataset": train_tok,
    "eval_dataset": val_tok,
    "data_collator": collator,
    "compute_metrics": compute_metrics,
    "callbacks": [
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ],
}

# Transformers versions differ between tokenizer and
# processing_class. Use whichever the installed Trainer accepts.
if (
    "processing_class"
    in supported_trainer_args
):
    trainer_kwargs[
        "processing_class"
    ] = tokenizer

elif (
    "tokenizer"
    in supported_trainer_args
):
    trainer_kwargs[
        "tokenizer"
    ] = tokenizer


trainer = Trainer(
    **trainer_kwargs
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\n" + "=" * 70)

print("[5/6] STARTING TRAINING...")

print(
    "This is the actual V2 training run."
)

print(
    "Do not close this terminal while training."
)

print("=" * 70)

print()

print(
    "Training samples:",
    f"{len(train_df):,}",
)

print(
    "Validation samples:",
    f"{len(val_df):,}",
)

print(
    "Approximate optimizer steps:",
    f"{approx_total_steps:,}",
)

print(
    "Epochs:",
    EPOCHS,
)

print(
    "Batch size:",
    TRAIN_BATCH,
)

print(
    "Gradient accumulation:",
    GRAD_ACCUM,
)

print(
    "Effective batch size:",
    TRAIN_BATCH * GRAD_ACCUM,
)

print(
    "Max tokens:",
    MAX_LENGTH,
)

print()


trainer.train()


# ============================================================
# 6. EVALUATION
# ============================================================

print("\n" + "=" * 70)

print("[6/6] EVALUATING V2")

print("=" * 70)


def evaluate_named(
    dataset,
    name,
):
    metrics = trainer.evaluate(
        eval_dataset=dataset,
        metric_key_prefix=name,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        name.upper()
    )

    print(
        "=" * 70
    )

    for key, value in metrics.items():

        if isinstance(
            value,
            float,
        ):
            print(
                f"{key}: {value:.4f}"
            )

        else:
            print(
                f"{key}: {value}"
            )

    return metrics


welfake_metrics = evaluate_named(
    test_tok,
    "welfake_test",
)

isot_metrics = evaluate_named(
    isot_tok,
    "isot_external",
)


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving V2 model..."
)

Path(
    OUTPUT_DIR
).mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "training_metadata.txt",
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "TruthLens AI V2\n"
    )

    f.write(
        "============================\n"
    )

    f.write(
        f"Base model: {BASE_MODEL}\n"
    )

    f.write(
        "Datasets: WELFake + cleaned ISOT\n"
    )

    f.write(
        "Labels: 0=FAKE, 1=REAL\n"
    )

    f.write(
        f"Max length: {MAX_LENGTH}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Training batch: {TRAIN_BATCH}\n"
    )

    f.write(
        f"Gradient accumulation: {GRAD_ACCUM}\n"
    )

    f.write(
        f"Effective batch: "
        f"{TRAIN_BATCH * GRAD_ACCUM}\n"
    )

    f.write(
        f"Warmup steps: {warmup_steps}\n"
    )

    f.write(
        f"Combined training samples: "
        f"{len(train_df)}\n"
    )

    f.write(
        f"WELFake validation samples: "
        f"{len(val_df)}\n"
    )

    f.write(
        f"WELFake test samples: "
        f"{len(test_df)}\n"
    )

    f.write(
        f"ISOT external samples: "
        f"{len(isot_val)}\n"
    )

    f.write(
        "\nWELFake TEST METRICS\n"
    )

    for key, value in welfake_metrics.items():

        f.write(
            f"{key}: {value}\n"
        )

    f.write(
        "\nISOT EXTERNAL METRICS\n"
    )

    for key, value in isot_metrics.items():

        f.write(
            f"{key}: {value}\n"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print("V2 TRAINING COMPLETE")

print("=" * 70)

print(
    "Model saved to:",
    OUTPUT_DIR,
)

print(
    "Metadata saved to:",
    metadata_path,
)

print()

print(
    "Next stage:"
)

print(
    "1. Update predict.py to load V2."
)

print(
    "2. Add sliding-window inference."
)

print(
    "3. Test BBC / Reuters / AP / CNN."
)

print(
    "4. Compare V1 vs V2."
)

print(
    "5. Only then connect V2 to the backend."
)

print("=" * 70)
