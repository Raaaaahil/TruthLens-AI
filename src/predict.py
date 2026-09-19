import os
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "truthlens-bert-v2"
)

MAX_LENGTH = 256
STRIDE = 64
INFERENCE_BATCH_SIZE = 4


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("TRUTHLENS AI - V2 INFERENCE")
print("=" * 70)

print(f"Model path : {MODEL_PATH}")
print(f"Device     : {DEVICE}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"V2 model not found at:\n{MODEL_PATH}"
    )

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print(f"Labels     : {model.config.id2label}")

print("=" * 70)


# ============================================================
# LABEL MAPPING
# ============================================================

def get_label_mapping():

    id2label = getattr(
        model.config,
        "id2label",
        {}
    )

    mapping = {}

    for key, value in id2label.items():

        try:
            index = int(key)
        except Exception:
            index = key

        mapping[index] = str(value).upper()

    if not mapping:
        mapping = {
            0: "FAKE",
            1: "REAL"
        }

    return mapping


LABEL_MAP = get_label_mapping()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if text is None:
        return ""

    text = str(text)

    return " ".join(
        text.split()
    ).strip()


# ============================================================
# CREATE MANUAL SLIDING WINDOWS
# ============================================================

def create_chunks(
    text: str,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE
) -> List[Dict]:
    """
    Deterministic sliding-window tokenization.

    BERT format:

        [CLS] content [SEP]

    MAX_LENGTH = 256

    Therefore content capacity = 254 tokens.

    STRIDE = 64 tokens of overlap.
    """

    # --------------------------------------------------------
    # Special token IDs
    # --------------------------------------------------------

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    if cls_token_id is None:
        raise ValueError(
            "Tokenizer does not have a CLS token."
        )

    if sep_token_id is None:
        raise ValueError(
            "Tokenizer does not have a SEP token."
        )

    if pad_token_id is None:
        pad_token_id = 0

    # --------------------------------------------------------
    # Content capacity
    # --------------------------------------------------------

    special_tokens_count = 2

    content_window_size = (
        max_length -
        special_tokens_count
    )

    if content_window_size <= 0:
        raise ValueError(
            "MAX_LENGTH is too small."
        )

    if stride >= content_window_size:
        raise ValueError(
            "STRIDE must be smaller than the content window."
        )

    # --------------------------------------------------------
    # Tokenize COMPLETE article
    # --------------------------------------------------------

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        verbose=False
    )

    all_token_ids = encoded["input_ids"]

    total_tokens = len(
        all_token_ids
    )

    if total_tokens == 0:
        return []

    # --------------------------------------------------------
    # Sliding-window step
    # --------------------------------------------------------

    step = (
        content_window_size -
        stride
    )

    chunks = []

    start = 0

    while start < total_tokens:

        end = min(
            start + content_window_size,
            total_tokens
        )

        content_ids = all_token_ids[
            start:end
        ]

        # ----------------------------------------------------
        # Manually construct:
        #
        # [CLS] content [SEP]
        # ----------------------------------------------------

        input_ids = (
            [cls_token_id] +
            content_ids +
            [sep_token_id]
        )

        attention_mask = [
            1
            for _ in input_ids
        ]

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        padding_length = (
            max_length -
            len(input_ids)
        )

        if padding_length > 0:

            input_ids.extend(
                [pad_token_id] *
                padding_length
            )

            attention_mask.extend(
                [0] *
                padding_length
            )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        input_ids = input_ids[
            :max_length
        ]

        attention_mask = attention_mask[
            :max_length
        ]

        chunks.append({

            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long
            ),

            "token_count": len(
                content_ids
            ),

            "start_token": start,

            "end_token": end
        })

        # ----------------------------------------------------
        # Stop when entire article is covered
        # ----------------------------------------------------

        if end >= total_tokens:
            break

        start += step

    return chunks


# ============================================================
# BATCH INFERENCE
# ============================================================

@torch.no_grad()
def predict_chunks(
    chunks: List[Dict]
) -> List[torch.Tensor]:

    probabilities = []

    for start in range(
        0,
        len(chunks),
        INFERENCE_BATCH_SIZE
    ):

        batch = chunks[
            start:
            start + INFERENCE_BATCH_SIZE
        ]

        input_ids = torch.stack([
            item["input_ids"]
            for item in batch
        ]).to(DEVICE)

        attention_mask = torch.stack([
            item["attention_mask"]
            for item in batch
        ]).to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(
            outputs.logits,
            dim=-1
        )

        probabilities.extend(
            probs.detach().cpu()
        )

    return probabilities


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_news(
    text: str
) -> Dict:
    """
    TruthLens V2 prediction.

    Returns:

        prediction
        fake_probability
        real_probability
        confidence
        chunks_analyzed
        coverage
        truncated
    """

    text = clean_text(
        text
    )

    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not text:

        return {
            "prediction": "UNKNOWN",
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "confidence": 0.0,
            "chunks_analyzed": 0,
            "coverage": 0.0,
            "truncated": False
        }

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    chunks = create_chunks(
        text
    )

    if not chunks:

        return {
            "prediction": "UNKNOWN",
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "confidence": 0.0,
            "chunks_analyzed": 0,
            "coverage": 0.0,
            "truncated": False
        }

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    chunk_probabilities = predict_chunks(
        chunks
    )

    # --------------------------------------------------------
    # Weighted aggregation
    # --------------------------------------------------------

    weighted_sum = torch.zeros(
        2,
        dtype=torch.float32
    )

    total_weight = 0.0

    for chunk, probabilities in zip(
        chunks,
        chunk_probabilities
    ):

        weight = max(
            float(
                chunk["token_count"]
            ),
            1.0
        )

        weighted_sum += (
            probabilities *
            weight
        )

        total_weight += weight

    final_probabilities = (
        weighted_sum /
        total_weight
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    fake_probability = float(
        final_probabilities[0].item()
    )

    real_probability = float(
        final_probabilities[1].item()
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_id = int(
        torch.argmax(
            final_probabilities
        ).item()
    )

    prediction = LABEL_MAP.get(
        predicted_id,
        "FAKE"
        if predicted_id == 0
        else "REAL"
    )

    confidence = float(
        final_probabilities[
            predicted_id
        ].item()
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    # Windows are explicitly generated until the final token.
    # Therefore every tokenized part of the article is covered.

    coverage = 100.0

    # More than one window means the original article was
    # longer than one BERT context window.

    truncated = (
        len(chunks) > 1
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "prediction": prediction,

        "fake_probability": round(
            fake_probability,
            6
        ),

        "real_probability": round(
            real_probability,
            6
        ),

        "confidence": round(
            confidence,
            6
        ),

        "chunks_analyzed": len(
            chunks
        ),

        "coverage": coverage,

        "truncated": truncated
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nRunning local V2 test...\n"
    )

    test_text = """
    The government announced a new policy today after several
    months of consultation with industry representatives.
    Officials said the policy would be implemented gradually
    and that additional details would be released later this week.
    """

    result = predict_news(
        test_text
    )

    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    for key, value in result.items():

        print(
            f"{key:20}: {value}"
        )

    print("=" * 70)