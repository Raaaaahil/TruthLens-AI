import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "truthlens-bert-v2"
)

MAX_LENGTH = 256
STRIDE = 64
TOP_K = 15


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("[XAI] Loading TruthLens BERT V2...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    output_attentions=True
)

model.to(DEVICE)
model.eval()

print(f"[XAI] Model loaded on {DEVICE}")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# CREATE SLIDING WINDOWS
# ============================================================

def create_windows(text):

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False
    )

    token_ids = encoded["input_ids"]

    if not token_ids:
        return []

    windows = []

    start = 0

    content_length = MAX_LENGTH - 2

    while start < len(token_ids):

        content_ids = token_ids[
            start:start + content_length
        ]

        if not content_ids:
            break

        input_ids = (
            [tokenizer.cls_token_id]
            + content_ids
            + [tokenizer.sep_token_id]
        )

        attention_mask = [
            1
        ] * len(input_ids)

        padding_length = (
            MAX_LENGTH - len(input_ids)
        )

        if padding_length > 0:

            input_ids += [
                tokenizer.pad_token_id
            ] * padding_length

            attention_mask += [
                0
            ] * padding_length

        windows.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_count": len(content_ids)
        })

        if start + content_length >= len(token_ids):
            break

        start += STRIDE

    return windows


# ============================================================
# TOKEN IMPORTANCE
#
# IMPORTANT:
# This performs ONE XAI model pass per window.
#
# Sentence analysis below REUSES these results.
# It does NOT call the model again.
# ============================================================

def calculate_token_importance(text):

    windows = create_windows(text)

    if not windows:
        return [], {}

    token_scores = {}

    for window in windows:

        input_ids = torch.tensor(
            [window["input_ids"]],
            dtype=torch.long,
            device=DEVICE
        )

        attention_mask = torch.tensor(
            [window["attention_mask"]],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

        attentions = output.attentions

        if not attentions:
            continue

        # Last transformer layer
        last_attention = attentions[-1]

        # [batch, heads, sequence, sequence]
        cls_attention = last_attention[
            0,
            :,
            0,
            :
        ]

        # Average attention across heads
        importance = cls_attention.mean(
            dim=0
        )

        ids = window["input_ids"]

        for index, token_id in enumerate(ids):

            if index == 0:
                continue

            if token_id in (
                tokenizer.pad_token_id,
                tokenizer.sep_token_id,
                tokenizer.cls_token_id
            ):
                continue

            if attention_mask[0, index].item() == 0:
                continue

            token = tokenizer.convert_ids_to_tokens(
                token_id
            )

            if not token:
                continue

            token = token.replace(
                "##",
                ""
            ).strip()

            if not token:
                continue

            # Ignore punctuation
            if not re.search(
                r"[A-Za-z0-9\u0900-\u097F\u0600-\u06FF]",
                token
            ):
                continue

            score = float(
                importance[index].item()
            )

            key = token.lower()

            if key not in token_scores:

                token_scores[key] = {
                    "token": token,
                    "importance": score,
                    "count": 1
                }

            else:

                token_scores[key]["importance"] += score

                token_scores[key]["count"] += 1

    # --------------------------------------------------------
    # Average repeated tokens
    # --------------------------------------------------------

    results = []

    for key, item in token_scores.items():

        score = (
            item["importance"]
            / max(item["count"], 1)
        )

        results.append({
            "token": item["token"],
            "importance": round(
                float(score),
                6
            )
        })

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results[:TOP_K], token_scores


# ============================================================
# SENTENCE IMPORTANCE
#
# IMPORTANT:
# NO MODEL CALL HERE.
#
# It reuses the token importance calculated above.
# ============================================================

def calculate_sentence_importance(
    text,
    token_scores
):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return []

    results = []

    for sentence in sentences:

        words = re.findall(
            r"[A-Za-z0-9\u0900-\u097F\u0600-\u06FF]+",
            sentence
        )

        scores = []

        for word in words:

            key = word.lower()

            if key in token_scores:

                item = token_scores[key]

                score = (
                    item["importance"]
                    / max(item["count"], 1)
                )

                scores.append(score)

        if scores:

            sentence_score = (
                sum(scores)
                / len(scores)
            )

        else:

            sentence_score = 0.0

        results.append({

            "sentence": sentence,

            "importance": round(
                float(sentence_score),
                6
            )
        })

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results[:5]


# ============================================================
# EXPLAIN NEWS
# ============================================================

def explain_news(text):

    text = clean_text(text)

    if not text:

        return {
            "important_tokens": [],
            "important_sentences": [],
            "explanation_method": (
                "BERT Transformer attention analysis"
            ),
            "model": "truthlens-bert-v2"
        }

    try:

        # ====================================================
        # ONE XAI CALCULATION
        # ====================================================

        important_tokens, token_scores = (
            calculate_token_importance(text)
        )

        # ====================================================
        # REUSE SAME TOKEN SCORES
        #
        # No second model inference.
        # ====================================================

        important_sentences = (
            calculate_sentence_importance(
                text,
                token_scores
            )
        )

        return {

            # Frontend uses this
            "important_tokens": important_tokens,

            # Additional XAI information
            "important_sentences": important_sentences,

            "explanation_method": (
                "BERT Transformer attention analysis"
            ),

            "model": "truthlens-bert-v2",

            "description": (
                "Important tokens are identified using "
                "attention from the final BERT transformer layer."
            )
        }

    except Exception as e:

        print(
            f"[XAI ERROR] {e}"
        )

        return {

            "important_tokens": [],

            "important_sentences": [],

            "explanation_method": (
                "BERT Transformer attention analysis"
            ),

            "model": "truthlens-bert-v2",

            "error": str(e)
        }