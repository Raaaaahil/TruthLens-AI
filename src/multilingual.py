"""
TruthLens AI - Multilingual Text Processing

Supported languages:
- English (en)
- Hindi (hi)
- Urdu (ur)

Design:
1. Detect language primarily from Unicode script so OCR text is handled reliably.
2. Translate Hindi/Urdu to English only when required.
3. Keep English unchanged.
4. Return a consistent structure for both text and image/OCR pipelines.
"""

from functools import lru_cache
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# SCRIPT DETECTION
# ============================================================

def _script_counts(text: str):
    devanagari = 0
    arabic = 0
    latin = 0

    for ch in text or "":
        if "\u0900" <= ch <= "\u097F":
            devanagari += 1
        elif (
            "\u0600" <= ch <= "\u06FF"
            or "\u0750" <= ch <= "\u077F"
            or "\u08A0" <= ch <= "\u08FF"
        ):
            arabic += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1

    return devanagari, arabic, latin


def _ratios(text: str):
    devanagari, arabic, latin = _script_counts(text)
    total = devanagari + arabic + latin

    if total == 0:
        return 0.0, 0.0, 0.0

    return (
        devanagari / total,
        arabic / total,
        latin / total,
    )


def detect_language(text: str):
    """
    Robust language detection for English/Hindi/Urdu.

    Unicode script is intentionally preferred over a word-list heuristic.
    This is especially important for OCR output, where punctuation,
    numbers and OCR noise can otherwise cause an Indic-language result
    to be incorrectly reported as Unknown/English.
    """
    text = (text or "").strip()

    if not text:
        return {
            "code": "unknown",
            "name": "Unknown",
            "confidence": 0.0,
        }

    devanagari_ratio, arabic_ratio, latin_ratio = _ratios(text)

    # Hindi / Devanagari
    if devanagari_ratio >= 0.10 and devanagari_ratio >= arabic_ratio:
        confidence = min(0.99, max(0.70, 0.65 + devanagari_ratio * 0.34))
        return {
            "code": "hi",
            "name": "Hindi",
            "confidence": round(confidence, 3),
        }

    # Urdu / Arabic-derived script
    if arabic_ratio >= 0.10 and arabic_ratio >= devanagari_ratio:
        confidence = min(0.99, max(0.70, 0.65 + arabic_ratio * 0.34))
        return {
            "code": "ur",
            "name": "Urdu",
            "confidence": round(confidence, 3),
        }

    # English / Latin
    if latin_ratio >= 0.25:
        confidence = min(0.99, max(0.70, 0.65 + latin_ratio * 0.34))
        return {
            "code": "en",
            "name": "English",
            "confidence": round(confidence, 3),
        }

    return {
        "code": "unknown",
        "name": "Unknown",
        "confidence": 0.0,
    }


# ============================================================
# TRANSLATION
# ============================================================

MODEL_NAMES = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "ur": "Helsinki-NLP/opus-mt-ur-en",
}


@lru_cache(maxsize=2)
def _load_translation_model(language_code: str):
    model_name = MODEL_NAMES[language_code]

    print(f"[MULTILINGUAL] Loading translation model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def _translate_to_english(text: str, language_code: str):
    if not text.strip():
        return ""

    tokenizer, model, device = _load_translation_model(language_code)

    # Keep chunks reasonably small so long OCR/text inputs do not overflow
    # the translation model context window.
    sentences = re.split(r"(?<=[.!?।؟])\s+|\n+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [text.strip()]

    translated_parts = []

    with torch.inference_mode():
        for sentence in sentences:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=2,
            )

            translated = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0].strip()

            if translated:
                translated_parts.append(translated)

    return " ".join(translated_parts).strip()


# ============================================================
# PUBLIC PREPARATION FUNCTION
# ============================================================

def prepare_multilingual_text(text: str):
    """
    Detect language and normalize content for the English BERT model.

    English:
        original text is returned unchanged.

    Hindi/Urdu:
        translated_text contains English translation.

    The returned keys are compatible with the TruthLens backend.
    """
    original_text = (text or "").strip()

    detected = detect_language(original_text)
    code = detected["code"]

    if code == "en":
        return {
            "detected_language": detected,
            "analysis_language": {
                "code": "en",
                "name": "English",
            },
            "translation_applied": False,
            "original_text": original_text,
            "translated_text": original_text,
        }

    if code in ("hi", "ur"):
        translated = _translate_to_english(
            original_text,
            code,
        )

        # If translation unexpectedly returns empty, preserve the OCR/text
        # rather than crashing the complete analysis pipeline.
        analysis_text = translated or original_text

        return {
            "detected_language": detected,
            "analysis_language": {
                "code": "en",
                "name": "English",
            },
            "translation_applied": bool(translated),
            "original_text": original_text,
            "translated_text": analysis_text,
        }

    # Unknown: don't claim a language. Let the existing BERT pipeline
    # operate on the original content when possible.
    return {
        "detected_language": detected,
        "analysis_language": {
            "code": "en",
            "name": "English",
        },
        "translation_applied": False,
        "original_text": original_text,
        "translated_text": original_text,
    }