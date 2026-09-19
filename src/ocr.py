"""
TruthLens AI - FINAL MULTILINGUAL OCR

Reliable OCR flow for English / Hindi / Urdu news images.

Key fixes:
- Never trust the OCR engine's language hint.
- Try script-specific OCR on multiple image variants.
- Use vertical column crops for newspaper pages.
- Prefer a strong script match over an unrelated OCR result.
- Preserve line/segment order.
- Filter tiny/low-confidence garbage.
"""

import re
from functools import lru_cache

import cv2
import easyocr


# ============================================================
# SCRIPT HELPERS
# ============================================================

def script_counts(text: str):
    dev = arabic = latin = 0

    for ch in text or "":
        if "\u0900" <= ch <= "\u097F":
            dev += 1
        elif (
            "\u0600" <= ch <= "\u06FF"
            or "\u0750" <= ch <= "\u077F"
            or "\u08A0" <= ch <= "\u08FF"
        ):
            arabic += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1

    return dev, arabic, latin


def script_ratios(text: str):
    dev, arabic, latin = script_counts(text)
    total = dev + arabic + latin

    if total == 0:
        return 0.0, 0.0, 0.0

    return (
        dev / total,
        arabic / total,
        latin / total,
    )


def clean_ocr_text(text: str):
    text = str(text or "")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _script_strength(text: str, code: str):
    dev, arabic, latin = script_ratios(text)

    if code == "hi":
        return dev

    if code == "ur":
        return arabic

    return latin


# ============================================================
# READERS
# ============================================================

@lru_cache(maxsize=3)
def get_reader(language: str):
    if language == "hi":
        languages = ["hi", "en"]

    elif language == "ur":
        languages = ["ur", "en"]

    else:
        languages = ["en"]

    try:
        import torch
        gpu = bool(torch.cuda.is_available())

    except Exception:
        gpu = False

    print(
        f"[OCR] Loading EasyOCR reader {languages}, GPU={gpu}"
    )

    return easyocr.Reader(
        languages,
        gpu=gpu,
        verbose=False,
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read the uploaded image."
        )

    h, w = image.shape[:2]

    # Keep enough resolution for newspaper body text.
    scale = (
        3.0
        if max(h, w) < 1800
        else 1.5
    )

    image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # CLAHE improves faded newspaper ink
    # without blurring strokes.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_gray = clahe.apply(gray)

    # High-contrast variant for printed pages.
    threshold = cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    return (
        image,
        enhanced_gray,
        threshold,
    )


# ============================================================
# OCR
# ============================================================

def run_reader(
    reader,
    image,
    min_confidence=0.25,
):
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.50,
        low_text=0.20,
        link_threshold=0.20,
        mag_ratio=1.0,
        canvas_size=3000,
    )

    segments = []

    for bbox, raw_text, confidence in results:

        text = clean_ocr_text(raw_text)
        confidence = float(confidence)

        if not text:
            continue

        if confidence < min_confidence:
            continue

        xs = [
            float(p[0])
            for p in bbox
        ]

        ys = [
            float(p[1])
            for p in bbox
        ]

        segments.append(
            {
                "text": text,
                "confidence": round(
                    confidence,
                    4,
                ),
                "x": int(min(xs)),
                "y": int(min(ys)),
            }
        )

    # Reading order:
    # top-to-bottom, then left-to-right.
    segments.sort(
        key=lambda s: (
            s["y"],
            s["x"],
        )
    )

    return segments


def segments_to_text(segments):
    return "\n".join(
        s["text"]
        for s in segments
        if s.get("text")
    ).strip()


# ============================================================
# NEWSPAPER COLUMN CROPPING
# ============================================================

def make_regions(image):
    """
    Return full-page + vertical regions.

    Newspaper OCR often fails because several narrow
    columns are processed together.

    Column regions give EasyOCR larger characters
    and cleaner reading.
    """

    h, w = image.shape[:2]

    regions = [
        ("full", image)
    ]

    if w >= 900:

        margin = max(
            10,
            int(w * 0.015)
        )

        usable_left = margin
        usable_right = w - margin

        column_width = (
            usable_right - usable_left
        ) / 3.0

        for i in range(3):

            left = int(
                usable_left
                + i * column_width
            )

            right = int(
                usable_left
                + (i + 1) * column_width
            )

            # Small overlap prevents
            # clipping at boundaries.
            overlap = int(
                column_width * 0.04
            )

            left = max(
                0,
                left - overlap
            )

            right = min(
                w,
                right + overlap
            )

            regions.append(
                (
                    f"column_{i + 1}",
                    image[:, left:right],
                )
            )

    return regions


def deduplicate_segments(segments):
    """
    Remove repeated OCR fragments caused
    by overlapping newspaper column crops.
    """

    output = []
    seen = set()

    for segment in segments:

        text = clean_ocr_text(
            segment["text"]
        )

        key = re.sub(
            r"\W+",
            "",
            text,
            flags=re.UNICODE,
        ).lower()

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        segment["text"] = text

        output.append(segment)

    return output


def collect_script_ocr(
    reader,
    image,
    code,
):
    all_segments = []

    for (
        region_name,
        region,
    ) in make_regions(image):

        segments = run_reader(
            reader,
            region,
            min_confidence=0.25,
        )

        for segment in segments:
            segment["_region"] = region_name

        all_segments.extend(
            segments
        )

    all_segments = deduplicate_segments(
        all_segments
    )

    # Remove obvious one/two-character noise.
    filtered = []

    for segment in all_segments:

        text = segment["text"]

        dev, arabic, latin = (
            script_counts(text)
        )

        if len(text) <= 2:

            if code == "hi" and dev == 0:
                continue

            if code == "ur" and arabic == 0:
                continue

            if code == "en" and latin == 0:
                continue

        filtered.append(segment)

    filtered.sort(
        key=lambda s: (
            s.get("y", 0),
            s.get("x", 0),
        )
    )

    return filtered


# ============================================================
# CANDIDATE SCORING
# ============================================================

def candidate_score(
    code,
    text,
    segments,
):
    if not text or not segments:
        return -1.0

    strength = _script_strength(
        text,
        code,
    )

    avg_conf = (
        sum(
            s["confidence"]
            for s in segments
        )
        / len(segments)
    )

    useful_chars = len(
        re.sub(
            r"[^\w\u0900-\u097F\u0600-\u06FF]",
            "",
            text,
            flags=re.UNICODE,
        )
    )

    # Strong script presence dominates.
    return (
        strength * 100.0
        + avg_conf * 25.0
        + min(
            useful_chars,
            1500
        ) * 0.02
        + min(
            len(segments),
            80
        ) * 0.10
    )


def language_confidence(
    code,
    text,
    segments,
):
    if not segments:
        return 0.0

    strength = _script_strength(
        text,
        code,
    )

    avg_conf = (
        sum(
            s["confidence"]
            for s in segments
        )
        / len(segments)
    )

    return round(
        min(
            0.99,
            max(
                0.50,
                strength * 0.60
                + avg_conf * 0.40,
            ),
        ),
        3,
    )


# ============================================================
# STANDARD ENGLISH OCR
# ============================================================

def extract_text_from_image(
    image_path: str,
):
    color, gray, _ = preprocess_image(
        image_path
    )

    reader = get_reader("en")

    segments = run_reader(
        reader,
        color,
        min_confidence=0.30,
    )

    return {
        "text": segments_to_text(
            segments
        ),
        "segments": segments,
        "segment_count": len(
            segments
        ),
    }


# ============================================================
# MULTILINGUAL OCR
# ============================================================

def extract_text_from_image_multilingual(
    image_path: str,
):
    """
    Robust English / Hindi / Urdu OCR.

    Urdu is attempted before English fallback,
    and Hindi/Urdu selection is based on actual
    Unicode script produced by the language-specific
    recognizer.

    Newspaper pages use column OCR to improve
    small-text recognition.
    """

    color, gray, threshold = (
        preprocess_image(
            image_path
        )
    )

    candidates = []

    # ========================================================
    # URDU
    # ========================================================

    ur_reader = get_reader("ur")

    ur_segments = collect_script_ocr(
        ur_reader,
        color,
        "ur",
    )

    ur_text = segments_to_text(
        ur_segments
    )

    ur_score = candidate_score(
        "ur",
        ur_text,
        ur_segments,
    )

    candidates.append(
        (
            "ur",
            "Urdu",
            ur_text,
            ur_segments,
            ur_score,
        )
    )

    # Threshold fallback if Urdu
    # script is weak.
    if _script_strength(
        ur_text,
        "ur",
    ) < 0.15:

        ur_threshold_segments = (
            collect_script_ocr(
                ur_reader,
                threshold,
                "ur",
            )
        )

        ur_threshold_text = (
            segments_to_text(
                ur_threshold_segments
            )
        )

        ur_threshold_score = (
            candidate_score(
                "ur",
                ur_threshold_text,
                ur_threshold_segments,
            )
        )

        if (
            ur_threshold_score
            > ur_score
        ):

            candidates.append(
                (
                    "ur",
                    "Urdu",
                    ur_threshold_text,
                    ur_threshold_segments,
                    ur_threshold_score,
                )
            )

    # ========================================================
    # HINDI
    # ========================================================

    hi_reader = get_reader("hi")

    hi_segments = collect_script_ocr(
        hi_reader,
        color,
        "hi",
    )

    hi_text = segments_to_text(
        hi_segments
    )

    hi_score = candidate_score(
        "hi",
        hi_text,
        hi_segments,
    )

    candidates.append(
        (
            "hi",
            "Hindi",
            hi_text,
            hi_segments,
            hi_score,
        )
    )

    # Threshold fallback if Hindi
    # script is weak.
    if _script_strength(
        hi_text,
        "hi",
    ) < 0.15:

        hi_threshold_segments = (
            collect_script_ocr(
                hi_reader,
                threshold,
                "hi",
            )
        )

        hi_threshold_text = (
            segments_to_text(
                hi_threshold_segments
            )
        )

        hi_threshold_score = (
            candidate_score(
                "hi",
                hi_threshold_text,
                hi_threshold_segments,
            )
        )

        if (
            hi_threshold_score
            > hi_score
        ):

            candidates.append(
                (
                    "hi",
                    "Hindi",
                    hi_threshold_text,
                    hi_threshold_segments,
                    hi_threshold_score,
                )
            )

    # ========================================================
    # ENGLISH
    # ========================================================

    en_reader = get_reader("en")

    en_segments = collect_script_ocr(
        en_reader,
        color,
        "en",
    )

    en_text = segments_to_text(
        en_segments
    )

    en_score = candidate_score(
        "en",
        en_text,
        en_segments,
    )

    candidates.append(
        (
            "en",
            "English",
            en_text,
            en_segments,
            en_score,
        )
    )

    # ========================================================
    # SELECT BEST LANGUAGE
    # ========================================================

    best = max(
        candidates,
        key=lambda item: item[4],
    )

    (
        code,
        name,
        text,
        segments,
        score,
    ) = best

    # Safety rule:
    # If Indic recognizer produced almost
    # no matching script, don't label it
    # as Hindi/Urdu.
    strength = _script_strength(
        text,
        code,
    )

    if (
        code in ("hi", "ur")
        and strength < 0.10
    ):

        english_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] == "en"
        ]

        if english_candidates:

            best = max(
                english_candidates,
                key=lambda item: item[4],
            )

            (
                code,
                name,
                text,
                segments,
                score,
            ) = best

    return {
        "text": text,
        "segments": segments,
        "segment_count": len(
            segments
        ),
        "ocr_language_hint": code,
        "ocr_language_name": name,
        "ocr_language_confidence": (
            language_confidence(
                code,
                text,
                segments,
            )
        ),
    }