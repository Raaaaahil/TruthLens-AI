import re


def extract_claims(text: str, max_claims: int = 5):
    text = re.sub(r"\s+", " ", (text or "")).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    candidates = []

    factual_signals = [
        "said", "says", "reported", "announced", "according",
        "confirmed", "warned", "found", "showed", "revealed",
        "will", "has", "have", "had", "was", "were", "is", "are",
        "increased", "decreased", "targeted", "attacked",
        "government", "president", "official", "study", "data",
        "research", "scientists", "scientist", "company",
        "country", "minister", "police", "court", "officials",
        "experts", "according to"
    ]

    noise_patterns = [
        "subscribe",
        "cookie",
        "sign up",
        "advertisement",
        "follow us",
        "all rights reserved",
        "privacy policy",
        "terms of service"
    ]

    for sentence in sentences:
        sentence = sentence.strip(" \t\r\n-•")

        words = sentence.split()

        # Ignore very short or extremely long sentences
        if len(words) < 8 or len(words) > 80:
            continue

        lower = sentence.lower()

        # Remove webpage noise
        if any(noise in lower for noise in noise_patterns):
            continue

        signal_count = sum(
            1 for signal in factual_signals
            if signal in lower
        )

        # Additional indicators of factual claims
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", sentence))
        has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", sentence))
        has_percentage = "%" in sentence

        # Every sufficiently long clean sentence can be a claim.
        # Factual signals/numbers simply increase its priority.
        claim_score = signal_count

        if has_number:
            claim_score += 1

        if has_year:
            claim_score += 1

        if has_percentage:
            claim_score += 1

        candidates.append({
            "claim": sentence,
            "signal_count": claim_score,
            "word_count": len(words)
        })

    # Prefer sentences that look more factual,
    # but don't completely discard ordinary factual statements.
    candidates.sort(
        key=lambda item: (
            item["signal_count"],
            min(item["word_count"], 40)
        ),
        reverse=True
    )

    selected = []
    seen = set()

    for item in candidates:
        normalized = re.sub(
            r"[^a-z0-9 ]",
            "",
            item["claim"].lower()
        )
        normalized = re.sub(
            r"\s+",
            " ",
            normalized
        ).strip()

        if normalized in seen:
            continue

        seen.add(normalized)
        selected.append(item)

        if len(selected) >= max_claims:
            break

    return selected


if __name__ == "__main__":
    test = (
        "The Earth revolves around the Sun and completes "
        "one orbit approximately every 365 days."
    )

    print(extract_claims(test, max_claims=1))