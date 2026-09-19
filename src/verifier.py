import torch

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"

MAX_RETRIEVAL_WORKERS = 3

NLI_BATCH_SIZE = 4

MAX_EVIDENCE_CHARS = 3000

MIN_EVIDENCE_CHARS = 100

MAX_EVIDENCE_RESULTS = 5


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def _load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# LABEL MAPPING
# ============================================================

@lru_cache(maxsize=1)
def _get_label_mapping_cached():

    _, model, _ = _load_model()

    id2label = model.config.id2label

    mapping = {}

    for index, label in id2label.items():

        normalized = str(label).lower()

        if "contradiction" in normalized:
            mapping["contradiction"] = int(index)

        elif "entailment" in normalized:
            mapping["entailment"] = int(index)

        elif "neutral" in normalized:
            mapping["neutral"] = int(index)

    return mapping


# ============================================================
# SINGLE NLI
# ============================================================

def _nli(
    evidence: str,
    claim: str
):

    tokenizer, model, device = _load_model()

    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        logits = model(
            **inputs
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0].detach().cpu().tolist()

    mapping = _get_label_mapping_cached()

    contradiction_index = mapping.get(
        "contradiction",
        0
    )

    entailment_index = mapping.get(
        "entailment",
        1
    )

    neutral_index = mapping.get(
        "neutral",
        2
    )

    contradiction = float(
        probabilities[contradiction_index]
    )

    entailment = float(
        probabilities[entailment_index]
    )

    neutral = float(
        probabilities[neutral_index]
    )

    scores = {
        "CONTRADICTION": contradiction,
        "ENTAILMENT": entailment,
        "NEUTRAL": neutral,
    }

    label = max(
        scores,
        key=scores.get
    )

    return {
        "label": label,
        "contradiction": contradiction,
        "entailment": entailment,
        "neutral": neutral,
    }


# ============================================================
# BATCH NLI
# ============================================================

def _nli_batch(pairs):

    if not pairs:
        return []

    tokenizer, model, device = _load_model()

    results = []

    mapping = _get_label_mapping_cached()

    contradiction_index = mapping.get(
        "contradiction",
        0
    )

    entailment_index = mapping.get(
        "entailment",
        1
    )

    neutral_index = mapping.get(
        "neutral",
        2
    )

    for start in range(
        0,
        len(pairs),
        NLI_BATCH_SIZE
    ):

        batch_pairs = pairs[
            start:start + NLI_BATCH_SIZE
        ]

        evidence_texts = [
            item[0]
            for item in batch_pairs
        ]

        claims = [
            item[1]
            for item in batch_pairs
        ]

        try:

            inputs = tokenizer(
                evidence_texts,
                claims,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=384,
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                logits = model(
                    **inputs
                ).logits

                probabilities = torch.softmax(
                    logits,
                    dim=-1
                ).detach().cpu()

            for row in probabilities:

                values = row.tolist()

                contradiction = float(
                    values[
                        contradiction_index
                    ]
                )

                entailment = float(
                    values[
                        entailment_index
                    ]
                )

                neutral = float(
                    values[
                        neutral_index
                    ]
                )

                scores = {
                    "CONTRADICTION":
                        contradiction,

                    "ENTAILMENT":
                        entailment,

                    "NEUTRAL":
                        neutral,
                }

                label = max(
                    scores,
                    key=scores.get
                )

                results.append(
                    {
                        "label": label,
                        "contradiction":
                            contradiction,
                        "entailment":
                            entailment,
                        "neutral":
                            neutral,
                    }
                )

        except Exception:

            for evidence, claim in batch_pairs:

                try:

                    results.append(
                        _nli(
                            evidence,
                            claim
                        )
                    )

                except Exception:

                    results.append(
                        {
                            "label": "NEUTRAL",
                            "contradiction": 0.0,
                            "entailment": 0.0,
                            "neutral": 1.0,
                        }
                    )

    return results


# ============================================================
# CLAIM VERIFICATION
# ============================================================

def verify_claim(
    claim,
    evidence_items
):

    valid_items = []

    for item in evidence_items:

        # New evidence_retriever.py uses "text".
        # Keep compatibility with older evidence formats too.
        evidence_text = (
            item.get(
                "text",
                ""
            )
            or item.get(
                "evidence_text",
                ""
            )
            or item.get(
                "snippet",
                ""
            )
            or ""
        ).strip()

        if len(evidence_text) < MIN_EVIDENCE_CHARS:
            continue

        evidence_text = (
            evidence_text[
                :MAX_EVIDENCE_CHARS
            ]
        )

        valid_items.append(
            (
                item,
                evidence_text
            )
        )

    if not valid_items:

        return {
            "assessment":
                "INSUFFICIENT_EVIDENCE",

            "support_score": 0.0,

            "refute_score": 0.0,

            "evidence": [],
        }

    pairs = [
        (
            evidence_text,
            claim
        )
        for _, evidence_text
        in valid_items
    ]

    nli_results = _nli_batch(
        pairs
    )

    checks = []

    for (
        item,
        evidence_text
    ), nli_result in zip(
        valid_items,
        nli_results
    ):

        checks.append(
            {
                "source": item.get(
                    "source",
                    item.get(
                        "publisher",
                        item.get(
                            "title",
                            ""
                        )
                    )
                ),

                "url": item.get(
                    "url",
                    ""
                ),

                "domain": item.get(
                    "domain",
                    ""
                ),

                "source_hint": item.get(
                    "source_hint",
                    "general-source"
                ),

                "published": item.get(
                    "published",
                    ""
                ),

                "relevance_score": item.get(
                    "relevance_score",
                    0.0
                ),

                "evidence_type": item.get(
                    "evidence_type",
                    "ARTICLE"
                ),

                "nli": nli_result,

                "evidence_excerpt":
                    evidence_text[:700],
            }
        )

    if not checks:

        return {
            "assessment":
                "INSUFFICIENT_EVIDENCE",

            "support_score": 0.0,

            "refute_score": 0.0,

            "evidence": [],
        }

    # ========================================================
    # SCORING
    # ========================================================

    support_scores = [
        item["nli"]["entailment"]
        for item in checks
    ]

    refute_scores = [
        item["nli"]["contradiction"]
        for item in checks
    ]

    strong_support = sum(
        score >= 0.70
        for score in support_scores
    )

    strong_refute = sum(
        score >= 0.70
        for score in refute_scores
    )

    max_support = max(
        support_scores
    )

    max_refute = max(
        refute_scores
    )

    support_domains = {
        item["domain"]
        for item in checks
        if (
            item["nli"]["entailment"]
            >= 0.70
        )
        and item["domain"]
    }

    refute_domains = {
        item["domain"]
        for item in checks
        if (
            item["nli"]["contradiction"]
            >= 0.70
        )
        and item["domain"]
    }

    # ========================================================
    # ASSESSMENT
    # ========================================================

    if (
        len(support_domains) >= 2
        and len(support_domains)
        > len(refute_domains)
    ):

        assessment = "SUPPORTED"

    elif (
        len(refute_domains) >= 2
        and len(refute_domains)
        > len(support_domains)
    ):

        assessment = "REFUTED"

    elif (
        max_support >= 0.85
        and max_support
        > max_refute + 0.15
    ):

        assessment = "SUPPORTED"

    elif (
        max_refute >= 0.85
        and max_refute
        > max_support + 0.15
    ):

        assessment = "REFUTED"

    else:

        assessment = "UNCERTAIN"

    return {
        "assessment":
            assessment,

        "support_score":
            round(
                max_support,
                4
            ),

        "refute_score":
            round(
                max_refute,
                4
            ),

        "supporting_sources":
            len(
                support_domains
            ),

        "refuting_sources":
            len(
                refute_domains
            ),

        "evidence":
            checks,
    }


# ============================================================
# PARALLEL CLAIM PROCESSING
# ============================================================

def _retrieve_and_verify_claim(
    item,
    title=""
):

    from src.evidence_retriever import (
        retrieve_evidence
    )

    claim = item["claim"]

    try:

        # ----------------------------------------------------
        # Evidence retrieval
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # The new evidence retriever does not require title.
        # Keep title in this function for backward compatibility
        # with callers, but do not pass it to retrieve_evidence().
        # ----------------------------------------------------

        evidence = retrieve_evidence(
            claim=claim,
            max_results=MAX_EVIDENCE_RESULTS
        )

        # ----------------------------------------------------
        # NLI verification
        # ----------------------------------------------------

        verification = verify_claim(
            claim,
            evidence
        )

        return {
            "claim": claim,
            "verification": verification,
        }

    except Exception as exc:

        return {
            "claim": claim,

            "verification": {
                "assessment":
                    "VERIFICATION_ERROR",

                "support_score": 0.0,

                "refute_score": 0.0,

                "evidence": [],

                "error": str(exc),
            },
        }


# ============================================================
# ARTICLE VERIFICATION
# ============================================================

def verify_article(
    text,
    title="",
    max_claims=3
):

    from src.claim_extractor import (
        extract_claims
    )

    # --------------------------------------------------------
    # Claim extraction
    # --------------------------------------------------------

    claims = extract_claims(
        text,
        max_claims=max_claims
    )

    if not claims:

        return {
            "overall_assessment":
                "INSUFFICIENT_EVIDENCE",

            "claims_checked": 0,

            "claims": [],

            "method":
                "Open-web evidence retrieval + NLI",

            "note": (
                "Evidence assessment is separate "
                "from the BERT fake/real classification. "
                "Retrieved web evidence can be incomplete, "
                "conflicting, or incorrect."
            ),
        }

    # ========================================================
    # PARALLEL EVIDENCE RETRIEVAL
    # ========================================================

    verified_claims = [
        None
        for _ in claims
    ]

    worker_count = min(
        MAX_RETRIEVAL_WORKERS,
        len(claims)
    )

    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:

        futures = {}

        for index, item in enumerate(
            claims
        ):

            future = executor.submit(
                _retrieve_and_verify_claim,
                item,
                title
            )

            futures[future] = index

        for future in as_completed(
            futures
        ):

            index = futures[future]

            try:

                verified_claims[index] = (
                    future.result()
                )

            except Exception as exc:

                claim = claims[index][
                    "claim"
                ]

                verified_claims[index] = {
                    "claim": claim,

                    "verification": {
                        "assessment":
                            "VERIFICATION_ERROR",

                        "support_score": 0.0,

                        "refute_score": 0.0,

                        "evidence": [],

                        "error": str(exc),
                    },
                }

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    verified_claims = [
        item
        for item in verified_claims
        if item is not None
    ]

    # ========================================================
    # OVERALL ASSESSMENT
    # ========================================================

    assessments = [
        item["verification"]["assessment"]

        for item in verified_claims

        if item["verification"]["assessment"]
        not in {
            "VERIFICATION_ERROR",
            "INSUFFICIENT_EVIDENCE",
        }
    ]

    if not assessments:

        overall = (
            "INSUFFICIENT_EVIDENCE"
        )

    elif (
        assessments.count(
            "REFUTED"
        )
        >
        assessments.count(
            "SUPPORTED"
        )
    ):

        overall = "REFUTED"

    elif (
        assessments.count(
            "SUPPORTED"
        )
        >
        assessments.count(
            "REFUTED"
        )
    ):

        overall = "SUPPORTED"

    else:

        overall = "UNCERTAIN"

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "overall_assessment":
            overall,

        "claims_checked":
            len(
                verified_claims
            ),

        "claims":
            verified_claims,

        "method":
            "Open-web evidence retrieval + NLI",

        "note": (
            "Evidence assessment is separate "
            "from the BERT fake/real classification. "
            "Retrieved web evidence can be incomplete, "
            "conflicting, or incorrect."
        ),
    }