"""
TruthLens AI - Evidence Retriever
=================================

Google News RSS based evidence retrieval.

GDELT is intentionally NOT used because it is unreachable in the
current development environment.

The retriever:
    1. Searches Google News RSS.
    2. Resolves Google News redirect URLs.
    3. Attempts publisher article extraction.
    4. Falls back to RSS title/description evidence.
    5. Scores relevance.
    6. Preserves domain diversity.
    7. Returns evidence compatible with verifier.py.

Important:
    Retrieval does NOT decide whether a claim is true or false.
    NLI verification remains responsible for that.
"""

from __future__ import annotations

import concurrent.futures
import html
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

import trafilatura


# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search"
)

REQUEST_TIMEOUT = 5

REDIRECT_TIMEOUT = 4

MAX_SEARCH_RESULTS = 10

MAX_RESULTS_PER_CLAIM = 5

ARTICLE_WORKERS = 5

MIN_RELEVANCE_SCORE = 0.05

MAX_ARTICLE_CHARS = 5000

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36"
)

logger = logging.getLogger(__name__)


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "is",
    "been",
    "being",
    "will",
    "would",
    "could",
    "should",
    "about",
    "after",
    "before",
    "into",
    "over",
    "under",
    "between",
    "through",
    "during",
    "their",
    "there",
    "they",
    "them",
    "than",
    "then",
    "also",
    "more",
    "most",
    "some",
    "such",
    "only",
    "very",
    "said",
    "says",
    "according",
    "reported",
    "reports",
    "officials",
    "official",
    "today",
    "yesterday",
    "news",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = html.unescape(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text: str) -> list[str]:

    if not text:
        return []

    words = re.findall(
        r"[A-Za-z][A-Za-z0-9'-]{2,}",
        text.lower()
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
    ]


def build_keywords(
    claim: str,
    max_keywords: int = 10
) -> list[str]:

    words = tokenize(claim)

    seen = set()
    unique = []

    for word in words:

        if word in seen:
            continue

        seen.add(word)
        unique.append(word)

    return unique[:max_keywords]


def build_search_query(
    claim: str
) -> str:

    keywords = build_keywords(
        claim,
        max_keywords=8
    )

    if not keywords:
        return claim[:150]

    return " ".join(keywords)


# ============================================================
# HTTP
# ============================================================

def fetch_url(
    url: str,
    timeout: int = REQUEST_TIMEOUT
) -> str | None:

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": (
                    "application/rss+xml, "
                    "application/xml, "
                    "text/xml, "
                    "text/html;q=0.9, "
                    "*/*;q=0.8"
                ),
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:

            content = response.read()

            if not content:
                return None

            return content.decode(
                "utf-8",
                errors="replace"
            )

    except Exception as exc:

        logger.debug(
            "HTTP request failed: %s",
            exc
        )

        return None


# ============================================================
# REDIRECT RESOLUTION
# ============================================================

def resolve_redirect(
    url: str
) -> str:

    """
    Resolve Google News redirect URLs to the publisher URL.

    If resolution fails, return the original URL.
    """

    if not url:
        return ""

    if "news.google.com" not in url:
        return url

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=REDIRECT_TIMEOUT
        ) as response:

            final_url = response.geturl()

            if final_url:
                return final_url

    except Exception as exc:

        logger.debug(
            "Redirect resolution failed: %s",
            exc
        )

    return url


# ============================================================
# DOMAIN
# ============================================================

def extract_domain(
    url: str
) -> str:

    try:

        parsed = urllib.parse.urlparse(
            url
        )

        domain = parsed.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:

        return ""


def is_google_news_url(
    url: str
) -> bool:

    domain = extract_domain(url)

    return (
        domain == "news.google.com"
        or domain.endswith(".google.com")
    )


# ============================================================
# GOOGLE NEWS RSS
# ============================================================

def build_google_news_url(
    query: str,
    days: int = 7
) -> str:

    query_with_window = (
        f"{query} when:{days}d"
    )

    params = urllib.parse.urlencode(
        {
            "q": query_with_window,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
    )

    return (
        f"{GOOGLE_NEWS_RSS}?{params}"
    )


def parse_rss(
    xml_text: str
) -> list[dict[str, Any]]:

    if not xml_text:
        return []

    try:

        root = ET.fromstring(
            xml_text
        )

    except Exception as exc:

        logger.debug(
            "RSS parsing failed: %s",
            exc
        )

        return []

    results = []

    for item in root.findall(
        ".//item"
    ):

        title = item.findtext(
            "title"
        ) or ""

        link = item.findtext(
            "link"
        ) or ""

        description = item.findtext(
            "description"
        ) or ""

        pub_date = item.findtext(
            "pubDate"
        ) or ""

        source_name = ""

        source_node = item.find(
            "source"
        )

        if source_node is not None:

            source_name = (
                source_node.text or ""
            ).strip()

        title = clean_text(
            title
        )

        description = clean_text(
            description
        )

        link = link.strip()

        source_name = clean_text(
            source_name
        )

        if not link or not title:
            continue

        results.append(
            {
                "title": title,
                "url": link,
                "description": description,
                "published": pub_date,
                "publisher": source_name,
            }
        )

    return results


def search_google_news(
    query: str,
    max_records: int = MAX_SEARCH_RESULTS
) -> list[dict[str, Any]]:

    url = build_google_news_url(
        query
    )

    started = time.perf_counter()

    xml_text = fetch_url(
        url,
        timeout=REQUEST_TIMEOUT
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if not xml_text:

        logger.info(
            "Google News search failed in %.2fs",
            elapsed
        )

        return []

    articles = parse_rss(
        xml_text
    )

    logger.info(
        "Google News returned %d results in %.2fs",
        len(articles),
        elapsed
    )

    return articles[:max_records]


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_article(
    item: dict[str, Any]
) -> dict[str, Any] | None:

    original_url = item.get(
        "url",
        ""
    )

    if not original_url:
        return None

    title = clean_text(
        item.get(
            "title",
            ""
        )
    )

    description = clean_text(
        item.get(
            "description",
            ""
        )
    )

    publisher = clean_text(
        item.get(
            "publisher",
            ""
        )
    )

    # --------------------------------------------------------
    # Resolve Google News redirect.
    # --------------------------------------------------------

    resolved_url = resolve_redirect(
        original_url
    )

    # --------------------------------------------------------
    # Try article extraction.
    # --------------------------------------------------------

    article_text = ""

    if (
        resolved_url
        and not is_google_news_url(
            resolved_url
        )
    ):

        try:

            downloaded = (
                trafilatura.fetch_url(
                    resolved_url
                )
            )

            if downloaded:

                article_text = (
                    trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=False,
                        favor_precision=True,
                    )
                    or ""
                )

        except Exception as exc:

            logger.debug(
                "Trafilatura extraction failed: %s",
                exc
            )

    article_text = clean_text(
        article_text
    )

    # --------------------------------------------------------
    # Fallback evidence.
    #
    # Google News RSS frequently gives redirect URLs and little
    # usable body text. Build a meaningful evidence passage from
    # title + description instead of returning empty evidence.
    # --------------------------------------------------------

    if not article_text:

        fallback_parts = []

        if publisher:
            fallback_parts.append(
                f"Source: {publisher}."
            )

        if title:
            fallback_parts.append(
                f"Headline: {title}."
            )

        if description:
            fallback_parts.append(
                f"Reported information: {description}."
            )

        article_text = " ".join(
            fallback_parts
        )

    # --------------------------------------------------------
    # Last-resort evidence.
    # This guarantees verifier-compatible text.
    # --------------------------------------------------------

    if not article_text:

        article_text = (
            f"News report titled "
            f"'{title}' "
            f"from {publisher or 'an external source'}."
        )

    domain = extract_domain(
        resolved_url
    )

    if (
        not domain
        or is_google_news_url(
            resolved_url
        )
    ):

        domain = (
            publisher.lower()
            if publisher
            else extract_domain(
                original_url
            )
        )

    return {
        "title": title,

        "url": (
            resolved_url
            if resolved_url
            else original_url
        ),

        "domain": domain,

        "publisher": publisher,

        "published": item.get(
            "published",
            ""
        ),

        "text": article_text,

        "evidence_text": article_text,

        "snippet": article_text,

        "description": description,
    }


# ============================================================
# RELEVANCE
# ============================================================

def calculate_relevance(
    claim: str,
    article: dict[str, Any]
) -> float:

    claim_words = set(
        build_keywords(
            claim,
            max_keywords=15
        )
    )

    if not claim_words:
        return 0.0

    title_words = set(
        tokenize(
            article.get(
                "title",
                ""
            )
        )
    )

    text_words = set(
        tokenize(
            article.get(
                "text",
                ""
            )
        )
    )

    description_words = set(
        tokenize(
            article.get(
                "description",
                ""
            )
        )
    )

    body_overlap = (
        len(
            claim_words
            & text_words
        )
        / len(claim_words)
    )

    title_overlap = (
        len(
            claim_words
            & title_words
        )
        / len(claim_words)
    )

    description_overlap = (
        len(
            claim_words
            & description_words
        )
        / len(claim_words)
    )

    score = (
        body_overlap * 0.50
        + title_overlap * 0.35
        + description_overlap * 0.15
    )

    return round(
        min(
            1.0,
            score
        ),
        4
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_articles(
    articles: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    seen_urls = set()
    seen_titles = set()

    output = []

    for article in articles:

        url = article.get(
            "url",
            ""
        ).strip()

        title = article.get(
            "title",
            ""
        ).strip().lower()

        url_key = url.lower()

        if (
            url_key
            and url_key in seen_urls
        ):
            continue

        if (
            title
            and title in seen_titles
        ):
            continue

        if url_key:
            seen_urls.add(
                url_key
            )

        if title:
            seen_titles.add(
                title
            )

        output.append(
            article
        )

    return output


# ============================================================
# EVIDENCE FORMAT
# ============================================================

def article_to_evidence(
    article: dict[str, Any],
    relevance: float
) -> dict[str, Any]:

    text = clean_text(
        article.get(
            "text",
            ""
        )
    )

    if len(text) > MAX_ARTICLE_CHARS:

        text = (
            text[:MAX_ARTICLE_CHARS]
            + "..."
        )

    # A full article gets ARTICLE type.
    # A title/description fallback gets TITLE_SNIPPET.
    evidence_type = (
        "ARTICLE"
        if len(text) >= 500
        else "TITLE_SNIPPET"
    )

    return {
        "title": article.get(
            "title",
            ""
        ),

        "url": article.get(
            "url",
            ""
        ),

        "domain": article.get(
            "domain",
            ""
        ),

        "publisher": article.get(
            "publisher",
            ""
        ),

        "published": article.get(
            "published",
            ""
        ),

        "snippet": text,

        "text": text,

        "evidence_text": text,

        "relevance_score": relevance,

        "evidence_type": evidence_type,

        "source": article.get(
            "publisher",
            article.get(
                "domain",
                ""
            )
        ),
    }


# ============================================================
# MAIN RETRIEVAL
# ============================================================

def retrieve_evidence(
    claim: str,
    max_results: int = MAX_RESULTS_PER_CLAIM
) -> list[dict[str, Any]]:

    started = time.perf_counter()

    claim = clean_text(
        claim
    )

    if not claim:
        return []

    logger.info(
        "Evidence retrieval started: %s",
        claim[:200]
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    query = build_search_query(
        claim
    )

    search_results = search_google_news(
        query=query,
        max_records=MAX_SEARCH_RESULTS
    )

    if not search_results:

        logger.info(
            "No Google News results."
        )

        return []

    search_results = (
        deduplicate_articles(
            search_results
        )
    )

    # --------------------------------------------------------
    # Resolve and extract articles concurrently.
    # --------------------------------------------------------

    extracted = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=ARTICLE_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                extract_article,
                item
            )
            for item in search_results
        ]

        for future in concurrent.futures.as_completed(
            futures
        ):

            try:

                article = future.result()

                if article:
                    extracted.append(
                        article
                    )

            except Exception as exc:

                logger.debug(
                    "Article worker failed: %s",
                    exc
                )

    if not extracted:

        logger.info(
            "No usable evidence extracted."
        )

        return []

    # --------------------------------------------------------
    # Relevance scoring.
    # --------------------------------------------------------

    scored = []

    for article in extracted:

        score = calculate_relevance(
            claim,
            article
        )

        if score >= MIN_RELEVANCE_SCORE:

            scored.append(
                (
                    score,
                    article
                )
            )

    # --------------------------------------------------------
    # Safety fallback.
    # Never discard all evidence merely because relevance
    # threshold was too strict.
    # --------------------------------------------------------

    if not scored:

        scored = [
            (
                calculate_relevance(
                    claim,
                    article
                ),
                article
            )
            for article in extracted
        ]

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Domain diversity.
    # --------------------------------------------------------

    final_results = []

    used_domains = set()

    for score, article in scored:

        domain = (
            article.get(
                "domain",
                ""
            )
            or article.get(
                "publisher",
                ""
            )
        ).lower().strip()

        if (
            domain
            and domain in used_domains
        ):
            continue

        if domain:
            used_domains.add(
                domain
            )

        final_results.append(
            article_to_evidence(
                article,
                score
            )
        )

        if len(final_results) >= max_results:
            break

    # --------------------------------------------------------
    # Fill remaining slots if domain filtering removed too many.
    # --------------------------------------------------------

    if len(final_results) < max_results:

        existing_urls = {
            item.get(
                "url",
                ""
            )
            for item in final_results
        }

        for score, article in scored:

            url = article.get(
                "url",
                ""
            )

            if url in existing_urls:
                continue

            final_results.append(
                article_to_evidence(
                    article,
                    score
                )
            )

            if len(final_results) >= max_results:
                break

    elapsed = (
        time.perf_counter()
        - started
    )

    logger.info(
        "Evidence retrieval completed: "
        "%d results in %.2fs",
        len(final_results),
        elapsed
    )

    return final_results


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def search_evidence(
    claim: str,
    max_results: int = MAX_RESULTS_PER_CLAIM
) -> list[dict[str, Any]]:

    return retrieve_evidence(
        claim=claim,
        max_results=max_results
    )


def get_evidence(
    claim: str,
    max_results: int = MAX_RESULTS_PER_CLAIM
) -> list[dict[str, Any]]:

    return retrieve_evidence(
        claim=claim,
        max_results=max_results
    )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_claim = (
        "NASA Artemis II mission will conduct "
        "a lunar flyby."
    )

    print("=" * 70)
    print(
        "TruthLens AI Evidence Retriever Test"
    )
    print("=" * 70)

    start = time.perf_counter()

    results = retrieve_evidence(
        claim=test_claim,
        max_results=5
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()

    print(
        f"Time: {elapsed:.2f} sec"
    )

    print(
        f"Results: {len(results)}"
    )

    print()

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"[{index}] "
            f"{result.get('domain')} | "
            f"score="
            f"{result.get('relevance_score')} | "
            f"type="
            f"{result.get('evidence_type')}"
        )

        print(
            result.get(
                "title",
                ""
            )[:160]
        )

        print(
            result.get(
                "url",
                ""
            )[:200]
        )

        print(
            "Evidence chars:",
            len(
                result.get(
                    "evidence_text",
                    ""
                )
            )
        )

        print()