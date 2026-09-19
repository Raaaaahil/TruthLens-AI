import json
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import trafilatura


# ============================================================
# URL VALIDATION
# ============================================================

def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)

        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.netloc)
        )

    except Exception:
        return False


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # Normalize spaces but preserve paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_for_comparison(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip(
        " \t\n\r-–—:|"
    )

    return text


# ============================================================
# REMOVE DUPLICATE TITLE
# ============================================================

def remove_duplicate_title(
    title: str,
    text: str
) -> str:

    if not title or not text:
        return text.strip()

    clean_title = clean_text(title)
    clean_article = clean_text(text)

    normalized_title = normalize_for_comparison(
        clean_title
    )

    normalized_article = normalize_for_comparison(
        clean_article
    )

    if not normalized_title or not normalized_article:
        return clean_article

    if normalized_article.startswith(
        normalized_title
    ):

        remaining = normalized_article[
            len(normalized_title):
        ].strip()

        if remaining:
            return remaining

        return ""

    escaped_title = re.escape(
        clean_title
    )

    pattern = (
        rf"^\s*{escaped_title}"
        rf"\s*(?:[-–—:|]+\s*)?"
    )

    cleaned = re.sub(
        pattern,
        "",
        clean_article,
        count=1,
        flags=re.IGNORECASE
    )

    return cleaned.strip()


# ============================================================
# HTTP DOWNLOAD
# ============================================================

def download_page(url: str) -> str:
    """
    Download a webpage with a browser-like User-Agent.

    Trafilatura's built-in fetcher can fail on sites that reject
    its default request. This fallback uses urllib directly.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language": (
            "en-US,en;q=0.9"
        ),
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    request = Request(
        url,
        headers=headers,
        method="GET"
    )

    try:
        with urlopen(
            request,
            timeout=20
        ) as response:

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                )
                .lower()
            )

            if (
                "text/html" not in content_type
                and "application/xhtml" not in content_type
            ):
                raise ValueError(
                    "URL did not return an HTML webpage."
                )

            raw = response.read()

            charset = (
                response.headers.get_content_charset()
                or "utf-8"
            )

            return raw.decode(
                charset,
                errors="replace"
            )

    except Exception as e:
        raise ValueError(
            f"Could not retrieve webpage: {e}"
        )


# ============================================================
# JSON-LD FALLBACK
# ============================================================

def extract_jsonld_article(html: str):
    """
    Extract articleBody/headline from JSON-LD.

    Many news websites expose structured Article data even
    when normal article extraction fails.
    """

    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>'
        r'(.*?)'
        r'</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL
    )

    candidates = []

    for raw_script in scripts:

        raw_script = raw_script.strip()

        if not raw_script:
            continue

        try:
            data = json.loads(
                raw_script
            )

        except Exception:
            continue

        if isinstance(data, dict):

            candidates.append(data)

            graph = data.get(
                "@graph"
            )

            if isinstance(graph, list):
                candidates.extend(
                    item
                    for item in graph
                    if isinstance(item, dict)
                )

        elif isinstance(data, list):

            candidates.extend(
                item
                for item in data
                if isinstance(item, dict)
            )

    best_title = ""
    best_body = ""

    for item in candidates:

        item_type = item.get(
            "@type",
            ""
        )

        if isinstance(
            item_type,
            list
        ):
            item_type = " ".join(
                str(x)
                for x in item_type
            )

        item_type = str(
            item_type
        ).lower()

        headline = str(
            item.get(
                "headline",
                ""
            )
            or ""
        ).strip()

        article_body = str(
            item.get(
                "articleBody",
                ""
            )
            or ""
        ).strip()

        if headline and not best_title:
            best_title = clean_text(
                headline
            )

        if (
            article_body
            and len(article_body) > len(best_body)
        ):
            best_body = clean_text(
                article_body
            )

        # Prefer actual article/news structured data.
        if (
            article_body
            and (
                "article" in item_type
                or "newsarticle" in item_type
            )
        ):
            best_body = clean_text(
                article_body
            )

            if headline:
                best_title = clean_text(
                    headline
                )

    if best_body:
        return {
            "title": best_title,
            "text": best_body
        }

    return None


# ============================================================
# TITLE EXTRACTION
# ============================================================

def extract_title(
    downloaded: str,
    extracted_data: dict,
    text: str
) -> str:

    title = (
        extracted_data.get(
            "title"
        )
        or ""
    )

    if title.strip():
        return clean_text(
            title
        )

    # Trafilatura metadata.
    try:

        metadata = trafilatura.extract_metadata(
            downloaded
        )

        if metadata:

            metadata_title = (
                metadata.title
                or ""
            )

            if metadata_title.strip():
                return clean_text(
                    metadata_title
                )

    except Exception:
        pass

    # HTML <title> fallback.
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        downloaded,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:

        html_title = clean_text(
            re.sub(
                r"<[^>]+>",
                " ",
                match.group(1)
            )
        )

        if html_title:
            return html_title

    # First meaningful text line.
    if text.strip():

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if lines:

            possible_title = clean_text(
                lines[0]
            )

            if len(possible_title) <= 200:
                return possible_title

            words = possible_title.split()

            return (
                " ".join(
                    words[:30]
                )
                + "..."
            )

    return ""


# ============================================================
# TRAFILATURA EXTRACTION
# ============================================================

def extract_with_trafilatura(
    downloaded: str
):
    """
    Run Trafilatura on already-downloaded HTML.
    """

    result = trafilatura.extract(
        downloaded,
        output_format="json",
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_precision=False,
        favor_recall=True
    )

    if not result:
        return None

    try:

        data = json.loads(
            result
        )

    except Exception:
        return None

    raw_text = (
        data.get("text")
        or ""
    )

    text = clean_text(
        raw_text
    )

    if not text:
        return None

    return {
        "data": data,
        "text": text
    }


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_article_from_url(
    url: str
):
    """
    Robust article extraction pipeline:

    1. Trafilatura built-in fetch
    2. Browser-like HTTP request
    3. Trafilatura extraction
    4. JSON-LD Article fallback
    """

    url = url.strip()

    if not validate_url(url):

        raise ValueError(
            "Invalid URL. Please enter a valid HTTP/HTTPS URL."
        )

    downloaded = None

    # --------------------------------------------------------
    # METHOD 1: Trafilatura fetch
    # --------------------------------------------------------

    try:

        downloaded = trafilatura.fetch_url(
            url
        )

    except Exception:
        downloaded = None

    # --------------------------------------------------------
    # METHOD 2: Browser-like HTTP request
    # --------------------------------------------------------

    if not downloaded:

        try:

            downloaded = download_page(
                url
            )

        except Exception as fetch_error:

            raise ValueError(
                str(fetch_error)
            )

    # --------------------------------------------------------
    # METHOD 3: Trafilatura extraction
    # --------------------------------------------------------

    extracted = extract_with_trafilatura(
        downloaded
    )

    title = ""
    article_text = ""

    if extracted:

        data = extracted["data"]
        text = extracted["text"]

        title = extract_title(
            downloaded,
            data,
            text
        )

        article_text = remove_duplicate_title(
            title,
            text
        )

        if not article_text:
            article_text = text

    # --------------------------------------------------------
    # METHOD 4: JSON-LD fallback
    # --------------------------------------------------------

    if not article_text:

        jsonld = extract_jsonld_article(
            downloaded
        )

        if jsonld:

            title = (
                jsonld.get(
                    "title"
                )
                or title
            )

            article_text = clean_text(
                jsonld.get(
                    "text",
                    ""
                )
            )

            article_text = (
                remove_duplicate_title(
                    title,
                    article_text
                )
                or article_text
            )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not article_text:

        raise ValueError(
            "Could not extract readable article content from this webpage."
        )

    # --------------------------------------------------------
    # Combined BERT text
    # --------------------------------------------------------

    if title:

        combined_text = (
            f"{title}\n\n"
            f"{article_text}"
        )

    else:

        combined_text = article_text

    combined_text = clean_text(
        combined_text
    )

    word_count = len(
        combined_text.split()
    )

    character_count = len(
        combined_text
    )

    return {
        "url": url,
        "title": title,
        "text": combined_text,
        "article_text": article_text,
        "character_count": character_count,
        "word_count": word_count
    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    url = input(
        "\nEnter article URL: "
    ).strip()

    try:

        result = extract_article_from_url(
            url
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "TRUTHLENS AI - URL EXTRACTION"
        )

        print(
            "=" * 60
        )

        print(
            "\nTitle:"
        )

        print(
            "-" * 40
        )

        print(
            result["title"]
            or "Title not detected"
        )

        print(
            "\nArticle:"
        )

        print(
            "-" * 40
        )

        print(
            result["article_text"]
        )

        print(
            "\nStatistics:"
        )

        print(
            "-" * 40
        )

        print(
            f"Words: "
            f"{result['word_count']}"
        )

        print(
            f"Characters: "
            f"{result['character_count']}"
        )

        print(
            "\nStatus: SUCCESS"
        )

    except Exception as e:

        print(
            f"\nERROR: {e}"
        )