import ipaddress
import json
import logging
import os
import socket
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================
# MODEL SERVING
# ============================================================

from src.model_server import (
    predict_news,
    model_health,
)

# ============================================================
# OTHER TRUTHLENS MODULES
# ============================================================

from src.explain import explain_news
from src.ocr import extract_text_from_image_multilingual
from src.url_extractor import extract_article_from_url
from src.verifier import verify_article
from src.multilingual import (
    detect_language,
    prepare_multilingual_text,
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_VERSION = os.getenv(
    "TRUTHLENS_VERSION",
    "1.2.0",
)

HOST = os.getenv(
    "TRUTHLENS_HOST",
    "127.0.0.1",
)

PORT = int(
    os.getenv(
        "TRUTHLENS_PORT",
        "8000",
    )
)

DATABASE_PATH = os.getenv(
    "TRUTHLENS_DB",
    "data/truthlens_jobs.db",
)

MAX_TEXT_CHARS = int(
    os.getenv(
        "TRUTHLENS_MAX_TEXT_CHARS",
        "20000",
    )
)

MAX_URL_LENGTH = int(
    os.getenv(
        "TRUTHLENS_MAX_URL_LENGTH",
        "2048",
    )
)

MAX_IMAGE_BYTES = int(
    os.getenv(
        "TRUTHLENS_MAX_IMAGE_BYTES",
        str(8 * 1024 * 1024),
    )
)

JOB_RETENTION_HOURS = int(
    os.getenv(
        "TRUTHLENS_JOB_RETENTION_HOURS",
        "24",
    )
)

VERIFY_WORKERS = max(
    1,
    int(
        os.getenv(
            "TRUTHLENS_VERIFY_WORKERS",
            "2",
        )
    ),
)

DEFAULT_CORS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "TRUTHLENS_CORS_ORIGINS",
        ",".join(DEFAULT_CORS),
    ).split(",")
    if origin.strip()
]

os.makedirs(
    os.path.dirname(DATABASE_PATH) or ".",
    exist_ok=True,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv(
        "TRUTHLENS_LOG_LEVEL",
        "INFO",
    ).upper(),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "truthlens"
)


# ============================================================
# EXECUTORS / LOCKS
# ============================================================

verification_executor = ThreadPoolExecutor(
    max_workers=VERIFY_WORKERS,
    thread_name_prefix="truthlens-verify",
)

db_lock = threading.Lock()


# ============================================================
# JSON SAFETY
# ============================================================

def make_json_safe(value: Any):

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return make_json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return value.item()

    return value


# ============================================================
# DATABASE
# ============================================================

def db_connect():

    conn = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    with db_lock, db_connect() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS verification_jobs (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                article_text TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_url
            ON verification_jobs(url)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_updated
            ON verification_jobs(updated_at)
            """
        )

        conn.commit()


def cleanup_old_jobs():

    cutoff = (
        time.time()
        - JOB_RETENTION_HOURS * 3600
    )

    with db_lock, db_connect() as conn:

        conn.execute(
            """
            DELETE FROM verification_jobs
            WHERE updated_at < ?
            """,
            (cutoff,),
        )

        conn.commit()


def save_job(
    job_id: str,
    url: str,
    title: str,
    article_text: str,
    status: str,
    result: dict,
):

    now = time.time()

    payload = json.dumps(
        make_json_safe(result),
        ensure_ascii=False,
    )

    with db_lock, db_connect() as conn:

        conn.execute(
            """
            INSERT INTO verification_jobs (
                id,
                url,
                title,
                article_text,
                status,
                result_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(id)
            DO UPDATE SET
                status=excluded.status,
                result_json=excluded.result_json,
                updated_at=excluded.updated_at
            """,
            (
                job_id,
                url,
                title,
                article_text,
                status,
                payload,
                now,
                now,
            ),
        )

        conn.commit()


def get_job(job_id: str):

    with db_lock, db_connect() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM verification_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

    if row is None:
        return None

    result = json.loads(
        row["result_json"] or "{}"
    )

    result["verification_id"] = row["id"]
    result["url"] = row["url"]
    result["status"] = row["status"]

    return make_json_safe(result)


def get_latest_job_for_url(url: str):

    normalized = url.rstrip("/")

    with db_lock, db_connect() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM verification_jobs
            WHERE rtrim(url, '/') = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()

    if row is None:
        return None

    result = json.loads(
        row["result_json"] or "{}"
    )

    result["verification_id"] = row["id"]
    result["url"] = row["url"]
    result["status"] = row["status"]

    return make_json_safe(result)


def get_restartable_jobs():

    with db_lock, db_connect() as conn:

        return conn.execute(
            """
            SELECT *
            FROM verification_jobs
            WHERE status IN ('PENDING', 'PROCESSING')
            """
        ).fetchall()


# ============================================================
# URL SECURITY
# ============================================================

def validate_public_url(
    value: str,
) -> str:

    value = value.strip()

    if not value:
        raise ValueError(
            "URL is required."
        )

    if len(value) > MAX_URL_LENGTH:
        raise ValueError(
            f"URL is too long. Maximum length is "
            f"{MAX_URL_LENGTH} characters."
        )

    parsed = urlparse(value)

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
    ):
        raise ValueError(
            "Only valid HTTP/HTTPS URLs are supported."
        )

    if parsed.username or parsed.password:
        raise ValueError(
            "URLs containing embedded credentials "
            "are not allowed."
        )

    host = (
        parsed.hostname
        .lower()
        .rstrip(".")
    )

    if host in {
        "localhost",
        "localhost.localdomain",
    }:
        raise ValueError(
            "Local URLs are not allowed."
        )

    try:

        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(
                host,
                None,
            )
        }

    except socket.gaierror as exc:

        raise ValueError(
            "The URL hostname could not be resolved."
        ) from exc

    for address in addresses:

        ip = ipaddress.ip_address(
            address
        )

        if any(
            [
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            ]
        ):
            raise ValueError(
                "URLs pointing to private or local "
                "network addresses are not allowed."
            )

    return value


# ============================================================
# REQUEST MODELS
# ============================================================

class StrictModel(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )


class NewsRequest(StrictModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_CHARS,
    )

    @field_validator("text")
    @classmethod
    def clean_text(
        cls,
        value: str,
    ):

        value = value.strip()

        if not value:
            raise ValueError(
                "Please provide news text."
            )

        return value


class URLRequest(StrictModel):

    url: str = Field(
        ...,
        min_length=1,
        max_length=MAX_URL_LENGTH,
    )

    @field_validator("url")
    @classmethod
    def clean_url(
        cls,
        value: str,
    ):

        return validate_public_url(
            value
        )


# ============================================================
# BACKGROUND VERIFICATION
# ============================================================

def create_verification_id():

    return uuid.uuid4().hex


def run_background_verification(
    verification_id: str,
    url: str,
    article_text: str,
    title: str,
):

    logger.info(
        "Verification started id=%s",
        verification_id,
    )

    processing = {
        "status": "PROCESSING",
        "verification_id": verification_id,
        "url": url,
        "overall_assessment": "PROCESSING",
        "claims_checked": 0,
        "claims": [],
        "note": (
            "Evidence verification is "
            "running in the background."
        ),
    }

    save_job(
        verification_id,
        url,
        title,
        article_text,
        "PROCESSING",
        processing,
    )

    try:

        result = verify_article(
            article_text,
            title=title,
            max_claims=3,
        )

        if not isinstance(result, dict):

            result = {
                "overall_assessment":
                    "INSUFFICIENT_EVIDENCE",
                "claims_checked":
                    0,
                "claims":
                    [],
                "note":
                    "Verification returned "
                    "an unexpected response.",
            }

        result = make_json_safe(
            result
        )

        result.update(
            {
                "status":
                    "COMPLETED",
                "verification_id":
                    verification_id,
                "url":
                    url,
            }
        )

        save_job(
            verification_id,
            url,
            title,
            article_text,
            "COMPLETED",
            result,
        )

        logger.info(
            "Verification completed id=%s",
            verification_id,
        )

    except Exception as exc:

        logger.exception(
            "Verification failed id=%s",
            verification_id,
        )

        result = {
            "status":
                "ERROR",
            "verification_id":
                verification_id,
            "url":
                url,
            "overall_assessment":
                "ERROR",
            "claims_checked":
                0,
            "claims":
                [],
            "note":
                (
                    f"Verification failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
        }

        save_job(
            verification_id,
            url,
            title,
            article_text,
            "ERROR",
            result,
        )


def submit_verification(
    url: str,
    title: str,
    article_text: str,
):

    job_id = create_verification_id()

    pending = {
        "status":
            "PENDING",
        "verification_id":
            job_id,
        "url":
            url,
        "overall_assessment":
            "PENDING",
        "claims_checked":
            0,
        "claims":
            [],
        "note":
            (
                "Evidence verification is "
                "running in the background."
            ),
    }

    save_job(
        job_id,
        url,
        title,
        article_text,
        "PENDING",
        pending,
    )

    verification_executor.submit(
        run_background_verification,
        job_id,
        url,
        article_text,
        title,
    )

    return pending


def resume_jobs():

    rows = get_restartable_jobs()

    for row in rows:

        job_id = row["id"]

        pending = {
            "status":
                "PENDING",
            "verification_id":
                job_id,
            "url":
                row["url"],
            "overall_assessment":
                "PENDING",
            "claims_checked":
                0,
            "claims":
                [],
            "note":
                (
                    "Verification resumed "
                    "after server restart."
                ),
        }

        save_job(
            job_id,
            row["url"],
            row["title"],
            row["article_text"],
            "PENDING",
            pending,
        )

        verification_executor.submit(
            run_background_verification,
            job_id,
            row["url"],
            row["article_text"],
            row["title"],
        )

    if rows:

        logger.info(
            "Resumed %d verification jobs",
            len(rows),
        )


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    init_db()

    cleanup_old_jobs()

    resume_jobs()

    logger.info(
        "TruthLens API started | "
        "version=%s | verification_workers=%s",
        APP_VERSION,
        VERIFY_WORKERS,
    )

    yield

    verification_executor.shutdown(
        wait=False,
        cancel_futures=False,
    )

    logger.info(
        "TruthLens API stopped"
    )


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="TruthLens AI",
    description=(
        "Explainable Fake News Detection "
        "& Evidence Verification API"
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Accept",
        "X-Request-ID",
    ],
)


# ============================================================
# REQUEST MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_context(
    request: Request,
    call_next,
):

    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or uuid.uuid4().hex
    )

    started = time.perf_counter()

    try:

        response = await call_next(
            request
        )

    except Exception:

        logger.exception(
            "Unhandled request error "
            "request_id=%s path=%s",
            request_id,
            request.url.path,
        )

        raise

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "strict-origin-when-cross-origin"

    logger.info(
        "%s %s -> %s %.1fms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )

    return response


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    return JSONResponse(
        status_code=422,
        content={
            "error":
                "Invalid request.",
            "details":
                make_json_safe(
                    exc.errors()
                ),
        },
    )


@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error":
                str(exc.detail)
        },
    )


@app.exception_handler(
    Exception
)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unhandled exception path=%s",
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error":
                (
                    "Internal server error. "
                    "Please try again later."
                )
        },
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "TruthLens AI API is running",

        "status":
            "success",

        "version":
            APP_VERSION,

        "model":
            "truthlens-bert-v2",

        "model_serving":
            "enabled",

        "features": [
            "BERT Classification",
            "Production Model Serving",
            "GPU Inference",
            "FP16 Inference",
            "Micro-Batching",
            "Inference Queue",
            "Sliding-Window Inference",
            "XAI Explanation",
            "OCR",
            "Multilingual Analysis",
            "URL Extraction",
            "Claim Extraction",
            "Evidence Retrieval",
            "NLI Verification",
            "Persistent Verification Jobs",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "service":
            "TruthLens AI",

        "model":
            "truthlens-bert-v2",

        "model_serving":
            "enabled",

        "verification_workers":
            VERIFY_WORKERS,
    }


# ============================================================
# READINESS
# ============================================================

@app.get("/ready")
def ready():

    try:

        with db_connect() as conn:

            conn.execute(
                "SELECT 1"
            ).fetchone()

        serving = model_health()

        return {
            "status":
                "ready",

            "database":
                "ok",

            "model":
                serving,
        }

    except Exception as exc:

        logger.exception(
            "Readiness check failed"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                f"Service not ready: {exc}"
            ),
        ) from exc


# ============================================================
# MODEL SERVING HEALTH
# ============================================================

@app.get("/model-health")
def model_serving_health():

    try:

        return make_json_safe(
            model_health()
        )

    except Exception as exc:

        logger.exception(
            "Model health check failed"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                f"Model server unavailable: {exc}"
            ),
        ) from exc


# ============================================================
# TEXT PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    request: NewsRequest,
):

    try:

        result = predict_news(
            request.text
        )

        return make_json_safe(
            {
                "source":
                    "text",

                "analysis":
                    result,
            }
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except TimeoutError as exc:

        raise HTTPException(
            status_code=504,
            detail=(
                "Model inference timed out."
            ),
        ) from exc

    except RuntimeError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. "
                "Please try again."
            ),
        ) from exc


# ============================================================
# TEXT + XAI
# ============================================================

@app.post("/explain")
def explain(
    request: NewsRequest,
):

    try:

        # Authoritative prediction comes
        # from the production model server.
        prediction = predict_news(
            request.text
        )

        explanation = explain_news(
            request.text
        )

        # Prediction is intentionally merged LAST
        # so XAI cannot overwrite model output.
        analysis = {
            **explanation,
            **prediction,
        }

        return make_json_safe(
            {
                "source":
                    "text",

                "analysis":
                    analysis,

                "model_analysis":
                    prediction,

                "explanation":
                    explanation,
            }
        )

    except Exception as exc:

        logger.exception(
            "Analysis failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis failed. "
                "Please try again."
            ),
        ) from exc


# ============================================================
# TEXT VERIFICATION
# ============================================================

@app.post("/verify")
def verify(
    request: NewsRequest,
):

    try:

        result = verify_article(
            request.text,
            max_claims=3,
        )

        return make_json_safe(
            {
                "source":
                    "text",

                "verification":
                    result,

                "pipeline": [
                    "Text Input",
                    "Claim Extraction",
                    "Open-Web Evidence Retrieval",
                    "NLI Evidence Verification",
                ],
            }
        )

    except Exception as exc:

        logger.exception(
            "Verification failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Verification failed. "
                "Please try again."
            ),
        ) from exc


# ============================================================
# MULTILINGUAL
# ============================================================

@app.post(
    "/analyze-multilingual"
)
def analyze_multilingual(
    request: NewsRequest,
):

    try:

        text = request.text

        language = detect_language(
            text
        )

        multilingual_result = (
            prepare_multilingual_text(
                text
            )
        )

        analysis_text = (
            multilingual_result.get(
                "translated_text"
            )
            or text
        )

        translation_applied = bool(
            multilingual_result.get(
                "translation_applied",
                False,
            )
        )

        prediction = predict_news(
            analysis_text
        )

        explanation = explain_news(
            analysis_text
        )

        analysis = {
            **explanation,
            **prediction,
        }

        verification = {
            "status":
                "PENDING",

            "overall_assessment":
                "PENDING",

            "claims_checked":
                0,

            "claims":
                [],

            "method":
                (
                    "Open-web evidence retrieval "
                    "+ NLI Evidence Verification"
                ),

            "note":
                (
                    "Evidence verification is "
                    "separate from BERT classification."
                ),
        }

        return make_json_safe(
            {
                "source":
                    "multilingual",

                "language":
                    language,

                "analysis_language":
                    {
                        "code":
                            "en",

                        "name":
                            "English",
                    },

                "translation_applied":
                    translation_applied,

                "original_text":
                    text,

                "translated_text":
                    multilingual_result.get(
                        "translated_text",
                        text,
                    ),

                "analysis":
                    analysis,

                "model_analysis":
                    prediction,

                "explanation":
                    explanation,

                "verification":
                    verification,

                "pipeline": [
                    "Text Input",
                    "Language Detection",
                    (
                        "Translation to English"
                        if translation_applied
                        else "Language Normalization"
                    ),
                    "Production BERT Model Server",
                    "XAI Explanation",
                    "Claim Extraction",
                    "Open-Web Evidence Retrieval",
                    "NLI Evidence Verification",
                ],
            }
        )

    except Exception as exc:

        logger.exception(
            "Multilingual analysis failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Multilingual analysis failed. "
                "Please try again."
            ),
        ) from exc


# ============================================================
# IMAGE / OCR
# ============================================================

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
):

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image format. "
                "Please upload JPG, PNG or WEBP."
            ),
        )

    temp_path = None
    total = 0

    try:

        suffix = os.path.splitext(
            file.filename or ".png"
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_path = temp_file.name

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total += len(chunk)

                if total > MAX_IMAGE_BYTES:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Image is too large. "
                            f"Maximum size is "
                            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB."
                        ),
                    )

                temp_file.write(
                    chunk
                )

        ocr_result = (
            extract_text_from_image_multilingual(
                temp_path
            )
        )

        extracted_text = str(
            ocr_result.get(
                "text",
                "",
            )
        ).strip()

        if not extracted_text:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text "
                    "found in the image."
                ),
            )

        extracted_text = (
            extracted_text[
                :MAX_TEXT_CHARS
            ]
        )

        detected_language = detect_language(
            extracted_text
        )

        multilingual_result = (
            prepare_multilingual_text(
                extracted_text
            )
        )

        analysis_text = (
            multilingual_result.get(
                "translated_text"
            )
            or extracted_text
        )

        translation_applied = bool(
            multilingual_result.get(
                "translation_applied",
                False,
            )
        )

        prediction = predict_news(
            analysis_text
        )

        explanation = explain_news(
            analysis_text
        )

        analysis = {
            **explanation,
            **prediction,
        }

        verification = {
            "status":
                "PENDING",

            "overall_assessment":
                "PENDING",

            "claims_checked":
                0,

            "claims":
                [],

            "method":
                (
                    "Open-web evidence retrieval "
                    "+ NLI Evidence Verification"
                ),

            "note":
                (
                    "Evidence verification does not "
                    "block image analysis."
                ),
        }

        response = {
            "source":
                "image",

            "filename":
                file.filename,

            "ocr": {
                "text":
                    extracted_text,

                "segments":
                    ocr_result.get(
                        "segments",
                        [],
                    ),

                "segment_count":
                    ocr_result.get(
                        "segment_count",
                        0,
                    ),

                "language": {
                    "code":
                        detected_language.get(
                            "code",
                            "unknown",
                        ),

                    "name":
                        detected_language.get(
                            "name",
                            "Unknown",
                        ),

                    "confidence":
                        detected_language.get(
                            "confidence",
                            0.0,
                        ),
                },

                "translation_applied":
                    translation_applied,

                "translated_text":
                    multilingual_result.get(
                        "translated_text",
                        extracted_text,
                    ),
            },

            "language":
                detected_language,

            "analysis_language":
                multilingual_result.get(
                    "analysis_language",
                    {
                        "code":
                            "en",

                        "name":
                            "English",
                    },
                ),

            "translation_applied":
                translation_applied,

            "original_text":
                multilingual_result.get(
                    "original_text",
                    extracted_text,
                ),

            "translated_text":
                multilingual_result.get(
                    "translated_text",
                    extracted_text,
                ),

            "analysis":
                analysis,

            "model_analysis":
                prediction,

            "explanation":
                explanation,

            "verification":
                verification,

            "pipeline": [
                "Image Upload",
                "EasyOCR",
                "Script Detection",
                "Text Extraction",
                "Language Normalization",
                "Translation to English",
                "Production BERT Model Server",
                "XAI Explanation",
                "Claim Extraction",
                "Evidence Verification Available Separately",
            ],
        }

        return make_json_safe(
            response
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Image analysis failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Image analysis failed. "
                "Please try again."
            ),
        ) from exc

    finally:

        try:
            await file.close()
        except Exception:
            pass

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:
                os.remove(
                    temp_path
                )
            except OSError:
                logger.warning(
                    "Could not remove "
                    "temporary image: %s",
                    temp_path,
                )


# ============================================================
# URL ANALYSIS
# ============================================================

@app.post("/analyze-url")
def analyze_url(
    request: URLRequest,
):

    try:

        article = extract_article_from_url(
            request.url
        )

        article_text = str(
            article.get(
                "text",
                "",
            )
        ).strip()

        if not article_text:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable article text "
                    "was extracted from this URL."
                ),
            )

        article_text = (
            article_text[
                :MAX_TEXT_CHARS
            ]
        )

        title = str(
            article.get(
                "title",
                "",
            )
        )[:1000]

        actual_url = str(
            article.get(
                "url",
                request.url,
            )
        )

        # Production model server.
        prediction = predict_news(
            article_text
        )

        explanation = explain_news(
            article_text
        )

        analysis = {
            **explanation,
            **prediction,
        }

        # Verification is deliberately asynchronous.
        pending = submit_verification(
            actual_url,
            title,
            article_text,
        )

        return make_json_safe(
            {
                "source":
                    "url",

                "url":
                    actual_url,

                "article": {
                    "title":
                        title,

                    "text":
                        article.get(
                            "article_text",
                            article_text,
                        ),

                    "word_count":
                        article.get(
                            "word_count",
                            len(
                                article_text.split()
                            ),
                        ),

                    "character_count":
                        article.get(
                            "character_count",
                            len(article_text),
                        ),
                },

                "analysis":
                    analysis,

                "model_analysis":
                    prediction,

                "explanation":
                    explanation,

                "verification":
                    pending,

                "verification_id":
                    pending[
                        "verification_id"
                    ],

                "pipeline": [
                    "URL Input",
                    "Webpage Retrieval",
                    "Article Extraction",
                    "Text Cleaning",
                    "Production BERT Model Server",
                    "XAI Explanation",
                    "Background Claim Extraction",
                    "Background Evidence Retrieval",
                    "Background NLI Verification",
                ],
            }
        )

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "URL analysis failed"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not analyze "
                "the supplied URL."
            ),
        ) from exc


# ============================================================
# VERIFICATION STATUS
# ============================================================

@app.get(
    "/verification-status/{verification_id}"
)
def verification_status_by_id(
    verification_id: str,
):

    result = get_job(
        verification_id
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Verification job "
                "not found."
            ),
        )

    return result


@app.post(
    "/verification-status"
)
def verification_status(
    request: URLRequest,
):

    result = get_latest_job_for_url(
        request.url
    )

    if result is None:

        return {
            "status":
                "NOT_FOUND",

            "url":
                request.url,

            "claims_checked":
                0,

            "claims":
                [],

            "overall_assessment":
                "NOT_FOUND",

            "note":
                (
                    "No verification job "
                    "was found for this URL."
                ),
        }

    return result


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )