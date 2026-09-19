from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    os.getenv(
        "TRUTHLENS_MODEL_PATH",
        "models/truthlens-bert-v2",
    )
)

MAX_LENGTH = int(
    os.getenv(
        "TRUTHLENS_MODEL_MAX_LENGTH",
        "256",
    )
)

STRIDE = int(
    os.getenv(
        "TRUTHLENS_MODEL_STRIDE",
        "64",
    )
)

INFERENCE_BATCH_SIZE = max(
    1,
    int(
        os.getenv(
            "TRUTHLENS_MODEL_BATCH_SIZE",
            "4",
        )
    ),
)

MAX_PENDING_REQUESTS = max(
    1,
    int(
        os.getenv(
            "TRUTHLENS_MODEL_QUEUE_SIZE",
            "32",
        )
    ),
)

BATCH_WAIT_MS = max(
    0,
    int(
        os.getenv(
            "TRUTHLENS_MODEL_BATCH_WAIT_MS",
            "15",
        )
    ),
)

USE_FP16 = (
    torch.cuda.is_available()
    and os.getenv(
        "TRUTHLENS_MODEL_FP16",
        "1",
    ) != "0"
)


# ============================================================
# REQUEST TYPE
# ============================================================

@dataclass
class _PredictionRequest:

    text: str
    event: threading.Event

    result: Optional[dict] = None
    error: Optional[BaseException] = None


# ============================================================
# MODEL SERVER
# ============================================================

class TruthLensModelServer:

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"TruthLens model not found: {MODEL_PATH}"
            )

        print(
            "[MODEL SERVER] Loading tokenizer:",
            MODEL_PATH,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )

        print(
            "[MODEL SERVER] Loading model:",
            MODEL_PATH,
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        for parameter in self.model.parameters():

            parameter.requires_grad_(False)

        self.fp16 = (
            USE_FP16
            and self.device.type == "cuda"
        )

        self.queue = queue.Queue(
            maxsize=MAX_PENDING_REQUESTS
        )

        self._stop_event = (
            threading.Event()
        )

        self._worker = threading.Thread(
            target=self._batch_worker,
            name="truthlens-model-worker",
            daemon=True,
        )

        self._worker.start()

        self.model_name = (
            "truthlens-bert-v2"
        )

        print(
            "[MODEL SERVER] Ready | "
            f"device={self.device} | "
            f"fp16={self.fp16} | "
            f"batch={INFERENCE_BATCH_SIZE} | "
            f"queue={MAX_PENDING_REQUESTS}"
        )

    # ========================================================
    # SLIDING WINDOW TOKENIZATION
    # ========================================================

    def _build_windows(
        self,
        text: str,
    ):

        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        if not token_ids:

            raise ValueError(
                "No usable tokens were found in the input."
            )

        content_length = (
            MAX_LENGTH - 2
        )

        if content_length < 8:

            raise ValueError(
                "Model max length must be at least 10."
            )

        step = (
            content_length - STRIDE
        )

        if step <= 0:

            raise ValueError(
                "Model stride must be smaller "
                "than MAX_LENGTH - 2."
            )

        windows = []

        start = 0

        while start < len(token_ids):

            chunk = token_ids[
                start:start + content_length
            ]

            if not chunk:
                break

            input_ids = [
                self.tokenizer.cls_token_id,
                *chunk,
                self.tokenizer.sep_token_id,
            ]

            attention_mask = [
                1
            ] * len(input_ids)

            padding = (
                MAX_LENGTH
                - len(input_ids)
            )

            if padding > 0:

                input_ids.extend(
                    [self.tokenizer.pad_token_id]
                    * padding
                )

                attention_mask.extend(
                    [0]
                    * padding
                )

            windows.append(
                (
                    input_ids,
                    attention_mask,
                    len(chunk),
                )
            )

            if (
                start + content_length
                >= len(token_ids)
            ):

                break

            start += step

        return windows

    # ========================================================
    # GPU INFERENCE
    # ========================================================

    def _infer_windows(
        self,
        windows,
    ):

        window_probabilities = []

        for batch_start in range(
            0,
            len(windows),
            INFERENCE_BATCH_SIZE,
        ):

            batch = windows[
                batch_start:
                batch_start
                + INFERENCE_BATCH_SIZE
            ]

            input_ids = torch.tensor(
                [
                    item[0]
                    for item in batch
                ],
                dtype=torch.long,
                device=self.device,
            )

            attention_mask = torch.tensor(
                [
                    item[1]
                    for item in batch
                ],
                dtype=torch.long,
                device=self.device,
            )

            with torch.inference_mode():

                if self.fp16:

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):

                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                else:

                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

            probabilities = torch.softmax(
                output.logits,
                dim=-1,
            )

            window_probabilities.extend(
                probabilities
                .detach()
                .float()
                .cpu()
                .tolist()
            )

        return window_probabilities

    # ========================================================
    # PROCESS REQUESTS
    # ========================================================

    def _process_requests(
        self,
        requests,
    ):

        prepared = []

        for request in requests:

            try:

                windows = self._build_windows(
                    request.text
                )

                prepared.append(
                    (
                        request,
                        windows,
                    )
                )

            except BaseException as exc:

                request.error = exc
                request.event.set()

        if not prepared:
            return

        combined_windows = []

        ranges = []

        for request, windows in prepared:

            start = len(
                combined_windows
            )

            combined_windows.extend(
                windows
            )

            end = len(
                combined_windows
            )

            ranges.append(
                (
                    request,
                    windows,
                    start,
                    end,
                )
            )

        try:

            started = time.perf_counter()

            probabilities = (
                self._infer_windows(
                    combined_windows
                )
            )

            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000

            for (
                request,
                windows,
                start,
                end,
            ) in ranges:

                request_probabilities = (
                    probabilities[start:end]
                )

                fake_probability_sum = 0.0
                real_probability_sum = 0.0
                token_weight_sum = 0

                for (
                    probability,
                    window,
                ) in zip(
                    request_probabilities,
                    windows,
                ):

                    weight = window[2]

                    fake_probability_sum += (
                        float(probability[0])
                        * weight
                    )

                    real_probability_sum += (
                        float(probability[1])
                        * weight
                    )

                    token_weight_sum += (
                        weight
                    )

                if token_weight_sum <= 0:

                    raise RuntimeError(
                        "Model inference produced "
                        "no weighted tokens."
                    )

                fake_probability = (
                    fake_probability_sum
                    / token_weight_sum
                )

                real_probability = (
                    real_probability_sum
                    / token_weight_sum
                )

                total = (
                    fake_probability
                    + real_probability
                )

                if total > 0:

                    fake_probability /= total
                    real_probability /= total

                prediction = (
                    "REAL"
                    if real_probability
                    >= fake_probability
                    else "FAKE"
                )

                confidence = max(
                    fake_probability,
                    real_probability,
                )

                request.result = {

                    "prediction":
                        prediction,

                    "fake_probability":
                        fake_probability,

                    "real_probability":
                        real_probability,

                    "confidence":
                        confidence,

                    "chunks_analyzed":
                        len(windows),

                    "coverage":
                        100.0,

                    "truncated":
                        len(windows) > 1,

                    "model":
                        self.model_name,

                    "model_serving": {

                        "device":
                            str(self.device),

                        "fp16":
                            self.fp16,

                        "batch_size":
                            INFERENCE_BATCH_SIZE,

                        "batched_requests":
                            len(prepared),

                        "windows_in_batch":
                            len(combined_windows),

                        "queue_enabled":
                            True,

                        "latency_ms":
                            round(
                                elapsed_ms,
                                2,
                            ),
                    },
                }

        except BaseException as exc:

            for (
                request,
                _,
                _,
                _,
            ) in ranges:

                request.error = exc

        finally:

            for (
                request,
                _,
                _,
                _,
            ) in ranges:

                request.event.set()

    # ========================================================
    # MICRO-BATCH WORKER
    # ========================================================

    def _batch_worker(self):

        while not self._stop_event.is_set():

            try:

                first = self.queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                continue

            requests = [
                first
            ]

            deadline = (
                time.perf_counter()
                + BATCH_WAIT_MS / 1000.0
            )

            while (
                len(requests)
                < INFERENCE_BATCH_SIZE
                and time.perf_counter()
                < deadline
            ):

                remaining = max(
                    0.0,
                    deadline
                    - time.perf_counter(),
                )

                try:

                    requests.append(
                        self.queue.get(
                            timeout=remaining
                        )
                    )

                except queue.Empty:

                    break

            self._process_requests(
                requests
            )

    # ========================================================
    # PUBLIC PREDICTION API
    # ========================================================

    def predict(
        self,
        text: str,
        timeout: float = 120.0,
    ) -> dict:

        if not isinstance(
            text,
            str,
        ):

            raise TypeError(
                "text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Please provide news text."
            )

        request = _PredictionRequest(
            text=text,
            event=threading.Event(),
        )

        try:

            self.queue.put(
                request,
                timeout=2.0,
            )

        except queue.Full:

            raise RuntimeError(
                "Model inference queue is busy. "
                "Please retry shortly."
            )

        if not request.event.wait(
            timeout=timeout
        ):

            raise TimeoutError(
                "Model inference timed out."
            )

        if request.error is not None:

            raise request.error

        if request.result is None:

            raise RuntimeError(
                "Model server returned no result."
            )

        return request.result


# ============================================================
# SINGLETON
# ============================================================

_server_lock = threading.Lock()

_server: Optional[
    TruthLensModelServer
] = None


def get_model_server():

    global _server

    if _server is None:

        with _server_lock:

            if _server is None:

                _server = (
                    TruthLensModelServer()
                )

    return _server


def predict_news(
    text: str,
) -> dict:

    return get_model_server().predict(
        text
    )


def model_health():

    server = get_model_server()

    return {

        "status":
            "ready",

        "model":
            server.model_name,

        "device":
            str(server.device),

        "fp16":
            server.fp16,

        "batch_size":
            INFERENCE_BATCH_SIZE,

        "queue_size":
            MAX_PENDING_REQUESTS,
    }