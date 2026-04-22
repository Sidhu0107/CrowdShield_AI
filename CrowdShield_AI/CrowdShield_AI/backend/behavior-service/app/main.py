"""Behavior service entrypoint.

Responsibilities (TDD §1.4):
- Consume Feature Events from Redis stream: feature_stream
- Maintain 30-frame sequence buffers per tracked person
- Run custom LSTM inference for behavior classification
- Publish Behavior Events to Redis stream: behavior_stream
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn as nn
from fastapi import FastAPI
from redis import Redis

SERVICE_NAME = os.getenv("SERVICE_NAME", "behavior-service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
FEATURE_STREAM = "feature_stream"
BEHAVIOR_STREAM = "behavior_stream"

SEQUENCE_LEN = int(os.getenv("BEHAVIOR_WINDOW_SIZE", "30"))
FEATURE_VECTOR_SIZE = int(os.getenv("FEATURE_VECTOR_SIZE", "9"))
HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "128"))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "model.pt")

CLASS_NAMES = ["normal", "violence", "fighting", "stampede"]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


@dataclass(frozen=True)
class LSTMState:
    """Container for hidden/cell states of a single LSTM layer."""

    h: torch.Tensor
    c: torch.Tensor


class CustomLSTMCell(nn.Module):
    """Manual LSTM cell implementation (no nn.LSTM)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x_t: torch.Tensor, state: LSTMState) -> LSTMState:
        gates = self.x_proj(x_t) + self.h_proj(state.h)
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_t = (f_t * state.c) + (i_t * g_t)
        h_t = o_t * torch.tanh(c_t)
        return LSTMState(h=h_t, c=c_t)


class CustomLSTMClassifier(nn.Module):
    """Two-layer custom LSTM classifier for 4-class behavior output."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.layer1 = CustomLSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.layer2 = CustomLSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> LSTMState:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return LSTMState(h=zeros, c=zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, input_size]
        if x.dim() != 3:
            raise ValueError("Expected input tensor shape [batch, seq_len, input_size]")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {x.shape[-1]}")

        batch_size = x.shape[0]
        state1 = self._init_state(batch_size, x.device, x.dtype)
        state2 = self._init_state(batch_size, x.device, x.dtype)

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            state1 = self.layer1(x_t, state1)
            state2 = self.layer2(state1.h, state2)

        return self.head(state2.h)


class BehaviorRuntime:
    """Service runtime state (Redis, model, worker, sequence buffers)."""

    def __init__(self) -> None:
        self.redis: Redis | None = None
        self.model: CustomLSTMClassifier | None = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.running = False
        self.last_feature_id = "$"

        self.buffers: dict[int, deque[list[float]]] = {}
        self.lock = threading.Lock()


runtime = BehaviorRuntime()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


def _resolve_model_path(model_path: str) -> Path | None:
    """Resolve model path from common locations inside the service container."""
    candidates = [
        Path(model_path),
        Path(__file__).resolve().parent / model_path,
        Path(__file__).resolve().parent.parent / model_path,
        Path(__file__).resolve().parent.parent.parent / model_path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _prepare_model() -> None:
    """Initialize classifier and load checkpoint if available."""
    runtime.model = CustomLSTMClassifier(
        input_size=FEATURE_VECTOR_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=len(CLASS_NAMES),
    ).to(runtime.device)

    resolved = _resolve_model_path(LSTM_MODEL_PATH)
    if resolved is None:
        logger.warning("model_not_found path=%r using_random_weights=true", LSTM_MODEL_PATH)
        runtime.model.eval()
        return

    state_dict = torch.load(resolved, map_location=runtime.device)
    runtime.model.load_state_dict(state_dict)
    runtime.model.eval()
    logger.info("model_loaded path=%s", resolved)


def _update_sequence_buffers(features: list[dict[str, object]]) -> list[tuple[int, torch.Tensor]]:
    """Append features per person and return ready [1,30,input_size] sequences."""
    ready: list[tuple[int, torch.Tensor]] = []

    with runtime.lock:
        for item in features:
            person_id = int(item.get("person_id", -1))
            vector = item.get("feature_vector", [])
            if person_id < 0:
                continue
            if not isinstance(vector, list) or len(vector) != FEATURE_VECTOR_SIZE:
                continue

            if person_id not in runtime.buffers:
                runtime.buffers[person_id] = deque(maxlen=SEQUENCE_LEN)

            cast_vector = [float(v) for v in vector]
            runtime.buffers[person_id].append(cast_vector)

            if len(runtime.buffers[person_id]) == SEQUENCE_LEN:
                seq_tensor = torch.tensor(
                    [list(runtime.buffers[person_id])],
                    dtype=torch.float32,
                    device=runtime.device,
                )
                ready.append((person_id, seq_tensor))

    return ready


@torch.no_grad()
def _predict(sequence: torch.Tensor) -> tuple[str, float]:
    """Run model inference on one sequence and return class + confidence."""
    assert runtime.model is not None
    logits = runtime.model(sequence)
    probs = torch.softmax(logits, dim=1)
    cls_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, cls_idx].item())
    return CLASS_NAMES[cls_idx], confidence


def _consumer_loop() -> None:
    """Consume feature_stream, infer behavior, publish behavior_stream."""
    assert runtime.redis is not None
    logger.info("consumer_started input_stream=%s output_stream=%s", FEATURE_STREAM, BEHAVIOR_STREAM)
    runtime.running = True

    try:
        while not runtime.stop_event.is_set():
            events = runtime.redis.xread({FEATURE_STREAM: runtime.last_feature_id}, block=1000, count=30)
            if not events:
                continue

            for _stream, entries in events:
                for entry_id, fields in entries:
                    runtime.last_feature_id = entry_id

                    features_raw = fields.get("features")
                    if not features_raw:
                        logger.warning("invalid_feature_event entry_id=%s", entry_id)
                        continue

                    try:
                        features = json.loads(features_raw)
                    except Exception:
                        logger.warning("feature_parse_failed entry_id=%s", entry_id)
                        continue

                    ready_sequences = _update_sequence_buffers(features)
                    for person_id, sequence in ready_sequences:
                        prediction, confidence = _predict(sequence)

                        # TDD Behavior Event contract:
                        # {
                        #   "sequence_id": "uuid",
                        #   "prediction": "normal|violence|fighting|stampede",
                        #   "confidence": 0.0
                        # }
                        behavior_event = {
                            "sequence_id": str(uuid4()),
                            "prediction": prediction,
                            "confidence": f"{confidence:.6f}",
                        }
                        runtime.redis.xadd(BEHAVIOR_STREAM, behavior_event, maxlen=10000, approximate=True)

                        logger.info(
                            "behavior_published person_id=%s prediction=%s confidence=%.4f",
                            person_id,
                            prediction,
                            confidence,
                        )
    except Exception:
        logger.exception("consumer_loop_failed")
    finally:
        runtime.running = False
        logger.info("consumer_stopped")


@app.on_event("startup")
def startup() -> None:
    """Initialize Redis + model and start stream consumer."""
    runtime.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    runtime.redis.ping()
    _prepare_model()

    runtime.stop_event.clear()
    runtime.worker = threading.Thread(target=_consumer_loop, daemon=True)
    runtime.worker.start()


@app.on_event("shutdown")
def shutdown() -> None:
    """Stop consumer thread and release Redis client."""
    runtime.stop_event.set()
    if runtime.worker and runtime.worker.is_alive():
        runtime.worker.join(timeout=2)
    if runtime.redis is not None:
        runtime.redis.close()


@app.get("/health", tags=["system"])
async def health() -> dict[str, str | bool]:
    """Liveness probe endpoint for container orchestrators and monitors."""
    redis_status = "down"
    if runtime.redis is not None:
        try:
            runtime.redis.ping()
            redis_status = "up"
        except Exception:
            redis_status = "down"

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "redis": redis_status,
        "running": runtime.running,
    }


@app.get("/status", tags=["system"])
async def status() -> dict[str, str | int | bool]:
    """Readiness endpoint containing service metadata."""
    with runtime.lock:
        tracked_people = len(runtime.buffers)

    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "state": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_stream": FEATURE_STREAM,
        "output_stream": BEHAVIOR_STREAM,
        "sequence_len": SEQUENCE_LEN,
        "tracked_people": tracked_people,
        "running": runtime.running,
    }