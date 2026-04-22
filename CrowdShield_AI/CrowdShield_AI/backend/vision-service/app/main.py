"""Vision service entrypoint.

Responsibilities (TDD §1.2):
- Consume Frame Event payloads from Redis stream: frame_stream
- Run YOLOv8 + ByteTrack for person detection/tracking
- Publish Detection Event payloads to Redis stream: detection_stream
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI
from redis import Redis
from ultralytics import YOLO

SERVICE_NAME = os.getenv("SERVICE_NAME", "vision-service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
FRAME_STREAM = "frame_stream"
DETECTION_STREAM = "detection_stream"

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


class VisionRuntime:
    """Runtime state for Redis, model, and consumer worker."""

    def __init__(self) -> None:
        self.redis: Redis | None = None
        self.model: YOLO | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.last_stream_id = "$"  # consume only new events after startup


runtime = VisionRuntime()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


def _decode_frame(frame_data_b64: str) -> np.ndarray | None:
    """Decode base64 image payload from Frame Event into OpenCV BGR frame."""
    try:
        raw = base64.b64decode(frame_data_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        logger.exception("frame_decode_failed")
        return None


def _extract_detections(frame: np.ndarray) -> list[dict[str, object]]:
    """Run YOLO + ByteTrack and return TDD contract detection items."""
    assert runtime.model is not None

    results = runtime.model.track(
        source=frame,
        conf=YOLO_CONF,
        classes=[0],  # person class only
        tracker="bytetrack.yaml",
        persist=True,
        device=YOLO_DEVICE,
        verbose=False,
    )

    detections: list[dict[str, object]] = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            # TDD requires person_id as int in detection payload.
            # ByteTrack can emit None on very early unstable tracks; use -1.
            person_id = int(box.id[0]) if box.id is not None else -1

            detections.append(
                {
                    "person_id": person_id,
                    "bbox": [x1, y1, width, height],
                }
            )

    return detections


def _consume_frames_loop() -> None:
    """Background loop: consume frame_stream, detect persons, publish events."""
    assert runtime.redis is not None

    logger.info("consumer_started input_stream=%s output_stream=%s", FRAME_STREAM, DETECTION_STREAM)
    runtime.running = True

    try:
        while not runtime.stop_event.is_set():
            events = runtime.redis.xread({FRAME_STREAM: runtime.last_stream_id}, block=1000, count=20)
            if not events:
                continue

            for stream_name, entries in events:
                _ = stream_name
                for entry_id, fields in entries:
                    runtime.last_stream_id = entry_id

                    frame_id = fields.get("frame_id")
                    frame_data = fields.get("frame_data")
                    if not frame_id or not frame_data:
                        logger.warning("invalid_frame_event entry_id=%s", entry_id)
                        continue

                    frame = _decode_frame(frame_data)
                    if frame is None:
                        continue

                    detections = _extract_detections(frame)

                    # TDD Detection Event contract:
                    # {
                    #   "frame_id": "uuid",
                    #   "detections": [{"person_id": int, "bbox": [x, y, w, h]}]
                    # }
                    detection_event = {
                        "frame_id": frame_id,
                        "detections": json.dumps(detections),
                    }

                    runtime.redis.xadd(DETECTION_STREAM, detection_event, maxlen=10000, approximate=True)
    except Exception:
        logger.exception("consumer_loop_failed")
    finally:
        runtime.running = False
        logger.info("consumer_stopped")


@app.on_event("startup")
def startup() -> None:
    """Initialize Redis and model, then start consumer worker."""
    runtime.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    runtime.redis.ping()

    logger.info("model_loading model=%s", YOLO_MODEL)
    runtime.model = YOLO(YOLO_MODEL)
    logger.info("model_ready model=%s", YOLO_MODEL)

    runtime.stop_event.clear()
    runtime.worker = threading.Thread(target=_consume_frames_loop, daemon=True)
    runtime.worker.start()


@app.on_event("shutdown")
def shutdown() -> None:
    """Stop worker and close Redis client."""
    runtime.stop_event.set()
    if runtime.worker and runtime.worker.is_alive():
        runtime.worker.join(timeout=2)
    if runtime.redis is not None:
        runtime.redis.close()


@app.get("/health", tags=["system"])
async def health() -> dict[str, str | bool]:
    """Liveness probe endpoint with worker/Redis readiness signal."""
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
async def status() -> dict[str, str | bool]:
    """Readiness endpoint containing service metadata."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "state": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_stream": FRAME_STREAM,
        "output_stream": DETECTION_STREAM,
        "running": runtime.running,
    }