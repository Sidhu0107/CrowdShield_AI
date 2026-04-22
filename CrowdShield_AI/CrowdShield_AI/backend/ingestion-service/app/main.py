"""Ingestion service entrypoint.

Responsibilities (TDD §1.1):
- Capture frames from webcam / video / RTSP
- Convert frame to base64 payload
- Publish Frame Event to Redis stream: frame_stream
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from redis import Redis

SERVICE_NAME = os.getenv("SERVICE_NAME", "ingestion-service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
FRAME_STREAM = "frame_stream"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


class StartRequest(BaseModel):
    """Request payload for starting capture/publish loop."""

    source: str | int = Field(default=0, description="Webcam index, video path, or RTSP URL")
    camera_id: str = Field(default="cam_1", description="Logical camera identifier")
    fps: int = Field(default=10, ge=1, le=60, description="Target publish FPS")
    jpeg_quality: int = Field(default=80, ge=30, le=100)


class IngestionRuntime:
    """Holds mutable runtime state for capture and Redis publishing."""

    def __init__(self) -> None:
        self.redis: Redis | None = None
        self.capture_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.running = False
        self.camera_id = ""
        self.source: str | int = 0
        self.target_fps = 10

    def configure_redis(self) -> None:
        self.redis = Redis.from_url(REDIS_URL, decode_responses=True)
        self.redis.ping()
        logger.info("redis_connected redis_url=%s", REDIS_URL)


runtime = IngestionRuntime()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


def _coerce_source(source: str | int) -> str | int:
    """Treat numeric strings as webcam index; keep others as paths/URLs."""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _capture_and_publish(camera_id: str, source: str | int, target_fps: int, jpeg_quality: int) -> None:
    """Background worker that captures frames and pushes strict Frame Events."""
    assert runtime.redis is not None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("capture_open_failed camera_id=%s source=%r", camera_id, source)
        with runtime.lock:
            runtime.running = False
        return

    logger.info("capture_started camera_id=%s source=%r fps=%s", camera_id, source, target_fps)

    frame_interval = 1.0 / max(target_fps, 1)
    next_tick = time.perf_counter()

    try:
        while not runtime.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                logger.warning("capture_frame_failed camera_id=%s", camera_id)
                break

            encoded_ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not encoded_ok:
                logger.warning("frame_encode_failed camera_id=%s", camera_id)
                continue

            frame_event = {
                # Strict TDD Frame Event contract fields:
                "frame_id": str(uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "camera_id": camera_id,
                "frame_data": base64.b64encode(buffer.tobytes()).decode("ascii"),
            }

            runtime.redis.xadd(FRAME_STREAM, frame_event, maxlen=10000, approximate=True)

            now = time.perf_counter()
            next_tick += frame_interval
            sleep_s = max(0.0, next_tick - now)
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = now
    except Exception:
        logger.exception("capture_loop_error camera_id=%s", camera_id)
    finally:
        cap.release()
        with runtime.lock:
            runtime.running = False
        logger.info("capture_stopped camera_id=%s", camera_id)


@app.on_event("startup")
def startup() -> None:
    """Initialize Redis client at service startup."""
    runtime.configure_redis()


@app.on_event("shutdown")
def shutdown() -> None:
    """Stop background capture loop and close Redis client."""
    runtime.stop_event.set()
    if runtime.capture_thread and runtime.capture_thread.is_alive():
        runtime.capture_thread.join(timeout=2)
    if runtime.redis is not None:
        runtime.redis.close()


@app.post("/start", tags=["ingestion"])
async def start_ingestion(request: StartRequest) -> dict[str, str | int]:
    """Start frame capture from webcam/video and publish to Redis frame_stream."""
    if runtime.redis is None:
        raise HTTPException(status_code=500, detail="Redis not initialized")

    with runtime.lock:
        if runtime.running:
            raise HTTPException(status_code=409, detail="Ingestion already running")

        runtime.stop_event.clear()
        runtime.running = True
        runtime.camera_id = request.camera_id
        runtime.source = _coerce_source(request.source)
        runtime.target_fps = request.fps

        runtime.capture_thread = threading.Thread(
            target=_capture_and_publish,
            args=(request.camera_id, runtime.source, request.fps, request.jpeg_quality),
            daemon=True,
        )
        runtime.capture_thread.start()

    logger.info(
        "start_requested camera_id=%s source=%r fps=%s",
        request.camera_id,
        request.source,
        request.fps,
    )
    return {
        "status": "started",
        "camera_id": request.camera_id,
        "stream": FRAME_STREAM,
        "fps": request.fps,
    }


@app.get("/health", tags=["system"])
async def health() -> dict[str, str | bool]:
    """Service liveness + Redis readiness state."""
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
        "capturing": runtime.running,
    }


@app.get("/status", tags=["system"])
async def status() -> dict[str, str | int | bool]:
    """Readiness endpoint containing service metadata and capture state."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "state": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stream": FRAME_STREAM,
        "capturing": runtime.running,
        "camera_id": runtime.camera_id,
        "target_fps": runtime.target_fps,
    }