"""Pose service entrypoint.

Responsibilities (TDD §1.3):
- Consume Detection Events from Redis stream: detection_stream
- Run MediaPipe Pose on person regions
- Generate per-person feature vectors
- Publish Feature Events to Redis stream: feature_stream
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
from redis import Redis

SERVICE_NAME = os.getenv("SERVICE_NAME", "pose-service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
FRAME_STREAM = "frame_stream"
DETECTION_STREAM = "detection_stream"
FEATURE_STREAM = "feature_stream"

FRAME_CACHE_SIZE = int(os.getenv("FRAME_CACHE_SIZE", "300"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


class PoseRuntime:
    """Holds service-wide runtime state."""

    def __init__(self) -> None:
        self.redis: Redis | None = None
        self.pose: mp.solutions.pose.Pose | None = None

        self.stop_event = threading.Event()
        self.frame_worker: threading.Thread | None = None
        self.detect_worker: threading.Thread | None = None

        self.last_frame_id = "$"
        self.last_detection_id = "$"

        self.frame_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_lock = threading.Lock()

        self.running = False


runtime = PoseRuntime()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


def _decode_frame(frame_data_b64: str) -> np.ndarray | None:
    """Decode frame_data base64 to OpenCV BGR frame."""
    try:
        raw = base64.b64decode(frame_data_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        logger.exception("frame_decode_failed")
        return None


def _cache_frame(frame_id: str, frame: np.ndarray) -> None:
    """Insert frame into bounded LRU cache keyed by frame_id."""
    with runtime.cache_lock:
        runtime.frame_cache[frame_id] = frame
        runtime.frame_cache.move_to_end(frame_id)
        while len(runtime.frame_cache) > FRAME_CACHE_SIZE:
            runtime.frame_cache.popitem(last=False)


def _get_frame(frame_id: str) -> np.ndarray | None:
    """Fetch frame from cache by frame_id."""
    with runtime.cache_lock:
        frame = runtime.frame_cache.get(frame_id)
        if frame is not None:
            runtime.frame_cache.move_to_end(frame_id)
        return frame


def _compute_feature_vector(
    keypoints: list[tuple[float, float, float]],
    bbox: list[float],
    frame_w: int,
    frame_h: int,
) -> list[float]:
    """Build compact feature vector from pose keypoints + bbox geometry."""
    x, y, w, h = bbox
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)

    # Normalize bbox geometry.
    fw = float(max(frame_w, 1))
    fh = float(max(frame_h, 1))
    area = float(max(frame_w * frame_h, 1))

    bbox_features = [
        cx / fw,
        cy / fh,
        w / fw,
        h / fh,
        (w * h) / area,
    ]

    # Aggregate pose signal; 0.0 if pose missing.
    if not keypoints:
        pose_features = [0.0, 0.0, 0.0, 0.0]
    else:
        vis = [kp[2] for kp in keypoints]
        xs = [kp[0] for kp in keypoints]
        ys = [kp[1] for kp in keypoints]
        pose_features = [
            float(np.mean(vis)),
            float(np.std(vis)),
            float(np.std(xs) / fw),
            float(np.std(ys) / fh),
        ]

    return bbox_features + pose_features


def _run_pose_on_roi(frame: np.ndarray, bbox: list[float]) -> list[tuple[float, float, float]]:
    """Run MediaPipe Pose on detection ROI and return 33 frame-space keypoints."""
    assert runtime.pose is not None

    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = bbox

    x1 = int(max(0, min(frame_w - 1, x)))
    y1 = int(max(0, min(frame_h - 1, y)))
    x2 = int(max(0, min(frame_w, x + w)))
    y2 = int(max(0, min(frame_h, y + h)))

    if x2 <= x1 or y2 <= y1:
        return []

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = runtime.pose.process(roi_rgb)
    if not result.pose_landmarks:
        return []

    roi_h, roi_w = roi.shape[:2]
    keypoints: list[tuple[float, float, float]] = []

    for lm in result.pose_landmarks.landmark:
        px = float(x1 + (lm.x * roi_w))
        py = float(y1 + (lm.y * roi_h))
        keypoints.append((px, py, float(lm.visibility)))

    return keypoints


def _frame_consumer_loop() -> None:
    """Continuously consume frame_stream and keep recent frames cached."""
    assert runtime.redis is not None

    logger.info("frame_consumer_started input_stream=%s", FRAME_STREAM)
    try:
        while not runtime.stop_event.is_set():
            events = runtime.redis.xread({FRAME_STREAM: runtime.last_frame_id}, block=1000, count=30)
            if not events:
                continue

            for _stream, entries in events:
                for entry_id, fields in entries:
                    runtime.last_frame_id = entry_id
                    frame_id = fields.get("frame_id")
                    frame_data = fields.get("frame_data")
                    if not frame_id or not frame_data:
                        continue

                    frame = _decode_frame(frame_data)
                    if frame is None:
                        continue
                    _cache_frame(frame_id, frame)
    except Exception:
        logger.exception("frame_consumer_failed")
    finally:
        logger.info("frame_consumer_stopped")


def _detection_consumer_loop() -> None:
    """Consume detection_stream, build features, publish feature_stream."""
    assert runtime.redis is not None

    logger.info("detection_consumer_started input_stream=%s output_stream=%s", DETECTION_STREAM, FEATURE_STREAM)
    runtime.running = True

    try:
        while not runtime.stop_event.is_set():
            events = runtime.redis.xread({DETECTION_STREAM: runtime.last_detection_id}, block=1000, count=30)
            if not events:
                continue

            for _stream, entries in events:
                for entry_id, fields in entries:
                    runtime.last_detection_id = entry_id

                    frame_id = fields.get("frame_id")
                    detections_raw = fields.get("detections")
                    if not frame_id or not detections_raw:
                        logger.warning("invalid_detection_event entry_id=%s", entry_id)
                        continue

                    try:
                        detections = json.loads(detections_raw)
                    except Exception:
                        logger.warning("detection_parse_failed entry_id=%s", entry_id)
                        continue

                    frame = _get_frame(frame_id)
                    if frame is None:
                        logger.warning("frame_not_found frame_id=%s", frame_id)
                        continue

                    frame_h, frame_w = frame.shape[:2]
                    features_payload: list[dict[str, object]] = []

                    for det in detections:
                        person_id = int(det.get("person_id", -1))
                        bbox = det.get("bbox", [])
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            continue

                        keypoints = _run_pose_on_roi(frame, [float(v) for v in bbox])
                        feature_vector = _compute_feature_vector(keypoints, [float(v) for v in bbox], frame_w, frame_h)
                        features_payload.append(
                            {
                                "person_id": person_id,
                                "feature_vector": feature_vector,
                            }
                        )

                    # TDD Feature Event contract:
                    # {
                    #   "frame_id": "uuid",
                    #   "features": [{"person_id": int, "feature_vector": [float]}]
                    # }
                    feature_event = {
                        "frame_id": frame_id,
                        "features": json.dumps(features_payload),
                    }
                    runtime.redis.xadd(FEATURE_STREAM, feature_event, maxlen=10000, approximate=True)
    except Exception:
        logger.exception("detection_consumer_failed")
    finally:
        runtime.running = False
        logger.info("detection_consumer_stopped")


@app.on_event("startup")
def startup() -> None:
    """Initialize Redis and MediaPipe, then start worker threads."""
    runtime.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    runtime.redis.ping()

    runtime.pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    runtime.stop_event.clear()
    runtime.frame_worker = threading.Thread(target=_frame_consumer_loop, daemon=True)
    runtime.detect_worker = threading.Thread(target=_detection_consumer_loop, daemon=True)

    runtime.frame_worker.start()
    runtime.detect_worker.start()


@app.on_event("shutdown")
def shutdown() -> None:
    """Stop workers and close resources."""
    runtime.stop_event.set()

    if runtime.frame_worker and runtime.frame_worker.is_alive():
        runtime.frame_worker.join(timeout=2)
    if runtime.detect_worker and runtime.detect_worker.is_alive():
        runtime.detect_worker.join(timeout=2)

    if runtime.pose is not None:
        runtime.pose.close()
    if runtime.redis is not None:
        runtime.redis.close()


@app.get("/health", tags=["system"])
async def health() -> dict[str, str | bool | int]:
    """Liveness endpoint with worker and cache state."""
    redis_status = "down"
    if runtime.redis is not None:
        try:
            runtime.redis.ping()
            redis_status = "up"
        except Exception:
            redis_status = "down"

    with runtime.cache_lock:
        cache_size = len(runtime.frame_cache)

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "redis": redis_status,
        "running": runtime.running,
        "frame_cache_size": cache_size,
    }


@app.get("/status", tags=["system"])
async def status() -> dict[str, str | bool]:
    """Readiness endpoint containing service metadata."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "state": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_stream": DETECTION_STREAM,
        "output_stream": FEATURE_STREAM,
        "running": runtime.running,
    }