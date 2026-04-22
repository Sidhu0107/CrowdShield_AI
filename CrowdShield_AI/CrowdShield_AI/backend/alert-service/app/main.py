"""Alert service entrypoint.

Responsibilities (TDD §1.5):
- Consume Behavior Events from Redis stream: behavior_stream
- Apply temporal smoothing (N-frame persistence rule)
- Generate Alert Events
- Store alerts in PostgreSQL
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
from fastapi import FastAPI
from redis.asyncio import Redis

SERVICE_NAME = os.getenv("SERVICE_NAME", "alert-service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
BEHAVIOR_STREAM = "behavior_stream"
ALERT_STREAM = "alert_stream"

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "crowdshield")
POSTGRES_USER = os.getenv("POSTGRES_USER", "crowdshield")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "change_me")

ALERT_PERSISTENCE_FRAMES = int(os.getenv("ALERT_PERSISTENCE_FRAMES", "10"))
ALERT_CONFIDENCE_THRESHOLD = float(os.getenv("ALERT_CONFIDENCE_THRESHOLD", "0.60"))
DEFAULT_CAMERA_ID = os.getenv("DEFAULT_CAMERA_ID", "cam_1")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


class AlertRuntime:
    """Mutable runtime state for stream consumption and smoothing."""

    def __init__(self) -> None:
        self.redis: Redis | None = None
        self.db_pool: asyncpg.Pool | None = None
        self.worker_task: asyncio.Task[None] | None = None

        self.running = False
        self.last_behavior_id = "$"  # consume only new events after startup

        # Temporal smoothing state:
        # - last prediction per camera
        # - consecutive count for that prediction
        self.last_prediction_by_camera: dict[str, str] = {}
        self.streak_by_camera: dict[str, int] = {}


runtime = AlertRuntime()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


def _database_dsn() -> str:
    """Build PostgreSQL DSN for asyncpg."""
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


def _severity_from_confidence(confidence: float) -> str:
    """Map confidence score to alert severity."""
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


def _apply_smoothing(camera_id: str, prediction: str, confidence: float) -> bool:
    """Apply N-frame persistence rule and return whether to emit an alert."""
    if prediction == "normal" or confidence < ALERT_CONFIDENCE_THRESHOLD:
        runtime.last_prediction_by_camera[camera_id] = "normal"
        runtime.streak_by_camera[camera_id] = 0
        return False

    last_pred = runtime.last_prediction_by_camera.get(camera_id)
    if last_pred == prediction:
        runtime.streak_by_camera[camera_id] = runtime.streak_by_camera.get(camera_id, 0) + 1
    else:
        runtime.last_prediction_by_camera[camera_id] = prediction
        runtime.streak_by_camera[camera_id] = 1

    # Emit when anomaly persists for N frames.
    return runtime.streak_by_camera[camera_id] == ALERT_PERSISTENCE_FRAMES


async def _ensure_alerts_table() -> None:
    """Create alerts table if it does not already exist."""
    assert runtime.db_pool is not None
    async with runtime.db_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                camera_id TEXT NOT NULL,
                sequence_id TEXT NOT NULL
            )
            """
        )


async def _insert_alert(
    alert_id: str,
    alert_type: str,
    severity: str,
    confidence: float,
    timestamp_iso: str,
    camera_id: str,
    sequence_id: str,
) -> None:
    """Persist one alert record in PostgreSQL."""
    assert runtime.db_pool is not None
    async with runtime.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO alerts (alert_id, type, severity, confidence, timestamp, camera_id, sequence_id)
            VALUES ($1, $2, $3, $4, $5::timestamptz, $6, $7)
            """,
            alert_id,
            alert_type,
            severity,
            confidence,
            timestamp_iso,
            camera_id,
            sequence_id,
        )


async def _consumer_loop() -> None:
    """Consume behavior events, smooth, emit alerts, and store in DB."""
    assert runtime.redis is not None
    runtime.running = True
    logger.info("consumer_started input_stream=%s output_stream=%s", BEHAVIOR_STREAM, ALERT_STREAM)

    try:
        while True:
            events = await runtime.redis.xread(
                streams={BEHAVIOR_STREAM: runtime.last_behavior_id},
                block=1000,
                count=50,
            )
            if not events:
                continue

            for _stream_name, entries in events:
                for entry_id, fields in entries:
                    runtime.last_behavior_id = entry_id

                    sequence_id = str(fields.get("sequence_id", ""))
                    prediction = str(fields.get("prediction", "normal"))
                    confidence_raw = fields.get("confidence", "0")
                    camera_id = str(fields.get("camera_id", DEFAULT_CAMERA_ID))

                    try:
                        confidence = float(confidence_raw)
                    except (TypeError, ValueError):
                        logger.warning("invalid_confidence entry_id=%s value=%r", entry_id, confidence_raw)
                        continue

                    if not _apply_smoothing(camera_id, prediction, confidence):
                        continue

                    alert_id = str(uuid4())
                    timestamp_iso = datetime.now(timezone.utc).isoformat()
                    severity = _severity_from_confidence(confidence)

                    # TDD Alert Event contract.
                    alert_event = {
                        "alert_id": alert_id,
                        "type": prediction,
                        "severity": severity,
                        "confidence": f"{confidence:.6f}",
                        "timestamp": timestamp_iso,
                        "camera_id": camera_id,
                    }

                    await runtime.redis.xadd(ALERT_STREAM, alert_event, maxlen=10000, approximate=True)
                    await _insert_alert(
                        alert_id=alert_id,
                        alert_type=prediction,
                        severity=severity,
                        confidence=confidence,
                        timestamp_iso=timestamp_iso,
                        camera_id=camera_id,
                        sequence_id=sequence_id,
                    )

                    logger.info(
                        "alert_generated alert_id=%s prediction=%s severity=%s confidence=%.4f camera_id=%s",
                        alert_id,
                        prediction,
                        severity,
                        confidence,
                        camera_id,
                    )
    except asyncio.CancelledError:
        logger.info("consumer_cancelled")
        raise
    except Exception:
        logger.exception("consumer_loop_failed")
    finally:
        runtime.running = False
        logger.info("consumer_stopped")


@app.on_event("startup")
async def startup() -> None:
    """Initialize Redis + PostgreSQL and start consumer task."""
    runtime.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    await runtime.redis.ping()

    runtime.db_pool = await asyncpg.create_pool(dsn=_database_dsn(), min_size=1, max_size=5)
    await _ensure_alerts_table()

    runtime.worker_task = asyncio.create_task(_consumer_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    """Stop consumer and close external resources."""
    if runtime.worker_task is not None:
        runtime.worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runtime.worker_task

    if runtime.redis is not None:
        await runtime.redis.close()
    if runtime.db_pool is not None:
        await runtime.db_pool.close()


@app.get("/health", tags=["system"])
async def health() -> dict[str, str | bool]:
    """Liveness and readiness snapshot for alert service dependencies."""
    redis_status = "down"
    db_status = "down"

    if runtime.redis is not None:
        try:
            await runtime.redis.ping()
            redis_status = "up"
        except Exception:
            redis_status = "down"

    if runtime.db_pool is not None:
        try:
            async with runtime.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_status = "up"
        except Exception:
            db_status = "down"

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "redis": redis_status,
        "database": db_status,
        "running": runtime.running,
    }


@app.get("/status", tags=["system"])
async def status() -> dict[str, str | int | bool]:
    """Readiness endpoint containing service metadata."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "state": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_stream": BEHAVIOR_STREAM,
        "output_stream": ALERT_STREAM,
        "smoothing_frames": ALERT_PERSISTENCE_FRAMES,
        "running": runtime.running,
    }