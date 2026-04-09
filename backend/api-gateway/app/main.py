"""API Gateway entrypoint.

Responsibilities (TDD §1.6):
- Expose REST APIs for alerts and health
- Provide WebSocket stream for live alerts/status updates
- Connect to Redis and PostgreSQL as integration boundaries
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from redis.asyncio import Redis

SERVICE_NAME = os.getenv("SERVICE_NAME", "api-gateway")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
ALERT_STREAM = "alert_stream"

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "crowdshield")
POSTGRES_USER = os.getenv("POSTGRES_USER", "crowdshield")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "change_me")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


class GatewayRuntime:
    """Mutable runtime state for external connections and streaming worker."""

    def __init__(self) -> None:
        self.redis: Redis | None = None
        self.db_pool: asyncpg.Pool | None = None
        self.stream_task: asyncio.Task[None] | None = None
        self.last_alert_id = "$"  # consume only new alerts after startup

        self.ws_clients: set[WebSocket] = set()
        self.ws_lock = asyncio.Lock()


runtime = GatewayRuntime()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


def _database_dsn() -> str:
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


async def _register_ws(websocket: WebSocket) -> None:
    async with runtime.ws_lock:
        runtime.ws_clients.add(websocket)


async def _unregister_ws(websocket: WebSocket) -> None:
    async with runtime.ws_lock:
        runtime.ws_clients.discard(websocket)


async def _broadcast(payload: dict[str, Any]) -> None:
    """Fan out alert/status payload to all connected frontend clients."""
    async with runtime.ws_lock:
        clients = list(runtime.ws_clients)

    if not clients:
        return

    stale: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)

    if stale:
        async with runtime.ws_lock:
            for ws in stale:
                runtime.ws_clients.discard(ws)


def _parse_alert_event(fields: dict[str, str]) -> dict[str, Any]:
    """Normalize Redis stream fields to frontend-friendly JSON payload."""
    confidence_raw = fields.get("confidence", "0")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "alert_id": fields.get("alert_id", ""),
        "type": fields.get("type", ""),
        "severity": fields.get("severity", "low"),
        "confidence": confidence,
        "timestamp": fields.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "camera_id": fields.get("camera_id", "cam_1"),
    }


async def _alert_stream_worker() -> None:
    """Continuously consume alert_stream and broadcast events over WebSocket."""
    assert runtime.redis is not None
    logger.info("stream_worker_started stream=%s", ALERT_STREAM)

    try:
        while True:
            events = await runtime.redis.xread(
                streams={ALERT_STREAM: runtime.last_alert_id},
                block=1000,
                count=100,
            )
            if not events:
                continue

            for _stream_name, entries in events:
                for entry_id, fields in entries:
                    runtime.last_alert_id = entry_id
                    payload = _parse_alert_event(fields)
                    await _broadcast(payload)
    except asyncio.CancelledError:
        logger.info("stream_worker_cancelled")
        raise
    except Exception:
        logger.exception("stream_worker_failed")
    finally:
        logger.info("stream_worker_stopped")


@app.on_event("startup")
async def startup() -> None:
    """Connect Redis and DB, then start live alert stream worker."""
    runtime.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    await runtime.redis.ping()

    runtime.db_pool = await asyncpg.create_pool(dsn=_database_dsn(), min_size=1, max_size=5)

    runtime.stream_task = asyncio.create_task(_alert_stream_worker())


@app.on_event("shutdown")
async def shutdown() -> None:
    """Stop worker and release external resources."""
    if runtime.stream_task is not None:
        runtime.stream_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runtime.stream_task

    if runtime.redis is not None:
        await runtime.redis.close()
    if runtime.db_pool is not None:
        await runtime.db_pool.close()


@app.get("/alerts", tags=["alerts"])
async def get_alerts(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Fetch latest alerts from PostgreSQL for dashboard/table rendering."""
    if runtime.db_pool is None:
        return []

    query = """
        SELECT alert_id, type, severity, confidence, timestamp, camera_id
        FROM alerts
        ORDER BY timestamp DESC
        LIMIT $1
    """
    async with runtime.db_pool.acquire() as conn:
        rows = await conn.fetch(query, limit)

    return [
        {
            "alert_id": row["alert_id"],
            "type": row["type"],
            "severity": row["severity"],
            "confidence": float(row["confidence"]),
            "timestamp": row["timestamp"].isoformat(),
            "camera_id": row["camera_id"],
        }
        for row in rows
    ]


@app.get("/health", tags=["system"])
async def health() -> dict[str, str | int]:
    """Gateway health including Redis/DB integration checks."""
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

    async with runtime.ws_lock:
        ws_clients = len(runtime.ws_clients)

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "redis": redis_status,
        "database": db_status,
        "ws_clients": ws_clients,
    }


@app.websocket("/ws")
async def websocket_alerts(websocket: WebSocket) -> None:
    """Live alert stream endpoint for frontend subscriptions."""
    await websocket.accept()
    await _register_ws(websocket)

    # Send initial status update for frontend service health panel.
    await websocket.send_json(
        {
            "services": {
                "api-gateway": "up",
                "alert-service": "up",
                "vision": "up",
                "behavior": "up",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    try:
        while True:
            # Keep connection open; optional messages from frontend are ignored.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await _unregister_ws(websocket)
    except Exception:
        await _unregister_ws(websocket)


@app.get("/status", tags=["system"])
async def status() -> dict[str, str]:
    """Readiness endpoint containing service metadata."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "state": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }