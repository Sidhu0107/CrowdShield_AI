"""Shared Redis stream helper abstractions."""

from __future__ import annotations

from redis.asyncio import Redis


def build_redis_client(redis_url: str) -> Redis:
    """Create a Redis client with sensible defaults for stream workloads."""
    return Redis.from_url(redis_url, decode_responses=True)