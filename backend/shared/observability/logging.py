"""Centralized structured logging setup for all services."""

from __future__ import annotations

import logging

import structlog


def configure_logging() -> None:
    """Configure stdlib + structlog for JSON-like structured output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )