"""Canonical event contract models shared across services."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FrameEvent(BaseModel):
    """Frame payload emitted by ingestion service."""

    frame_id: str
    timestamp: datetime
    camera_id: str
    frame_data: str


class Detection(BaseModel):
    """A tracked person detection in a single frame."""

    person_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)


class DetectionEvent(BaseModel):
    """Detections emitted by vision service."""

    frame_id: str
    detections: list[Detection]


class Feature(BaseModel):
    """Pose and movement feature vector for one tracked identity."""

    person_id: int
    feature_vector: list[float]


class FeatureEvent(BaseModel):
    """Feature event emitted by pose service."""

    frame_id: str
    features: list[Feature]


class BehaviorEvent(BaseModel):
    """Behavior prediction emitted by behavior service."""

    sequence_id: str
    prediction: Literal["normal", "violence", "fighting", "stampede", "panic"]
    confidence: float = Field(ge=0.0, le=1.0)


class AlertEvent(BaseModel):
    """Alert payload emitted by alert service and consumed by API/UI."""

    alert_id: str
    type: str
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    camera_id: str