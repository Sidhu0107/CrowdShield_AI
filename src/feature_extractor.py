"""Offline feature extraction utilities for training data preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe
import numpy as np


def extract_features_from_video(video_path: str, output_json_path: str) -> None:
    """Extract per-frame pose features from a video and save them as JSON.

    This helper is designed for offline batch preprocessing of training videos.
    It runs MediaPipe pose extraction on each frame and stores a frame-wise
    feature list that can later be grouped into sequences for model training.

    Args:
        video_path: Path to the source video file.
        output_json_path: Path to the output JSON file.

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    video_path_obj = Path(video_path)
    output_path_obj = Path(output_json_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    pose = mediapipe.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_features: list[list[float]] = []
    total_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks is None:
            vec = np.zeros((118,), dtype=np.float32)
        else:
            keypoints = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                dtype=np.float32,
            )
            key_flat = keypoints.reshape(-1)[:99]

            # Build simple 17-angle orientation features from landmark pair vectors.
            idx = mediapipe.solutions.pose.PoseLandmark
            pairs = [
                (idx.LEFT_SHOULDER.value, idx.LEFT_ELBOW.value),
                (idx.LEFT_ELBOW.value, idx.LEFT_WRIST.value),
                (idx.RIGHT_SHOULDER.value, idx.RIGHT_ELBOW.value),
                (idx.RIGHT_ELBOW.value, idx.RIGHT_WRIST.value),
                (idx.LEFT_HIP.value, idx.LEFT_KNEE.value),
                (idx.LEFT_KNEE.value, idx.LEFT_ANKLE.value),
                (idx.RIGHT_HIP.value, idx.RIGHT_KNEE.value),
                (idx.RIGHT_KNEE.value, idx.RIGHT_ANKLE.value),
                (idx.LEFT_SHOULDER.value, idx.LEFT_HIP.value),
                (idx.RIGHT_SHOULDER.value, idx.RIGHT_HIP.value),
                (idx.LEFT_HIP.value, idx.RIGHT_HIP.value),
                (idx.LEFT_SHOULDER.value, idx.RIGHT_SHOULDER.value),
                (idx.NOSE.value, idx.LEFT_SHOULDER.value),
                (idx.NOSE.value, idx.RIGHT_SHOULDER.value),
                (idx.LEFT_HIP.value, idx.LEFT_ANKLE.value),
                (idx.RIGHT_HIP.value, idx.RIGHT_ANKLE.value),
                (idx.LEFT_SHOULDER.value, idx.RIGHT_HIP.value),
            ]
            angles = []
            for a, b in pairs:
                dx = float(keypoints[b, 0] - keypoints[a, 0])
                dy = float(keypoints[b, 1] - keypoints[a, 1])
                angles.append(float(np.arctan2(dy, dx)))
            angles = np.asarray(angles, dtype=np.float32)

            if frame_features:
                prev_key = np.asarray(frame_features[-1][:99], dtype=np.float32)
                motion = float(np.mean(np.abs(key_flat - prev_key)))
            else:
                motion = 0.0

            crowd_count = 1.0
            vec = np.concatenate(
                [
                    key_flat,
                    angles,
                    np.asarray([motion], dtype=np.float32),
                    np.asarray([crowd_count], dtype=np.float32),
                ]
            ).astype(np.float32)

            if vec.shape[0] != 118:
                fixed = np.zeros((118,), dtype=np.float32)
                fixed[: min(vec.shape[0], 118)] = vec[:118]
                vec = fixed

        frame_features.append(vec.tolist())

    cap.release()
    pose.close()

    payload: dict[str, Any] = {
        "video_path": str(video_path_obj),
        "total_frames": total_frames,
        "sequences": [frame_features],
    }

    with output_path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
