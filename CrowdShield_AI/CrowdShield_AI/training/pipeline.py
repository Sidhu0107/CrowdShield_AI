"""
CrowdShield AI — End-to-end single-script detection pipeline.

Reads a video source (file path, RTSP URL, or webcam index), runs YOLOv8
person detection + ByteTrack tracking, overlays bounding boxes with stable
person IDs, and displays the annotated result in a live window.

Usage:
    python pipeline.py                         # webcam 0
    python pipeline.py --source path/to/video.mp4
    python pipeline.py --source rtsp://...
    python pipeline.py --model yolov8n.pt --conf 0.4

Development order (per TDD section 10):
    1. Validate full pipeline in this single script  ← you are here
    2. Split into microservices once flow is verified
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterator

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import torch
from torch import Tensor
from ultralytics import YOLO

from scripts.custom_lstm import CustomLSTMClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    person_id: int = -1  # ByteTrack assigned ID; -1 = not yet confirmed


@dataclass
class DetectionResult:
    """All person detections produced for a single frame."""
    frame: np.ndarray
    boxes: list[BoundingBox] = field(default_factory=list)


@dataclass
class PoseResult:
    """Pose keypoints for one detected person in full-frame coordinates."""
    person_id: int
    keypoints: list[tuple[int, int, float]] = field(default_factory=list)


@dataclass
class FeatureResult:
    """Feature vector computed for one person in one frame."""
    person_id: int
    feature_vector: list[float] = field(default_factory=list)


@dataclass
class SequenceResult:
    """LSTM-ready feature sequence for one tracked person."""
    person_id: int
    sequence: np.ndarray  # Shape: [window_size, feature_dim]


@dataclass
class PredictionResult:
    """Behavior classification output for one tracked person."""
    person_id: int
    class_name: str
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# VideoReader
# ─────────────────────────────────────────────────────────────────────────────

class VideoReader:
    """
    Thin wrapper around OpenCV VideoCapture.

    Handles file paths, RTSP URLs, and webcam indices uniformly.
    Acts as a context manager and an iterator so the caller can simply do:

        with VideoReader(source) as reader:
            for frame in reader:
                ...

    Attributes:
        source: Video source passed to cv2.VideoCapture.
        fps:    Capture FPS reported by the source (read-only after open).
        width:  Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, source: str | int = 0) -> None:
        self.source = source
        self._cap: cv2.VideoCapture | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        self._assert_open()
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        self._assert_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        self._assert_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the capture device. Raises RuntimeError if it fails."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")

    def close(self) -> None:
        """Release the underlying capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Iterator ──────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield decoded BGR frames until the source is exhausted or 'q' quits."""
        self._assert_open()
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_open(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoReader is not open. Call open() first.")


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class Detector:
    """
    YOLOv8 + ByteTrack person detector backed by Ultralytics.

    Uses model.track() with persist=True so the ByteTrack state is preserved
    across successive calls, yielding stable person_id values that match the
    Detection Event contract defined in TDD §3.2.

    Only the 'person' class (COCO class 0) is returned.
    Ultralytics downloads model weights automatically on first use.

    Args:
        model_path: Path to a .pt weight file or a model name such as
                    'yolov8n.pt', 'yolov8s.pt', etc.
        confidence: Minimum detection confidence threshold (0–1).
        device:     Inference device ('cpu', 'cuda', 'mps', or '' for auto).
    """

    # COCO class index for "person"
    _PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        device: str = "",
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self._model: YOLO | None = None

    def load(self) -> None:
        """Load (or download) the YOLO weights. Call once before detect()."""
        print(f"[Detector] Loading model: {self.model_path}")
        self._model = YOLO(self.model_path)
        print("[Detector] Model ready.")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run YOLOv8 detection + ByteTrack tracking on a single BGR frame.

        persist=True is the key flag: it keeps the ByteTrack state object alive
        between calls so track IDs are stable across the full video sequence
        rather than reassigned each frame.

        Args:
            frame: Raw BGR image as a NumPy array (H, W, 3).

        Returns:
            DetectionResult with boxes that carry stable person_id values.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        # track() runs YOLO inference then feeds detections into ByteTrack.
        # persist=True preserves tracker state between successive calls.
        results = self._model.track(
            source=frame,
            conf=self.confidence,
            classes=[self._PERSON_CLASS_ID],
            tracker="bytetrack.yaml",
            persist=True,
            device=self.device,
            verbose=False,
        )

        boxes: list[BoundingBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                # box.id is None for detections ByteTrack has not yet confirmed
                # as stable tracks (typically the first 1-2 frames of a new ID).
                person_id = int(box.id[0]) if box.id is not None else -1
                boxes.append(
                    BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf, person_id=person_id,
                    )
                )

        return DetectionResult(frame=frame, boxes=boxes)


# ─────────────────────────────────────────────────────────────────────────────
# PoseEstimator
# ─────────────────────────────────────────────────────────────────────────────

class PoseEstimator:
    def __init__(self, min_detection_confidence: float=0.5, min_tracking_confidence: float=0.5) -> None:
        model_path = "pose_landmarker.task"
        if not os.path.exists(model_path):
            print("[PoseEstimator] Downloading mediapipe task model (one-time setup)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                print(f"Failed to download task model: {e}")

        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False)
            self._detector = vision.PoseLandmarker.create_from_options(options)
        else:
            self._detector = None

    def close(self) -> None:
        if self._detector:
            self._detector.close()

    def estimate(self, frame: np.ndarray, boxes: list[BoundingBox]) -> list[PoseResult]:
        h, w = frame.shape[:2]
        pose_results: list[PoseResult] = []

        if getattr(self, "_detector", None) is None:
            return [PoseResult(person_id=b.person_id, keypoints=[]) for b in boxes]

        for box in boxes:
            x1 = max(0, min(box.x1, w - 1))
            y1 = max(0, min(box.y1, h - 1))
            x2 = max(0, min(box.x2, w - 1))
            y2 = max(0, min(box.y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                pose_results.append(PoseResult(person_id=box.person_id, keypoints=[]))
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                pose_results.append(PoseResult(person_id=box.person_id, keypoints=[]))
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
            mp_result = self._detector.detect(mp_image)

            if not mp_result.pose_landmarks:
                pose_results.append(PoseResult(person_id=box.person_id, keypoints=[]))
                continue

            roi_h, roi_w = roi.shape[:2]
            keypoints = []

            for lm in mp_result.pose_landmarks[0]:
                px = max(0, min(int(x1 + (lm.x * roi_w)), w - 1))
                py = max(0, min(int(y1 + (lm.y * roi_h)), h - 1))
                # New Tasks API uses visibility and presence
                keypoints.append((px, py, float(lm.visibility or lm.presence)))

            pose_results.append(PoseResult(person_id=box.person_id, keypoints=keypoints))

        return pose_results


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Compute per-person features for each frame.

    Features currently include:
    - joint angles (upper/lower body)
    - velocity between frames (from tracked bbox center)
    - simple crowd density (people per normalized frame area)

    Output ordering follows detection ordering so box[i] -> feature[i] mapping
    remains stable for downstream service splitting.
    """

    # MediaPipe Pose landmark indices (33-point model)
    _L_SHOULDER = 11
    _R_SHOULDER = 12
    _L_ELBOW = 13
    _R_ELBOW = 14
    _L_WRIST = 15
    _R_WRIST = 16
    _L_HIP = 23
    _R_HIP = 24
    _L_KNEE = 25
    _R_KNEE = 26
    _L_ANKLE = 27
    _R_ANKLE = 28

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps if fps > 0 else 30.0
        self._prev_centers: dict[int, tuple[float, float]] = {}

    def extract(
        self,
        boxes: list[BoundingBox],
        poses: list[PoseResult],
        frame_shape: tuple[int, int, int],
    ) -> list[FeatureResult]:
        """Return a feature vector for each detected person in the current frame."""
        frame_h, frame_w = frame_shape[:2]
        frame_diag = float(np.hypot(frame_w, frame_h)) if frame_w > 0 and frame_h > 0 else 1.0

        # Simple density: number of detections normalized by image area.
        # Scale factor keeps values in a practical numeric range.
        crowd_density = (len(boxes) / float(max(frame_w * frame_h, 1))) * 100000.0

        results: list[FeatureResult] = []

        for idx, box in enumerate(boxes):
            pose = poses[idx] if idx < len(poses) else PoseResult(person_id=box.person_id, keypoints=[])

            # Joint-angle features in degrees.
            angles = self._compute_joint_angles(pose)

            # Velocity feature in normalized-diagonal units per second.
            velocity = self._compute_velocity(box, frame_diag)

            feature_vector = angles + [velocity, crowd_density]
            results.append(FeatureResult(person_id=box.person_id, feature_vector=feature_vector))

        return results

    def _compute_joint_angles(self, pose: PoseResult) -> list[float]:
        """Compute robust joint angles; returns 0.0 for missing/low-confidence joints."""
        return [
            self._safe_angle(pose, self._L_SHOULDER, self._L_ELBOW, self._L_WRIST),
            self._safe_angle(pose, self._R_SHOULDER, self._R_ELBOW, self._R_WRIST),
            self._safe_angle(pose, self._L_ELBOW, self._L_SHOULDER, self._L_HIP),
            self._safe_angle(pose, self._R_ELBOW, self._R_SHOULDER, self._R_HIP),
            self._safe_angle(pose, self._L_HIP, self._L_KNEE, self._L_ANKLE),
            self._safe_angle(pose, self._R_HIP, self._R_KNEE, self._R_ANKLE),
            self._safe_angle(pose, self._L_SHOULDER, self._L_HIP, self._L_KNEE),
            self._safe_angle(pose, self._R_SHOULDER, self._R_HIP, self._R_KNEE),
        ]

    def _safe_angle(self, pose: PoseResult, i: int, j: int, k: int) -> float:
        """Angle at landmark j formed by i-j-k, in degrees [0, 180]."""
        if len(pose.keypoints) != 33:
            return 0.0

        ax, ay, av = pose.keypoints[i]
        bx, by, bv = pose.keypoints[j]
        cx, cy, cv = pose.keypoints[k]
        if av < _POSE_MIN_VISIBILITY or bv < _POSE_MIN_VISIBILITY or cv < _POSE_MIN_VISIBILITY:
            return 0.0

        return self._angle_from_points((ax, ay), (bx, by), (cx, cy))

    @staticmethod
    def _angle_from_points(
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
    ) -> float:
        """Compute the angle ABC in degrees."""
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)

        ba_norm = float(np.linalg.norm(ba))
        bc_norm = float(np.linalg.norm(bc))
        if ba_norm == 0.0 or bc_norm == 0.0:
            return 0.0

        cos_angle = float(np.dot(ba, bc) / (ba_norm * bc_norm))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return float(np.degrees(np.arccos(cos_angle)))

    def _compute_velocity(self, box: BoundingBox, frame_diag: float) -> float:
        """Track-aware velocity using bbox center displacement between frames."""
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0

        # Without stable track IDs, temporal velocity is not reliable.
        if box.person_id < 0:
            return 0.0

        prev = self._prev_centers.get(box.person_id)
        self._prev_centers[box.person_id] = (cx, cy)
        if prev is None:
            return 0.0

        dx = cx - prev[0]
        dy = cy - prev[1]
        displacement = float(np.hypot(dx, dy))
        return (displacement / max(frame_diag, 1.0)) * self.fps


# ─────────────────────────────────────────────────────────────────────────────
# SequenceBuilder
# ─────────────────────────────────────────────────────────────────────────────

class SequenceBuilder:
    """
    Maintain per-person sliding windows and emit LSTM-ready sequences.

    Window semantics:
    - One buffer per stable ByteTrack person_id.
    - Each new frame appends one feature vector to that person's buffer.
    - Buffer max length is fixed (default 30), so it naturally slides.
    - Once full, every new frame emits a fresh sequence for inference.
    """

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._buffers: dict[int, Deque[list[float]]] = {}

    def update(self, features: list[FeatureResult]) -> list[SequenceResult]:
        """
        Update buffers with current-frame features and return ready sequences.

        Returns:
            List of sequences that are full and ready for LSTM input.
            Each sequence has shape [window_size, feature_dim].
        """
        ready_sequences: list[SequenceResult] = []
        active_ids: set[int] = set()

        for feature in features:
            person_id = feature.person_id
            # Sequence continuity requires stable track IDs.
            if person_id < 0:
                continue

            active_ids.add(person_id)
            if person_id not in self._buffers:
                self._buffers[person_id] = deque(maxlen=self.window_size)

            self._buffers[person_id].append(feature.feature_vector)

            if len(self._buffers[person_id]) == self.window_size:
                sequence = np.array(self._buffers[person_id], dtype=np.float32)
                ready_sequences.append(SequenceResult(person_id=person_id, sequence=sequence))

        # Keep buffers only for currently visible tracked people.
        # This avoids memory growth and stale sequences for departed tracks.
        stale_ids = [pid for pid in self._buffers if pid not in active_ids]
        for pid in stale_ids:
            del self._buffers[pid]

        return ready_sequences


# ─────────────────────────────────────────────────────────────────────────────
# BehaviorPredictor
# ─────────────────────────────────────────────────────────────────────────────

class BehaviorPredictor:
    """Run trained custom LSTM inference on 30-frame sequences."""

    def __init__(
        self,
        model_path: str,
        input_size: int,
        hidden_size: int = 128,
        class_names: list[str] | None = None,
        device: str = "",
    ) -> None:
        self.class_names = class_names or ["normal", "violence", "fighting", "stampede"]
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enabled = False

        self._model = CustomLSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=len(self.class_names),
        ).to(self.device)
        self._model.eval()

        resolved_path = self._resolve_model_path(model_path)
        if resolved_path is None:
            print(f"[BehaviorPredictor] Model file not found: {model_path!r}. Predictions disabled.")
            return

        state_dict = torch.load(resolved_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self.enabled = True
        print(f"[BehaviorPredictor] Loaded model: {resolved_path}")

    def _resolve_model_path(self, model_path: str) -> Path | None:
        """Resolve model path against common runtime locations."""
        candidates = [
            Path(model_path),
            Path(__file__).resolve().parent / model_path,
            Path(__file__).resolve().parent / "scripts" / "model.pt",
            Path(__file__).resolve().parent / "model.pt",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @torch.no_grad()
    def predict(self, sequences: list[SequenceResult]) -> dict[int, PredictionResult]:
        """Predict class/confidence for each ready sequence."""
        if not self.enabled or not sequences:
            return {}

        predictions: dict[int, PredictionResult] = {}
        for seq in sequences:
            x = torch.from_numpy(seq.sequence).float().unsqueeze(0).to(self.device)  # [1, 30, feat_dim]
            logits: Tensor = self._model(x)
            probs: Tensor = torch.softmax(logits, dim=1)

            cls_idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, cls_idx].item())
            predictions[seq.person_id] = PredictionResult(
                person_id=seq.person_id,
                class_name=self.class_names[cls_idx],
                confidence=conf,
            )

        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

# ── Visual constants ───────────────────────────────────────────────────────
_LABEL_BG   = (30, 32, 50)
_LABEL_TEXT = (248, 250, 252)
_BOX_THICK  = 2
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICK = 1
_POSE_MIN_VISIBILITY = 0.5

# Distinct BGR colours cycling per person_id for visual differentiation.
# Each ID maps to the same colour on every frame for perceptual stability.
_PALETTE: list[tuple[int, int, int]] = [
    ( 99, 102, 241),  # indigo
    ( 16, 185, 129),  # emerald
    (245, 158,  11),  # amber
    (239,  68,  68),  # red
    ( 56, 189, 248),  # sky
    (168,  85, 247),  # violet
    (249, 115,  22),  # orange
    ( 20, 184, 166),  # teal
]

_POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]


def _draw_pose_skeleton(
    frame: np.ndarray,
    pose_result: PoseResult,
    color: tuple[int, int, int],
) -> None:
    """Draw MediaPipe skeleton and visible landmarks for one detected person."""
    if len(pose_result.keypoints) != 33:
        return

    # Draw limbs first so keypoints appear above them.
    for start_idx, end_idx in _POSE_CONNECTIONS:
        sx, sy, sv = pose_result.keypoints[start_idx]
        ex, ey, ev = pose_result.keypoints[end_idx]
        if sv >= _POSE_MIN_VISIBILITY and ev >= _POSE_MIN_VISIBILITY:
            cv2.line(frame, (sx, sy), (ex, ey), color, 2, cv2.LINE_AA)

    # Draw visible landmarks.
    for x, y, vis in pose_result.keypoints:
        if vis >= _POSE_MIN_VISIBILITY:
            cv2.circle(frame, (x, y), 2, color, cv2.FILLED, cv2.LINE_AA)


def _id_color(person_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a given ByteTrack person_id."""
    if person_id < 0:
        return (160, 160, 160)  # grey for unconfirmed / pending tracks
    return _PALETTE[person_id % len(_PALETTE)]


def draw_detections(
    result: DetectionResult,
    poses: list[PoseResult],
    predictions: dict[int, PredictionResult] | None = None,
) -> np.ndarray:
    """
    Overlay per-person bounding boxes and ID labels onto the frame.

    Each person_id receives a consistent colour so identities are visually
    distinguishable across frames. Labels show "#ID  conf" matching the
    Detection Event contract (TDD §3.2: person_id + bbox + confidence).

    Returns a new annotated copy; the original frame is not modified.
    """
    annotated = result.frame.copy()

    for idx, box in enumerate(result.boxes):
        color = _id_color(box.person_id)
        prediction = predictions.get(box.person_id) if predictions else None

        # Draw skeleton mapped from this exact detection ROI before box overlay.
        if idx < len(poses):
            _draw_pose_skeleton(annotated, poses[idx], color)

        # Bounding rectangle drawn in the person's assigned colour
        cv2.rectangle(
            annotated,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color,
            _BOX_THICK,
        )

        # Label: "#ID  conf" — mirrors Detection Event person_id field
        id_str = str(box.person_id) if box.person_id >= 0 else "?"
        # Include keypoint count to verify pose extraction quality per track.
        kp_count = len(poses[idx].keypoints) if idx < len(poses) else 0
        label = f"#{id_str}  {box.confidence:.2f}  kp:{kp_count}"
        if prediction is not None:
            label += f"  {prediction.class_name}:{prediction.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        label_y = max(box.y1 - 4, th + 4)  # stay inside frame top edge

        cv2.rectangle(
            annotated,
            (box.x1, label_y - th - baseline - 2),
            (box.x1 + tw + 4, label_y + 2),
            _LABEL_BG,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (box.x1 + 2, label_y - baseline),
            _FONT,
            _FONT_SCALE,
            _LABEL_TEXT,
            _FONT_THICK,
            cv2.LINE_AA,
        )

    # Frame-level overlay: count of confirmed tracked persons only
    confirmed = sum(1 for b in result.boxes if b.person_id >= 0)
    cv2.putText(
        annotated,
        f"Tracked: {confirmed}",
        (10, 24),
        _FONT,
        0.65,
        _PALETTE[0],  # indigo
        2,
        cv2.LINE_AA,
    )

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    source: str | int,
    model_path: str,
    confidence: float,
    device: str,
    lstm_model_path: str,
) -> None:
    """
    Main pipeline loop: read → detect → draw → display.

    Press 'q' in the display window to exit cleanly.
    """
    detector = Detector(model_path=model_path, confidence=confidence, device=device)
    detector.load()
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
    sequence_builder = SequenceBuilder(window_size=30)
    behavior_predictor = BehaviorPredictor(
        model_path=lstm_model_path,
        input_size=10,
        hidden_size=128,
        class_names=["normal", "violence", "fighting", "stampede"],
        device=device,
    )

    print(f"[Pipeline] Opening source: {source!r}")
    print("[Pipeline] Press 'q' to quit.")

    try:
        with VideoReader(source) as reader:
            print(f"[Pipeline] Source opened — {reader.width}x{reader.height} @ {reader.fps:.1f} FPS")
            feature_extractor.fps = reader.fps if reader.fps > 0 else 30.0

            for frame in reader:
                result = detector.detect(frame)
                poses = pose_estimator.estimate(frame, result.boxes)
                features = feature_extractor.extract(result.boxes, poses, frame.shape)
                sequences = sequence_builder.update(features)
                predictions = behavior_predictor.predict(sequences)
                annotated = draw_detections(result, poses, predictions)

                # `features` now contains one feature vector per detected person.
                # This is the exact payload source for the future feature stream.
                if features:
                    first = features[0]
                    cv2.putText(
                        annotated,
                        f"fv[{first.person_id}]: {len(first.feature_vector)} dims",
                        (10, 48),
                        _FONT,
                        0.55,
                        _LABEL_TEXT,
                        1,
                        cv2.LINE_AA,
                    )

                # `sequences` contains only full sliding windows [30, feature_dim].
                # This is directly consumable by the behavior-service LSTM model.
                if sequences:
                    first_seq = sequences[0]
                    seq_len, feat_dim = first_seq.sequence.shape
                    cv2.putText(
                        annotated,
                        f"seq[{first_seq.person_id}]: {seq_len}x{feat_dim}",
                        (10, 70),
                        _FONT,
                        0.55,
                        _LABEL_TEXT,
                        1,
                        cv2.LINE_AA,
                    )

                # Print class + confidence on frame for the most recent predictions.
                if predictions:
                    y_offset = 92
                    for person_id, pred in sorted(predictions.items()):
                        cv2.putText(
                            annotated,
                            f"pred[{person_id}] = {pred.class_name} ({pred.confidence:.2f})",
                            (10, y_offset),
                            _FONT,
                            0.52,
                            _LABEL_TEXT,
                            1,
                            cv2.LINE_AA,
                        )
                        y_offset += 18

                cv2.imshow("CrowdShield AI — Detection + Pose Pipeline", annotated)

                # 'q' exits; waitKey delay keeps window responsive without busy-looping
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Pipeline] User requested exit.")
                    break
    finally:
        pose_estimator.close()
        cv2.destroyAllWindows()

    print("[Pipeline] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrowdShield AI — single-script detection pipeline"
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Video source: file path, RTSP URL, or webcam index (default: 0)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model name or path to .pt file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold 0–1 (default: 0.35)",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Inference device: 'cpu', 'cuda', 'mps', or '' for auto (default: '')",
    )
    parser.add_argument(
        "--lstm-model",
        default="scripts/model.pt",
        help="Path to trained custom LSTM weights (default: scripts/model.pt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Convert numeric strings to int so cv2.VideoCapture treats them as webcam indices
    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run_pipeline(
        source=source,
        model_path=args.model,
        confidence=args.conf,
        device=args.device,
        lstm_model_path=args.lstm_model,
    )
