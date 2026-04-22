#!/usr/bin/env python
"""
test_pipeline.py — CrowdShield AI end-to-end visual pipeline tester.

Runs the full detection → pose → feature → LSTM pipeline on a webcam or video
file, overlaying all visual debug information and printing per-frame stats to
the console so every stage can be verified at a glance.

Displays on screen:
  • Bounding boxes per tracked person (colour-coded by ID)
  • ByteTrack tracking IDs
  • MediaPipe pose skeletons
  • LSTM prediction label + confidence per person
  • Bottom-left debug panel (frame stats, FPS, pose rate, LSTM state)

Console logs (every --log-interval frames):
  • Detection count
  • Pose success rate  (persons with full 33-keypoint skeleton / total detections)
  • Feature vector dimensionality
  • Number of sequences ready for LSTM
  • LSTM prediction per person-id

Usage:
    python test_pipeline.py --source 0                   # webcam index 0
    python test_pipeline.py --source path/to/video.mp4
    python test_pipeline.py --source 1 --conf 0.3
    python test_pipeline.py --no-window                   # headless, logs only

Press 'q' in the display window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np

# ─── sys.path: allow running from the project root ───────────────────────────
_HERE = Path(__file__).resolve().parent
_TRAINING_DIR = _HERE / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

# ─── Stage 0: import pipeline classes ────────────────────────────────────────
print("─" * 60)
print("  CrowdShield AI — Pipeline Tester")
print("─" * 60)
print("[Stage 0] Importing pipeline classes...", end=" ", flush=True)

try:
    from pipeline import (  # type: ignore[import]
        BehaviorPredictor,
        Detector,
        FeatureExtractor,
        PoseEstimator,
        PoseResult,
        SequenceBuilder,
        VideoReader,
        draw_detections,
    )
    from ultralytics import YOLO
    print("OK")
except ImportError as exc:
    print(f"FAIL\n\n  {exc}")
    print("\n  Check that 'training/pipeline.py' exists and all deps are installed:")
    print("      pip install ultralytics mediapipe torch opencv-python numpy")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Per-frame and cumulative statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameStats:
    """Debug data collected for one processed frame."""

    frame_idx: int = 0
    detection_count: int = 0
    pose_success_count: int = 0
    feature_dim: int = 0
    sequences_ready: int = 0
    lstm_outputs: list[tuple[str, float]] = field(default_factory=list)
    weapon_detections: list[tuple[str, float, tuple[int, int, int, int]]] = field(default_factory=list)

    @property
    def pose_rate(self) -> float:
        if self.detection_count == 0:
            return 0.0
        return self.pose_success_count / self.detection_count

    def console_line(self) -> str:
        lstm_str = (
            " | ".join(f"{cls}({conf:.2f})" for cls, conf in self.lstm_outputs)
            or "none"
        )
        weapon_str = (
            " | ".join(f"{cls}({conf:.2f})" for cls, conf, _ in self.weapon_detections)
            or "none"
        )
        return (
            f"[Frame {self.frame_idx:05d}] "
            f"dets={self.detection_count} | "
            f"pose={self.pose_success_count}/{self.detection_count}"
            f" ({self.pose_rate:.0%}) | "
            f"feat_dim={self.feature_dim} | "
            f"seqs={self.sequences_ready} | "
            f"lstm={lstm_str} | "
            f"weapons={weapon_str}"
        )


@dataclass
class CumulativeStats:
    """Accumulated stats across all processed frames."""

    total_frames: int = 0
    total_detections: int = 0
    total_pose_successes: int = 0
    total_lstm_outputs: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    def update(self, fs: FrameStats) -> None:
        self.total_frames += 1
        self.total_detections += fs.detection_count
        self.total_pose_successes += fs.pose_success_count
        self.total_lstm_outputs += len(fs.lstm_outputs)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def avg_fps(self) -> float:
        return self.total_frames / max(self.elapsed, 1e-6)

    @property
    def avg_detections(self) -> float:
        return self.total_detections / max(self.total_frames, 1)

    @property
    def overall_pose_rate(self) -> float:
        return self.total_pose_successes / max(self.total_detections, 1)

    def print_summary(self) -> None:
        print("\n" + "─" * 60)
        print("  Test Summary")
        print("─" * 60)
        print(f"  Frames processed  : {self.total_frames}")
        print(f"  Elapsed time      : {self.elapsed:.1f}s  ({self.avg_fps:.1f} FPS avg)")
        print(f"  Total detections  : {self.total_detections}  ({self.avg_detections:.1f}/frame avg)")
        print(f"  Pose success rate : {self.overall_pose_rate:.1%}")
        print(f"  LSTM predictions  : {self.total_lstm_outputs}")
        print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# On-screen debug panel
# ─────────────────────────────────────────────────────────────────────────────

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_DBG_BG     = (20, 20, 30)
_DBG_FG     = (200, 230, 200)   # soft green
_DBG_WARN   = (60, 160, 255)    # amber
_DBG_OK     = (140, 230, 140)   # bright green


def _draw_debug_panel(
    frame: np.ndarray,
    fs: FrameStats,
    cumulative: CumulativeStats,
    predictor_enabled: bool,
) -> None:
    """Draw a semi-transparent debug info panel in the bottom-left corner."""
    h = frame.shape[0]

    # Build (text, colour) pairs
    if not predictor_enabled:
        lstm_text, lstm_color = "LSTM : disabled (no model.pt)", _DBG_WARN
    elif fs.lstm_outputs:
        parts = " | ".join(f"{c}({v:.2f})" for c, v in fs.lstm_outputs)
        lstm_text, lstm_color = f"LSTM : {parts}", _DBG_OK
    else:
        lstm_text, lstm_color = "LSTM : awaiting 30-frame window...", _DBG_FG

    pose_color = _DBG_OK if fs.pose_rate >= 0.5 else _DBG_WARN

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("─ CrowdShield Debug ─────────", _DBG_FG),
        (f"Frame      : {fs.frame_idx}", _DBG_FG),
        (f"Detections : {fs.detection_count}", _DBG_FG),
        (
            f"Pose OK    : {fs.pose_success_count}/{fs.detection_count}"
            f"  ({fs.pose_rate:.0%})",
            pose_color,
        ),
        (f"Feat dim   : {fs.feature_dim}", _DBG_FG),
        (f"Seqs ready : {fs.sequences_ready}", _DBG_FG),
        (lstm_text, lstm_color),
        (f"Avg FPS    : {cumulative.avg_fps:.1f}", _DBG_FG),
    ]

    line_h   = 17
    margin   = 6
    panel_h  = line_h * len(lines) + margin * 2
    panel_w  = 300

    # Semi-transparent overlay via addWeighted
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (panel_w, h), _DBG_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = h - panel_h + margin + (i + 1) * line_h - 3
        cv2.putText(frame, text, (margin + 2, y), _FONT, 0.40, color, 1, cv2.LINE_AA)


def _draw_alert_overlay(frame: np.ndarray, fs: FrameStats) -> None:
    """Draw a massive CAUTION overlay if severe anomaly or weapon is detected."""
    anomalies = [
        (cls, conf) for cls, conf in fs.lstm_outputs 
        if cls in ["violence", "stampede"] and conf >= 0.70
    ]
    
    msg = ""
    is_weapon_alert = False
    
    if fs.weapon_detections:
        is_weapon_alert = True
        best_weapon = max(fs.weapon_detections, key=lambda x: x[1])
        msg = f" CAUTION: LETHAL WEAPON ({best_weapon[0].upper()}) DETECTED - PREDICTIVE VIOLENCE ALERT "
        
        # Draw bounding boxes for weapons
        for cls_name, conf, (wx1, wy1, wx2, wy2) in fs.weapon_detections:
            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)
            lbl = f"WEAPON: {cls_name} {conf:.2f}"
            cv2.putText(frame, lbl, (wx1, max(wy1 - 5, 10)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
            
    elif anomalies:
        anomalies.sort(key=lambda x: x[1], reverse=True)
        worst_cls, worst_conf = anomalies[0]
        msg = f" CAUTION: {worst_cls.upper()} DETECTED ({worst_conf:.0%}) "
    else:
        return
        
    h, w = frame.shape[:2]
    banner_h = 60
    
    # Use flashing red/black for weapon alert, solid red for normal anomaly
    bg_color = (0, 0, 200)
    if is_weapon_alert and fs.frame_idx % 10 < 5:
        bg_color = (0, 0, 0)
        
    cv2.rectangle(frame, (0, 0), (w, banner_h), bg_color, -1)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(msg, font, font_scale, thickness)
    
    text_x = (w - text_w) // 2
    text_y = (banner_h + text_h) // 2
    cv2.putText(frame, msg, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Stage initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_weapon_detector(model_path: str, device: str) -> Optional[Any]:
    """Stage 1b — load generic YOLO for weapon detection."""
    print(f"[Stage 1b] Loading Weapon Detector model={model_path!r}...", end=" ", flush=True)
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("OK")
        return model
    except Exception as exc:
        print(f"FAIL\n\n  {exc}")
        return None


def _init_detector(model_path: str, confidence: float, device: str) -> Optional[Detector]:
    """Stage 1 — load YOLO + ByteTrack detector."""
    print(f"[Stage 1] Loading Detector (YOLO+ByteTrack)  model={model_path!r}...",
          end=" ", flush=True)
    try:
        det = Detector(model_path=model_path, confidence=confidence, device=device)
        det.load()
        print("OK")
        return det
    except Exception as exc:
        print(f"FAIL\n\n  {exc}")
        print("  → pip install ultralytics")
        return None


def _init_pose() -> Optional[PoseEstimator]:
    """Stage 2 — initialise MediaPipe pose estimator."""
    print("[Stage 2] Loading PoseEstimator (MediaPipe)...", end=" ", flush=True)
    try:
        pe = PoseEstimator()
        print("OK")
        return pe
    except Exception as exc:
        print(f"FAIL\n\n  {exc}")
        print("  → pip install mediapipe==0.10.21")
        return None


def _init_predictor(lstm_model_path: str, device: str) -> BehaviorPredictor:
    """Stage 4 — load LSTM behavior predictor (non-fatal if weights absent)."""
    print(
        f"[Stage 4] Loading BehaviorPredictor (LSTM)  weights={lstm_model_path!r}...",
        end=" ",
        flush=True,
    )
    try:
        bp = BehaviorPredictor(
            model_path=lstm_model_path,
            input_size=10,
            hidden_size=128,
            class_names=["normal", "violence", "fighting", "stampede"],
            device=device,
        )
        if bp.enabled:
            print("OK (weights loaded)")
        else:
            print("OK (random weights — model.pt not found, predictions are random)")
        return bp
    except Exception as exc:
        print(f"FAIL\n\n  {exc}")
        print("  → pip install torch")
        raise


def _open_source(source: str | int) -> VideoReader:
    """Stage 0b — open video source with a clear error on failure."""
    print(f"[Stage 0] Opening VideoReader  source={source!r}...", end=" ", flush=True)
    reader = VideoReader(source)
    try:
        reader.open()
        print(f"OK  ({reader.width}x{reader.height} @ {reader.fps:.1f} FPS)")
        return reader
    except RuntimeError as exc:
        print(f"FAIL\n\n  {exc}")
        if isinstance(source, int):
            print(
                f"  → No webcam at index {source}. "
                "Try --source 0 or --source path/to/video.mp4"
            )
        else:
            print(f"  → File not found or unreadable: {source!r}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

def _process_frame(
    frame: np.ndarray,
    frame_idx: int,
    detector: Detector,
    weapon_detector: Optional[Any],
    pose_estimator: Optional[PoseEstimator],
    feature_extractor: FeatureExtractor,
    sequence_builder: SequenceBuilder,
    behavior_predictor: BehaviorPredictor,
) -> tuple[np.ndarray, FrameStats]:
    """
    Run one BGR frame through all five pipeline stages.

    Returns the annotated frame (bboxes + skeleton + labels) and a FrameStats
    snapshot.  Per-stage exceptions are caught and reported without crashing
    the loop so a single bad frame never terminates the test.
    """
    fs = FrameStats(frame_idx=frame_idx)

    # ── Stage 1: Detection ────────────────────────────────────────────────────
    try:
        det_result = detector.detect(frame)
        fs.detection_count = len(det_result.boxes)
    except Exception as exc:
        print(f"[Frame {frame_idx:05d}] Stage 1 (Detection) ERROR: {exc}")
        return frame.copy(), fs

    # ── Stage 1b: Weapon Detection ────────────────────────────────────────────
    if weapon_detector is not None:
        try:
            # COCO classes: 43=knife, 34=baseball bat, 76=scissors
            w_results = weapon_detector.predict(
                source=frame, conf=0.25, classes=[34, 43, 76],
                device=detector.device, verbose=False
            )
            weapons = []
            for r in w_results:
                if r.boxes:
                    for b in r.boxes:
                        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                        conf = float(b.conf[0])
                        cls_idx = int(b.cls[0])
                        cls_name = weapon_detector.names[cls_idx]
                        weapons.append((cls_name, conf, (x1, y1, x2, y2)))
            fs.weapon_detections = weapons
        except Exception as exc:
            pass

    # ── Stage 2: Pose estimation ──────────────────────────────────────────────
    if pose_estimator is not None:
        try:
            poses: list[PoseResult] = pose_estimator.estimate(frame, det_result.boxes)
            # A full skeleton has exactly 33 MediaPipe landmarks.
            fs.pose_success_count = sum(1 for p in poses if len(p.keypoints) == 33)
        except Exception as exc:
            print(f"[Frame {frame_idx:05d}] Stage 2 (Pose) ERROR: {exc}")
            poses = [PoseResult(person_id=b.person_id) for b in det_result.boxes]
    else:
        # Pose disabled but still need empty placeholders to keep index alignment.
        poses = [PoseResult(person_id=b.person_id) for b in det_result.boxes]

    # ── Stage 3: Feature extraction ───────────────────────────────────────────
    try:
        features = feature_extractor.extract(det_result.boxes, poses, frame.shape)
        fs.feature_dim = len(features[0].feature_vector) if features else 0
    except Exception as exc:
        print(f"[Frame {frame_idx:05d}] Stage 3 (FeatureExtractor) ERROR: {exc}")
        features = []

    # ── Stage 3b: Sequence builder ────────────────────────────────────────────
    try:
        sequences = sequence_builder.update(features)
        fs.sequences_ready = len(sequences)
    except Exception as exc:
        print(f"[Frame {frame_idx:05d}] Stage 3b (SequenceBuilder) ERROR: {exc}")
        sequences = []

    # ── Stage 4: LSTM behavior prediction ────────────────────────────────────
    predictions: dict = {}
    try:
        predictions = behavior_predictor.predict(sequences)
        fs.lstm_outputs = [
            (pred.class_name, pred.confidence) for pred in predictions.values()
        ]
    except Exception as exc:
        print(f"[Frame {frame_idx:05d}] Stage 4 (BehaviorPredictor) ERROR: {exc}")

    # ── Stage 5: Rendering ────────────────────────────────────────────────────
    try:
        annotated = draw_detections(det_result, poses, predictions)
    except Exception as exc:
        print(f"[Frame {frame_idx:05d}] Stage 5 (Rendering) ERROR: {exc}")
        annotated = frame.copy()

    return annotated, fs


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_test(args: argparse.Namespace) -> None:
    """Initialise all stages, then run the frame loop with debug output."""

    # ── Resolve source type ───────────────────────────────────────────────────
    if args.source is None:
        print("\n[ABORT] No --source provided.")
        print("  test_pipeline.py is a local tester, not the backend API server.")
        print("  Use one of:")
        print("    python test_pipeline.py --source 0")
        print("    python test_pipeline.py --source stampede.mp4")
        sys.exit(1)

    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # ── Stage init ────────────────────────────────────────────────────────────
    detector = _init_detector(args.model, args.conf, args.device)
    if detector is None:
        print("\n[ABORT] Detector failed to initialise. Cannot continue.")
        sys.exit(1)

    weapon_detector = _init_weapon_detector(args.model, args.device)

    pose_estimator = _init_pose()
    if pose_estimator is None:
        print("[WARN]  Pose stage disabled — skeletons will not be drawn.\n")

    feature_extractor = FeatureExtractor()
    sequence_builder  = SequenceBuilder(window_size=30)
    print("[Stage 3] FeatureExtractor + SequenceBuilder: OK")

    try:
        behavior_predictor = _init_predictor(args.lstm_model, args.device)
    except Exception:
        print("\n[ABORT] BehaviorPredictor failed to initialise. Cannot continue.")
        sys.exit(1)

    # ── Open source ───────────────────────────────────────────────────────────
    try:
        reader = _open_source(source)
    except RuntimeError:
        sys.exit(1)

    feature_extractor.fps = reader.fps if reader.fps > 0 else 30.0

    cumulative = CumulativeStats()
    print(f"\n[Pipeline] Running — press 'q' in the window to quit.\n")

    # ── Frame loop ────────────────────────────────────────────────────────────
    try:
        for frame_idx, frame in enumerate(reader):
            annotated, fs = _process_frame(
                frame, frame_idx,
                detector, weapon_detector, pose_estimator,
                feature_extractor, sequence_builder, behavior_predictor,
            )
            cumulative.update(fs)

            # Throttled console output
            if frame_idx % args.log_interval == 0:
                print(fs.console_line())

            if not args.no_window:
                _draw_debug_panel(annotated, fs, cumulative, behavior_predictor.enabled)
                _draw_alert_overlay(annotated, fs)
                cv2.imshow("CrowdShield AI — Pipeline Test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[Pipeline] User exit.")
                    break

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")
    finally:
        reader.close()
        if pose_estimator is not None:
            pose_estimator.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    cumulative.print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CrowdShield AI — full pipeline visual tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_pipeline.py --source 0\n"
            "  python test_pipeline.py --source crowd.mp4 --conf 0.3\n"
            "  python test_pipeline.py --no-window --log-interval 1\n"
        ),
    )
    p.add_argument(
        "--source",
        default=None,
        help="Webcam index (0, 1, ...) or path/RTSP URL to a video (required)",
    )
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model name or .pt file path (default: yolov8n.pt)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold 0–1 (default: 0.35)",
    )
    p.add_argument(
        "--device",
        default="",
        help="Inference device: 'cpu', 'cuda', 'mps', or '' for auto-select (default: '')",
    )
    p.add_argument(
        "--lstm-model",
        default="scripts/model.pt",
        help=(
            "Path to trained LSTM weights, relative to training/ "
            "(default: scripts/model.pt)"
        ),
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print a debug line to the console every N frames (default: 10)",
    )
    p.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the display window; useful for headless/SSH environments",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_test(_parse_args())
