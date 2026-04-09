"""Unified crowd anomaly inference module.

This module combines a YOLOv8 person detector and a custom two-layer LSTM
classifier into a single high-level inference class suitable for online or
offline video analysis in the CrowdShield project.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable

import cv2
import mediapipe
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


CONFIG: dict[str, Any] = {
    "YOLO_MODEL_PATH": "models/yolo_crowdhuman_best.pt",
    "LSTM_MODEL_PATH": "models/crowd_anomaly_lstm_v1.pt",
    "FEATURE_CONFIG_PATH": "models/feature_config.json",
    "LABEL_ENCODER_PATH": "models/label_encoder.json",
    "CONFIDENCE_THRESHOLD": 0.65,
    "SEQUENCE_LENGTH": 30,
    "ANOMALY_CLASSES": [1],
}


class LSTMCell(nn.Module):
    """Manual LSTM cell implementation using explicit gate linear layers."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize gate layers for one recurrent cell.

        Args:
            input_size: Number of features in each input frame vector.
            hidden_size: Number of hidden units in this recurrent layer.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single recurrent update step.

        Args:
            x_t: Input tensor for the current time step.
            h_prev: Previous hidden state.
            c_prev: Previous cell state.

        Returns:
            A tuple containing the next hidden state and next cell state.
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        g_t = torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))

        c_next = (f_t * c_prev) + (i_t * g_t)
        h_next = o_t * torch.tanh(c_next)
        return h_next, c_next


class CustomLSTM(nn.Module):
    """Two-layer custom LSTM encoder with LayerNorm and Dropout."""

    def __init__(self, input_size: int = 118) -> None:
        """Construct the recurrent encoder layers.

        Args:
            input_size: Feature dimension per frame.
        """
        super().__init__()
        self.hidden1 = 256
        self.hidden2 = 128

        self.layer1 = LSTMCell(input_size=input_size, hidden_size=self.hidden1)
        self.layer2 = LSTMCell(input_size=self.hidden1, hidden_size=self.hidden2)

        self.norm1 = nn.LayerNorm(self.hidden1)
        self.norm2 = nn.LayerNorm(self.hidden2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a full sequence and return the final hidden state.

        Args:
            x: Tensor with shape ``(batch, sequence_length, input_size)``.

        Returns:
            Final hidden state from layer 2 with shape ``(batch, 128)``.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h1 = torch.zeros(batch_size, self.hidden1, device=device)
        c1 = torch.zeros(batch_size, self.hidden1, device=device)
        h2 = torch.zeros(batch_size, self.hidden2, device=device)
        c2 = torch.zeros(batch_size, self.hidden2, device=device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h1, c1 = self.layer1(x_t, h1, c1)
            h1 = self.norm1(h1)
            h1 = self.dropout(h1)

            h2, c2 = self.layer2(h1, h2, c2)
            h2 = self.norm2(h2)

        return h2


class CrowdAnomalyClassifier(nn.Module):
    """Sequence classifier head on top of the custom recurrent encoder."""

    def __init__(self, input_size: int = 118, num_classes: int = 2) -> None:
        """Initialize classifier head layers.

        Args:
            input_size: Feature dimension for each frame in a sequence.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.temporal = CustomLSTM(input_size=input_size)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward inference and return output logits.

        Args:
            x: Input sequence tensor with shape ``(batch, 30, 118)``.

        Returns:
            Logits tensor with shape ``(batch, num_classes)``.
        """
        z = self.temporal(x)
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.relu(self.fc2(z))
        return self.fc3(z)

    def get_confidence_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Return anomaly-class softmax probability.

        Args:
            logits: Raw classifier logits with shape ``(batch, 2)``.

        Returns:
            Tensor of anomaly probabilities for class index 1.
        """
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]


class CrowdDetector:
    """Unified detector that combines YOLO person detection and LSTM anomaly inference."""

    def __init__(self) -> None:
        """Load all models, metadata, and runtime buffers for inference.

        This initializer builds the full inference pipeline:
        1. YOLO model for person detection.
        2. Feature and label metadata from JSON files.
        3. Custom LSTM classifier architecture and trained weights.
        4. MediaPipe pose extractor for frame-level keypoint features.
        5. Rolling frame buffer and alert log state.
        """
        self.yolo_model = YOLO(CONFIG["YOLO_MODEL_PATH"])

        with Path(CONFIG["FEATURE_CONFIG_PATH"]).open("r", encoding="utf-8") as f:
            self.feature_config = json.load(f)

        with Path(CONFIG["LABEL_ENCODER_PATH"]).open("r", encoding="utf-8") as f:
            raw_labels = json.load(f)

        # Normalize label mapping to int -> string.
        self.label_names: dict[int, str] = {}
        for k, v in raw_labels.items():
            try:
                self.label_names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        if not self.label_names:
            self.label_names = {0: "normal", 1: "anomaly"}

        self.input_size = int(self.feature_config.get("input_size", 118))
        self.sequence_length = int(
            self.feature_config.get("sequence_length", CONFIG["SEQUENCE_LENGTH"])
        )

        self.lstm_model = CrowdAnomalyClassifier(input_size=self.input_size, num_classes=2)
        loaded = torch.load(CONFIG["LSTM_MODEL_PATH"], map_location="cpu")
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            self.lstm_model.load_state_dict(loaded["model_state_dict"])
        else:
            self.lstm_model.load_state_dict(loaded)
        self.lstm_model.eval()

        self.pose = mediapipe.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.frame_buffer: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.alert_log: list[dict[str, Any]] = []

    @staticmethod
    def _angle_from_pair(a: np.ndarray, b: np.ndarray) -> float:
        """Compute pairwise orientation angle using ``np.arctan2``.

        Args:
            a: First landmark ``[x, y, z]``.
            b: Second landmark ``[x, y, z]``.

        Returns:
            Orientation angle in radians for vector ``b - a``.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        return float(np.arctan2(dy, dx))

    def extract_pose_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract a 118-dimensional pose feature vector from one frame.

        Steps implemented:
        - MediaPipe Pose inference on RGB frame.
        - 99 keypoint xyz values.
        - 17 pairwise orientation angles via ``np.arctan2``.
        - Motion estimate from previous frame keypoints in rolling buffer.
        - Crowd count proxy as a scalar feature.

        Args:
            frame: Input BGR frame from OpenCV.

        Returns:
            NumPy array with shape ``(118,)``.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if result.pose_landmarks is None:
            return np.zeros((118,), dtype=np.float32)

        keypoints = np.asarray(
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
            dtype=np.float32,
        )

        # Ensure strict 33x3 shape before flattening.
        if keypoints.shape != (33, 3):
            padded = np.zeros((33, 3), dtype=np.float32)
            flat = keypoints.reshape(-1)
            padded.reshape(-1)[: min(flat.size, 99)] = flat[:99]
            keypoints = padded

        kf = keypoints.reshape(-1)[:99]

        idx = mediapipe.solutions.pose.PoseLandmark
        angle_pairs = [
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
        angles = np.asarray(
            [self._angle_from_pair(keypoints[a], keypoints[b]) for a, b in angle_pairs],
            dtype=np.float32,
        )

        if self.frame_buffer:
            prev_keypoints = self.frame_buffer[-1][:99]
            motion = float(np.mean(np.abs(kf - prev_keypoints)))
        else:
            motion = 0.0

        crowd_count = 1.0

        feat = np.concatenate(
            [kf, angles, np.asarray([motion], dtype=np.float32), np.asarray([crowd_count], dtype=np.float32)]
        )
        if feat.shape[0] != 118:
            # Fallback safeguard for strict contract compliance.
            out = np.zeros((118,), dtype=np.float32)
            out[: min(118, feat.shape[0])] = feat[:118]
            return out
        return feat.astype(np.float32)

    def detect_persons(self, frame: np.ndarray) -> list[list[float]]:
        """Detect person bounding boxes with YOLO and return sorted detections.

        Args:
            frame: Input BGR frame.

        Returns:
            List of bounding boxes as ``[x1, y1, x2, y2, conf]`` sorted by
            descending confidence.
        """
        results = self.yolo_model.predict(frame, conf=0.4, classes=[0], verbose=False)

        boxes: list[list[float]] = []
        if results:
            r0 = results[0]
            if r0.boxes is not None:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                conf = r0.boxes.conf.cpu().numpy()
                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    boxes.append([x1, y1, x2, y2, float(conf[i])])

        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Process one frame end-to-end and return inference metadata.

        Workflow:
        1. Person detection via YOLO.
        2. Pose feature extraction on the full frame.
        3. Sequence buffering for LSTM input.
        4. Anomaly inference once sequence length is satisfied.
        5. Alert generation and annotated visualization.

        Args:
            frame: Input BGR frame.

        Returns:
            Dictionary with frame status, predictions, detection metadata, and
            annotated frame.
        """
        boxes = self.detect_persons(frame)
        feat = self.extract_pose_features(frame)
        self.frame_buffer.append(feat)

        if len(self.frame_buffer) < self.sequence_length:
            return {
                "status": "buffering",
                "frames_needed": self.sequence_length - len(self.frame_buffer),
                "persons": len(boxes),
                "boxes": boxes,
            }

        sequence = np.stack(list(self.frame_buffer), axis=0).astype(np.float32)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0)

        with torch.no_grad():
            logits = self.lstm_model(sequence_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, pred].item())

        class_name = self.label_names.get(pred, str(pred))
        is_alert = bool(
            pred in CONFIG["ANOMALY_CLASSES"]
            and conf > float(CONFIG["CONFIDENCE_THRESHOLD"])
        )

        if is_alert:
            alert = {
                "timestamp": time.time(),
                "class_name": class_name,
                "confidence": float(conf),
                "persons_in_frame": len(boxes),
            }
            self.alert_log.append(alert)

        annotated = frame.copy()
        color = (0, 0, 255) if is_alert else (0, 255, 0)

        for box in boxes:
            x1, y1, x2, y2, _ = box
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, color, 2)
            cv2.putText(
                annotated,
                f"{class_name} {conf:.2f}",
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if is_alert:
            cv2.putText(
                annotated,
                "ANOMALY DETECTED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        person_text = f"Persons: {len(boxes)}"
        text_size, _ = cv2.getTextSize(person_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            annotated,
            person_text,
            (max(10, annotated.shape[1] - text_size[0] - 10), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return {
            "annotated_frame": annotated,
            "status": "anomaly" if is_alert else "normal",
            "class_name": class_name,
            "confidence": conf,
            "persons": len(boxes),
            "boxes": boxes,
            "is_alert": is_alert,
            "alerts_total": len(self.alert_log),
        }

    def process_video_file(
        self,
        video_path: str,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Process an entire video file and optionally write annotated output.

        Args:
            video_path: Input video file path.
            output_path: Optional path to write annotated video.

        Returns:
            List of result dictionaries, one entry per processed frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        writer = None
        if output_path is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = 25.0 if fps <= 0 else fps
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        results: list[dict[str, Any]] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = self.process_frame(frame)
            results.append(res)

            if writer is not None:
                to_write = res.get("annotated_frame", frame)
                writer.write(to_write)

        cap.release()
        if writer is not None:
            writer.release()

        return results

    def process_rtsp_stream(
        self,
        rtsp_url: str,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Process an RTSP stream with reconnect logic for unstable connections.

        Args:
            rtsp_url: RTSP stream URL.
            callback: Optional callback invoked with each frame result dict.

        Notes:
            The method retries dropped connections up to three times, waiting
            two seconds between attempts, and uses a minimal decode buffer to
            reduce stream latency.
        """
        retries = 0

        while retries < 3:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                retries += 1
                time.sleep(2)
                continue

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                res = self.process_frame(frame)
                if callback is not None:
                    callback(res)

            cap.release()
            retries += 1
            time.sleep(2)

    def get_alert_summary(self) -> dict[str, Any]:
        """Return aggregate alert statistics for monitoring or API responses.

        Returns:
            Dictionary with total alerts, class-wise counts, latest alert entry,
            and mean confidence over all alerts.
        """
        if not self.alert_log:
            return {
                "total_alerts": 0,
                "alerts_by_class": {},
                "last_alert": None,
                "avg_confidence": 0.0,
            }

        counts = Counter(alert["class_name"] for alert in self.alert_log)
        avg_conf = float(np.mean([a["confidence"] for a in self.alert_log]))

        return {
            "total_alerts": len(self.alert_log),
            "alerts_by_class": dict(counts),
            "last_alert": self.alert_log[-1],
            "avg_confidence": avg_conf,
        }

    def reset_buffer(self) -> None:
        """Clear temporal inference state and accumulated alert history."""
        self.frame_buffer.clear()
        self.alert_log.clear()
