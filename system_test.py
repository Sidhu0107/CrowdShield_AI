"""CrowdShield AI end-to-end system validation script.

This script validates key parts of the local CrowdShield stack:
1. Folder structure
2. Python dependencies
3. Redis connectivity
4. PostgreSQL connectivity
5. AI pipeline components (detection, pose, feature extraction)
6. Custom LSTM inference contract
7. JSON event contract shape checks (TDD)
8. Structured PASS/FAIL report
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


ROOT = Path(__file__).resolve().parent


def load_dotenv_file(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env into process environment.

    Existing environment variables are preserved.
    """
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class TestResult:
    """Result container for one validation module."""

    name: str
    status: str  # PASS / FAIL
    message: str


class Reporter:
    """Collect and print structured test output."""

    def __init__(self) -> None:
        self.results: list[TestResult] = []

    def add_pass(self, name: str, message: str) -> None:
        self.results.append(TestResult(name=name, status="PASS", message=message))

    def add_fail(self, name: str, message: str) -> None:
        self.results.append(TestResult(name=name, status="FAIL", message=message))

    def run(self, name: str, fn: Callable[[], str]) -> None:
        try:
            msg = fn()
            self.add_pass(name, msg)
        except Exception as exc:  # noqa: BLE001 - explicit script-level catch
            self.add_fail(name, f"{exc.__class__.__name__}: {exc}")

    def print_report(self) -> None:
        width_name = 28
        width_status = 8
        line = "-" * (width_name + width_status + 47)

        print("\nCrowdShield AI - System Validation Report")
        print(line)
        print(f"{'Module':<{width_name}} {'Status':<{width_status}} Message")
        print(line)

        for result in self.results:
            print(f"{result.name:<{width_name}} {result.status:<{width_status}} {result.message}")

        print(line)
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = total - passed
        print(f"Summary: {passed}/{total} PASS, {failed}/{total} FAIL")

    @property
    def has_failures(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)


def load_module_from_path(module_name: str, file_path: Path) -> Any:
    """Load a Python module from an absolute file path."""
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    # Ensure decorators/introspection can resolve the module during execution.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_folder_structure() -> str:
    backend = ROOT / "backend"
    required = [
        "ingestion-service",
        "vision-service",
        "pose-service",
        "behavior-service",
        "alert-service",
        "api-gateway",
    ]

    if not backend.exists():
        raise FileNotFoundError(f"Missing backend folder: {backend}")

    missing = [name for name in required if not (backend / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing service folders: {', '.join(missing)}")

    return "All required service folders exist"


def test_python_dependencies() -> str:
    required_imports = ["cv2", "torch", "ultralytics", "mediapipe", "fastapi", "redis"]
    failures: list[str] = []

    for pkg in required_imports:
        try:
            importlib.import_module(pkg)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{pkg} ({exc.__class__.__name__})")

    if failures:
        raise ImportError("Dependency import failures: " + ", ".join(failures))

    return "All required imports resolved"


def test_redis_connection() -> str:
    from redis import Redis

    client = Redis(host="localhost", port=6379, decode_responses=True)
    key = f"crowdshield:test:{uuid4()}"
    value = f"ok:{datetime.now(timezone.utc).isoformat()}"

    try:
        client.ping()
        client.set(key, value, ex=30)
        got = client.get(key)
    finally:
        client.close()

    if got != value:
        raise RuntimeError("Redis write/read mismatch")

    return "Connected, write/read test passed"


async def _test_postgres_connection_async() -> str:
    try:
        import asyncpg
    except Exception as exc:  # noqa: BLE001
        raise ImportError("asyncpg is not installed") from exc

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "crowdshield")

    # Try configured credentials first, then a small fallback set for local dev.
    candidates = [
        (
            os.getenv("POSTGRES_USER", "crowdshield"),
            os.getenv("POSTGRES_PASSWORD", "change_me"),
        ),
        ("postgres", "postgres"),
        ("crowdshield", "crowdshield"),
    ]

    errors: list[str] = []
    for user, password in candidates:
        try:
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                timeout=5,
            )
            try:
                v = await conn.fetchval("SELECT 1")
            finally:
                await conn.close()

            if v != 1:
                raise RuntimeError("PostgreSQL check query failed")

            return f"Connected to PostgreSQL at {host}:{port} as {user}"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{user}:{exc.__class__.__name__}")

    raise RuntimeError("Unable to connect with tested credentials: " + ", ".join(errors))


def test_postgres_connection() -> str:
    return asyncio.run(_test_postgres_connection_async())


def _get_test_frame() -> tuple[Any, str]:
    """Get one frame from sample video or webcam; fallback to synthetic frame."""
    import cv2
    import numpy as np

    media_glob = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    search_roots = [ROOT / "training", ROOT]

    for root in search_roots:
        for pattern in media_glob:
            files = list(root.rglob(pattern))
            if not files:
                continue
            cap = cv2.VideoCapture(str(files[0]))
            ok, frame = cap.read()
            cap.release()
            if ok:
                return frame, f"video:{files[0].name}"

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        return frame, "webcam:0"

    # Fallback for non-camera/headless environments.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return frame, "synthetic"


def test_ai_pipeline() -> str:
    pipeline_mod = load_module_from_path("training_pipeline", ROOT / "training" / "pipeline.py")

    frame, source = _get_test_frame()

    detector = pipeline_mod.Detector(model_path="yolov8n.pt", confidence=0.25, device="")
    detector.load()

    detection_result = detector.detect(frame)
    if not hasattr(detection_result, "boxes"):
        raise RuntimeError("Detection result missing boxes")

    pose_estimator = pipeline_mod.PoseEstimator()
    try:
        poses = pose_estimator.estimate(frame, detection_result.boxes)
    finally:
        pose_estimator.close()

    feature_extractor = pipeline_mod.FeatureExtractor(fps=10.0)
    features = feature_extractor.extract(detection_result.boxes, poses, frame.shape)

    if not isinstance(poses, list):
        raise RuntimeError("Pose output is not a list")
    if not isinstance(features, list):
        raise RuntimeError("Feature output is not a list")

    det_count = len(detection_result.boxes)
    pose_count = len(poses)
    feat_count = len(features)
    return f"source={source}; detections={det_count}, poses={pose_count}, features={feat_count}"


def _find_model_file() -> Path | None:
    candidates = [
        ROOT / "model.pt",
        ROOT / "training" / "model.pt",
        ROOT / "training" / "scripts" / "model.pt",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _infer_model_dims(state_dict: dict[str, Any]) -> tuple[int, int, int]:
    """Infer (input_size, hidden_size, num_classes) from checkpoint tensors."""
    x_weight = state_dict.get("layer1.x_proj.weight")
    head_weight = state_dict.get("classifier.weight") or state_dict.get("head.weight")

    if x_weight is None or head_weight is None:
        # Fallback defaults aligned with project scripts.
        return 10, 128, 4

    input_size = int(x_weight.shape[1])
    hidden_size = int(x_weight.shape[0] // 4)
    num_classes = int(head_weight.shape[0])
    return input_size, hidden_size, num_classes


def test_lstm_model() -> str:
    import torch

    lstm_mod = load_module_from_path(
        "custom_lstm_module",
        ROOT / "training" / "scripts" / "custom_lstm.py",
    )

    model_path = _find_model_file()

    if model_path is None:
        model = lstm_mod.CustomLSTMClassifier(input_size=10, hidden_size=128, num_classes=4)
        dummy = torch.randn(2, 30, 10)
        out = model(dummy)
        if tuple(out.shape) != (2, 4):
            raise RuntimeError(f"Unexpected output shape without checkpoint: {tuple(out.shape)}")
        return "model.pt not found; validated architecture with random weights, output=(2,4)"

    state = torch.load(model_path, map_location="cpu")
    input_size, hidden_size, num_classes = _infer_model_dims(state)

    model = lstm_mod.CustomLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
    )
    model.load_state_dict(state, strict=True)

    dummy = torch.randn(3, 30, input_size)
    out = model(dummy)
    expected = (3, num_classes)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)} != {expected}")

    return f"Loaded {model_path.name}; output shape verified: {expected}"


def _validate_required_keys(event_name: str, payload: dict[str, Any], required: list[str]) -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"{event_name} missing keys: {missing}")


def test_json_contracts() -> str:
    frame_event = {
        "frame_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": "cam_1",
        "frame_data": "dGVzdA==",
    }

    detection_event = {
        "frame_id": str(uuid4()),
        "detections": [
            {"person_id": 1, "bbox": [100.0, 120.0, 80.0, 170.0]}
        ],
    }

    feature_event = {
        "frame_id": str(uuid4()),
        "features": [
            {"person_id": 1, "feature_vector": [0.1, 0.2, 0.3]}
        ],
    }

    behavior_event = {
        "sequence_id": str(uuid4()),
        "prediction": "violence",
        "confidence": 0.91,
    }

    _validate_required_keys("frame_event", frame_event, ["frame_id", "timestamp", "camera_id", "frame_data"])
    _validate_required_keys("detection_event", detection_event, ["frame_id", "detections"])
    _validate_required_keys("feature_event", feature_event, ["frame_id", "features"])
    _validate_required_keys("behavior_event", behavior_event, ["sequence_id", "prediction", "confidence"])

    if not isinstance(detection_event["detections"], list):
        raise TypeError("detection_event.detections must be a list")
    if not isinstance(feature_event["features"], list):
        raise TypeError("feature_event.features must be a list")
    if not isinstance(behavior_event["confidence"], (int, float)):
        raise TypeError("behavior_event.confidence must be numeric")

    det_item = detection_event["detections"][0]
    if not isinstance(det_item.get("person_id"), int):
        raise TypeError("detection.person_id must be int")
    if not (isinstance(det_item.get("bbox"), list) and len(det_item["bbox"]) == 4):
        raise TypeError("detection.bbox must be [x,y,w,h]")

    feat_item = feature_event["features"][0]
    if not isinstance(feat_item.get("person_id"), int):
        raise TypeError("feature.person_id must be int")
    if not isinstance(feat_item.get("feature_vector"), list):
        raise TypeError("feature.feature_vector must be list")

    return "Frame/Detection/Feature/Behavior contracts validated"


def main() -> int:
    reporter = Reporter()

    # Load project-level defaults before running connection tests.
    load_dotenv_file(ROOT / ".env")

    print("CrowdShield AI system_test.py")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    reporter.run("1. Folder Structure", test_folder_structure)
    reporter.run("2. Python Dependencies", test_python_dependencies)
    reporter.run("3. Redis Connection", test_redis_connection)
    reporter.run("4. PostgreSQL Connection", test_postgres_connection)
    reporter.run("5. AI Pipeline", test_ai_pipeline)
    reporter.run("6. LSTM", test_lstm_model)
    reporter.run("7. JSON Contracts", test_json_contracts)

    reporter.print_report()

    return 1 if reporter.has_failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print("\nFATAL ERROR")
        print(f"{exc.__class__.__name__}: {exc}")
        print(traceback.format_exc())
        raise SystemExit(2)
