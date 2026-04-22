import os
import sys
import uuid
import time
import threading
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path for imports from 'training'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
training_dir = os.path.join(project_root, "training")
if project_root not in sys.path:
    sys.path.append(project_root)
if training_dir not in sys.path:
    sys.path.append(training_dir)

from training.pipeline import (  # noqa: E402
    BehaviorPredictor,
    Detector,
    FeatureExtractor,
    PoseEstimator,
    SequenceBuilder,
    VideoReader,
    draw_detections,
)
from training.detectors import get_default_registry  # noqa: E402
from training.triage import SeverityTriageEngine  # noqa: E402

app = FastAPI(title="CrowdShield AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LiveStartRequest(BaseModel):
    source: str | int = Field(default=0, description="Webcam index or video/RTSP source")
    clear_events: bool = Field(default=False, description="Clear existing event feed before start")


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Shared frame/event outputs used by the frontend.
        self.latest_annotated_frame: Optional[bytes] = None
        self.events: List[Dict[str, Any]] = []
        self.latest_report: Optional[Dict[str, Any]] = None

        # Analysis (uploaded video) state.
        self.analysis_frame_idx = 0
        self.analysis_total_frames = 0
        self.analysis_status = "idle"
        self.analysis_fps = 25.0
        self.video_duration_s = 0.0

        # Live monitor state.
        self.live_running = False
        self.live_status = "stopped"
        self.live_source: str | int | None = None
        self.live_frame_idx = 0
        self.live_thread: Optional[threading.Thread] = None
        self.live_stop_event = threading.Event()

        # UI configuration state.
        self.config: Dict[str, Any] = {
            "thresholds": {
                "crowdDensity": 82,
                "loiteringSeconds": 90,
                "counterFlow": 70,
                "fireSmokeConfidence": 76,
            },
            "stream": {
                "source": "camera_01",
                "ingestFps": 25,
                "detectionStride": "1",
                "retentionDays": 14,
            },
            "notifications": {
                "inApp": True,
                "email": True,
                "sms": False,
                "webhook": False,
                "webhookUrl": "",
            },
            "governance": {
                "autoEscalateCritical": True,
                "blurFacesInExports": True,
                "saveSnapshots": True,
            },
        }


state = AppState()


@app.on_event("shutdown")
def shutdown() -> None:
    """Best-effort cleanup so webcam handles are released when API stops."""
    state.live_stop_event.set()
    thread = state.live_thread
    if thread and thread.is_alive():
        thread.join(timeout=2.0)


def _coerce_source(source: str | int) -> str | int:
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _build_pipeline_context() -> Dict[str, Any]:
    detector = Detector(model_path="yolov8n.pt", confidence=0.35)
    detector.load()

    return {
        "detector": detector,
        "pose_estimator": PoseEstimator(),
        "feature_extractor": FeatureExtractor(),
        "sequence_builder": SequenceBuilder(window_size=30),
        "behavior_predictor": BehaviorPredictor(
            model_path="model.pt",
            input_size=10,
            hidden_size=128,
        ),
        "registry": get_default_registry(),
        "triage_engine": SeverityTriageEngine(),
    }


def _append_events(triage_report: Any, frame_idx: int, fps: float) -> None:
    new_events = [
        {
            "detector_name": event.detector_name,
            "severity": event.severity,
            "confidence": event.confidence * 100,
            "description": event.description,
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx / fps if fps > 0 else 0,
        }
        for event in triage_report.red_alerts + triage_report.yellow_alerts
    ]

    state.events.extend(new_events)
    if len(state.events) > 200:
        state.events = state.events[-200:]


def _run_frame_pipeline(
    frame: Any,
    frame_idx: int,
    fps: float,
    context: Dict[str, Any],
) -> tuple[bytes | None, Dict[str, Any], Any]:
    detector = context["detector"]
    pose_estimator = context["pose_estimator"]
    feature_extractor = context["feature_extractor"]
    sequence_builder = context["sequence_builder"]
    behavior_predictor = context["behavior_predictor"]
    registry = context["registry"]
    triage_engine = context["triage_engine"]

    det_result = detector.detect(frame)
    poses = pose_estimator.estimate(frame, det_result.boxes)
    features = feature_extractor.extract(det_result.boxes, poses, frame.shape)
    sequences = sequence_builder.update(features)
    predictions = behavior_predictor.predict(sequences)

    tracked_persons = []
    for index, box in enumerate(det_result.boxes):
        if box.person_id >= 0:
            tracked_persons.append(
                {
                    "id": box.person_id,
                    "bbox": (box.x1, box.y1, box.x2, box.y2),
                    "pose_keypoints": poses[index].keypoints if index < len(poses) else [],
                    "velocity_vector": (0, 0),
                    "history_frames": [],
                }
            )

    registry_events = registry.run_all(tracked_persons, frame, frame_idx, yolo_detections=[])
    triage_report = triage_engine.process(registry_events, frame_idx, fps)

    annotated = draw_detections(det_result, poses, predictions)

    # Draw alert banner if severe anomalies detected
    anomalies = [
        (pred.class_name, pred.confidence) 
        for pred in predictions.values() 
        if pred.class_name in ["violence", "stampede", "fighting"] and pred.confidence >= 0.70
    ]
    if anomalies:
        anomalies.sort(key=lambda x: x[1], reverse=True)
        worst_cls, worst_conf = anomalies[0]
        msg = f" CAUTION: {worst_cls.upper()} DETECTED ({worst_conf:.0%}) "
        
        h, w = annotated.shape[:2]
        banner_h = 60
        bg_color = (0, 0, 200)  # Red
        
        cv2.rectangle(annotated, (0, 0), (w, banner_h), bg_color, -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(msg, font, font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = (banner_h + text_h) // 2
        
        cv2.putText(annotated, msg, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    for event in triage_report.red_alerts + triage_report.yellow_alerts:
        if event.bbox:
            x, y, w, h = event.bbox
            color = (0, 0, 255) if event.severity == "RED" else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                annotated,
                event.detector_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    encoded_ok, buffer = cv2.imencode(".jpg", annotated)
    frame_bytes = buffer.tobytes() if encoded_ok else None

    return frame_bytes, triage_report.to_dict(), triage_report


def run_analysis_worker(source_path: str, cleanup_file: bool = True) -> None:
    context = _build_pipeline_context()

    try:
        with VideoReader(source_path) as reader:
            with state.lock:
                state.analysis_status = "processing"
                state.analysis_total_frames = int(reader._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                state.analysis_fps = reader.fps if reader.fps > 0 else 25.0
                state.video_duration_s = (
                    state.analysis_total_frames / state.analysis_fps if state.analysis_fps > 0 else 0
                )

            for frame_idx, frame in enumerate(reader):
                frame_bytes, report_dict, triage_report = _run_frame_pipeline(
                    frame,
                    frame_idx,
                    state.analysis_fps,
                    context,
                )

                with state.lock:
                    state.analysis_frame_idx = frame_idx + 1
                    if frame_bytes is not None:
                        state.latest_annotated_frame = frame_bytes
                    state.latest_report = report_dict
                    _append_events(triage_report, frame_idx, state.analysis_fps)

        with state.lock:
            state.analysis_status = "completed"
    except Exception as exc:
        print(f"Error in analysis worker: {exc}")
        with state.lock:
            state.analysis_status = "error"
    finally:
        if cleanup_file:
            try:
                os.remove(source_path)
            except OSError:
                pass


class AsyncCameraReader:
    def __init__(self, source: str | int):
        self.cap = cv2.VideoCapture(source)
        self.is_opened = self.cap.isOpened()
        self.ok = True
        self.frame = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        if self.is_opened:
            # Buffer first frame
            self.ok, self.frame = self.cap.read()
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            
    def _update_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                with self._lock:
                    self.ok = False
                break
            with self._lock:
                self.frame = frame
                
    def read(self):
        with self._lock:
            if self.frame is not None:
                return self.ok, self.frame.copy()
            return self.ok, None
            
    def release(self):
        self._stop_event.set()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

def run_live_worker(source: str | int) -> None:
    context = _build_pipeline_context()
    reader = AsyncCameraReader(source)

    if not reader.is_opened:
        with state.lock:
            state.live_running = False
            state.live_status = "error"
        return

    fps = reader.fps
    if fps <= 0 or fps != fps:  # NaN check
        fps = 25.0

    with state.lock:
        state.live_status = "running"

    frame_idx = 0

    try:
        while not state.live_stop_event.is_set():
            ok, frame = reader.read()
            if not ok or frame is None:
                break

            frame_bytes, report_dict, triage_report = _run_frame_pipeline(frame, frame_idx, fps, context)

            with state.lock:
                state.live_frame_idx = frame_idx + 1
                if frame_bytes is not None:
                    state.latest_annotated_frame = frame_bytes
                state.latest_report = report_dict
                _append_events(triage_report, frame_idx, fps)

            frame_idx += 1
    except Exception as exc:
        print(f"Error in live worker: {exc}")
        with state.lock:
            state.live_status = "error"
    finally:
        reader.release()
        with state.lock:
            state.live_running = False
            if state.live_status != "error":
                state.live_status = "stopped"


@app.post("/api/analyze")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, str]:
    suffix = Path(file.filename or "").suffix or ".mp4"
    fd, input_path = tempfile.mkstemp(prefix="crowdshield_input_", suffix=suffix)

    with os.fdopen(fd, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    with state.lock:
        state.analysis_frame_idx = 0
        state.analysis_total_frames = 0
        state.analysis_status = "started"
        state.latest_report = None
        if not state.live_running:
            state.events = []

    background_tasks.add_task(run_analysis_worker, input_path, True)
    return {"status": "started", "job_id": str(uuid.uuid4())}


@app.post("/api/live/start")
async def start_live_stream(request: LiveStartRequest) -> Dict[str, Any]:
    source = _coerce_source(request.source)

    with state.lock:
        if state.live_running:
            return {
                "status": "already_running",
                "source": state.live_source,
                "frame": state.live_frame_idx,
            }

        state.live_stop_event.clear()
        state.live_running = True
        state.live_status = "starting"
        state.live_source = source
        state.live_frame_idx = 0
        if request.clear_events:
            state.events = []

        state.live_thread = threading.Thread(target=run_live_worker, args=(source,), daemon=True)
        state.live_thread.start()

    return {
        "status": "started",
        "source": source,
    }


@app.post("/api/live/stop")
async def stop_live_stream() -> Dict[str, Any]:
    with state.lock:
        thread = state.live_thread
        running = state.live_running
        state.live_stop_event.set()
        state.live_status = "stopping" if running else "stopped"

    if thread and thread.is_alive():
        thread.join(timeout=2.0)

    with state.lock:
        if not state.live_running:
            state.live_status = "stopped"

    return {
        "status": "stopped",
        "running": state.live_running,
    }


@app.get("/api/live/status")
async def live_status() -> Dict[str, Any]:
    with state.lock:
        return {
            "running": state.live_running,
            "status": state.live_status,
            "source": state.live_source,
            "frame": state.live_frame_idx,
        }


def generate_frames():
    while True:
        with state.lock:
            frame = state.latest_annotated_frame

        if frame:
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.04)


@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/progress")
async def get_progress() -> Dict[str, Any]:
    with state.lock:
        percent = (
            (state.analysis_frame_idx / state.analysis_total_frames * 100)
            if state.analysis_total_frames > 0
            else 0
        )
        return {
            "frame": state.analysis_frame_idx,
            "total": state.analysis_total_frames,
            "status": state.analysis_status,
            "percent": percent,
        }


@app.get("/api/events/live")
async def get_live_events() -> List[Dict[str, Any]]:
    with state.lock:
        return state.events[-50:]


@app.get("/api/report/latest")
async def get_latest_report():
    with state.lock:
        if not state.latest_report:
            return JSONResponse(content={"status": "processing"}, status_code=200)

        report = state.latest_report.copy()
        report["video_duration_s"] = state.video_duration_s
        report["fps"] = state.analysis_fps
        report["analysis_status"] = state.analysis_status
        report["live_status"] = state.live_status
        return report


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    with state.lock:
        return deepcopy(state.config)


@app.post("/api/config")
async def set_config(config: Dict[str, Any]) -> Dict[str, Any]:
    with state.lock:
        state.config = deepcopy(config)
        return {"status": "saved", "config": deepcopy(state.config)}


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    with state.lock:
        return {
            "status": "ok",
            "analysis_status": state.analysis_status,
            "live_status": state.live_status,
            "live_running": state.live_running,
            "events_buffer": len(state.events),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
