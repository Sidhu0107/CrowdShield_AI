import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
from scipy.spatial import ConvexHull

@dataclass
class AnomalyEvent:
    """Represents an anomaly detected by one of the registered detectors."""
    detector_name: str
    severity: str
    confidence: float
    person_ids: List[int]
    description: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    context: Dict[str, Any] = field(default_factory=dict)


class LoiteringDetector:
    """
    Detects if a person has remained within a spatial convex hull of area < 2500px^2
    for > 90s. Resets if they leave the area.
    """
    def __init__(self, fps: int = 25):
        self.fps = fps
        # pid -> { 'history': [(frame, x, y)], 'start_frame': int, 'gaps': int, 'last_seen': int }
        self.tracker = {}

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        current_ids = set()

        for p in tracked_persons:
            pid = p.get('id', -1)
            bbox = p.get('bbox')
            if pid < 0 or not bbox:
                continue

            current_ids.add(pid)
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if pid not in self.tracker:
                self.tracker[pid] = {'history': [], 'start_frame': frame_idx, 'gaps': 0, 'last_seen': frame_idx}

            t_info = self.tracker[pid]
            
            # Check for gaps if a frame was missed
            frames_missed = frame_idx - t_info['last_seen'] - 1
            if frames_missed > 0:
                t_info['gaps'] += frames_missed
                
            t_info['last_seen'] = frame_idx
            t_info['history'].append((frame_idx, cx, cy))

            # Keep history for max length (e.g. 5 minutes)
            min_frame = frame_idx - (300 * self.fps)
            t_info['history'] = [h for h in t_info['history'] if h[0] >= min_frame]

            pts = np.array([(h[1], h[2]) for h in t_info['history']])
            
            # Calculate centroid of the hull / points
            if len(pts) > 0:
                centroid_x = np.mean(pts[:, 0])
                centroid_y = np.mean(pts[:, 1])
                
                # Check for zone change reset (>100px from centroid)
                if math.hypot(cx - centroid_x, cy - centroid_y) > 100:
                    t_info['start_frame'] = frame_idx
                    t_info['history'] = [(frame_idx, cx, cy)]
                    t_info['gaps'] = 0
                    pts = np.array([(cx, cy)])

            area = 0.0
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    area = hull.volume # For 2D points, volume is area
                except Exception:
                    area = 0.0

            seconds_loitering = (frame_idx - t_info['start_frame']) / self.fps

            if seconds_loitering >= 90 and area < 2500.0:
                severity = "RED" if seconds_loitering >= 180 else "YELLOW"
                
                # Base confidence scales with time
                conf = 0.5 + 0.5 * min((seconds_loitering - 90) / 90.0, 1.0)
                # Confidence decay based on gaps
                conf -= min(0.1 * t_info['gaps'], 0.4)
                conf = max(conf, 0.1)

                events.append(AnomalyEvent(
                    detector_name="LoiteringDetector",
                    severity=severity,
                    confidence=round(conf, 2),
                    person_ids=[pid],
                    description=f"Person {pid} loitering for {int(seconds_loitering)}s (Area: {int(area)}px²)",
                    bbox=tuple(bbox)
                ))

        # Cleanup
        for pid in list(self.tracker.keys()):
            if pid not in current_ids and (frame_idx - self.tracker[pid]['last_seen']) > 2 * self.fps:
                del self.tracker[pid]

        return events


class FallingDetector:
    """
    3-phase detection: Upright -> Transition -> Fallen
    Also recovers fallen person to Upright -> Fall Recovered (GREEN).
    """
    def __init__(self, fps: int = 25):
        self.fps = fps
        # pid -> {'state': str, 'last_y': float, 'last_ratio': float, 'velocity': float}
        self.states = {}

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        current_ids = set()
        frame_h = frame.shape[0]

        for p in tracked_persons:
            pid = p.get('id', -1)
            bbox = p.get('bbox')
            kps = p.get('pose_keypoints', [])
            if pid < 0 or not bbox:
                continue

            current_ids.add(pid)
            if pid not in self.states:
                self.states[pid] = {'state': 'Upright', 'last_y': 0.0, 'velocity': 0.0}

            state_info = self.states[pid]
            
            x1, y1, x2, y2 = bbox
            width = max(x2 - x1, 1)
            height = y2 - y1
            ratio = height / float(width)

            hip_y = 0.0
            if kps and len(kps) > 24:
                hip_l = kps[23] if len(kps[23]) >= 2 else (0, 0)
                hip_r = kps[24] if len(kps[24]) >= 2 else (0, 0)
                hip_y = (hip_l[1] + hip_r[1]) / 2.0
            else:
                hip_y = y1 + height / 2.0 # Fallback to center Y

            # Calculate vertical velocity
            vel = hip_y - state_info['last_y']
            state_info['velocity'] = vel
            state_info['last_y'] = hip_y

            # State transitions
            old_state = state_info['state']
            if old_state == 'Upright':
                if ratio < 1.0 and vel > 15:
                    state_info['state'] = 'Transition'
                    events.append(AnomalyEvent(
                        detector_name="FallingDetector",
                        severity="YELLOW",
                        confidence=0.6,
                        person_ids=[pid],
                        description=f"Person {pid} transitioning to fall (High vertical velocity).",
                        bbox=tuple(bbox)
                    ))
            elif old_state == 'Transition':
                if ratio < 0.6 and hip_y > 0.7 * frame_h and abs(vel) < 5:
                    state_info['state'] = 'Fallen'
                    events.append(AnomalyEvent(
                        detector_name="FallingDetector",
                        severity="RED",
                        confidence=0.9,
                        person_ids=[pid],
                        description=f"Person {pid} confirmed fallen (Ratio < 0.6, still).",
                        bbox=tuple(bbox)
                    ))
                elif ratio >= 1.2:
                    state_info['state'] = 'Upright'
            elif old_state == 'Fallen':
                if ratio > 1.2:
                    state_info['state'] = 'Upright'
                    events.append(AnomalyEvent(
                        detector_name="FallingDetector",
                        severity="GREEN",
                        confidence=0.9,
                        person_ids=[pid],
                        description=f"Person {pid} Fall Recovered.",
                        bbox=tuple(bbox)
                    ))

        # Cleanup
        for pid in list(self.states.keys()):
            if pid not in current_ids:
                del self.states[pid]

        return events


class CrowdDensityDetector:
    """
    Dynamic grid size and density velocity.
    """
    def __init__(self, fps: int = 25):
        self.fps = fps
        # time -> {(r,c): count}
        self.history = deque(maxlen=10 * fps)

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        h, w = frame.shape[:2]
        num_people = sum(1 for p in tracked_persons if p.get('bbox'))

        if num_people < 10:
            grid_n = 3
        elif num_people <= 30:
            grid_n = 4
        else:
            grid_n = 6

        cell_w = w / grid_n
        cell_h = h / grid_n
        
        # 100m^2 total area
        cell_area_m2 = 100.0 / (grid_n * grid_n)
        
        grid_counts = defaultdict(int)
        grid_pids = defaultdict(list)
        
        for p in tracked_persons:
            bbox = p.get('bbox')
            pid = p.get('id', -1)
            if bbox is not None and pid >= 0:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                c = min(grid_n - 1, max(0, int(cx / cell_w)))
                r = min(grid_n - 1, max(0, int(cy / cell_h)))
                grid_counts[(r, c)] += 1
                grid_pids[(r, c)].append(pid)

        self.history.append((frame_idx, dict(grid_counts)))
        
        # Check density and surges
        for (r, c), count in grid_counts.items():
            density = count / cell_area_m2
            
            # Get historical count 2 seconds ago
            old_count = 0
            for hist_fidx, hist_grid in self.history:
                if frame_idx - hist_fidx <= 2 * self.fps:
                    old_count = hist_grid.get((r,c), 0)
                    break
                    
            if density > 4.0:
                severity = "RED" if density > 6.0 else "YELLOW"
                x1, y1 = int(c * cell_w), int(r * cell_h)
                x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                
                events.append(AnomalyEvent(
                    detector_name="CrowdDensityDetector",
                    severity=severity,
                    confidence=1.0,
                    person_ids=grid_pids[(r, c)],
                    description=f"OVERCROWDED: cell ({r},{c}) has {density:.1f} p/m²",
                    bbox=(x1, y1, x2, y2)
                ))

            if old_count < 2 and count > 5:
                events.append(AnomalyEvent(
                    detector_name="CrowdDensityDetector",
                    severity="YELLOW",
                    confidence=0.8,
                    person_ids=grid_pids[(r, c)],
                    description=f"RAPID_DENSITY_SURGE in cell ({r},{c}): {old_count}->{count} in <2s."
                ))

        return events


class CounterFlowDetector:
    """
    Weighted median dominant flow, tracks sequential counterflow frames.
    """
    def __init__(self):
        self.counter_history = defaultdict(int)

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        valid_vectors = []
        pid_data = {}
        
        for p in tracked_persons:
            vec = p.get('velocity_vector')
            pid = p.get('id', -1)
            if vec is not None and pid >= 0 and len(vec) == 2:
                dx, dy = vec
                mag = math.hypot(dx, dy)
                valid_vectors.append((dx, dy, mag))
                pid_data[pid] = (dx, dy, mag, p.get('bbox'))
                
        if not valid_vectors:
            return events

        # Weighted median for dominant flow
        # Simple weighted sum over total mass (magnitudes)
        sum_mag = sum(v[2] for v in valid_vectors)
        if sum_mag == 0: return events
        
        wx = sum(v[0] * v[2] for v in valid_vectors) / sum_mag
        wy = sum(v[1] * v[2] for v in valid_vectors) / sum_mag
        dom_mag = math.hypot(wx, wy)
        
        if dom_mag < 1.0:
            return events

        # Assess chaos (variance of vectors from dominant)
        variances = [math.hypot(v[0]-wx, v[1]-wy) for v in valid_vectors]
        speed_var = np.var(variances) if len(variances) > 1 else 0
        
        # High chaotic variance -> require > 160 degree reverse. Base threshold 140.
        angle_threshold = 160.0 if speed_var > 20.0 else 140.0

        counter_flows = []
        current_pids = set(pid_data.keys())

        for pid, (dx, dy, mag, bbox) in pid_data.items():
            if mag > 8.0:
                dot = (dx * wx + dy * wy)
                cos_theta = dot / (mag * dom_mag)
                cos_theta = max(-1.0, min(1.0, cos_theta))
                angle_deg = math.degrees(math.acos(cos_theta))
                
                if angle_deg > angle_threshold:
                    self.counter_history[pid] += 1
                    if self.counter_history[pid] >= 5:
                        conf = (1.0 - cos_theta) / 2.0
                        counter_flows.append((pid, bbox, conf, angle_deg))
                else:
                    self.counter_history[pid] = 0
            else:
                self.counter_history[pid] = 0
                
        # Group handling
        if counter_flows:
            # 5+ people counterflow = RED event
            is_coordinated = len(counter_flows) >= 5
            sev = "RED" if is_coordinated else "YELLOW"
            desc_prefix = "Coordinated counter-flow — possible intruder group. " if is_coordinated else "CounterFlow: "
            
            for pid, bbox, conf, angle_deg in counter_flows:
                events.append(AnomalyEvent(
                    detector_name="CounterFlowDetector",
                    severity=sev,
                    confidence=round(conf, 2),
                    person_ids=[pid],
                    description=f"{desc_prefix}{angle_deg:.1f} deg opposite.",
                    bbox=tuple(bbox) if bbox else None
                ))

        # Cleanup
        for pid in list(self.counter_history.keys()):
            if pid not in current_pids:
                del self.counter_history[pid]

        return events


class AbandonedObjectDetector:
    """
    CV2 histogram Re-ID, Retrieval detection, sigmoid confidence.
    """
    def __init__(self, fps: int = 25):
        self.fps = fps
        self.objects = {} 
        self.next_obj_id = 1

    def _compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3); yi1 = max(y1, y3)
        xi2 = min(x2, x4); yi2 = min(y2, y4)
        if xi2 <= xi1 or yi2 <= yi1: return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        return inter / float(((x2-x1)*(y2-y1)) + ((x4-x3)*(y4-y3)) - inter)

    def _get_hist(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1: return None
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        if yolo_detections is None: return []
        events = []
        
        current_detections = [d for d in yolo_detections if d.get('class_id') in [24, 26, 28]]
        
        updated_objects = {}
        for det in current_detections:
            det_bbox = det.get('bbox')
            hist = self._get_hist(frame, det_bbox)
            if hist is None: continue

            best_id = None
            best_score = 0.0
            
            for obj_id, obj in self.objects.items():
                iou = self._compute_iou(det_bbox, obj['bbox'])
                hist_score = 0.0
                if obj.get('hist') is not None:
                    hist_score = cv2.compareHist(hist, obj['hist'], cv2.HISTCMP_CORREL)

                if iou > 0.5 or hist_score > 0.85:
                    score = iou + hist_score * 2.0
                    if score > best_score:
                        best_score = score
                        best_id = obj_id

            if best_id:
                obj = self.objects[best_id].copy()
                
                # Retrieval check: Moved after being unattended and person is close
                ocx, ocy = (obj['bbox'][0] + obj['bbox'][2])/2, (obj['bbox'][1] + obj['bbox'][3])/2
                dcx, dcy = (det_bbox[0] + det_bbox[2])/2, (det_bbox[1] + det_bbox[3])/2
                moved = math.hypot(ocx - dcx, ocy - dcy) > 15

                if moved:
                    if obj.get('flagged', False) and obj.get('person_near', False):
                        events.append(AnomalyEvent(
                            detector_name="AbandonedObjectDetector",
                            severity="GREEN",
                            confidence=0.9,
                            person_ids=[],
                            description=f"Object Retrieved (Class {obj['class_id']})",
                            bbox=tuple(det_bbox),
                        ))
                    # Reset stationary
                    obj['stationary_since'] = frame_idx
                    obj['flagged'] = False

                obj['bbox'] = det_bbox
                obj['last_seen_frame'] = frame_idx
                if hist_score > 0: obj['hist'] = hist
                updated_objects[best_id] = obj
            else:
                updated_objects[self.next_obj_id] = {
                    'bbox': det_bbox,
                    'start_frame': frame_idx,
                    'last_seen_frame': frame_idx,
                    'stationary_since': frame_idx,
                    'unattended_since': frame_idx,
                    'class_id': det.get('class_id'),
                    'hist': hist,
                    'flagged': False,
                    'person_near': False
                }
                self.next_obj_id += 1
                
        # Carry over objects that might have been occluded briefly (keep for 5s)
        for obj_id, obj in self.objects.items():
            if obj_id not in updated_objects:
                if frame_idx - obj['last_seen_frame'] < 5 * self.fps:
                    updated_objects[obj_id] = obj

        self.objects = updated_objects
        
        for obj_id, obj in self.objects.items():
            # Check near persons
            ocx, ocy = (obj['bbox'][0] + obj['bbox'][2])/2, (obj['bbox'][1] + obj['bbox'][3])/2
            person_close = False
            for p in tracked_persons:
                pb = p.get('bbox')
                if pb:
                    pcx, pcy = (pb[0] + pb[2])/2, (pb[1] + pb[3])/2
                    if math.hypot(ocx - pcx, ocy - pcy) < 80:
                        person_close = True
                        break
            
            obj['person_near'] = person_close
            if person_close:
                obj['unattended_since'] = frame_idx
            
            unattended_sec = (frame_idx - obj['unattended_since']) / self.fps
            stationary_sec = (frame_idx - obj['stationary_since']) / self.fps
            
            if stationary_sec >= 60 and unattended_sec >= 60:
                obj['flagged'] = True
                sev = "RED" if unattended_sec > 120 else "YELLOW"
                # conf = 1 / (1 + exp(-0.05 * (seconds_unattended - 60)))
                conf = 1.0 / (1.0 + math.exp(-0.05 * (unattended_sec - 60)))
                
                events.append(AnomalyEvent(
                    detector_name="AbandonedObjectDetector",
                    severity=sev,
                    confidence=round(conf, 2),
                    person_ids=[],
                    description=f"ABANDONED OBJECT for {int(unattended_sec)}s",
                    bbox=tuple(obj['bbox']),
                ))

        return events


class FireSmokeDetector:
    """
    Combined YOLO and HSV smoke/fire with flicker (temporal variance) detection.
    """
    def __init__(self):
        self._model = None
        try:
            from ultralytics import YOLO
            self._model = YOLO("keremberke/yolov8n-fire-detection")
        except Exception:
            pass
        self.prev_gray = None

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_mask = None
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray

        # Base Fire YOLO prediction
        fire_boxes = []
        if self._model:
            try:
                res = self._model.predict(frame, conf=0.4, verbose=False)
                for r in res:
                    for box in r.boxes:
                        fire_boxes.append([int(v) for v in box.xyxy[0].tolist()])
            except: pass

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Smoke Heuristics
        # Gray pixels: S < 50, V in [80, 200]
        smoke_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 200]))
        if diff_mask is not None:
            smoke_mask = cv2.bitwise_and(smoke_mask, diff_mask)

        smoke_ratio = np.count_nonzero(smoke_mask) / (frame.shape[0] * frame.shape[1])
        smoke_bbox = None
        if smoke_ratio > 0.005:
            coords = cv2.findNonZero(smoke_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                smoke_bbox = [x, y, x+w, y+h]
                events.append(AnomalyEvent(
                    detector_name="FireSmokeDetector",
                    severity="YELLOW",
                    confidence=min(smoke_ratio * 80.0, 1.0),
                    person_ids=[],
                    description="SMOKE DETECTED (High Variance Region)",
                    bbox=tuple(smoke_bbox)
                ))

        # Fire Heuristics + Flicker check
        fire_mask = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([30, 255, 255]))
        fire_bboxes = fire_boxes
        
        if not fire_bboxes and np.count_nonzero(fire_mask) > 0.003 * frame.size / 3:
            coords = cv2.findNonZero(fire_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                fire_bboxes.append([x, y, x+w, y+h])

        for fb in fire_bboxes:
            if diff_mask is not None:
                # check flicker variance
                fx1, fy1, fx2, fy2 = fb
                roi_diff = diff_mask[fy1:fy2, fx1:fx2]
                flicker_var = np.var(roi_diff) if roi_diff.size > 0 else 0
                if flicker_var < 10:
                    continue # Likely a red object (no flicker)

            has_smoke = False
            if smoke_bbox:
                # Check intersection
                sx1, sy1, sx2, sy2 = smoke_bbox
                ix1, iy1 = max(fb[0], sx1), max(fb[1], sy1)
                ix2, iy2 = min(fb[2], sx2), min(fb[3], sy2)
                if ix1 < ix2 and iy1 < iy2:
                    has_smoke = True

            if has_smoke:
                events.append(AnomalyEvent(
                    detector_name="FireSmokeDetector",
                    severity="RED",
                    confidence=0.95,
                    person_ids=[],
                    description="CONFIRMED_FIRE_AND_SMOKE",
                    bbox=tuple(fb)
                ))
            else:
                events.append(AnomalyEvent(
                    detector_name="FireSmokeDetector",
                    severity="RED",
                    confidence=0.8,
                    person_ids=[],
                    description="FIRE DETECTED (YOLO/Heuristic)",
                    bbox=tuple(fb)
                ))

        return events


class VandalismDetector:
    """
    Repetitive motion + MOG surface change persistence.
    """
    def __init__(self, fps: int = 25):
        self.fps = fps
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.wrist_history = defaultdict(list)
        self.zones = [] # [(bbox, start_frame, last_active, id)]

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        fgMask = self.backSub.apply(frame)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 2000:
                self.zones.append((cv2.boundingRect(cnt), frame_idx, frame_idx, len(self.zones)))

        # Clean old zones
        self.zones = [z for z in self.zones if frame_idx - z[1] < 30 * self.fps]

        current_pids = set()

        for p in tracked_persons:
            pid = p.get('id', -1)
            kps = p.get('pose_keypoints', [])
            bbox = p.get('bbox')
            if pid < 0 or not kps or len(kps) < 24 or not bbox: continue
            
            current_pids.add(pid)
            rw = kps[16] # Right wrist
            lw = kps[15] # Left wrist
            
            self.wrist_history[pid].append((frame_idx, rw, lw))
            self.wrist_history[pid] = [h for h in self.wrist_history[pid] if frame_idx - h[0] < 5 * self.fps]

            wh = self.wrist_history[pid]
            if len(wh) > 3 * self.fps:
                # Check for oscillating wrist motion in X or Y
                rx_vals = [h[1][0] for h in wh]
                ry_vals = [h[1][1] for h in wh]
                rx_var = np.var(rx_vals)
                ry_var = np.var(ry_vals)
                
                # High variance indicates sweeping motion
                if rx_var > 100 or ry_var > 100:
                    events.append(AnomalyEvent(
                        detector_name="VandalismDetector",
                        severity="YELLOW",
                        confidence=0.7,
                        person_ids=[pid],
                        description="Repetitive arm motion detected.",
                        bbox=tuple(bbox)
                    ))

        # Check surface change persistence
        # If a zone was created recently, and nobody is around it anymore, but it STILL has MOG foreground -> permanent change
        for z in self.zones:
            zb = z[0]
            zx = zb[0] + zb[2]/2
            zy = zb[1] + zb[3]/2

            person_near = False
            for p in tracked_persons:
                if p.get('bbox'):
                    px1, py1, px2, py2 = p['bbox']
                    pcx, pcy = (px1+px2)/2, (py1+py2)/2
                    if math.hypot(zx-pcx, zy-pcy) < 150:
                        person_near = True
                        break
            
            if not person_near:
                # Zone is abandoned. Is there still motion mask active AT the center of this zone?
                # Actually MOG2 will adapt over time. But if it persists for a bit immediately after they leave:
                if frame_idx - z[2] > 2 * self.fps:
                    # check if fgMask is active in the zone
                    x, y, w, h = zb
                    roi = fgMask[y:y+h, x:x+w]
                    if np.count_nonzero(roi) > (w*h*0.3):
                        events.append(AnomalyEvent(
                            detector_name="VandalismDetector",
                            severity="RED",
                            confidence=0.9,
                            person_ids=[],
                            description="CONFIRMED_VANDALISM: Surface alteration persists after subject departure.",
                            bbox=tuple(zb)
                        ))

        # Cleanup
        for pid in list(self.wrist_history.keys()):
            if pid not in current_pids:
                del self.wrist_history[pid]

        return events


class WeaponDetector:
    """
    Second-pass inference on crops using YOLOv8. Proxies knife/scissors.
    """
    def __init__(self):
        self._model = None
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
        except Exception:
            pass

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        if not self._model: return events
        
        for p in tracked_persons:
            bbox = p.get('bbox')
            pid = p.get('id', -1)
            if not bbox: continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1: continue
            crop = frame[y1:y2, x1:x2]
            
            # Predict crop at 320x320
            # Knife: 43, Scissors: 76 (COCO defaults mapped to YOLOv8)
            try:
                res = self._model.predict(crop, imgsz=320, classes=[43, 76], conf=0.45, verbose=False)
                for r in res:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        w_type = self._model.names[cls]
                        c = float(box.conf[0])
                        
                        events.append(AnomalyEvent(
                            detector_name="WeaponDetector",
                            severity="RED",
                            confidence=c,
                            person_ids=[pid],
                            description=f"Weapon detected on person: {w_type}",
                            bbox=tuple(bbox),
                            context={'weapon_type': w_type}
                        ))
            except Exception:
                pass
                
        return events


class SuspiciousRoamingDetector:
    """
    Flags persons visiting >4 zones in 5 mins without staying anywhere >30s, low avg speed.
    """
    def __init__(self, fps: int = 25):
        self.fps = fps
        # pid -> list of (frame, zone_id, cx, cy)
        self.roam = defaultdict(list)
    
    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        h, w = frame.shape[:2]
        # 4x4 Grid
        cell_w, cell_h = w / 4.0, h / 4.0
        current_pids = set()

        for p in tracked_persons:
            pid = p.get('id', -1)
            bbox = p.get('bbox')
            if pid < 0 or not bbox: continue
            
            current_pids.add(pid)
            x1, y1, x2, y2 = bbox
            cx = (x1+x2)/2.0
            cy = (y1+y2)/2.0
            
            r = min(3, max(0, int(cy / cell_h)))
            c = min(3, max(0, int(cx / cell_w)))
            zone = (r, c)
            
            self.roam[pid].append((frame_idx, zone, cx, cy))
            
            # Keep 5 mins history
            self.roam[pid] = [h for h in self.roam[pid] if frame_idx - h[0] <= 300 * self.fps]
            
            history = self.roam[pid]
            if len(history) < 2: continue
            
            # Unique zones
            unique_zones = set([h[1] for h in history])
            num_zones = len(unique_zones)
            
            if num_zones > 4:
                # Check avg speed
                tot_dist = sum(math.hypot(history[i][2]-history[i-1][2], history[i][3]-history[i-1][3]) for i in range(1, len(history)))
                tot_frames = history[-1][0] - history[0][0]
                avg_speed = tot_dist / tot_frames if tot_frames > 0 else 0
                
                if avg_speed < 5.0:
                    # Estimate max time spent in ANY consecutive single zone
                    max_stay = 0
                    current_z = history[0][1]
                    z_start = history[0][0]
                    
                    for i in range(1, len(history)):
                        if history[i][1] != current_z:
                            dur = history[i-1][0] - z_start
                            if dur > max_stay: max_stay = dur
                            current_z = history[i][1]
                            z_start = history[i][0]
                    # final segment
                    dur = history[-1][0] - z_start
                    if dur > max_stay: max_stay = dur
                    
                    max_stay_sec = max_stay / self.fps
                    
                    if max_stay_sec < 30:
                        conf = min(num_zones / 9.0, 1.0)
                        sev = "RED" if num_zones >= 7 or (tot_frames/self.fps) >= 600 else "YELLOW"
                        
                        events.append(AnomalyEvent(
                            detector_name="SuspiciousRoamingDetector",
                            severity=sev,
                            confidence=round(conf, 2),
                            person_ids=[pid],
                            description=f"Suspicious roaming: visited {num_zones} zones in {(tot_frames/self.fps):.0f}s, speed: {avg_speed:.1f}px",
                            bbox=tuple(bbox)
                        ))

        # Cleanup
        for pid in list(self.roam.keys()):
            if pid not in current_pids:
                del self.roam[pid]

        return events


class SuddenDispersalDetector:
    """
    Unchanged from previous except to fulfill the default registry needs.
    Operates on the full crowd.
    """
    def __init__(self):
        self.crowd_history: List[Tuple[int, List[Tuple[float, float]]]] = []
        
    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        centroids = []
        for p in tracked_persons:
            bbox = p.get('bbox')
            if bbox:
                centroids.append(((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0))
                
        self.crowd_history.append((frame_idx, centroids))
        self.crowd_history = [h for h in self.crowd_history if frame_idx - h[0] <= 30]
        
        target_hist = None
        for h in self.crowd_history:
            if target_hist is None or target_hist[0] > h[0]:
                target_hist = h
                
        def mean_pairwise_distance(pts: List[Tuple[float, float]]) -> float:
            if len(pts) < 2: return 0.0
            total_dist = 0.0
            count = 0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    total_dist += math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                    count += 1
            return total_dist / count if count > 0 else 0.0
            
        current_mean_dist = mean_pairwise_distance(centroids)
        if target_hist is not None and len(target_hist[1]) >= 2 and len(centroids) >= 2:
            old_mean_dist = mean_pairwise_distance(target_hist[1])
            if old_mean_dist > 0:
                increase_ratio = (current_mean_dist - old_mean_dist) / old_mean_dist
                if increase_ratio > 0.60:
                    conf = min(increase_ratio, 1.0)
                    events.append(AnomalyEvent(
                        detector_name="SuddenDispersalDetector",
                        severity="RED",
                        confidence=round(conf, 2),
                        person_ids=[],
                        description=f"STAMPEDE_DISPERSAL: Crowd dist increased by {int(increase_ratio * 100)}%"
                    ))
        return events

class ExitBlockingDetector:
    """
    Unchanged from previous.
    """
    EXITS: List[Dict[str, Any]] = []

    def __init__(self, fps: int = 25):
        self.fps = fps
        self.blocked_history = {}

    def detect(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        events = []
        for i, config in enumerate(self.EXITS):
            bbox = config.get('bbox')
            capacity = config.get('capacity', 5)
            if not bbox: continue
                
            ex1, ey1, ex2, ey2 = bbox
            count = 0
            pids = []
            
            for p in tracked_persons:
                p_bbox = p.get('bbox')
                pid = p.get('id', -1)
                if p_bbox and pid >= 0:
                    pcx = (p_bbox[0]+p_bbox[2])/2.0
                    pcy = (p_bbox[1]+p_bbox[3])/2.0
                    if ex1 <= pcx <= ex2 and ey1 <= pcy <= ey2:
                        count += 1
                        pids.append(pid)
                        
            if count > capacity:
                self.blocked_history[i] = self.blocked_history.get(i, 0) + 1
                seconds_blocked = self.blocked_history[i] / self.fps
                severity = "YELLOW"
                desc = f"Exit {i} blocked: {count}/{capacity} persons"
                if seconds_blocked > 30:
                    severity = "RED"
                    desc = f"FULLY_BLOCKED: Exit {i} blocked for {int(seconds_blocked)}s"
                    
                events.append(AnomalyEvent(
                    detector_name="ExitBlockingDetector",
                    severity=severity,
                    confidence=1.0,
                    person_ids=pids,
                    description=desc,
                    bbox=tuple(bbox)
                ))
            else:
                if i in self.blocked_history:
                    del self.blocked_history[i]
        return events


class DetectorRegistry:
    def __init__(self):
        self.detectors = []
        
    def register(self, detector: Any):
        self.detectors.append(detector)
        
    def run_all(self, tracked_persons: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, yolo_detections: List[Dict[str, Any]] = None) -> List[AnomalyEvent]:
        all_events = []
        for det in self.detectors:
            try:
                events = det.detect(tracked_persons, frame, frame_idx, yolo_detections)
                if events:
                    all_events.extend(events)
            except Exception as e:
                print(f"Error in {det.__class__.__name__}: {e}")
                pass
        return all_events


# Setup the default registry spanning all models outlined
DEFAULT_REGISTRY = DetectorRegistry()
DEFAULT_REGISTRY.register(LoiteringDetector())
DEFAULT_REGISTRY.register(FallingDetector())
DEFAULT_REGISTRY.register(SuddenDispersalDetector())
DEFAULT_REGISTRY.register(CrowdDensityDetector())
DEFAULT_REGISTRY.register(CounterFlowDetector())
DEFAULT_REGISTRY.register(ExitBlockingDetector())
DEFAULT_REGISTRY.register(FireSmokeDetector())
DEFAULT_REGISTRY.register(AbandonedObjectDetector())
DEFAULT_REGISTRY.register(VandalismDetector())
DEFAULT_REGISTRY.register(WeaponDetector())
DEFAULT_REGISTRY.register(SuspiciousRoamingDetector())

def get_default_registry() -> DetectorRegistry:
    """Returns the default populated detector registry instance."""
    return DEFAULT_REGISTRY
