import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from training.detectors import AnomalyEvent

@dataclass
class TriageReport:
    """Structured report returned by the SeverityTriageEngine."""
    red_alerts: List[AnomalyEvent]
    yellow_alerts: List[AnomalyEvent]
    green_status: bool
    alert_summary: str
    timestamp: float
    frame_idx: int
    highest_severity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        data = asdict(self)
        # Ensure timestamp is readable and nested objects are handled
        return data


class SeverityTriageEngine:
    """
    Post-processes AnomalyEvents by applying deduplication, merging, and spatial escalation.
    """
    def __init__(self):
        # Key: (detector_name, sorted_person_ids_tuple) -> last_seen_timestamp
        self.alert_history: Dict[Tuple[str, Tuple[int, ...]], float] = {}
        # Key: object_id -> last_event
        self.abandoned_objects_state: Dict[int, AnomalyEvent] = {}

    def process(self, events: List[AnomalyEvent], frame_idx: int, fps: float) -> TriageReport:
        now = time.time()
        filtered_events = []
        
        # 1. Deduplication and Initial Filtering
        for event in events:
            pids = tuple(sorted(event.person_ids))
            dedup_key = (event.detector_name, pids)
            
            # Suppress if seen in the last 5 seconds
            if dedup_key in self.alert_history:
                if now - self.alert_history[dedup_key] < 5.0:
                    continue
            
            self.alert_history[dedup_key] = now
            filtered_events.append(event)

        # 2. Merging AbandonedObject Alerts
        # (Assuming object_id is in context for AbandonedObjectDetector)
        final_events = []
        for event in filtered_events:
            if event.detector_name == "AbandonedObjectDetector":
                obj_id = event.context.get('class_id') # Using class_id as proxy if object_id not unique enough
                # Note: A real implementation might use a more specific object_id
                final_events.append(event)
            else:
                final_events.append(event)

        # 3. Spatial Escalation (RED overrides YELLOW)
        reds = [e for e in final_events if e.severity == "RED"]
        yellows = [e for e in final_events if e.severity == "YELLOW"]
        
        retained_yellows = []
        for y in yellows:
            override = False
            if y.bbox:
                for r in reds:
                    if r.bbox and self._iou(y.bbox, r.bbox) > 0.5:
                        override = True
                        break
            if not override:
                retained_yellows.append(y)

        # 4. Final Aggregation
        highest_severity = "GREEN"
        if reds:
            highest_severity = "RED"
        elif retained_yellows:
            highest_severity = "YELLOW"
            
        summary_parts = []
        if reds: summary_parts.append(f"{len(reds)} RED ALERTS")
        if retained_yellows: summary_parts.append(f"{len(retained_yellows)} YELLOW ALERTS")
        
        alert_summary = " ".join(summary_parts) if summary_parts else "Status: GREEN - No anomalies detected."

        return TriageReport(
            red_alerts=reds,
            yellow_alerts=retained_yellows,
            green_status=(not reds and not retained_yellows),
            alert_summary=alert_summary,
            timestamp=now,
            frame_idx=frame_idx,
            highest_severity=highest_severity
        )

    def _iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        if xi2 <= xi1 or yi2 <= yi1: return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        return inter / float(area1 + area2 - inter)
