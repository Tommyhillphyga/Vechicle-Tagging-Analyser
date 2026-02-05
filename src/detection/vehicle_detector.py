# src/detection/vehicle_detector.py
import numpy as np
import logging
from typing import List, Tuple
import supervision as sv

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

class VehicleDetector:
    """
    Wraps a YOLO detector (ultralytics) and converts results to supervision.Detections.
    If YOLO is not installed, returns empty detections.
    """
    def __init__(self, model_path: str = "yolov8m.pt", conf: float = 0.5):
        self.conf = conf
        self.model = None
        if _YOLO_AVAILABLE:
            try:
                logger.info("Loading YOLO vehicle model: %s", model_path)
                self.model = YOLO(model_path)
            except Exception as e:
                logger.warning("Failed to load YOLO model [%s]: %s", model_path, e)
                self.model = None
        else:
            logger.warning("ultralytics YOLO not available. Vehicle detection will be disabled.")

    def detect(self, frame) -> sv.Detections:
        """
        Returns supervision.Detections with fields xyxy (N,4) and confidence (N,)
        """
        if self.model is None:
            return sv.Detections.empty()

        try:
            results = self.model(frame, verbose=False, conf=self.conf, classes=[2,3,5,7])  # car, motorbike, bus, truck
            boxes = []
            scores = []
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for b in r.boxes:
                    coords = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else b.xyxy[0]
                    x1, y1, x2, y2 = map(int, coords)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(b.conf[0]))
            if not boxes:
                return sv.Detections.empty()
            return sv.Detections(xyxy=np.array(boxes), confidence=np.array(scores))
        except Exception as e:
            logger.exception("Vehicle detection failed: %s", e)
            return sv.Detections.empty()