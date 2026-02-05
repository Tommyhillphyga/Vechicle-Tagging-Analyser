# src/detection/face_detector.py
import cv2
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Try different backends
try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False

# YOLO face model via ultralytics is optional
try:
    from ultralytics import YOLO
    _HAS_YOLO_FACE = True
except Exception:
    _HAS_YOLO_FACE = False

class FaceDetector:
    """
    Multi-backend face detector:
      - Try YOLO face if provided (fast/good)
      - Else try face_recognition (dlib hog)
      - Else fallback to Haar cascade (OpenCV)
    """

    def __init__(self, yolov8_face_model: str = None):
        self.yolo_face = None
        if yolov8_face_model and _HAS_YOLO_FACE:
            try:
                self.yolo_face = YOLO(yolov8_face_model)
                logger.info("Loaded YOLO face model")
            except Exception as e:
                logger.warning("Could not load YOLO face model: %s", e)
                self.yolo_face = None

        if not _HAS_FR and self.yolo_face is None:
            # Haar cascade fallback
            try:
                self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                logger.info("Using OpenCV Haar cascade for faces")
            except Exception as e:
                logger.error("Haar cascade not available: %s", e)
                self.haar = None
        else:
            self.haar = None

    def detect_faces_in_region(self, region: np.ndarray) -> List[np.ndarray]:
        """
        Return list of face crops found inside the given region image.
        """
        faces = []

        if region is None or region.size == 0:
            return faces

        # 1) YOLO face detector (if loaded)
        if self.yolo_face is not None:
            try:
                results = self.yolo_face(region, verbose=False, conf=0.3)
                for r in results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    for b in r.boxes:
                        coords = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else b.xyxy[0]
                        x1, y1, x2, y2 = map(int, coords)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop = region[y1:y2, x1:x2]
                        if crop.size > 0 and self._check_face_quality(crop):
                            faces.append(crop)
                if faces:
                    return faces
            except Exception:
                logger.exception("YOLO face detection failed; falling back")

        # 2) face_recognition (dlib HOG)
        if _HAS_FR:
            try:
                rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model="hog")
                for (top, right, bottom, left) in locs:
                    crop = region[top:bottom, left:right]
                    if crop.size > 0 and self._check_face_quality(crop):
                        faces.append(crop)
                if faces:
                    return faces
            except Exception:
                logger.exception("face_recognition detection failed; falling back")

        # 3) Haar cascade
        if self.haar is not None:
            try:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                rects = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
                for (x, y, w, h) in rects:
                    crop = region[y:y+h, x:x+w]
                    if crop.size > 0 and self._check_face_quality(crop):
                        faces.append(crop)
                return faces
            except Exception:
                logger.exception("Haar cascade failed")

        return faces

    def detect_driver_faces(self, frame: np.ndarray, vehicle_bbox: Tuple[int,int,int,int]) -> List[np.ndarray]:
        """
        Multi-region strategy based on vehicle bbox.
        Only searches upper half of vehicle and tries right-side first (driver side for right-hand traffic).
        """
        x1, y1, x2, y2 = vehicle_bbox
        w = x2 - x1
        h = y2 - y1
        regions = []

        # right upper (driver side in right-hand traffic), prefer this
        rx1 = int(x1 + 0.4 * w)
        rx2 = x2
        ry1 = y1
        ry2 = int(y1 + 0.6 * h)
        regions.append(frame[ry1:ry2, rx1:rx2])

        # left upper
        lx1 = x1
        lx2 = int(x1 + 0.6 * w)
        ly1 = y1
        ly2 = int(y1 + 0.6 * h)
        regions.append(frame[ly1:ly2, lx1:lx2])

        # center upper
        cx1 = int(x1 + 0.25 * w)
        cx2 = int(x2 - 0.25 * w)
        cy1 = y1
        cy2 = int(y1 + 0.5 * h)
        regions.append(frame[cy1:cy2, cx1:cx2])

        # full vehicle
        regions.append(frame[y1:y2, x1:x2])

        # try regions in order and return as soon as any faces are found
        for region in regions:
            if region is None or region.size == 0:
                continue
            faces = self.detect_faces_in_region(region)
            if faces:
                return faces

        return []

    def _check_face_quality(self, face_crop: np.ndarray) -> bool:
        # Basic quality checks: size, brightness, variance
        try:
            h, w = face_crop.shape[:2]
            if h < 20 or w < 20:
                return False
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
            if np.std(gray) < 10:
                return False
            mean_brightness = np.mean(gray)
            if mean_brightness < 15 or mean_brightness > 240:
                return False
            return True
        except Exception:
            return False