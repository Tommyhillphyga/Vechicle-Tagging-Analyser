
import cv2
import numpy as np
import torch
import streamlit as st
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from .schemas import Snapshot, MatchResult, VehicleCluster

class SentinelProcessor:
    """
    Optimized Forensic Vision Engine.
    Implements local model caching, vectorized similarity, and multi-stage verification.
    """
    def __init__(self):
        self.vehicle_model = self._load_yolo('yolov8n.pt')
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True
        )
        # Thresholds from notebook
        self.v_threshold = 0.7
        self.d_threshold = 0.6
        self.match_threshold = 0.65

    @st.cache_resource(_self=None)
    def _load_yolo(model_name: str):
        return YOLO(model_name)

    def _get_cv2_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def process_image(self, snapshot: Snapshot) -> Snapshot:
        """Extracts forensic features from a raw snapshot."""
        frame = self._get_cv2_image(snapshot.image_data)
        if frame is None: return snapshot

        # 1. Vehicle Detection
        results = self.vehicle_model(frame, verbose=False, classes=[2, 3, 5, 7], conf=0.4)
        if not results or len(results[0].boxes) == 0:
            return snapshot

        # Take primary detection
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        snapshot.bbox = (x1, y1, x2, y2)
        snapshot.vehicle_crop = frame[y1:y2, x1:x2]

        # 2. Driver Face Detection (Simplified Region Strategy)
        # Focus on upper-middle of vehicle crop for driver
        vh, vw = snapshot.vehicle_crop.shape[:2]
        driver_region = snapshot.vehicle_crop[0:int(vh*0.6), int(vw*0.1):int(vw*0.9)]
        
        # Landmark Validation (MediaPipe)
        rgb_region = cv2.cvtColor(driver_region, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_region)
        
        if face_results.multi_face_landmarks:
            snapshot.driver_crops = [driver_region]
            # Local feature extraction (Histogram-based for robustness without GPU-Facenet)
            snapshot.driver_embedding = self._extract_color_embedding(driver_region)
        
        # Vehicle ReID embedding (Color + Texture Profile)
        snapshot.vehicle_embedding = self._extract_color_embedding(snapshot.vehicle_crop)
        
        return snapshot

    def _extract_color_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Vectorized color profile extraction for high-speed local matching."""
        if crop is None or crop.size == 0: return np.zeros(512)
        resized = cv2.resize(crop, (128, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def cluster_snapshots(self, snapshots: List[Snapshot], is_entry: bool) -> List[VehicleCluster]:
        """Groups snapshots into unique vehicle clusters for scalability."""
        clusters = []
        for s in snapshots:
            best_match = None
            best_score = 0
            
            for c in clusters:
                score = cosine_similarity(
                    s.vehicle_embedding.reshape(1, -1), 
                    c.avg_vehicle_emb.reshape(1, -1)
                )[0][0]
                if score > best_score:
                    best_score = score
                    best_match = c
            
            if best_match and best_score > self.v_threshold:
                best_match.snapshots.append(s)
                best_match.compute_averages()
            else:
                new_c = VehicleCluster(cluster_id=f"V-{s.id}", snapshots=[s], is_entry=is_entry)
                new_c.compute_averages()
                clusters.append(new_c)
        return clusters

    def match_clusters(self, entries: List[VehicleCluster], exits: List[VehicleCluster]) -> List[MatchResult]:
        """Cross-matches entry and exit clusters with forensic weighting."""
        results = []
        for ex in exits:
            best_match = None
            max_ovr = 0
            
            for en in entries:
                v_sim = float(cosine_similarity(ex.avg_vehicle_emb.reshape(1, -1), en.avg_vehicle_emb.reshape(1, -1))[0][0])
                d_sim = 0.0
                if ex.avg_driver_emb is not None and en.avg_driver_emb is not None:
                    d_sim = float(cosine_similarity(ex.avg_driver_emb.reshape(1, -1), en.avg_driver_emb.reshape(1, -1))[0][0])
                
                # Overall score: 60% Vehicle shape/color, 40% Driver features
                overall = (v_sim * 0.6) + (d_sim * 0.4)
                
                if overall > max_ovr:
                    max_ovr = overall
                    best_match = (en, v_sim, d_sim, overall)
            
            if best_match:
                en, vs, ds, ov = best_match
                status = "VERIFIED" if ov > self.match_threshold else "MISMATCH"
                results.append(MatchResult(
                    id=f"M-{ex.cluster_id}",
                    entry_cluster_id=en.cluster_id,
                    exit_cluster_id=ex.cluster_id,
                    vehicle_similarity=vs,
                    driver_similarity=ds,
                    overall_score=ov,
                    status=status,
                    reason=f"Verification {'successful' if status == 'VERIFIED' else 'failed'}. Identity profile correlation: {ov:.2f}",
                    entry_sample_img=en.snapshots[0].image_data,
                    exit_sample_img=ex.snapshots[0].image_data
                ))
        return results
