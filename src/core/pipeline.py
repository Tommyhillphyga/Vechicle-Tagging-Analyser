# src/core/pipeline.py
import os
import logging
import time
from typing import List, Dict, Any

from detection.vehicle_detector import VehicleDetector
from detection.face_detector import FaceDetector
from tracking.bytetrack_manager import ByteTrackManager
from embeddings.driver_embedder import DriverEmbedder
from embeddings.vehicle_embedder import VehicleEmbedder
from io.frame_loader import FrameLoader
from data_models.snapshot import VehicleSnapshot
from core.clustering import cluster_snapshots
from core.matcher import VehicleDriverMatcher

logger = logging.getLogger(__name__)

class VehicleDriverPipeline:
    def __init__(
        self,
        entry_frames_path: str,
        exit_frames_path: str,
        output_path: str = "./data/outputs",
        vehicle_model_path: str = "yolov8m.pt",
        face_model_path: str = None,
        reid_opts: str = None,
        reid_ckpt: str = None,
        vehicle_similarity_threshold: float = 0.7,
        driver_similarity_threshold: float = 0.6,
        overall_match_threshold: float = 0.5,
        video_fps: int = 10,
        verbose: bool = True
    ):
        self.entry_frames_path = entry_frames_path
        self.exit_frames_path = exit_frames_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = verbose

        # detectors / trackers / embedders
        self.vehicle_detector = VehicleDetector(model_path=vehicle_model_path)
        self.face_detector = FaceDetector(yolov8_face_model=face_model_path)
        self.tracker = ByteTrackManager(frame_rate=video_fps)
        self.driver_embedder = DriverEmbedder()
        self.vehicle_embedder = VehicleEmbedder(reid_opts=reid_opts, reid_ckpt=reid_ckpt)

        # thresholds
        self.vehicle_similarity_threshold = vehicle_similarity_threshold
        self.driver_similarity_threshold = driver_similarity_threshold
        self.overall_match_threshold = overall_match_threshold

        # stats
        self.stats = {
            'entry_frames_processed': 0,
            'exit_frames_processed': 0,
            'entry_vehicles_detected': 0,
            'exit_vehicles_detected': 0,
            'entry_clusters': 0,
            'exit_clusters': 0,
            'matches_found': 0,
            'mismatches_detected': 0,
            'no_match_found': 0
        }

    # main
    def run_analysis(self) -> Dict[str, Any]:
        logger.info("="*80)
        logger.info("STARTING ANALYSIS PIPELINE")
        logger.info("="*80)

        # Entry
        logger.info("Processing entry frames...")
        entry_snapshots = self._process_frames_batch(self.entry_frames_path, is_entry=True)
        self.stats['entry_frames_processed'] = len(list(filter(lambda f: f.lower().endswith(('.jpg','.png','.jpeg')), os.listdir(self.entry_frames_path))))
        self.stats['entry_vehicles_detected'] = len(entry_snapshots)
        logger.info("Found %d entry snapshots", len(entry_snapshots))

        # Exit
        logger.info("Processing exit frames...")
        exit_snapshots = self._process_frames_batch(self.exit_frames_path, is_entry=False)
        self.stats['exit_frames_processed'] = len(list(filter(lambda f: f.lower().endswith(('.jpg','.png','.jpeg')), os.listdir(self.exit_frames_path))))
        self.stats['exit_vehicles_detected'] = len(exit_snapshots)
        logger.info("Found %d exit snapshots", len(exit_snapshots))

        # Clustering
        logger.info("Clustering entry snapshots...")
        entry_clusters = cluster_snapshots(entry_snapshots, threshold=self.vehicle_similarity_threshold)
        for c in entry_clusters: c.finalize()
        self.stats['entry_clusters'] = len(entry_clusters)
        logger.info("Created %d entry clusters", len(entry_clusters))

        logger.info("Clustering exit snapshots...")
        exit_clusters = cluster_snapshots(exit_snapshots, threshold=self.vehicle_similarity_threshold)
        for c in exit_clusters: c.finalize()
        self.stats['exit_clusters'] = len(exit_clusters)
        logger.info("Created %d exit clusters", len(exit_clusters))

        # Matching
        logger.info("Matching exit clusters to entry clusters...")
        matcher = VehicleDriverMatcher(driver_threshold=self.driver_similarity_threshold, overall_threshold=self.overall_match_threshold)
        match_results = matcher.match(entry_clusters, exit_clusters)

        # Update stats & optional alerts
        for r in match_results:
            if r["is_match"]:
                self.stats['matches_found'] += 1
            else:
                if r["reason"] == "no_entry_driver_found":
                    self.stats['no_match_found'] += 1
                else:
                    self.stats['mismatches_detected'] += 1
                    # Optionally create an alert video (not implemented here to keep simple)

        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        self._print_summary()
        return {
            'entry_clusters': entry_clusters,
            'exit_clusters': exit_clusters,
            'match_results': match_results,
            'stats': self.stats
        }

    # frame batch processing with tracker
    def _process_frames_batch(self, frames_dir: str, is_entry: bool):
        snapshots = []
        loader = FrameLoader(frames_dir)
        self.tracker.reset()

        for img_path, frame in loader:
            try:
                detections = self.vehicle_detector.detect(frame)
                tracked = self.tracker.update_with_detections(detections)

                # tracked has attributes xyxy, confidence, tracker_id
                if getattr(tracked, "xyxy", None) is None or len(tracked.xyxy) == 0:
                    continue

                for i, tid in enumerate(tracked.tracker_id):
                    tid = int(tid)
                    # If snapshot already taken for this track, skip
                    if self.tracker.is_completed(tid):
                        continue

                    x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                    # sanity check
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:
                        continue

                    # Detect driver faces in vehicle
                    driver_crops = self.face_detector.detect_driver_faces(frame, (x1,y1,x2,y2))
                    if not driver_crops:
                        # wait for next frame
                        continue

                    # Extract embeddings
                    vehicle_emb = self.vehicle_embedder.embed(vehicle_crop)
                    driver_emb = self.driver_embedder.embed(driver_crops)

                    snapshot = VehicleSnapshot(
                        track_id=tid,
                        frame_path=img_path,
                        bbox=(x1,y1,x2,y2),
                        vehicle_crop=vehicle_crop,
                        driver_crops=driver_crops,
                        vehicle_embedding=vehicle_emb,
                        driver_embedding=driver_emb,
                        timestamp=time.time(),
                        is_entry=is_entry
                    )

                    snapshots.append(snapshot)
                    self.tracker.mark_completed(tid)
                    if self.verbose:
                        logger.info("Captured snapshot for track %s from frame %s", tid, os.path.basename(img_path))

            except Exception as e:
                logger.exception("Error processing frame %s: %s", img_path, e)
                continue

        return snapshots

    def _print_summary(self):
        logger.info("📊 ANALYSIS SUMMARY:")
        for k, v in self.stats.items():
            logger.info("   %s: %s", k, v)
        logger.info("📁 Output directory: %s", self.output_path)