from core.matcher import VehicleDriverMatcher
from core.clustering import cluster_snapshots
from io.frame_loader import FrameLoader
from tracking.bytetrack_manager import ByteTrackManager
from detection.vehicle_detector import VehicleDetector
from detection.face_detector import FaceDetector
from embeddings.vehicle_embedder import VehicleEmbedder
from embeddings.driver_embedder import DriverEmbedder
from data_models.snapshot import VehicleSnapshot
import time

class VehicleDriverPipeline:
    def __init__(self, entry_source, exit_source):
        self.entry_source = entry_source
        self.exit_source = exit_source

        self.detector = VehicleDetector()
        self.tracker = ByteTrackManager()
        self.face_detector = FaceDetector()
        self.vehicle_embedder = VehicleEmbedder()
        self.driver_embedder = DriverEmbedder()

    def _process_source(self, source):
        loader = FrameLoader(source)
        snapshots = []

        self.tracker.reset()

        for frame in loader:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)

            for track in tracks:
                if track.completed:
                    continue

                crop = track.crop(frame)
                faces = self.face_detector.detect_faces(crop)

                if not faces:
                    continue

                vehicle_emb = self.vehicle_embedder.embed(crop)
                driver_emb = self.driver_embedder.embed(faces)

                snapshot = VehicleSnapshot(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    vehicle_embedding=vehicle_emb,
                    driver_embedding=driver_emb,
                    timestamp=time.time()
                )

                snapshots.append(snapshot)
                self.tracker.mark_completed(track.track_id)

        return cluster_snapshots(snapshots)

    def run(self):
        entry_clusters = self._process_source(self.entry_source)
        exit_clusters = self._process_source(self.exit_source)

        matcher = VehicleDriverMatcher()
        results = matcher.match(entry_clusters, exit_clusters)

        for r in results:
            print(r)