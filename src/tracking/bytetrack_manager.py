# src/tracking/bytetrack_manager.py
import logging
from typing import Dict, List
import supervision as sv
from supervision ByteTrack
from .track_state import VehicleTrackState

logger = logging.getLogger(__name__)

class ByteTrackManager:
    """
    Thin wrapper around supervision ByteTrack to maintain per-track state.
    """

    def __init__(self, frame_rate:int = 10):
        self.tracker = ByteTrack(frame_rate=frame_rate)
        self.tracks: Dict[int, VehicleTrackState] = {}

    def reset(self):
        self.tracker.reset()
        self.tracks.clear()

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Pass detections to ByteTrack. Returns the tracked detections object
        which contains xyxy, confidence, and tracker_id arrays.
        """
        try:
            tracked = self.tracker.update_with_detections(detections)
            # update local track states
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                xyxy = tracked.xyxy[i]
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                if tid not in self.tracks:
                    self.tracks[tid] = VehicleTrackState(track_id=tid)
                st = self.tracks[tid]
                st.frames_seen += 1
                st.bbox = bbox
            return tracked
        except Exception as e:
            logger.exception("ByteTrack update failed: %s", e)
            return sv.Detections.empty()

    def is_completed(self, track_id: int) -> bool:
        return self.tracks.get(track_id, VehicleTrackState(track_id)).snapshot_taken

    def mark_completed(self, track_id: int):
        if track_id in self.tracks:
            self.tracks[track_id].snapshot_taken = True