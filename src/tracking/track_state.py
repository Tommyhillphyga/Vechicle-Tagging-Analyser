# src/tracking/track_state.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class VehicleTrackState:
    track_id: int
    bbox: Tuple[int, int, int, int] = None
    frames_seen: int = 0
    snapshot_taken: bool = False

    def crop(self, frame):
        if self.bbox is None:
            return None
        x1, y1, x2, y2 = self.bbox
        return frame[y1:y2, x1:x2]