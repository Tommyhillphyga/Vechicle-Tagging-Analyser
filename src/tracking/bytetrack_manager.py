from tracking.track_state import VehicleTrackState

class ByteTrackManager:
    def __init__(self):
        self.tracks = {}

    def reset(self):
        self.tracks.clear()

    def update(self, detections):
        # Placeholder for ByteTrack integration
        updated = []

        for det in detections:
            track_id = det["id"]
            state = self.tracks.setdefault(
                track_id, VehicleTrackState(track_id)
            )
            state.frames_seen += 1
            state.bbox = det["bbox"]
            updated.append(state)

        return updated

    def mark_completed(self, track_id):
        if track_id in self.tracks:
            self.tracks[track_id].completed = True