class VehicleTrackState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.frames_seen = 0
        self.completed = False
        self.bbox = None

    def crop(self, frame):
        x1, y1, x2, y2 = self.bbox
        return frame[y1:y2, x1:x2]