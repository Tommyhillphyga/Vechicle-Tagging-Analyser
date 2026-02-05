# src/embeddings/driver_embedder.py
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

# try keras-facenet first
try:
    from keras_facenet import FaceNet
    _HAS_FACENET = True
except Exception:
    _HAS_FACENET = False

# fallback to face_recognition encodings if available
try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False

class DriverEmbedder:
    """
    Produce normalized driver face embeddings.
    Primary: keras-facenet -> returns 512-d vector.
    Secondary: face_recognition.face_encodings -> 128-d vector (pad to 512).
    Final fallback: simple histogram-based pseudo-embedding.
    """

    def __init__(self):
        if _HAS_FACENET:
            self.model = FaceNet()
        else:
            self.model = None
            if _HAS_FR:
                logger.info("Using face_recognition for driver embeddings")
            else:
                logger.warning("No face embedding backend available; using fallback histograms")

    def embed(self, face_crops):
        if not face_crops:
            return np.zeros(512, dtype=np.float32)

        if self.model is not None:
            embs = []
            for f in face_crops:
                try:
                    resized = cv2.resize(f, (160,160))
                    resized = resized.astype("float32")
                    resized = np.expand_dims(resized, 0)
                    emb = self.model.embeddings(resized)[0]
                    embs.append(emb)
                except Exception:
                    continue
            if not embs:
                return np.zeros(512, dtype=np.float32)
            avg = np.mean(embs, axis=0)
            return (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)

        if _HAS_FR:
            embs = []
            for f in face_crops:
                try:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    enc = face_recognition.face_encodings(rgb)
                    if len(enc):
                        embs.append(enc[0])
                except Exception:
                    continue
            if not embs:
                return np.zeros(512, dtype=np.float32)
            avg = np.mean(embs, axis=0)
            # pad 128->512
            if avg.shape[0] == 128:
                padded = np.zeros(512, dtype=np.float32)
                padded[:128] = avg
                avg = padded
            return (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)

        # fallback: basic histogram
        hist = []
        for f in face_crops:
            try:
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256]).flatten()
                hist.append(h)
            except Exception:
                continue
        if not hist:
            return np.zeros(512, dtype=np.float32)
        avg = np.mean(hist, axis=0)
        # resize/pad to 512
        if avg.size < 512:
            padded = np.zeros(512, dtype=np.float32)
            padded[:avg.size] = avg
            avg = padded
        avg = avg.astype(np.float32)
        return (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)