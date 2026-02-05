# src/embeddings/vehicle_embedder.py
import numpy as np
import cv2
import logging
import torch

logger = logging.getLogger(__name__)

# Try to import a reid loader if available. If not, fallback to color-histogram.
try:
    from reid_model.load_reid_model import load_model_from_opts
    _HAS_REID = True
except Exception:
    _HAS_REID = False

class VehicleEmbedder:
    """
    Try to use a loaded ReID model (if user has their model). If not present,
    compute a color histogram based descriptor (normalized) of size 512.
    """

    def __init__(self, reid_opts=None, reid_ckpt=None, device='cpu'):
        self.device = device
        self.model = None
        if _HAS_REID and reid_opts and reid_ckpt:
            try:
                logger.info("Loading vehicle ReID model")
                self.model = load_model_from_opts(reid_opts, ckpt=reid_ckpt, remove_classifier=True)
                self.model.eval()
            except Exception as e:
                logger.warning("Failed to init ReID model: %s", e)
                self.model = None

    def embed(self, vehicle_crop):
        if vehicle_crop is None or vehicle_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        if self.model is not None:
            try:
                img = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224,224))
                img = img.transpose(2,0,1).astype("float32")
                tensor = torch.from_numpy(img).unsqueeze(0)
                with torch.no_grad():
                    out = self.model(tensor).cpu().numpy()[0]
                out = out.astype(np.float32)
                return out / (np.linalg.norm(out) + 1e-8)
            except Exception:
                logger.exception("ReID model failed; falling back to histogram")

        # Fallback histogram: multichannel concatenated histograms (size adjustable -> 512)
        try:
            hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0], None, [64], [0,180]).flatten()
            s = cv2.calcHist([hsv], [1], None, [64], [0,256]).flatten()
            v = cv2.calcHist([hsv], [2], None, [64], [0,256]).flatten()
            hist = np.concatenate([h, s, v])  # length 192
            if hist.size < 512:
                padded = np.zeros(512, dtype=np.float32)
                padded[:hist.size] = hist
                hist = padded
            hist = hist.astype(np.float32)
            hist = hist / (np.linalg.norm(hist) + 1e-8)
            return hist
        except Exception:
            return np.zeros(512, dtype=np.float32)