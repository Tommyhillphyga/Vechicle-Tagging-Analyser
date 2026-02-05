# src/io/frame_loader.py
import os
import cv2
from typing import Iterator, Tuple
import logging

logger = logging.getLogger(__name__)

class FrameLoader:
    """
    Iterates over image files in a directory (jpg, png, jpeg), returning (path, frame).
    """

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"FrameLoader: directory not found: {path}")
        self.files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __iter__(self) -> Iterator[Tuple[str, any]]:
        for f in self.files:
            full = os.path.join(self.path, f)
            frame = cv2.imread(full)
            if frame is None:
                logger.warning("Failed to read frame: %s", full)
                continue
            yield full, frame