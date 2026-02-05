# src/data_models/snapshot.py
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

@dataclass(frozen=True)
class VehicleSnapshot:
    track_id: int
    frame_path: str
    bbox: Tuple[int, int, int, int]
    vehicle_crop: np.ndarray
    driver_crops: List[np.ndarray]
    vehicle_embedding: np.ndarray
    driver_embedding: np.ndarray
    timestamp: float
    is_entry: bool