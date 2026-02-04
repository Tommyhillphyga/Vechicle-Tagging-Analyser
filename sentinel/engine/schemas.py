
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Any
import numpy as np
import uuid

@dataclass
class Snapshot:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    image_data: bytes = None
    file_name: str = ""
    # Forensic features
    vehicle_crop: Optional[np.ndarray] = None
    driver_crops: List[np.ndarray] = field(default_factory=list)
    vehicle_embedding: Optional[np.ndarray] = None
    driver_embedding: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class VehicleCluster:
    """Represents a group of snapshots identified as the same vehicle."""
    cluster_id: str
    snapshots: List[Snapshot] = field(default_factory=list)
    avg_vehicle_emb: Optional[np.ndarray] = None
    avg_driver_emb: Optional[np.ndarray] = None
    is_entry: bool = True

    def compute_averages(self):
        v_embs = [s.vehicle_embedding for s in self.snapshots if s.vehicle_embedding is not None]
        if v_embs:
            self.avg_vehicle_emb = np.mean(v_embs, axis=0)
            self.avg_vehicle_emb /= (np.linalg.norm(self.avg_vehicle_emb) + 1e-8)
        
        d_embs = [s.driver_embedding for s in self.snapshots if s.driver_embedding is not None]
        if d_embs:
            self.avg_driver_emb = np.mean(d_embs, axis=0)
            self.avg_driver_emb /= (np.linalg.norm(self.avg_driver_emb) + 1e-8)

@dataclass
class MatchResult:
    id: str
    entry_cluster_id: str
    exit_cluster_id: str
    vehicle_similarity: float
    driver_similarity: float
    overall_score: float
    status: str  # VERIFIED, MISMATCH, UNKNOWN
    reason: str
    entry_sample_img: Optional[bytes] = None
    exit_sample_img: Optional[bytes] = None
