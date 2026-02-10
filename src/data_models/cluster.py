# src/data_models/cluster.py
import numpy as np
from dataclasses import dataclass, field
from typing import List
from .snapshot import VehicleSnapshot

@dataclass
class VehicleCluster:
    cluster_id: str
    is_entry: bool
    snapshots: List[VehicleSnapshot] = field(default_factory=list)
    vehicle_embedding: np.ndarray = None
    driver_embedding: np.ndarray = None

    def add_snapshot(self, snapshot: VehicleSnapshot):
        self.snapshots.append(snapshot)

    def finalize(self):
        if not self.snapshots:
            return
        vehicle_embs = [s.vehicle_embedding for s in self.snapshots if s.vehicle_embedding is not None]
        driver_embs = [s.driver_embedding for s in self.snapshots if s.driver_embedding is not None]

        if vehicle_embs:
            self.vehicle_embedding = np.mean(vehicle_embs, axis=0)
            self.vehicle_embedding = self.vehicle_embedding / (np.linalg.norm(self.vehicle_embedding))

        if driver_embs:
            self.driver_embedding = np.mean(driver_embs, axis=0)
            self.driver_embedding = self.driver_embedding / (np.linalg.norm(self.driver_embedding))

    @classmethod
    def from_snapshots(cls, cluster_id: str, snapshots: List[VehicleSnapshot], is_entry: bool):
        c = cls(cluster_id=cluster_id, is_entry=is_entry, snapshots=list(snapshots))
        c.finalize()
        return c