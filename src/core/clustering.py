# src/core/clustering.py
"""
Simple centroid-style clustering for vehicle snapshots.
Clusters snapshots by vehicle_embedding cosine similarity threshold.
"""
from typing import List, Dict
import numpy as np
from data_models.snapshot import VehicleSnapshot
from data_models.cluster import VehicleCluster
from utils.similarity import cosine_similarity

def cluster_snapshots(snapshots: List[VehicleSnapshot], threshold: float = 0.7) -> List[VehicleCluster]:
    if not snapshots:
        return []

    clusters: List[VehicleCluster] = []
    centroids: List[np.ndarray] = []

    for snap in snapshots:
        emb = snap.vehicle_embedding
        if emb is None or emb.size == 0:
            # treat as its own cluster
            c = VehicleCluster.from_snapshots(cluster_id=str(snap.track_id), snapshots=[snap], is_entry=snap.is_entry)
            clusters.append(c)
            centroids.append(c.vehicle_embedding if c.vehicle_embedding is not None else np.zeros(512))
            continue

        if not centroids:
            c = VehicleCluster.from_snapshots(cluster_id=str(snap.track_id), snapshots=[snap], is_entry=snap.is_entry)
            clusters.append(c)
            centroids.append(c.vehicle_embedding)
            continue

        # compute similarities
        sims = [cosine_similarity(emb, cent) for cent in centroids]
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        if best_score >= threshold:
            clusters[best_idx].add_snapshot(snap)
            clusters[best_idx].finalize()
            centroids[best_idx] = clusters[best_idx].vehicle_embedding
        else:
            c = VehicleCluster.from_snapshots(cluster_id=str(snap.track_id), snapshots=[snap], is_entry=snap.is_entry)
            clusters.append(c)
            centroids.append(c.vehicle_embedding)

    return clusters