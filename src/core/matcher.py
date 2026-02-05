# src/core/matcher.py
from typing import List, Dict, Any
from data_models.cluster import VehicleCluster
from utils.similarity import cosine_similarity

class VehicleDriverMatcher:
    """
    Matches exit clusters to entry clusters using driver and vehicle embeddings.
    """

    def __init__(self, driver_threshold: float = 0.6, overall_threshold: float = 0.5):
        self.driver_threshold = driver_threshold
        self.overall_threshold = overall_threshold

    def match(self, entry_clusters: List[VehicleCluster], exit_clusters: List[VehicleCluster]) -> List[Dict[str, Any]]:
        results = []

        for exit_c in exit_clusters:
            best_entry = None
            best_driver_score = 0.0

            for entry_c in entry_clusters:
                # safe guards
                if exit_c.driver_embedding is None or entry_c.driver_embedding is None:
                    continue
                score = cosine_similarity(exit_c.driver_embedding, entry_c.driver_embedding)
                if score > best_driver_score:
                    best_driver_score = score
                    best_entry = entry_c

            if best_entry is None or best_driver_score < self.driver_threshold:
                results.append({
                    "exit_cluster": exit_c,
                    "entry_cluster": None,
                    "driver_score": best_driver_score,
                    "vehicle_score": 0.0,
                    "overall_score": 0.0,
                    "is_match": False,
                    "reason": "no_entry_driver_found"
                })
                continue

            vehicle_score = 0.0
            if exit_c.vehicle_embedding is not None and best_entry.vehicle_embedding is not None:
                vehicle_score = cosine_similarity(exit_c.vehicle_embedding, best_entry.vehicle_embedding)

            overall_score = 0.4 * best_driver_score + 0.6 * vehicle_score

            is_match = (best_driver_score >= self.driver_threshold) and (overall_score >= self.overall_threshold)

            results.append({
                "exit_cluster": exit_c,
                "entry_cluster": best_entry,
                "driver_score": best_driver_score,
                "vehicle_score": vehicle_score,
                "overall_score": overall_score,
                "is_match": is_match,
                "reason": "match" if is_match else "driver_mismatch"
            })

        return results