from utils.similarity import cosine_similarity

class VehicleDriverMatcher:
    def match(self, entry_clusters, exit_clusters):
        results = []

        for exit_cluster in exit_clusters:
            best_match = None
            best_score = -1

            for entry_cluster in entry_clusters:
                v_sim = cosine_similarity(
                    exit_cluster.vehicle_embedding,
                    entry_cluster.vehicle_embedding
                )
                d_sim = cosine_similarity(
                    exit_cluster.driver_embedding,
                    entry_cluster.driver_embedding
                )

                score = 0.5 * v_sim + 0.5 * d_sim

                if score > best_score:
                    best_score = score
                    best_match = entry_cluster

            results.append({
                "exit_id": exit_cluster.cluster_id,
                "entry_id": best_match.cluster_id if best_match else None,
                "score": best_score
            })

        return results