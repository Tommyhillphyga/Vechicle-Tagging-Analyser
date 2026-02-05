from data_models.cluster import VehicleCluster

def cluster_snapshots(snapshots):
    clusters = {}
    for snap in snapshots:
        clusters.setdefault(snap.track_id, []).append(snap)

    return [
        VehicleCluster.from_snapshots(cluster_id, snaps)
        for cluster_id, snaps in clusters.items()
    ]