Vehicle–Driver Re-Identification & Mismatch Detection Pipeline

A modular, production-oriented computer vision pipeline for vehicle and driver re-identification, designed to detect vehicle–driver mismatches between entry and exit points (e.g. gated facilities, parking lots, campuses, malls).

This system answers the question:

Did the same driver who entered with a vehicle also exit with it?

🚗🔍 Core Capabilities

Vehicle detection and tracking across frames

Driver face detection inside vehicles

Vehicle and driver embedding extraction

Snapshot-based clustering (entry & exit)

Cross-camera vehicle–driver matching

Mismatch and no-match detection

Clean, extensible architecture (research → production)


📐 High-Level Architecture

Frames
 ├── VehicleDetector
 ├── ByteTrackManager
 ├── FaceDetector
 ├── Snapshot (one per track)
 ├── Cluster (vehicle-level aggregation)
 ├── Matcher (entry ↔ exit)
 └── Alert / Result

 Each vehicle track produces exactly one snapshot, preventing noisy embeddings and enforcing deterministic behavior.

 src/
├── core/
│   ├── pipeline.py          # End-to-end orchestration
│   ├── clustering.py        # Vehicle snapshot clustering
│   └── matcher.py           # Entry–exit matching logic
│
├── detection/
│   ├── vehicle_detector.py  # Vehicle detection (YOLO-style)
│   └── face_detector.py     # Driver face detection
│
├── tracking/
│   └── bytetrack_manager.py # Multi-object tracking (ByteTrack wrapper)
│
├── embeddings/
│   ├── vehicle_embedder.py  # Vehicle ReID or histogram fallback
│   └── driver_embedder.py   # Driver face embedding
│
├── data_models/
│   ├── snapshot.py          # Single-track snapshot
│   └── cluster.py           # Aggregated vehicle cluster
│
├── io/
│   └── frame_loader.py      # Frame iteration utility
│
├── utils/
│   └── similarity.py        # Cosine similarity
│
└── main.py                  # Entry point


🚀 How the Pipeline Works

Load frames from entry and exit directories

Detect vehicles in each frame

Track vehicles using ByteTrack

Capture a snapshot once per track

Detect driver faces inside vehicle crop

Extract embeddings (vehicle + driver)

Cluster snapshots into unique vehicles

Match exit clusters to entry clusters

Report results

Match

Mismatch

No matching entry found

▶️ Running the Pipeline
1. Install Dependencies
```pip install numpy opencv-python torch ultralytics
```

2. Prepare Data
data/
├── entry_frames/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── exit_frames/
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── ...

