# Vehicle–Driver Re-Identification & Mismatch Detection Pipeline

A modular, production-oriented computer vision pipeline for **vehicle and driver re-identification**, designed to detect **vehicle–driver mismatches** between entry and exit points (e.g. gated facilities, parking lots, campuses, malls).

---

##  Core Capabilities

- Vehicle detection and tracking across frames
- Driver face detection inside vehicles
- Vehicle and driver embedding extraction
- Snapshot-based clustering (entry & exit)
- Cross-camera vehicle–driver matching
- Mismatch and no-match detection
- Clean, extensible architecture (research → production)

---

## High-Level Architecture

```
Frames
 ├── VehicleDetector
 ├── ByteTrackManager
 ├── FaceDetector
 ├── Snapshot (one per track)
 ├── Cluster (vehicle-level aggregation)
 ├── Matcher (entry ↔ exit)
 └── Alert / Result
```


