
export type DetectionStatus = 'idle' | 'processing' | 'completed' | 'error';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionResult {
  id: string;
  type: 'vehicle' | 'driver';
  confidence: number;
  bbox: BoundingBox;
}

export interface Snapshot {
  id: string;
  timestamp: string;
  imageUrl: string;
  detections: DetectionResult[];
  metadata: {
    vehicleMake?: string;
    vehicleColor?: string;
    licensePlate?: string;
  };
}

export interface MatchResult {
  id: string;
  entrySnapshotId: string;
  exitSnapshotId: string;
  vehicleSimilarity: number;
  driverSimilarity: number;
  overallScore: number;
  isMatch: boolean;
  status: 'VERIFIED' | 'MISMATCH' | 'UNKNOWN';
  reason?: string;
}

export interface SystemStats {
  totalDetections: number;
  verifiedMatches: number;
  mismatchesDetected: number;
  averageProcessingTime: number;
}
