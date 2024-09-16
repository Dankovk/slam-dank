import { VirtualObject } from './virtual_object';
import * as math from 'mathjs';
import { Frame } from './frame';

export interface MapPoint {
  position: math.Matrix;
  observations: number;
}

export class MapManager {
  private virtualObjects: VirtualObject[];
  private mapPoints: MapPoint[];
  private maxMapPoints: number;
  private lastCleanupTime: number = 0;
  private readonly CLEANUP_INTERVAL: number = 60000; // 1 minute

  constructor(maxMapPoints: number = 1000) {
    this.virtualObjects = [];
    this.mapPoints = [];
    this.maxMapPoints = maxMapPoints;
  }

  public placeVirtualObject(position: math.Matrix): void {
    const obj = new VirtualObject(position);
    this.virtualObjects.push(obj);
  }

  public getVirtualObjects(): VirtualObject[] {
    return this.virtualObjects;
  }

  public updateMap(frame: Frame): void {
    // Add new 3D points to the map
    const features = frame.getFeatures();
    if (features) {
      for (let i = 0; i < features.keypoints.size(); i++) {
        const keypoint = features.keypoints.get(i);
        const worldPoint = frame.unproject(keypoint.pt);
        this.addMapPoint(worldPoint);
      }
    }

    // Update virtual objects' positions
    const framePose = frame.getPoseMatrix();
    for (const obj of this.virtualObjects) {
      obj.updatePosition(framePose);
    }

    this.pruneMapPoints();
    this.removeOldMapPoints(frame.getId());
    console.log('Map updated with new frame');
  }

  public getMapPoints(): MapPoint[] {
    return this.mapPoints;
  }

  public removeVirtualObject(index: number): void {
    if (index >= 0 && index < this.virtualObjects.length) {
      this.virtualObjects.splice(index, 1);
    }
  }

  private addMapPoint(point: math.Matrix): void {
    const existingPoint = this.findNearestMapPoint(point);
    if (existingPoint) {
      existingPoint.observations++;
    } else {
      this.mapPoints.push({ position: point, observations: 1 });
    }
  }

  private findNearestMapPoint(point: math.Matrix, threshold: number = 0.1): MapPoint | null {
    for (const mapPoint of this.mapPoints) {
      const distance = math.norm(math.subtract(mapPoint.position, point) as math.Matrix);
      if (distance < threshold) {
        return mapPoint;
      }
    }
    return null;
  }

  private pruneMapPoints(): void {
    if (this.mapPoints.length > this.maxMapPoints) {
      // Sort map points by number of observations (descending)
      this.mapPoints.sort((a, b) => b.observations - a.observations);
      // Keep only the top maxMapPoints
      this.mapPoints = this.mapPoints.slice(0, this.maxMapPoints);
    }
  }

  private removeOldMapPoints(currentFrameId: number): void {
    const currentTime = Date.now();
    if (currentTime - this.lastCleanupTime > this.CLEANUP_INTERVAL) {
      this.mapPoints = this.mapPoints.filter(point => {
        const lastObservationAge = currentFrameId - Math.max(...point.observedFrom.map(f => f.getId()));
        return lastObservationAge < 100; // Remove points not seen in last 100 frames
      });
      this.lastCleanupTime = currentTime;
    }
  }
}
