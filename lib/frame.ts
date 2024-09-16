import * as cv from '@techstark/opencv-js';
import * as math from 'mathjs';

export class Frame {
  private id: number;
  private pose: math.Matrix; // 4x4 transformation matrix
  private features: { keypoints: cv.KeyPointVector, descriptors: cv.Mat } | null;
  private cameraMatrix: cv.Mat;
  private image: cv.Mat;

  constructor(id: number, image: cv.Mat, orbDetector: cv.ORB) {
    this.id = id;
    this.pose = math.identity(4) as math.Matrix;
    this.features = null;
    this.cameraMatrix = new cv.Mat();
    this.image = image.clone();
    this.initializeCameraMatrix(image.cols, image.rows);
  }

  private initializeCameraMatrix(imageWidth: number, imageHeight: number): void {
    const focalLength = 0.5 * (imageWidth + imageHeight);
    const centerX = imageWidth / 2;
    const centerY = imageHeight / 2;
    this.cameraMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
      focalLength, 0, centerX,
      0, focalLength, centerY,
      0, 0, 1,
    ]);
  }

  public getId(): number {
    return this.id;
  }

  public setPose(pose: math.Matrix): void {
    this.pose = pose;
  }

  public getPose(): math.Matrix {
    return this.pose;
  }

  public setFeatures(features: { keypoints: cv.KeyPointVector, descriptors: cv.Mat }): void {
    this.features = features;
  }

  public getFeatures(): { keypoints: cv.KeyPointVector, descriptors: cv.Mat } {
    if (!this.features) {
      throw new Error('Features have not been set for this frame');
    }
    return this.features;
  }

  public getCameraMatrix(): cv.Mat {
    return this.cameraMatrix;
  }

  public project(point3D: math.Matrix): cv.Point {
    const homogeneousPoint = math.matrix([
      [point3D.get([0])],
      [point3D.get([1])],
      [point3D.get([2])],
      [1]
    ]);
    const projectedPoint = math.multiply(this.pose, homogeneousPoint) as math.Matrix;
    const x = projectedPoint.get([0]) / projectedPoint.get([2]);
    const y = projectedPoint.get([1]) / projectedPoint.get([2]);
    return new cv.Point(x, y);
  }

  public unproject(pixel: cv.Point): math.Matrix {
    const { fx, fy, cx, cy } = this.getCameraIntrinsics();
    const xNormalized = (pixel.x - cx) / fx;
    const yNormalized = (pixel.y - cy) / fy;
    const dirCam = math.matrix([[xNormalized], [yNormalized], [1]]);
    const normalizedDirCam = math.divide(dirCam, math.norm(dirCam)) as math.Matrix;
    const rotationMatrix = this.pose.subset(math.index([0, 1, 2], [0, 1, 2])) as math.Matrix;
    const dirWorld = math.multiply(rotationMatrix, normalizedDirCam) as math.Matrix;
    const position = this.pose.subset(math.index([0, 1, 2], 3)) as math.Matrix;
    const distance = 1.0;
    return math.add(position, math.multiply(dirWorld, distance)) as math.Matrix;
  }

  public cleanup(): void {
    if (this.features) {
      this.features.keypoints.delete();
      this.features.descriptors.delete();
    }
    this.cameraMatrix.delete();
  }

  private getCameraIntrinsics(): { fx: number, fy: number, cx: number, cy: number } {
    return {
      fx: this.cameraMatrix.data64F[0],
      fy: this.cameraMatrix.data64F[4],
      cx: this.cameraMatrix.data64F[2],
      cy: this.cameraMatrix.data64F[5]
    };
  }

  public getPoseMatrix(): math.Matrix {
    return this.pose;
  }

  setImage(image: cv.Mat): void {
    this.image = image.clone();
  }

  public updatePose(R: cv.Mat, t: cv.Mat): void {
    const newPose = math.matrix([
      [R.data64F[0], R.data64F[1], R.data64F[2], t.data64F[0]],
      [R.data64F[3], R.data64F[4], R.data64F[5], t.data64F[1]],
      [R.data64F[6], R.data64F[7], R.data64F[8], t.data64F[2]],
      [0, 0, 0, 1]
    ]);
    this.pose = newPose;
  }

  public getProjectionMatrix(): cv.Mat {
    const P = new cv.Mat(3, 4, cv.CV_64F);
    const R = this.pose.subset(math.index([0, 1, 2], [0, 1, 2]));
    const t = this.pose.subset(math.index([0, 1, 2], 3));
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        P.data64F[i * 4 + j] = R.get([i, j]);
      }
      P.data64F[i * 4 + 3] = t.get([i]);
    }
    
    return P;
  }

  public getKeyPointFromDescriptor(descriptor: cv.Mat): cv.KeyPoint | null {
    if (!this.features) {
      return null;
    }
    
    const matches = new cv.DMatchVector();
    const matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
    matcher.match(descriptor, this.features.descriptors, matches);
    
    if (matches.size() > 0) {
      const bestMatch = matches.get(0);
      return this.features.keypoints.get(bestMatch.trainIdx);
    }
    
    return null;
  }
}
