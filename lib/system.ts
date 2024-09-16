import { IMUHandler, IMUData } from './imu_handler';
import { Frame } from './frame';
import { MapManager } from './map_manager';
import { VirtualObject } from './virtual_object';
import * as cv from '@techstark/opencv-js';
import * as math from 'mathjs';
import { MapPoint } from './map_manager';  // Add this import at the top of the file

import { CeresSolver, ReprojectionError } from './ceres_solver';
import { withMat, drawKeypoints, decomposeEssentialMatrix, logMatrixInfo } from './opencv_utils';
import { ensureMatrixDimensions, cvMatToMathMatrix, mathMatrixToCvMat } from './math_utils';
import { ransac } from './ransac';
import { Mutex } from 'async-mutex';
import { logger } from './logger';

const SVD_MODIFY_A = 1;
const SVD_FULL_UV = 4;

interface PoseGraphNode {
  id: number;
  pose: math.Matrix;
}

interface PoseGraph {
  nodes: PoseGraphNode[];
  edges: any[]; // Define a proper type for edges if needed
}



export class System {
  private imuHandler: IMUHandler;
  private currentFrame: Frame;
  private previousFrame: Frame | null;
  private mapManager: MapManager;
  private isTracking: boolean = false;
  private orbDetector: cv.ORB;
  private matcher: cv.BFMatcher;
  private mapPoints: Array<{
    position: math.Matrix,
    descriptor: cv.Mat,
    observedFrom: Frame[]
  }> = [];
  private keyframes: Frame[] = [];
  private loopClosures: Array<{ frame1: Frame, frame2: Frame, transform: math.Matrix }> = [];
  private virtualObjects: VirtualObject[] = [];
  private frameDatabase: any; // Placeholder for a frame database (e.g., DBoW2)
  private ceresSolver: CeresSolver;
  private frameMutex: Mutex;
  private keyframeMutex: Mutex;

  constructor() {
    this.imuHandler = new IMUHandler();
    this.mapManager = new MapManager();
    this.currentFrame = new Frame(0, new cv.Mat(), new cv.ORB(500));
    this.previousFrame = null;
    this.orbDetector = new cv.ORB(500);
    this.matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
    this.frameDatabase = {}; // Initialize frame database
    this.ceresSolver = new CeresSolver();
    this.frameMutex = new Mutex();
    this.keyframeMutex = new Mutex();
  }

  public getCurrentFrame(): Frame | null {
    // Return the current frame or null if not available
    return this.currentFrame || null;
  }

  public getMapPoints(): MapPoint[] {
    
    return this.mapManager.getMapPoints();
  }


  public getVirtualObjects(): VirtualObject[] {
    return this.mapManager.getVirtualObjects();
  }

  public async trackMonocular(srcMat: cv.Mat, timestamp: number): Promise<void> {
    await this.frameMutex.runExclusive(async () => {
      if (!this.isTracking) return;
      this.previousFrame = this.currentFrame;
      
      this.currentFrame = new Frame(timestamp, srcMat, this.orbDetector);
      const features = this.extractFeatures(srcMat);
      this.currentFrame.setFeatures(features);
      
      await this.performSLAM(features);

      logger.info('Frame update:', timestamp, this.currentFrame.getPose());
      if (await this.isNewKeyframe()) {
        await this.addKeyframe(this.currentFrame);
      }
    });
  }

  public trackMonocularIMU(srcMat: cv.Mat, imuData: IMUData, timestamp: number): void {
    if (!this.isTracking) return;

    this.imuHandler.processIMUData(imuData);
    const predictedPose = this.imuHandler.getPredictedPose();

    this.previousFrame = this.currentFrame;
    this.currentFrame = new Frame(timestamp, srcMat, this.orbDetector);
    this.currentFrame.setPose(predictedPose);

    const features = this.extractFeatures(srcMat);
    this.currentFrame.setFeatures(features);

    this.performSLAM(features);

    logger.info('Frame update:', timestamp, this.currentFrame.getPose());
  }

  private extractFeatures(image: cv.Mat): { keypoints: cv.KeyPointVector, descriptors: cv.Mat } {
    const keypoints = new cv.KeyPointVector();
    const descriptors = new cv.Mat();
    this.orbDetector.detectAndCompute(image, new cv.Mat(), keypoints, descriptors);
    return { keypoints, descriptors };
  }

  private async performSLAM(features: { keypoints: cv.KeyPointVector, descriptors: cv.Mat }): Promise<void> {
    try {
      if (this.previousFrame) {
        const matches = this.matchFeatures(features, this.previousFrame.getFeatures());
        this.estimatePose(matches, features.keypoints, this.previousFrame.getFeatures().keypoints);
        this.updateLocalMap(features);
        this.detectLoopClosure();
        this.globalOptimization();
        this.mapManager.updateMap(this.currentFrame);
      }

      this.currentFrame.setFeatures(features);

      if (await this.isNewKeyframe()) {
        await this.addKeyframe(this.currentFrame);
      }
    } catch (error) {
      logger.error('Error in performSLAM:', error);
      throw error;
    }
  }

  private async isNewKeyframe(): Promise<boolean> {
    return await this.keyframeMutex.runExclusive(() => {
      return this.keyframes.length === 0 || 
             this.currentFrame.getId() - this.keyframes[this.keyframes.length - 1].getId() > 20;
    });
  }

  private async addKeyframe(frame: Frame): Promise<void> {
    await this.keyframeMutex.runExclusive(() => {
      this.keyframes.push(frame);
      this.frameDatabase.add(frame, this.computeBoW(frame));
    });
  }

  private matchFeatures(features1: { keypoints: cv.KeyPointVector, descriptors: cv.Mat }, 
                        features2: { keypoints: cv.KeyPointVector, descriptors: cv.Mat }): cv.DMatchVector {
    const matches = new cv.DMatchVector();
    this.matcher.match(features1.descriptors, features2.descriptors, matches);
    return this.filterMatchesWithRANSAC(matches, features1.keypoints, features2.keypoints);
  }

  private filterMatchesWithRANSAC(matches: cv.DMatchVector, keypoints1: cv.KeyPointVector, keypoints2: cv.KeyPointVector): cv.DMatchVector {
    const points1 = [];
    const points2 = [];
    for (let i = 0; i < matches.size(); i++) {
      const match = matches.get(i);
      points1.push(keypoints1.get(match.queryIdx).pt);
      points2.push(keypoints2.get(match.trainIdx).pt);
    }

    const { inliers } = ransac(points1, points2, this.estimateFundamentalMatrix, 8, 1.0, 0.99, 1000);

    const filteredMatches = new cv.DMatchVector();
    for (let i = 0; i < inliers.length; i++) {
      if (inliers[i]) {
        filteredMatches.push_back(matches.get(i));
      }
    }

    return filteredMatches;
  }

  private estimateFundamentalMatrix(points1: cv.Point2[], points2: cv.Point2[]): cv.Mat {
    const points1Mat = cv.matFromArray(points1.length, 2, cv.CV_64F, points1.flatMap(p => [p.x, p.y]));
    const points2Mat = cv.matFromArray(points2.length, 2, cv.CV_64F, points2.flatMap(p => [p.x, p.y]));
    const F = cv.findFundamentalMat(points1Mat, points2Mat, cv.FM_8POINT);
    points1Mat.delete();
    points2Mat.delete();
    return F;
  }

  private estimatePose(matches: cv.DMatchVector, keypoints1: cv.KeyPointVector, keypoints2: cv.KeyPointVector): void {
    if (matches.size() < 8) {
      logger.warn('Not enough matches for pose estimation');
      return;
    }

    withMat(
      () => cv.matFromArray(matches.size(), 2, cv.CV_64F, this.extractMatchPoints(matches, keypoints1)),
      (points1Mat) => {
        withMat(
          () => cv.matFromArray(matches.size(), 2, cv.CV_64F, this.extractMatchPoints(matches, keypoints2)),
          (points2Mat) => {
            withMat(
              () => cv.findEssentialMat(points1Mat, points2Mat, this.currentFrame.getCameraMatrix()),
              (E) => {
                logMatrixInfo(E, 'Essential Matrix');
                const [R1, R2, t] = decomposeEssentialMatrix(E, this.currentFrame.getCameraMatrix());
                const [bestR, bestT] = this.chooseBestPose(R1, R2, t, points1Mat, points2Mat);
                this.currentFrame.updatePose(bestR, bestT);
              }
            );
          }
        );
      }
    );
  }

  private extractMatchPoints(matches: cv.DMatchVector, keypoints: cv.KeyPointVector): number[] {
    const points = [];
    for (let i = 0; i < matches.size(); i++) {
      const match = matches.get(i);
      const point = keypoints.get(match.queryIdx).pt;
      points.push(point.x, point.y);
    }
    return points;
  }

  private decomposeEssentialMatrix(E: cv.Mat, K: cv.Mat): [cv.Mat, cv.Mat, cv.Mat] {
    const w = new cv.Mat();
    const u = new cv.Mat();
    const vt = new cv.Mat();
    cv.SVDecomp(E, w, u, vt, SVD_MODIFY_A | SVD_FULL_UV);

    if (cv.determinant(u) < 0) u.col(2).mul(cv.matFromArray(3, 1, cv.CV_64F, [-1, -1, -1]));
    if (cv.determinant(vt) < 0) vt.row(2).mul(cv.matFromArray(1, 3, cv.CV_64F, [-1, -1, -1]));

    const W = cv.Mat.zeros(3, 3, cv.CV_64F);
    W.data64F[1] = -1;
    W.data64F[3] = 1;
    W.data64F[8] = 1;

    const R1 = new cv.Mat();
    const R2 = new cv.Mat();
    const t = u.col(2);

    cv.gemm(u, W, 1, new cv.Mat(), 0, R1, 0);
    cv.gemm(R1, vt, 1, new cv.Mat(), 0, R1, 0);

    cv.gemm(u, W.t(), 1, new cv.Mat(), 0, R2, 0);
    cv.gemm(R2, vt, 1, new cv.Mat(), 0, R2, 0);

    w.delete();
    u.delete();
    vt.delete();
    W.delete();

    const negT = new cv.Mat();
    cv.subtract(cv.Mat.zeros(3, 1, cv.CV_64F), t, negT);
    return [R1, R2, t, negT];
  }

  private chooseBestPose(R1: cv.Mat, R2: cv.Mat, t: cv.Mat, points1: cv.Mat, points2: cv.Mat): [cv.Mat, cv.Mat] {
    const poses = [
      { R: R1, t: t },
      { R: R1, t: (() => { const temp = new cv.Mat(); cv.subtract(cv.Mat.zeros(3, 1, cv.CV_64F), t, temp); return temp; })() },
      { R: R2, t: t },
      { R: R2, t: (() => { const temp = new cv.Mat(); cv.subtract(cv.Mat.zeros(3, 1, cv.CV_64F), t, temp); return temp; })() }
    ];

    let bestPose = poses[0];
    let maxInFront = 0;

    for (const pose of poses) {
      const inFront = this.countPointsInFront(pose.R, pose.t, points1, points2);
      if (inFront > maxInFront) {
        maxInFront = inFront;
        bestPose = pose;
      }
    }

    return [bestPose.R, bestPose.t];
  }

  private countPointsInFront(R: cv.Mat, t: cv.Mat, points1: cv.Mat, points2: cv.Mat): number {
    let count = 0;
    const P1 = cv.Mat.eye(3, 4, cv.CV_64F);
    const P2 = new cv.Mat(3, 4, cv.CV_64F);
    R.copyTo(P2.colRange(0, 3));
    t.copyTo(P2.col(3));

    for (let i = 0; i < points1.rows; i++) {
      const point3D = this.triangulatePoint(
        P1,
        P2,
        points1.row(i),
        points2.row(i)
      );

      if (point3D.get([2]) > 0 && point3D.get([2]) < 1000) { // Adjust depth threshold as needed
        count++;
      }
    }

    P2.delete();
    return count;
  }

  private triangulatePoint(P1: cv.Mat, P2: cv.Mat, point1: cv.Mat, point2: cv.Mat): math.Matrix {
    const A = new cv.Mat(4, 4, cv.CV_64F);

    for (let i = 0; i < 4; i++) {
      A.data64F[i] = point1.data64F[0] * P1.data64F[8 + i] - P1.data64F[i];
      A.data64F[4 + i] = point1.data64F[1] * P1.data64F[8 + i] - P1.data64F[4 + i];
      A.data64F[8 + i] = point2.data64F[0] * P2.data64F[8 + i] - P2.data64F[i];
      A.data64F[12 + i] = point2.data64F[1] * P2.data64F[8 + i] - P2.data64F[4 + i];
    }

    const w = new cv.Mat();
    const u = new cv.Mat();
    const vt = new cv.Mat();
    cv.SVDecomp(A, w, u, vt, SVD_MODIFY_A | SVD_FULL_UV);

    const point3D = math.matrix([
      vt.data64F[12] / vt.data64F[15],
      vt.data64F[13] / vt.data64F[15],
      vt.data64F[14] / vt.data64F[15]
    ]);

    A.delete();
    w.delete();
    u.delete();
    vt.delete();

    return point3D;
  }

  private updateLocalMap(features: { keypoints: cv.KeyPointVector, descriptors: cv.Mat }): void {
    for (let i = 0; i < features.keypoints.size(); i++) {
      const kp = features.keypoints.get(i);
      const descriptor = features.descriptors.row(i);
      
      const point3D = this.triangulateMapPoint(kp.pt, [this.currentFrame]);
      
      this.mapPoints.push({
        position: point3D,
        descriptor: descriptor.clone(),
        observedFrom: [this.currentFrame]
      });
    }
    
    this.localBundleAdjustment();
    this.cullingMapPoints();
  }

  private localBundleAdjustment(): void {
    logger.info('Performing local bundle adjustment');

    for (const mapPoint of this.mapPoints) {
      for (const observationFrame of mapPoint.observedFrom) {
        const observedKeypoint = observationFrame.getKeyPointFromDescriptor(mapPoint.descriptor);

        if (observedKeypoint) {
          const reprojectionError = new ReprojectionError(observedKeypoint.x, observedKeypoint.y, this.currentFrame.getCameraMatrix());
          this.ceresSolver.addResidualBlock(reprojectionError, observationFrame.getPose(), mapPoint.position);
        }
      }
    }

    this.ceresSolver.solve();
  }

  private cullingMapPoints(): void {
    this.mapPoints = this.mapPoints.filter(point => {
      const observationCount = point.observedFrom.length;
      const reprojectionError = this.computeReprojectionError(point);
      return observationCount > 2 && reprojectionError < 2.0;
    });
  }

  private computeReprojectionError(mapPoint: { position: math.Matrix, descriptor: cv.Mat, observedFrom: Frame[] }): number {
    let totalError = 0;
    let count = 0;

    for (const frame of mapPoint.observedFrom) {
      const projectedPoint = frame.project(mapPoint.position);
      const keypoint = this.findCorrespondingKeypoint(frame, mapPoint.descriptor);
      if (keypoint) {
        const error = Math.sqrt(
          Math.pow(projectedPoint.x - keypoint.x, 2) +
          Math.pow(projectedPoint.y - keypoint.y, 2)
        );
        totalError += error;
        count++;
      }
    }
    return count > 0 ? totalError / count : Infinity;
  }

  private findCorrespondingKeypoint(frame: Frame, descriptor: cv.Mat): cv.KeyPoint | null {
    const frameFeatures = frame.getFeatures();
    const matches = new cv.DMatchVector();
    this.matcher.match(descriptor, frameFeatures.descriptors, matches);
    if (matches.size() > 0) {
      const bestMatch = matches.get(0);
      return frameFeatures.keypoints.get(bestMatch.trainIdx);
    }
    return null;
  }

  private detectLoopClosure(): void {
    logger.info('Detecting loop closure');
    
    const currentBoW = this.computeBoW(this.currentFrame);
    const candidateFrames = this.frameDatabase.querySimilar(currentBoW);
    
    for (const candidateFrame of candidateFrames) {
      const matches = this.matchFrameFeatures(this.currentFrame, candidateFrame);
      
      if (matches.size() > 20) {  // Arbitrary threshold
        const [R, t] = this.estimateTransform(matches);
        
        if (this.isValidTransform(R, t)) {
          logger.info('Loop closure detected!');
          this.performLoopClosure(candidateFrame, R, t);
          break;
        }
      }
    }
  }

  private computeBoW(frame: Frame): any {
    const features = frame.getFeatures();
    return {
      frameId: frame.getId(),
            descriptors: features.descriptors
    };
  }

  private matchFrameFeatures(frame1: Frame, frame2: Frame): cv.DMatchVector {
    const features1 = frame1.getFeatures();
    const features2 = frame2.getFeatures();
    return this.matchFeatures(features1, features2);
  }

  private estimateTransform(matches: cv.DMatchVector): [cv.Mat, cv.Mat] {
    const points1 = [];
    const points2 = [];

    if (!this.previousFrame) {
      throw new Error("Previous frame is null");
    }

    for (let i = 0; i < matches.size(); i++) {
      const match = matches.get(i);
      points1.push(this.currentFrame.getFeatures().keypoints.get(match.queryIdx).pt);
      points2.push(this.previousFrame.getFeatures().keypoints.get(match.trainIdx).pt);
    }

    const points1Mat = cv.matFromArray(points1.length, 2, cv.CV_64F, points1.flat());
    const points2Mat = cv.matFromArray(points2.length, 2, cv.CV_64F, points2.flat());

    const E = cv.findEssentialMat(points1Mat, points2Mat, this.currentFrame.getCameraMatrix());
    const [R, t] = this.decomposeEssentialMatrix(E, this.currentFrame.getCameraMatrix());

    points1Mat.delete();
    points2Mat.delete();
    E.delete();

    return [R, t];
  }

  private isValidTransform(R: cv.Mat, t: cv.Mat): boolean {
    const traceR = cv.trace(R);
    const rotationMagnitude = Math.acos(((traceR?.at(0) ?? 3) - 1) / 2);
    const translationMagnitude = cv.norm(t);
    
    const maxRotation = Math.PI / 4; // 45 degrees
    const maxTranslation = 2.0; // 2 meters

    return rotationMagnitude < maxRotation && translationMagnitude < maxTranslation;
  }

  private performLoopClosure(candidateFrame: Frame, R: cv.Mat, t: cv.Mat): void {
    logger.info('Performing loop closure');
    
    const loopClosureTransform = new cv.Mat(4, 4, cv.CV_64F);
    R.copyTo(loopClosureTransform.rowRange(0, 3).colRange(0, 3));
    t.copyTo(loopClosureTransform.rowRange(0, 3).col(3));
    loopClosureTransform.data64F[15] = 1;

    this.loopClosures.push({
      frame1: this.currentFrame,
      frame2: candidateFrame,
      transform: math.matrix(Array.from(loopClosureTransform.data64F))
    });

    this.globalOptimization();

    loopClosureTransform.delete();
  }

  private globalOptimization(): void {
    logger.info('Performing global optimization');
    
    const poseGraph = this.constructPoseGraph();
    this.addLoopClosureConstraints(poseGraph);
    const optimizedPoses = this.optimizePoseGraph(poseGraph);
    this.updateKeyframePoses(optimizedPoses);
    this.updateMapPoints();
  }

  private constructPoseGraph(): PoseGraph {
    const poseGraph: PoseGraph = {
      nodes: [],
      edges: []
    };

    for (const keyframe of this.keyframes) {
      poseGraph.nodes.push({
        id: keyframe.getId(),
        pose: keyframe.getPose()
      });
    }

    for (let i = 1; i < this.keyframes.length; i++) {
      poseGraph.edges.push({
        from: this.keyframes[i - 1].getId(),
        to: this.keyframes[i].getId(),
        transform: this.computeRelativeTransform(this.keyframes[i - 1], this.keyframes[i])
      });
    }

    return poseGraph;
  }

  private computeRelativeTransform(frame1: Frame, frame2: Frame): math.Matrix {
    const pose1 = frame1.getPose();
    const pose2 = frame2.getPose();
    return math.multiply(math.inv(pose1), pose2);
  }

  private addLoopClosureConstraints(poseGraph: PoseGraph): void {
    for (const loopClosure of this.loopClosures) {
      poseGraph.edges.push({
        from: loopClosure.frame1.getId(),
        to: loopClosure.frame2.getId(),
        transform: loopClosure.transform
      });
    }
  }

  private optimizePoseGraph(poseGraph: PoseGraph): math.Matrix[] {
    logger.info('Optimizing pose graph');
    return poseGraph.nodes.map(node => node.pose);
  }

  private updateKeyframePoses(optimizedPoses: math.Matrix[]): void {
    for (let i = 0; i < this.keyframes.length; i++) {
      const pose = optimizedPoses[i];
      const R = cv.matFromArray(3, 3, cv.CV_64F, pose.subset(math.index([0, 1, 2], [0, 1, 2])).toArray().flat());
      const t = cv.matFromArray(3, 1, cv.CV_64F, pose.subset(math.index([0, 1, 2], 3)).toArray());
      this.keyframes[i].updatePose(R, t);
      R.delete();
      t.delete();
    }
  }

  private updateMapPoints(): void {
    for (const mapPoint of this.mapPoints) {
      const observations = mapPoint.observedFrom;
      if (observations.length > 0) {
        const newPosition = this.triangulateMapPoint(mapPoint, observations);
        ensureMatrixDimensions(newPosition, 3, 1);
        mapPoint.position = newPosition;
      }
    }
  }

  private triangulateMapPoint(mapPoint: any, observations: Frame[]): math.Matrix {
    const points2D = [];
    const projectionMatrices = [];

    for (const frame of observations) {
      const keypoint = this.findCorrespondingKeypoint(frame, mapPoint.descriptor);
      if (keypoint) {
        points2D.push(keypoint);
        projectionMatrices.push(frame.getProjectionMatrix());
      }
    }

    if (points2D.length < 2) {
      return mapPoint.position; // Not enough observations for triangulation
    }

    const A = new cv.Mat(points2D.length * 2, 4, cv.CV_64F);
    for (let i = 0; i < points2D.length; i++) {
      const P = projectionMatrices[i];
      const x = points2D[i].x;
      const y = points2D[i].y;

      A.data64F[i * 8 + 0] = x * P.data64F[8] - P.data64F[0];
      A.data64F[i * 8 + 1] = x * P.data64F[9] - P.data64F[1];
      A.data64F[i * 8 + 2] = x * P.data64F[10] - P.data64F[2];
      A.data64F[i * 8 + 3] = x * P.data64F[11] - P.data64F[3];
      A.data64F[i * 8 + 4] = y * P.data64F[8] - P.data64F[4];
      A.data64F[i * 8 + 5] = y * P.data64F[9] - P.data64F[5];
      A.data64F[i * 8 + 6] = y * P.data64F[10] - P.data64F[6];
      A.data64F[i * 8 + 7] = y * P.data64F[11] - P.data64F[7];
    }

    const w = new cv.Mat();
    const u = new cv.Mat();
    const vt = new cv.Mat();
    cv.SVDecomp(A, w, u, vt, SVD_MODIFY_A | SVD_FULL_UV);

    const homogeneousPoint = vt.row(3);
    const point3D = math.matrix([
      homogeneousPoint.data64F[0] / homogeneousPoint.data64F[3],
      homogeneousPoint.data64F[1] / homogeneousPoint.data64F[3],
      homogeneousPoint.data64F[2] / homogeneousPoint.data64F[3]
    ]);

    A.delete();
    w.delete();
    u.delete();
    vt.delete();
    homogeneousPoint.delete();

    return point3D;
  }

  public handleUserTap(tapPosition: cv.Point): void {
    const worldPoint = this.currentFrame.unproject(tapPosition);
    this.mapManager.placeVirtualObject(worldPoint);
  }
}
