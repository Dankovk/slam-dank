import * as cv from '@techstark/opencv-js';
import * as math from 'mathjs';
import { logger } from './logger';

export function ransac(
  points1: cv.Point2[], 
  points2: cv.Point2[], 
  modelEstimator: (pts1: cv.Point2[], pts2: cv.Point2[]) => cv.Mat, 
  minSamples: number, 
  threshold: number, 
  confidence: number, 
  maxIterations: number
) {
  let bestInliers: boolean[] = [];
  let bestModel: cv.Mat | null = null;

  for (let i = 0; i < maxIterations; i++) {
    const sample = selectRandomSample(points1, points2, minSamples);
    const model = modelEstimator(sample.set1, sample.set2);
    const inliers = evaluateModel(model, points1, points2, threshold);

    if (inliers.filter(Boolean).length > bestInliers.filter(Boolean).length) {
      bestInliers = inliers;
      bestModel = model;
    }

    if (checkTerminationCriteria(bestInliers, confidence, i, maxIterations)) break;
  }

  if (bestModel) {
    // Refine the model using all inliers
    const inlierPoints1 = points1.filter((_, i) => bestInliers[i]);
    const inlierPoints2 = points2.filter((_, i) => bestInliers[i]);
    bestModel = modelEstimator(inlierPoints1, inlierPoints2);
  }

  return { inliers: bestInliers, model: bestModel };
}

function selectRandomSample(points1: cv.Point2[], points2: cv.Point2[], sampleSize: number) {
  const indices = new Set<number>();
  while (indices.size < sampleSize) {
    indices.add(Math.floor(Math.random() * points1.length));
  }
  
  const set1 = Array.from(indices).map(i => points1[i]);
  const set2 = Array.from(indices).map(i => points2[i]);
  
  return { set1, set2 };
}

function evaluateModel(model: cv.Mat, points1: cv.Point2[], points2: cv.Point2[], threshold: number): boolean[] {
  const inliers: boolean[] = [];
  
  for (let i = 0; i < points1.length; i++) {
    const error = computeError(model, points1[i], points2[i]);
    inliers.push(error < threshold);
  }
  
  return inliers;
}

function computeError(model: cv.Mat, point1: cv.Point2, point2: cv.Point2): number {
  const homogeneous1 = new cv.Mat(3, 1, cv.CV_64F);
  const homogeneous2 = new cv.Mat(3, 1, cv.CV_64F);
  
  homogeneous1.data64F.set([point1.x, point1.y, 1]);
  homogeneous2.data64F.set([point2.x, point2.y, 1]);

  const result = new cv.Mat();
  cv.gemm(homogeneous2.t(), model, 1, new cv.Mat(), 0, result);
  cv.gemm(result, homogeneous1, 1, new cv.Mat(), 0, result);

  const error = Math.abs(result.data64F[0]);

  homogeneous1.delete();
  homogeneous2.delete();
  result.delete();

  return error;
}

function checkTerminationCriteria(inliers: boolean[], confidence: number, iteration: number, maxIterations: number): boolean {
  const inlierRatio = inliers.filter(Boolean).length / inliers.length;
  const estimatedIterations = Math.log(1 - confidence) / Math.log(1 - Math.pow(inlierRatio, 8));
  
  return iteration >= Math.min(estimatedIterations, maxIterations);
}