import * as cv from '@techstark/opencv-js';
import * as math from 'mathjs';
import { logger } from './logger';

/**
 * Safely executes a function that uses cv.Mat objects, ensuring they are properly deleted afterwards.
 * @param createMat A function that creates a cv.Mat object
 * @param action A function that uses the created cv.Mat object
 * @returns The result of the action function
 */
export function withMat<T>(createMat: () => cv.Mat, action: (mat: cv.Mat) => T): T {
  const mat = createMat();
  try {
    return action(mat);
  } finally {
    mat.delete();
  }
}

/**
 * Converts a cv.Mat to a math.js Matrix
 * @param cvMat The OpenCV matrix to convert
 * @returns A math.js Matrix
 */
export function cvMatToMathMatrix(cvMat: cv.Mat): math.Matrix {
  const rows = cvMat.rows;
  const cols = cvMat.cols;
  const data = new Float64Array(cvMat.data64F);
  return math.matrix(math.reshape(data, [rows, cols]));
}

/**
 * Converts a math.js Matrix to a cv.Mat
 * @param mathMatrix The math.js Matrix to convert
 * @returns An OpenCV matrix
 */
export function mathMatrixToCvMat(mathMatrix: math.Matrix): cv.Mat {
  const size = mathMatrix.size();
  const rows = size[0];
  const cols = size[1];
  const mat = new cv.Mat(rows, cols, cv.CV_64F);
  mat.data64F.set(mathMatrix.toArray().flat());
  return mat;
}

/**
 * Draws keypoints on an image
 * @param img The image to draw on
 * @param keypoints The keypoints to draw
 * @returns A new image with the keypoints drawn
 */
export function drawKeypoints(img: cv.Mat, keypoints: cv.KeyPointVector): cv.Mat {
  const result = img.clone();
  for (let i = 0; i < keypoints.size(); i++) {
    const point = keypoints.get(i).pt;
    cv.circle(result, point, 3, new cv.Scalar(0, 255, 0), -1);
  }
  return result;
}

/**
 * Draws matches between two images
 * @param img1 The first image
 * @param keypoints1 Keypoints from the first image
 * @param img2 The second image
 * @param keypoints2 Keypoints from the second image
 * @param matches The matches between the keypoints
 * @returns A new image showing the matches between the two input images
 */
export function drawMatches(
  img1: cv.Mat,
  keypoints1: cv.KeyPointVector,
  img2: cv.Mat,
  keypoints2: cv.KeyPointVector,
  matches: cv.DMatchVector
): cv.Mat {
  const result = new cv.Mat();
  const matchColor = new cv.Scalar(0, 255, 0);
  const singlePointColor = new cv.Scalar(255, 0, 0);
  cv.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    matches,
    result,
    matchColor,
    singlePointColor
  );
  return result;
}

/**
 * Converts a rotation matrix to Euler angles
 * @param R Rotation matrix (3x3 cv.Mat)
 * @returns An array of Euler angles [x, y, z] in radians
 */
export function rotationMatrixToEulerAngles(R: cv.Mat): [number, number, number] {
  const sy = Math.sqrt(R.data64F[0] * R.data64F[0] + R.data64F[3] * R.data64F[3]);
  const singular = sy < 1e-6;
  let x, y, z;
  if (!singular) {
    x = Math.atan2(R.data64F[7], R.data64F[8]);
    y = Math.atan2(-R.data64F[6], sy);
    z = Math.atan2(R.data64F[3], R.data64F[0]);
  } else {
    x = Math.atan2(-R.data64F[5], R.data64F[4]);
    y = Math.atan2(-R.data64F[6], sy);
    z = 0;
  }
  return [x, y, z];
}

/**
 * Converts Euler angles to a rotation matrix
 * @param angles An array of Euler angles [x, y, z] in radians
 * @returns A 3x3 rotation matrix (cv.Mat)
 */
export function eulerAnglesToRotationMatrix(angles: [number, number, number]): cv.Mat {
  const [x, y, z] = angles;
  const Rx = new cv.Mat(3, 3, cv.CV_64F);
  const Ry = new cv.Mat(3, 3, cv.CV_64F);
  const Rz = new cv.Mat(3, 3, cv.CV_64F);

  // Rotation around X-axis
  Rx.data64F.set([
    1, 0, 0,
    0, Math.cos(x), -Math.sin(x),
    0, Math.sin(x), Math.cos(x)
  ]);

  // Rotation around Y-axis
  Ry.data64F.set([
    Math.cos(y), 0, Math.sin(y),
    0, 1, 0,
    -Math.sin(y), 0, Math.cos(y)
  ]);

  // Rotation around Z-axis
  Rz.data64F.set([
    Math.cos(z), -Math.sin(z), 0,
    Math.sin(z), Math.cos(z), 0,
    0, 0, 1
  ]);

  const R = new cv.Mat();
  cv.gemm(Rz, Ry, 1, new cv.Mat(), 0, R);
  cv.gemm(R, Rx, 1, new cv.Mat(), 0, R);

  Rx.delete();
  Ry.delete();
  Rz.delete();

  return R;
}

/**
 * Decomposes an essential matrix into rotation and translation
 * @param E Essential matrix
 * @param K Camera intrinsic matrix
 * @returns An array containing two possible rotations and one translation [R1, R2, t]
 */
export function decomposeEssentialMatrix(E: cv.Mat, K: cv.Mat): [cv.Mat, cv.Mat, cv.Mat] {
  const w = new cv.Mat();
  const u = new cv.Mat();
  const vt = new cv.Mat();
  cv.SVDecomp(E, w, u, vt, cv.SVD_MODIFY_A);

  const W = new cv.Mat(3, 3, cv.CV_64F, [0, -1, 0, 1, 0, 0, 0, 0, 1]);
  const Wt = new cv.Mat(3, 3, cv.CV_64F, [0, 1, 0, -1, 0, 0, 0, 0, 1]);

  const R1 = new cv.Mat();
  const R2 = new cv.Mat();
  const t = new cv.Mat();

  cv.gemm(u, W, 1, new cv.Mat(), 0, R1);
  cv.gemm(R1, vt, 1, new cv.Mat(), 0, R1);

  cv.gemm(u, Wt, 1, new cv.Mat(), 0, R2);
  cv.gemm(R2, vt, 1, new cv.Mat(), 0, R2);

  t.data64F.set(u.col(2).data64F);

  W.delete();
  Wt.delete();
  w.delete();
  u.delete();
  vt.delete();

  return [R1, R2, t];
}

/**
 * Logs OpenCV matrix information
 * @param mat The matrix to log
 * @param name Optional name for the matrix
 */
export function logMatrixInfo(mat: cv.Mat, name: string = 'Matrix'): void {
  logger.debug(`${name} - Type: ${mat.type()}, Size: ${mat.rows}x${mat.cols}`);
  logger.debug(`${name} Data:`, mat.data64F);
}
