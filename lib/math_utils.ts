import * as math from 'mathjs';
import * as cv from '@techstark/opencv-js';

export function ensureMatrixDimensions(matrix: math.Matrix, rows: number, cols: number): void {
  const size = matrix.size();
  if (size[0] !== rows || size[1] !== cols) {
    throw new Error(`Matrix dimensions mismatch. Expected ${rows}x${cols}, got ${size[0]}x${size[1]}`);
  }
}

export function cvMatToMathMatrix(cvMat: cv.Mat): math.Matrix {
  const rows = cvMat.rows;
  const cols = cvMat.cols;
  const data = new Float64Array(cvMat.data64F);
  return math.matrix(math.reshape(data, [rows, cols]));
}

export function mathMatrixToCvMat(mathMatrix: math.Matrix): cv.Mat {
  const size = mathMatrix.size();
  const rows = size[0];
  const cols = size[1];
  const mat = new cv.Mat(rows, cols, cv.CV_64F);
  mat.data64F.set(mathMatrix.toArray().flat());
  return mat;
}

export function rotationMatrixToEulerAngles(R: math.Matrix): [number, number, number] {
  const sy = Math.sqrt(R.get([0, 0]) * R.get([0, 0]) + R.get([1, 0]) * R.get([1, 0]));
  const singular = sy < 1e-6;
  let x, y, z;
  if (!singular) {
    x = Math.atan2(R.get([2, 1]), R.get([2, 2]));
    y = Math.atan2(-R.get([2, 0]), sy);
    z = Math.atan2(R.get([1, 0]), R.get([0, 0]));
  } else {
    x = Math.atan2(-R.get([1, 2]), R.get([1, 1]));
    y = Math.atan2(-R.get([2, 0]), sy);
    z = 0;
  }
  return [x, y, z];
}

export function eulerAnglesToRotationMatrix(angles: [number, number, number]): math.Matrix {
  const [x, y, z] = angles;
  const Rx = math.matrix([
    [1, 0, 0],
    [0, Math.cos(x), -Math.sin(x)],
    [0, Math.sin(x), Math.cos(x)]
  ]);
  const Ry = math.matrix([
    [Math.cos(y), 0, Math.sin(y)],
    [0, 1, 0],
    [-Math.sin(y), 0, Math.cos(y)]
  ]);
  const Rz = math.matrix([
    [Math.cos(z), -Math.sin(z), 0],
    [Math.sin(z), Math.cos(z), 0],
    [0, 0, 1]
  ]);
  return math.multiply(Rz, math.multiply(Ry, Rx)) as math.Matrix;
}

export function quaternionToRotationMatrix(q: math.Matrix): math.Matrix {
  const [x, y, z, w] = q.toArray() as number[];
  return math.matrix([
    [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
    [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
  ]);
}

export function rotationMatrixToQuaternion(R: math.Matrix): math.Matrix {
  const tr = R.get([0, 0]) + R.get([1, 1]) + R.get([2, 2]);
  let q: number[];
  if (tr > 0) {
    const S = Math.sqrt(tr + 1.0) * 2;
    q = [
      (R.get([2, 1]) - R.get([1, 2])) / S,
      (R.get([0, 2]) - R.get([2, 0])) / S,
      (R.get([1, 0]) - R.get([0, 1])) / S,
      0.25 * S
    ];
  } else if (R.get([0, 0]) > R.get([1, 1]) && R.get([0, 0]) > R.get([2, 2])) {
    const S = Math.sqrt(1.0 + R.get([0, 0]) - R.get([1, 1]) - R.get([2, 2])) * 2;
    q = [
      0.25 * S,
      (R.get([0, 1]) + R.get([1, 0])) / S,
      (R.get([0, 2]) + R.get([2, 0])) / S,
      (R.get([2, 1]) - R.get([1, 2])) / S
    ];
  } else if (R.get([1, 1]) > R.get([2, 2])) {
    const S = Math.sqrt(1.0 + R.get([1, 1]) - R.get([0, 0]) - R.get([2, 2])) * 2;
    q = [
      (R.get([0, 1]) + R.get([1, 0])) / S,
      0.25 * S,
      (R.get([1, 2]) + R.get([2, 1])) / S,
      (R.get([0, 2]) - R.get([2, 0])) / S
    ];
  } else {
    const S = Math.sqrt(1.0 + R.get([2, 2]) - R.get([0, 0]) - R.get([1, 1])) * 2;
    q = [
      (R.get([0, 2]) + R.get([2, 0])) / S,
      (R.get([1, 2]) + R.get([2, 1])) / S,
      0.25 * S,
      (R.get([1, 0]) - R.get([0, 1])) / S
    ];
  }
  return math.matrix(q);
}

export function composePoses(pose1: math.Matrix, pose2: math.Matrix): math.Matrix {
  const R1 = pose1.subset(math.index([0, 1, 2], [0, 1, 2]));
  const t1 = pose1.subset(math.index([0, 1, 2], 3));
  const R2 = pose2.subset(math.index([0, 1, 2], [0, 1, 2]));
  const t2 = pose2.subset(math.index([0, 1, 2], 3));

  const R = math.multiply(R1, R2) as math.Matrix;
  const t = math.add(math.multiply(R1, t2) as math.Matrix, t1) as math.Matrix;

  const composedPose = math.identity(4) as math.Matrix;
  composedPose.subset(math.index([0, 1, 2], [0, 1, 2]), R);
  composedPose.subset(math.index([0, 1, 2], 3), t);

  return composedPose;
}