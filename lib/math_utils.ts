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