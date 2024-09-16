import * as math from 'mathjs';
import { ensureMatrixDimensions } from './math_utils';
import { cvMatToMathMatrix } from './opencv_utils'; // Add this import at the top of the file

export class CeresSolver {
  private residualBlocks: ResidualBlock[] = [];
  private lambda: number = 1e-3;
  private maxIterations: number = 100;
  private tolerance: number = 1e-6;

  public addResidualBlock(costFunction: ReprojectionError, cameraPose: math.Matrix, point3D: math.Matrix): void {
    ensureMatrixDimensions(cameraPose, 4, 4);
    ensureMatrixDimensions(point3D, 3, 1);
    const residualBlock = new ResidualBlock(costFunction, cameraPose, point3D);
    this.residualBlocks.push(residualBlock);
  }

  public solve(): void {
    for (let iter = 0; iter < this.maxIterations; iter++) {
      const [J, r] = this.computeJacobianAndResiduals();
      const H = math.multiply(math.transpose(J), J);
      const g = math.multiply(math.transpose(J), r);

      const damping = math.multiply(this.lambda, math.diag(math.diag(H)));
      const augmentedH = math.add(H, damping) as math.Matrix;

      const delta = math.lusolve(augmentedH, math.multiply(g, -1));
      this.updateParameters(delta);

      const newCost = this.computeCost();
      if (newCost < this.tolerance) {
        console.log(`Converged after ${iter + 1} iterations`);
        break;
      }

      if (newCost < this.computeCost()) {
        this.lambda *= 0.1;
      } else {
        this.lambda *= 10;
        this.revertUpdate(delta);
      }
    }
  }

  private computeJacobianAndResiduals(): [math.Matrix, math.Matrix] {
    const J = [];
    const r = [];
    for (const block of this.residualBlocks) {
      const [blockJ, blockR] = block.computeJacobianAndResidual();
      J.push(...blockJ);
      r.push(...blockR);
    }
    return [math.matrix(J), math.matrix(r)];
  }

  private computeCost(): number {
    let cost = 0;
    for (const block of this.residualBlocks) {
      const residual = block.computeResidual();
      cost += math.dot(residual, residual);
    }
    return cost;
  }

  private updateParameters(delta: math.Matrix): void {
    let offset = 0;
    for (const block of this.residualBlocks) {
      const poseSize = 6; // 3 for rotation, 3 for translation
      const pointSize = 3;
      
      const poseDelta = delta.subset(math.index(math.range(offset, offset + poseSize), 0));
      offset += poseSize;
      const pointDelta = delta.subset(math.index(math.range(offset, offset + pointSize), 0));
      offset += pointSize;

      block.updateParameters(poseDelta, pointDelta);
    }
  }

  private revertUpdate(delta: math.Matrix): void {
    this.updateParameters(math.multiply(delta, -1));
  }
}

export class ReprojectionError {
  private observedX: number;
  private observedY: number;
  private K: math.Matrix;

  constructor(observedX: number, observedY: number, K: cv.Mat) {
    this.observedX = observedX;
    this.observedY = observedY;
    this.K = cvMatToMathMatrix(K);
  }

  public compute(cameraPose: math.Matrix, point3D: math.Matrix): number[] {
    const projectedPoint = this.projectPoint(cameraPose, point3D);
    const error = [
      projectedPoint[0] - this.observedX,
      projectedPoint[1] - this.observedY
    ];
    return error;
  }

  private projectPoint(cameraPose: math.Matrix, point3D: math.Matrix): number[] {
    const R = cameraPose.subset(math.index([0, 1, 2], [0, 1, 2]));
    const t = cameraPose.subset(math.index([0, 1, 2], 3));
    
    const transformedPoint = math.add(math.multiply(R, point3D), t) as math.Matrix;
    const projectedPoint = math.multiply(this.K, transformedPoint) as math.Matrix;
    
    const z = projectedPoint.get([2]);
    if (Math.abs(z) < 1e-10) {
      throw new Error('Division by zero in point projection');
    }
    
    const x = projectedPoint.get([0]) / z;
    const y = projectedPoint.get([1]) / z;
    
    return [x, y];
  }
}

class ResidualBlock {
  private costFunction: ReprojectionError;
  private cameraPose: math.Matrix;
  private point3D: math.Matrix;

  constructor(costFunction: ReprojectionError, cameraPose: math.Matrix, point3D: math.Matrix) {
    this.costFunction = costFunction;
    this.cameraPose = cameraPose;
    this.point3D = point3D;
  }

  public computeResidual(): number[] {
    return this.costFunction.compute(this.cameraPose, this.point3D);
  }

  public getCameraPose(): math.Matrix {
    return this.cameraPose;
  }

  public getPoint3D(): math.Matrix {
    return this.point3D;
  }

  public computeJacobianAndResidual(): [number[][], number[]] {
    const residual = this.computeResidual();
    const J_pose = this.numericalJacobian(this.cameraPose, 6);
    const J_point = this.numericalJacobian(this.point3D, 3);
    return [J_pose.concat(J_point), residual];
  }

  private numericalJacobian(param: math.Matrix, paramSize: number): number[][] {
    const J = [];
    const h = 1e-8;
    const residual = this.computeResidual();

    for (let i = 0; i < paramSize; i++) {
      const paramPlus = param.clone();
      paramPlus.set([i], paramPlus.get([i]) + h);
      const residualPlus = this.costFunction.compute(
        param === this.cameraPose ? paramPlus : this.cameraPose,
        param === this.point3D ? paramPlus : this.point3D
      );
      const Jcol = math.divide(math.subtract(residualPlus, residual), h) as math.Matrix;
      J.push(Jcol.toArray());
    }

    return J;
  }

  public updateParameters(poseDelta: math.Matrix, pointDelta: math.Matrix): void {
    this.cameraPose = this.updatePose(this.cameraPose, poseDelta);
    this.point3D = math.add(this.point3D, pointDelta) as math.Matrix;
  }

  private updatePose(pose: math.Matrix, delta: math.Matrix): math.Matrix {
    const R = pose.subset(math.index([0, 1, 2], [0, 1, 2]));
    const t = pose.subset(math.index([0, 1, 2], 3));
    
    const deltaR = this.expSO3(delta.subset(math.index([0, 1, 2], 0)));
    const deltaT = delta.subset(math.index([3, 4, 5], 0));
    
    const newR = math.multiply(deltaR, R) as math.Matrix;
    const newT = math.add(t, deltaT) as math.Matrix;
    
    const newPose = math.identity(4) as math.Matrix;
    newPose.subset(math.index([0, 1, 2], [0, 1, 2]), newR);
    newPose.subset(math.index([0, 1, 2], 3), newT);
    
    return newPose;
  }

  private expSO3(omega: math.Matrix): math.Matrix {
    const theta = math.norm(omega);
    if (theta < 1e-8) {
      return math.identity(3) as math.Matrix;
    }
    const omega_hat = math.matrix([
      [0, -omega.get([2]), omega.get([1])],
      [omega.get([2]), 0, -omega.get([0])],
      [-omega.get([1]), omega.get([0]), 0]
    ]);
    return math.add(
      math.identity(3),
      math.multiply(omega_hat, Math.sin(theta) / theta),
      math.multiply(math.multiply(omega_hat, omega_hat), (1 - Math.cos(theta)) / (theta * theta))
    ) as math.Matrix;
  }
}

  