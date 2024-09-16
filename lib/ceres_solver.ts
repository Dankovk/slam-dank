import * as math from 'mathjs';


export class ReprojectionError {
    private observedX: number;
    private observedY: number;
    private K: math.Matrix;
  
    constructor(observedX: number, observedY: number, K: math.Matrix) {
      this.observedX = observedX;
      this.observedY = observedY;
      this.K = K; // Camera intrinsics matrix
    }
  
    // This function computes the reprojection error
    compute(cameraPose: math.Matrix, point3D: math.Matrix): number[] {
      // Project the 3D point using the camera pose and intrinsics
      const projected = math.multiply(cameraPose, point3D);
      const normalized = math.multiply(this.K, projected);
  
      // Normalize the projected point (perspective division)
      const xp = normalized.get([0]) / normalized.get([2]);
      const yp = normalized.get([1]) / normalized.get([2]);
  
      // Calculate the reprojection error
      const errorX = xp - this.observedX;
      const errorY = yp - this.observedY;
  
      return [errorX, errorY];
    }
  }
  
export class ResidualBlock {
    private costFunction: ReprojectionError;
    private cameraPose: math.Matrix;
    private point3D: math.Matrix;
  
    constructor(costFunction: ReprojectionError, cameraPose: math.Matrix, point3D: math.Matrix) {
      this.costFunction = costFunction;
      this.cameraPose = cameraPose;
      this.point3D = point3D;
    }
  
    // Compute the residual (reprojection error)
    computeResidual(): number[] {
      return this.costFunction.compute(this.cameraPose, this.point3D);
    }
  
    // Getters for the camera pose and 3D point
    getCameraPose(): math.Matrix {
      return this.cameraPose;
    }
  
    getPoint3D(): math.Matrix {
      return this.point3D;
    }
  }
  

export class CeresSolver {
    private residualBlocks: ResidualBlock[] = [];
    private lambda: number = 1e-3; // Damping factor for LM optimization
    private maxIterations: number = 100;
    private tolerance: number = 1e-6;
  
    constructor() {}
  
    // Add residual blocks (camera pose, 3D point, reprojection error)
    public addResidualBlock(costFunction: ReprojectionError, cameraPose: math.Matrix, point3D: math.Matrix): void {
      const residualBlock = new ResidualBlock(costFunction, cameraPose, point3D);
      this.residualBlocks.push(residualBlock);
    }
  
    // Solve the optimization problem using the Levenberg-Marquardt algorithm
    public solve(): void {
      for (let iter = 0; iter < this.maxIterations; iter++) {
        let totalError = 0;
  
        for (const block of this.residualBlocks) {
          const residual = block.computeResidual();
          const error = Math.sqrt(residual[0] * residual[0] + residual[1] * residual[1]);
  
          // Update total error to check convergence
          totalError += error;
  
          // Levenberg-Marquardt update step (simplified)
          this.updateCameraPose(block.getCameraPose(), residual);
          this.updatePoint3D(block.getPoint3D(), residual);
        }
  
        // Check for convergence
        if (totalError < this.tolerance) {
          console.log(`Converged after ${iter} iterations`);
          break;
        }
      }
    }
  
    // Update camera pose based on reprojection error
    private updateCameraPose(pose: math.Matrix, residual: number[]): void {
      // Simplified gradient descent-like update
      const delta = math.matrix([[-this.lambda * residual[0], 0, 0, 0], [0, -this.lambda * residual[1], 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
      math.add(pose, delta);
    }
  
    // Update 3D point based on reprojection error
    private updatePoint3D(point3D: math.Matrix, residual: number[]): void {
      const deltaPoint = math.matrix([this.lambda * residual[0], this.lambda * residual[1], 0]);
      math.add(point3D, deltaPoint);
    }
  }

  