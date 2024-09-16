import * as math from 'mathjs';

export interface IMUData {
  acceleration: math.Matrix; // 3x1 vector
  gyroscope: math.Matrix; // 3x1 vector
  timestamp: number;
}

export class IMUHandler {
  private orientation: math.Matrix; // 3x3 rotation matrix
  private position: math.Matrix; // 3x1 vector
  private velocity: math.Matrix; // 3x1 vector
  private lastTimestamp: number;

  constructor() {
    this.orientation = math.identity(3) as math.Matrix;
    this.position = math.zeros(3, 1) as math.Matrix;
    this.velocity = math.zeros(3, 1) as math.Matrix;
    this.lastTimestamp = -1;
  }

  public processIMUData(imuData: IMUData): void {
    if (this.lastTimestamp < 0) {
      this.lastTimestamp = imuData.timestamp;
      return;
    }

    const dt = imuData.timestamp - this.lastTimestamp;
    this.lastTimestamp = imuData.timestamp;

    // Integrate IMU data
    this.integrateIMUData(imuData, dt);

    console.log('IMU data processed:', {
      acceleration: imuData.acceleration.toString(),
      gyroscope: imuData.gyroscope.toString(),
      timestamp: imuData.timestamp
    });
  }

  private integrateIMUData(imuData: IMUData, dt: number): void {
    // Update orientation
    const omega = math.multiply(imuData.gyroscope, dt) as math.Matrix;
    const deltaTheta = math.norm(omega) as number;
    if (deltaTheta > 0) {
      const axis = math.divide(omega, deltaTheta) as math.Matrix;
      const deltaRotation = this.rotationMatrixFromAxisAngle(axis, deltaTheta);
      this.orientation = math.multiply(this.orientation, deltaRotation) as math.Matrix;
    }

    // Update velocity and position
    const accWorld = math.multiply(this.orientation, imuData.acceleration) as math.Matrix;
    this.velocity = math.add(this.velocity, math.multiply(accWorld, dt) as math.Matrix) as math.Matrix;
    this.position = math.add(
      this.position,
      math.add(
        math.multiply(this.velocity, dt) as math.Matrix,
        math.multiply(accWorld, 0.5 * dt * dt) as math.Matrix
      ) as math.Matrix
    ) as math.Matrix;
  }

  public getPredictedPose(): math.Matrix {
    const pose = math.identity(4) as math.Matrix;
    pose.subset(math.index([0, 1, 2], [0, 1, 2]), this.orientation);
    pose.subset(math.index([0, 1, 2], 3), this.position);
    return pose;
  }

  private rotationMatrixFromAxisAngle(axis: math.Matrix, angle: number): math.Matrix {
    // Rodrigues' rotation formula
    const K = math.matrix([
      [0, -axis.get([2]), axis.get([1])],
      [axis.get([2]), 0, -axis.get([0])],
      [-axis.get([1]), axis.get([0]), 0]
    ]);
    const I = math.identity(3) as math.Matrix;
    const R = math.add(
      math.add(
        I,
        math.multiply(K, Math.sin(angle))
      ),
      math.multiply(math.multiply(K, K), 1 - Math.cos(angle))
    ) as math.Matrix;
    return R;
  }

  public getOrientation(): math.Matrix {
    return this.orientation;
  }

  public getPosition(): math.Matrix {
    return this.position;
  }

  public reset(): void {
    this.orientation = math.identity(3) as math.Matrix;
    this.position = math.zeros(3, 1) as math.Matrix;
    this.velocity = math.zeros(3, 1) as math.Matrix;
    this.lastTimestamp = -1;
  }
}
