import * as math from 'mathjs';

export class VirtualObject {
  public position: math.Matrix;

  constructor(position: math.Matrix) {
    this.position = position;
  }

  public updatePosition(framePose: math.Matrix): void {
    // Transform the object's position based on the current frame pose
    this.position = math.multiply(framePose, this.position) as math.Matrix;
  }
}
