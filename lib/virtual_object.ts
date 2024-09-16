import * as math from 'mathjs';

export class VirtualObject {
  public position: math.Matrix;
  public orientation: math.Matrix;
  public scale: number;
  public id: string;

  constructor(position: math.Matrix, id: string, scale: number = 1) {
    this.position = position;
    this.orientation = math.identity(3) as math.Matrix;
    this.scale = scale;
    this.id = id;
  }

  public updatePosition(framePose: math.Matrix): void {
    this.position = math.multiply(framePose, this.position) as math.Matrix;
  }

  public updateOrientation(rotation: math.Matrix): void {
    this.orientation = math.multiply(rotation, this.orientation) as math.Matrix;
  }

  public setScale(scale: number): void {
    this.scale = scale;
  }
}
