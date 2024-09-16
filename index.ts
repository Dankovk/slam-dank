import { System } from "./lib/system";
import * as cv from '@techstark/opencv-js';
import * as math from 'mathjs';
import { IMUData } from './lib/imu_handler';
import { VirtualObject } from './lib/virtual_object';

class SlamDank {
    private system: System;

    constructor() {
        this.system = new System();
    }
    
    // User-facing API methods
    public initialize(): void {
        console.log("Initializing SlamDank...");
        // Any additional initialization logic can be added here
    }

    public processFrame(frame: ImageData): void {
        const mat = new cv.Mat(frame.height, frame.width, cv.CV_8UC4);
        mat.data.set(frame.data);
        const timestamp = performance.now() / 1000; // Convert to seconds
        this.system.trackMonocular(mat, timestamp);
        mat.delete();
    }

    public processFrameWithIMU(frame: ImageData, imuData: IMUData): void {
        const mat = new cv.Mat(frame.height, frame.width, cv.CV_8UC4);
        mat.data.set(frame.data);
        const timestamp = performance.now() / 1000; // Convert to seconds
        this.system.trackMonocularIMU(mat, imuData, timestamp);
        mat.delete();
    }

    public getPose(): { position: number[], rotation: number[] } {
        const currentFrame = this.system.getCurrentFrame();
        if (!currentFrame) {
            return { position: [0, 0, 0], rotation: [0, 0, 0] };
        }

        const pose = currentFrame.getPose();
        const position = pose.subset(math.index([0, 1, 2], 3)).toArray() as number[];
        
        // Extract rotation from the pose matrix
        const rotationMatrix = pose.subset(math.index([0, 1, 2], [0, 1, 2]));
        const rotation = this.rotationMatrixToEuler(rotationMatrix);

        return { position, rotation };
    }

    public getMap(): { points: number[][], features: any[] } {
        const mapPoints = this.system.getMapPoints();
        const points = mapPoints.map(point => point.position.toArray() as number[]);
        
        // For simplicity, we're not returning actual feature descriptors
        // as they might be too large and not directly useful for most users
        const features = mapPoints.map(point => ({
            position: point.position.toArray(),
            observationCount: typeof point.observations === 'number' ? point.observations : 0
        }));

        return { points, features };
    }

    public getVirtualObjects(): VirtualObject[] {
        return this.system.getVirtualObjects();
    }

    public handleUserTap(x: number, y: number): void {
        const tapPosition = new cv.Point(x, y);
        this.system.handleUserTap(tapPosition);
    }

    public reset(): void {
        console.log("Resetting SlamDank...");
        this.system = new System();
    }

    private rotationMatrixToEuler(R: math.Matrix): number[] {
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
}

export { SlamDank };
