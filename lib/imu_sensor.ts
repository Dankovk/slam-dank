import { IMUData } from './imu_handler';
import * as math from 'mathjs';

export function startIMUListener(callback: (imuData: IMUData) => void): void {
  if ('DeviceMotionEvent' in window && typeof (DeviceMotionEvent as any).requestPermission === 'function') {
    (DeviceMotionEvent as any).requestPermission()
      .then((response: string) => {
        if (response === 'granted') {
          window.addEventListener('devicemotion', deviceMotionHandler);
        }
      })
      .catch(console.error);
  } else {
    window.addEventListener('devicemotion', deviceMotionHandler);
  }

  function deviceMotionHandler(event: DeviceMotionEvent) {
    if (event.accelerationIncludingGravity && event.rotationRate) {
      const acceleration = math.matrix([
        event.accelerationIncludingGravity.x || 0,
        event.accelerationIncludingGravity.y || 0,
        event.accelerationIncludingGravity.z || 0
      ]);

      const gyroscope = math.matrix([
        (event.rotationRate.alpha || 0) * (Math.PI / 180),
        (event.rotationRate.beta || 0) * (Math.PI / 180),
        (event.rotationRate.gamma || 0) * (Math.PI / 180)
      ]);

      const imuData: IMUData = {
        acceleration,
        gyroscope,
        timestamp: performance.now() / 1000
      };
      callback(imuData);
    }
  }
}
