
# Slam Dank

**Slam Dank** is a real-time system for 3D tracking, sensor fusion, and SLAM (Simultaneous Localization and Mapping) that integrates data from computer vision and IMU (Inertial Measurement Unit) sensors. The system is designed to track 3D objects and estimate their position and orientation in real time.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [React-Three-Fiber (r3f) Example](#react-three-fiber-r3f-example)

## Features

- **IMU Data Processing**: Process real-time IMU sensor data (accelerometer and gyroscope) to estimate orientation and velocity.
- **Computer Vision Integration**: Extract and match ORB features from image frames using OpenCV.
- **SLAM Implementation**: Perform Simultaneous Localization and Mapping using keyframes and feature matching.
- **RANSAC Algorithm**: Use RANSAC for robust pose estimation with noisy sensor data.
- **Non-linear Optimization**: Integrate a custom Ceres Solver for optimizing camera poses and 3D point projections.

## Installation

### Prerequisites

- **Node.js** and **Bun**: Ensure you have Node.js installed. Install [Bun](https://bun.sh) as the package manager.
- **OpenCV**: This project requires OpenCV with JavaScript bindings. You can install it from npm:
  
  ```bash
  npm install @techstark/opencv-js
  ```

### Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/Dankovk/slam-dank.git
cd slam-dank
```

### Install Dependencies

Run the following command to install all required dependencies:

```bash
bun install
```

### Build the Project

To build the project, use:

```bash
bun build
```

## React-Three-Fiber (r3f) Example

This example demonstrates how to place a cube in a 3D space where the user taps on the camera feed, using the `SlamDank` class to perform SLAM tracking. The cube is placed at the point where the user taps, and its position and orientation are updated in real time using the SLAM tracking system.

### Example Code:

```jsx
import React, { useEffect, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { SlamDank } from './lib/slam_dank'; // Import SlamDank class
import * as cv from '@techstark/opencv-js';

const CameraFeed = ({ onTap }) => {
  const videoRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          video.srcObject = stream;
          video.play();
        })
        .catch(err => console.error("Error accessing camera: ", err));
    }
  }, []);

  return (
    <div
      style={{ position: 'relative' }}
      onClick={(e) => onTap(e.clientX, e.clientY)}
    >
      <video ref={videoRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

const Cube = ({ position, orientation }) => {
  const mesh = useRef();

  // Update the cube's position and orientation using SLAM data
  useFrame(() => {
    if (mesh.current) {
      mesh.current.position.set(...position);
      // Update rotation from orientation matrix
      mesh.current.rotation.setFromRotationMatrix(orientation);
    }
  });

  return (
    <mesh ref={mesh}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="orange" />
    </mesh>
  );
};

const App = () => {
  const [virtualObjects, setVirtualObjects] = useState([]);
  const slamDank = useRef(new SlamDank()); // Initialize the SlamDank system

  const handleUserTap = async (x, y) => {
    console.log(`User tapped at position: ${x}, ${y}`);

    // Capture the image frame from the camera feed
    const srcMat = new cv.Mat(); // This would be the actual image from the video feed
    const timestamp = Date.now();

    // Perform SLAM tracking using the SlamDank system
    const worldPoint = await slamDank.current.trackMonocularIMU(srcMat, { x, y }, timestamp);

    // Update the list of virtual objects with the new object at the world point
    setVirtualObjects([...virtualObjects, { position: worldPoint.position, orientation: worldPoint.orientation }]);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <CameraFeed onTap={handleUserTap} />
      <Canvas style={{ flex: 1 }}>
        <ambientLight />
        <pointLight position={[10, 10, 10]} />
        {virtualObjects.map((obj, index) => (
          <Cube key={index} position={obj.position} orientation={obj.orientation} />
        ))}
      </Canvas>
    </div>
  );
};

export default App;
```

### Explanation:

1. **Camera Feed**: The `CameraFeed` component captures live video from the camera and provides the ability for the user to tap on the feed. The tap coordinates are passed to the `handleUserTap` function.

2. **SLAM Tracking**: In `handleUserTap`, the system captures the current frame and uses the `SlamDank` class to perform SLAM tracking. The user tap's 2D coordinates are translated into 3D world coordinates using SLAM.

3. **Cube Placement**: A cube is placed in the virtual 3D world at the point corresponding to the tap. Its position and orientation are updated using the SLAM tracking data.

4. **Virtual Objects**: The `virtualObjects` state stores the list of virtual cubes that are placed at user tap positions. Each cube's position and orientation are tracked using SLAM and visualized using `React-Three-Fiber`.

### Key Components:

- **SlamDank Class**: The `SlamDank` class is responsible for performing the SLAM operations. When the user taps the camera feed, SLAM tracking is initiated, and the cube is placed at the corresponding 3D location.
- **Cube Visualization**: The `Cube` component uses the position and orientation provided by the SLAM system to render a cube in 3D space.

### Steps:

1. **Install React-Three-Fiber**:

   ```bash
   npm install @react-three/fiber three
   ```

2. **Install OpenCV**:

   ```bash
   npm install @techstark/opencv-js
   ```

3. **Run the App**: After installing the dependencies, you can run the app, and as you tap on the video feed, cubes will be placed at corresponding 3D points, and their positions and orientations will be updated in real-time using SLAM data.


## License

This project is licensed under the MIT License.
