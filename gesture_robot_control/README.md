# Gesture Robot Control

A system for controlling robots using hand gestures, built with Python.

## Overview

This project enables robot control through hand gesture recognition using a webcam. It combines computer vision (OpenCV), hand tracking (MediaPipe), and robot simulation (Robotics Toolbox) to create an intuitive gesture-based interface.

## Working Features

- Real-time hand gesture recognition and tracking
- Control of a simulated robot arm via hand position and gestures
- Supported gestures:
  - Point: Controls X,Y position of robot end-effector
  - Point + circle: Controls Z height (simulated but not fully implemented)
  - Open palm + wave: Toggles gripper state (simulated through console output)
- Visual feedback in OpenCV window

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- Robotics Toolbox for Python
- Additional dependencies listed in requirements.txt

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo:
   ```bash
   python src/main.py
   ```

3. Gesture controls:
   - Point at the camera to control X,Y position
   - Open/close your hand to toggle the gripper
   - Press 'q' to quit

## Project Structure

```
gesture_robot_control/
├── src/                      # core modules
│   ├── main.py               # entry point
│   ├── multimodal/           # gesture recognition components
│   │   └── gesture_control.py # gesture recognition and mapping
│   └── simulation/           # robot simulation
│       └── robot_sim.py      # robot simulator interface
├── requirements.txt          # project dependencies
```

## System Architecture and Flow

### Initialization

main.py creates two key objects:
GestureControl: handles hand tracking and gesture recognition
RobotSimulator: handles robot simulation and command execution

### Video Processing Loop

- captures frames from the webcam
- flips the frame horizontally (for more intuitive control)
- sends each frame to GestureControl.process_frame()

### Gesture Recognition

HandTracker uses MediaPipe to detect hands and landmarks
- detects both static gestures (pointing, open palm) and dynamic gestures (wave, circle)
- maintains motion history to recognize movements over time

### Command Generation

- converts recognized gestures to robot commands
- normalizes hand positions to robot workspace coordinates
- outputs RobotCommand objects with action and parameters

### Robot Execution

- RobotSimulator uses Robotics Toolbox for Python to simulate a Panda robot
- converts command coordinates to robot end-effector poses
- solves inverse kinematics to determine joint angles
- moves the simulated robot accordingly

### Key Gestures and Mapping

- pointing gesture:
  - detected when index finger is extended and other fingers are closed
  - index fingertip position (x,y) maps to robot end-effector position
  - normalized from pixel coordinates to robot workspace
- open palm + wave:
  - detected when all fingers are extended and moved in a waving pattern
  - toggles the gripper state (open/close)
- point + circle (partially implemented):
  - intended to control z-height when making circular motions
  - detection code exists but full implementation is simplified

### Technical Details

- hand tracking: uses MediaPipe Hands, which provides 21 hand landmarks
- coordinate normalization: camera coordinates (pixels) are normalized to robot workspace (meters)
- robot simulation: uses roboticstoolbox library with a simulated Panda robot in PyPlot environment
- inverse kinematics: the ik_LM method (Levenberg-Marquardt algorithm) solves for joint angles
- error handling:
  - handles cases where no hands are detected
  - reports when inverse kinematics cannot find a solution (position out of reach)
  - gracefully handles program termination with proper resource cleanup

## Development

- Create virtual environment: `python -m venv venv`
- Activate virtual environment: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
- The program uses MediaPipe for hand tracking and Robotics Toolbox for robot simulation

## Troubleshooting

- If positions are outside the robot's workspace, "IK solution not found" will be displayed
- Ensure good lighting for reliable hand tracking
- Keep your hand within the camera frame

## License

MIT License 