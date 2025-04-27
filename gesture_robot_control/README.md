# Gesture Robot Control

A system for controlling a wheeled robot using hand gestures via webcam input.

## Overview

This project provides real-time gesture command processing to control a simulated wheeled robot via webcam input.

## Working Features

- Real-time hand gesture recognition and tracking using MediaPipe
- Gesture control for wheeled robot:
  - Point to drive toward that direction
  - Make a fist; orientation controls rotation direction
  - Open palm + wave to toggle stop/go
- Visual feedback in OpenCV windows for camera feed and recognized gestures
- Configurable parameters via `control/config.py`

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- robotics-toolbox-python
- Additional dependencies listed in `requirements.txt`

## Quick Start

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate       # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the demo:
   ```bash
   python src/main.py            # Wheeled robot gesture control
   ```
4. Controls:
   - Point at the camera to drive toward that direction
   - Make a fist; orientation controls rotation direction
   - Open palm + wave to toggle stop/go
   - Press `q` to quit

## Project Structure

```
gesture_robot_control/
├── src/
│   ├── main.py
│   ├── control/
│   │   ├── gesture_control.py
│   │   └── config.py
│   ├── gestures/      # custom gesture definitions
│   └── simulation/
│       ├── wheeled_robot_sim.py
│       ├── sim_env.py
│       └── test_sim.py
├── requirements.txt
└── README.md
```

## System Architecture and Flow

1. **Initialization**:
   - Initialize `GestureControl`
   - Initialize `WheeledRobotSimulator`

2. **Input Loop**:
   - Capture video frames and process each frame for gesture recognition to generate `RobotCommand` objects

3. **Command Execution**:
   - Map gestures to robot commands (velocity and rotation)
   - Execute commands in simulation environment with visual feedback

4. **Cleanup**:
   - Release camera and close simulation windows

## Configuration

Edit parameters in `control/config.py`:
- Camera settings (width, height)
- Gesture recognition thresholds
- Robot workspace bounds

## Troubleshooting

- Ensure good lighting for reliable hand tracking
- Keep your hand in the camera frame
- Verify your setup (e.g., camera) and adjust settings in `control/config.py`

## License

MIT License 