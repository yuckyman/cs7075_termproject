# CS 7075: Artificial Intelligence & Robotics

## Term Project

### Gesture Robot Control

Members:
- Ian Taylor
- Matt Hensel   

### Overview

This project enables robot control through hand gesture recognition using a webcam. It combines computer vision (OpenCV), hand tracking (MediaPipe), and robot simulation (Robotics Toolbox) to create an intuitive gesture-based interface.

### Repository Structure

```
term_project/
├── gesture_robot_control/       # Main project code
│   ├── config/                  # Configuration files
│   ├── docs/                    # Documentation
│   ├── src/                     # Source code
│   │   ├── control/             # Robot control logic
│   │   ├── gestures/            # Gesture recognition
│   │   ├── multimodal/          # Multimodal integration
│   │   ├── simulation/          # Robot simulation
│   │   └── main.py              # Main application entry point
│   ├── requirements.txt         # Python dependencies
│   └── settings.py              # Global settings
├── articles/                    # Related research papers
├── report/                      # Project report files
└── README.md                    # This file
```

### Instructions

1. Clone the repository

```bash
git clone https://github.com/yuckyman/cs7075_termproject.git
cd cs7075_termproject/gesture_robot_control
```

2. Set up the environment and install the dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the application

```bash
python src/main.py
```
