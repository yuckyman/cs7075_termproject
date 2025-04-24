# gesture-controlled robot arm

this project combines gesture recognition with pid control to create an intuitive interface for controlling a robotic arm. it uses computer vision for gesture detection and a pid controller for smooth, precise movements.

## features
- real-time gesture recognition using opencv and mediapipe
- pid control system for smooth motion
- simulated robotic arm (initially using pybullet, later gazebo)
- visualization tools for system response

## setup
1. create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # on unix/mac
# or
.\venv\Scripts\activate  # on windows
```

2. install dependencies:
```bash
pip install -r requirements.txt
```

## project structure
```
gesture_pid/
├── src/                    # source code
│   ├── gestures/          # gesture recognition
│   ├── control/           # pid controller
│   ├── simulation/        # robot simulation
│   └── utils/             # shared utilities
├── config/                # configuration files
├── tests/                 # test cases
└── requirements.txt       # dependencies
```

## running the project
```bash
python src/main.py
``` 