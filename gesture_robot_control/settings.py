# pid controller settings
PID_CONFIG = {
    'kp': 1.0,  # proportional gain
    'ki': 0.1,  # integral gain
    'kd': 0.05  # derivative gain
}

# gesture recognition settings
GESTURE_CONFIG = {
    'camera_id': 0,  # webcam id
    'confidence_threshold': 0.5,  # minimum confidence for gesture detection
    'frame_width': 640,
    'frame_height': 480,
    'labels': {
        'gestures': [
            '<gesture_unknown>',
            '<gesture_point>',
            '<gesture_swipe>',
            '<gesture_circle>'
        ]
    }
}

# simulation settings
SIM_CONFIG = {
    'timestep': 1/240,  # simulation timestep
    'max_force': 100,   # maximum force for robot joints
    'target_velocity': 0.5  # target velocity for movements
} 