# config.py - Shared configuration

CONFIG = {
    # --- Classifier Paths ---
    "gesture_classifier_path": "models/gesture_clf.joblib",
    "tone_classifier_path": "models/tone_clf.joblib",

    # --- MediaPipe Pose Configuration ---
    "pose_config": {
        "static_image_mode": False,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5
    },
    
    # --- Gesture Recognition Settings ---
    "gesture_config": {
        "camera_id": 0,  # webcam id
        "confidence_threshold": 0.5,  # minimum confidence for gesture detection
        "frame_width": 640,
        "frame_height": 480,
        "labels": {
            "gestures": [
                '<gesture_unknown>',
                '<gesture_point>',
                '<gesture_swipe>',
                '<gesture_circle>'
            ]
        }
    },

    # --- Simulation Settings ---
    "sim_config": {
        "timestep": 1/240,  # simulation timestep
        "max_force": 100,   # maximum force for robot joints
        "target_velocity": 0.5  # target velocity for movements
    }
} 