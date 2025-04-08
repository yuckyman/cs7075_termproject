# gesture_module.py - Placeholder 

import cv2
import mediapipe as mp
import numpy as np
import math # Add math import for distance calculation

class GestureRecognizer:
    def __init__(self, config):
        """Initializes the MediaPipe Pose model and stores configuration."""
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=config["pose_config"]["static_image_mode"],
            min_detection_confidence=config["pose_config"]["min_detection_confidence"],
            min_tracking_confidence=config["pose_config"]["min_tracking_confidence"]
        )
        self.gesture_labels = config["labels"]["gestures"]
        # TODO: Load or initialize the actual gesture classifier if needed
        # self.classifier = load_classifier(config["gesture_classifier_path"])

    def process_frame(self, frame):
        """
        Takes a single video frame (numpy array BGR), detects poses,
        extracts keypoints, classifies gesture, and returns a gesture token string.
        Returns None if no pose is detected.
        """
        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True # Image is now writeable again

        gesture_token = None
        if results.pose_landmarks:
            # Extract relevant joint info here
            keypoints = self._extract_keypoints(results.pose_landmarks)
            if keypoints is not None:
                # Classify the gesture based on keypoints
                gesture_token = self._classify_gesture(keypoints)
        
        # Optional: Draw the pose annotation on the image (useful for debugging).
        # Convert the image back to BGR for OpenCV display
        # frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        # if results.pose_landmarks:
        #     mp.solutions.drawing_utils.draw_landmarks(
        #         frame_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return gesture_token

    def _extract_keypoints(self, pose_landmarks):
        """
        Extracts keypoints (x, y, z, visibility) from MediaPipe pose landmarks.

        Args:
            pose_landmarks: The pose landmarks detected by MediaPipe.

        Returns:
            A numpy array of shape (33, 4) containing the keypoints, 
            or None if extraction fails.
        """
        if not pose_landmarks:
            return None
        
        landmarks_list = []
        for landmark in pose_landmarks.landmark:
            landmarks_list.append([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        
        keypoints = np.array(landmarks_list)
        # print(f"Extracted keypoints shape: {keypoints.shape}") # Debug print
        return keypoints

    def _classify_gesture(self, keypoints):
        """
        Classifies the gesture based on the extracted keypoints using simple rules.
        Focuses on upper body gestures (shoulders, wrists).

        Args:
            keypoints: A numpy array of shape (33, 4) containing the keypoints (x, y, z, visibility).

        Returns:
            A string token representing the classified gesture (e.g., '<gesture_neutral>').
        """
        # --- Define Landmark Indices (based on MediaPipe Pose) ---
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        # LEFT_ELBOW = 13 # Might be useful later
        # RIGHT_ELBOW = 14 # Might be useful later

        # --- Visibility Threshold ---
        VISIBILITY_THRESHOLD = 0.3

        # --- Get Key Landmark Coordinates & Check Visibility ---
        try:
            l_shoulder = keypoints[LEFT_SHOULDER]
            r_shoulder = keypoints[RIGHT_SHOULDER]
            l_wrist = keypoints[LEFT_WRIST]
            r_wrist = keypoints[RIGHT_WRIST]

            # Check visibility of essential landmarks
            if (l_shoulder[3] < VISIBILITY_THRESHOLD or
                r_shoulder[3] < VISIBILITY_THRESHOLD or
                l_wrist[3] < VISIBILITY_THRESHOLD or
                r_wrist[3] < VISIBILITY_THRESHOLD):
                # print("Key landmarks not visible, defaulting to neutral.")
                return f"<gesture_{self.gesture_labels[0]}>" # Default to neutral

        except IndexError:
            # print("Key landmark index out of bounds.")
            return f"<gesture_{self.gesture_labels[0]}>" # Default to neutral

        # --- Calculate Distances (using x, y coordinates for simplicity) ---
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        shoulder_width = distance(l_shoulder, r_shoulder)
        wrist_dist = distance(l_wrist, r_wrist)

        # --- Calculate Relative Heights (Y coordinate, lower value is higher up) ---
        avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        avg_wrist_y = (l_wrist[1] + r_wrist[1]) / 2

        # --- Define Gesture Rules ---
        gesture = self.gesture_labels[0] # Default to neutral

        # Rule for 'open': Wrists are significantly wider than shoulders
        # Tunable threshold: 1.5 might mean arms wide open
        if wrist_dist > shoulder_width * 1.5:
            gesture = self.gesture_labels[2] # 'open'
        
        # Rule for 'angry': Wrists are close together and relatively high (crossed arms)
        # Tunable thresholds: 0.8 for closeness, 0.1 for height relative to shoulders
        elif wrist_dist < shoulder_width * 0.8 and avg_wrist_y < avg_shoulder_y + 0.1:
             gesture = self.gesture_labels[1] # 'angry'

        # Otherwise, it remains 'neutral'

        # print(f"Classified as: {gesture}, WristDist: {wrist_dist:.2f}, ShoulderWidth: {shoulder_width:.2f}, WristY: {avg_wrist_y:.2f}, ShoulderY: {avg_shoulder_y:.2f}") # Debug print
        return f"<gesture_{gesture}>"

    def close(self):
        """Releases resources used by the pose model."""
        self.pose.close()
        print("GestureRecognizer resources released.") 