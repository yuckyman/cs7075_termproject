# gesture_module.py - Placeholder 

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

class GestureRecognizer:
    def __init__(self, config):
        """Initializes the MediaPipe Hands model and stores configuration."""
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.gesture_labels = config["labels"]["gestures"]
        
        # Trail for dynamic gesture recognition
        self.trail_length = 15
        self.index_finger_trail = deque(maxlen=self.trail_length)
        
        # Gesture detection parameters
        self.min_swipe_distance = 0.1
        self.min_circle_points = 10
        self.circle_threshold = 0.6

    def process_frame(self, frame):
        """
        Takes a single video frame, detects hands, tracks gestures,
        and returns a gesture token string.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            self.index_finger_trail.clear()  # Clear trail when no hand detected
            return "<gesture_unknown>"

        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Check if we're in a pointing gesture before tracking
        if self._is_pointing(hand_landmarks):
            # Only track index finger when pointing
            index_tip = hand_landmarks.landmark[8]  # 8 is the index fingertip
            self.index_finger_trail.append((index_tip.x, index_tip.y))
        else:
            # Clear the trail when not pointing
            self.index_finger_trail.clear()
        
        # Classify the gesture
        gesture_token = self._classify_gesture(hand_landmarks)
        return gesture_token

    def _is_pointing(self, hand_landmarks):
        """Check if the hand is in a pointing gesture (index extended, others closed)."""
        index_tip = hand_landmarks.landmark[8]    # index fingertip
        index_pip = hand_landmarks.landmark[6]    # index pip (middle joint)
        middle_tip = hand_landmarks.landmark[12]  # middle fingertip
        ring_tip = hand_landmarks.landmark[16]    # ring fingertip
        pinky_tip = hand_landmarks.landmark[20]   # pinky fingertip
        wrist = hand_landmarks.landmark[0]        # wrist

        # Check if index is extended (using both tip and middle joint)
        index_extended = (index_tip.y < index_pip.y and index_pip.y < wrist.y)
        
        # More lenient check for other fingers (just need to be lower than index)
        others_closed = all(
            finger.y > index_tip.y
            for finger in [middle_tip, ring_tip, pinky_tip]
        )

        return index_extended and others_closed

    def _classify_gesture(self, hand_landmarks):
        """
        Classifies both static and dynamic hand gestures.
        """
        if self._is_pointing(hand_landmarks):
            # Check for dynamic gestures when pointing
            if len(self.index_finger_trail) >= self.min_circle_points:
                # Check for circular motion
                if self._detect_circle():
                    return "<gesture_circle>"
                
                # Check for swipe
                if self._detect_swipe():
                    return "<gesture_swipe>"
            
            return "<gesture_point>"
        
        return "<gesture_unknown>"

    def _is_finger_extended(self, finger_tip, wrist):
        """Check if a finger is extended by comparing y position to wrist."""
        return finger_tip.y < wrist.y

    def _detect_swipe(self):
        """Detect horizontal or vertical swipe gestures."""
        if len(self.index_finger_trail) < 2:
            return False

        start_point = self.index_finger_trail[0]
        end_point = self.index_finger_trail[-1]
        
        distance = math.sqrt(
            (end_point[0] - start_point[0])**2 + 
            (end_point[1] - start_point[1])**2
        )
        
        return distance > self.min_swipe_distance

    def _detect_circle(self):
        """Detect circular motion using the index finger trail."""
        if len(self.index_finger_trail) < self.min_circle_points:
            return False

        # Calculate center of the trail
        center_x = sum(p[0] for p in self.index_finger_trail) / len(self.index_finger_trail)
        center_y = sum(p[1] for p in self.index_finger_trail) / len(self.index_finger_trail)

        # Calculate average radius
        radii = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2)
                for p in self.index_finger_trail]
        avg_radius = sum(radii) / len(radii)

        # Check if points form a circle by comparing distances to average radius
        radius_diffs = [abs(r - avg_radius) for r in radii]
        avg_diff = sum(radius_diffs) / len(radius_diffs)

        return avg_diff < (1 - self.circle_threshold)

    def close(self):
        """Releases resources used by the hands model."""
        self.hands.close()
        print("GestureRecognizer resources released.") 