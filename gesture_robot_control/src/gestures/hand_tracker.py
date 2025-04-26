import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import time

class HandTracker:
    def __init__(self, max_hands=1, min_detection_confidence=0.7,
                 motion_history_size=20):
        """
        initialize hand tracking with mediapipe
        
        args:
            max_hands (int): maximum number of hands to detect
            min_detection_confidence (float): minimum confidence for detection
            motion_history_size (int): number of frames to keep for motion tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence
        )
        
        # motion tracking
        self.motion_history_size = motion_history_size
        self.index_tip_history = deque(maxlen=motion_history_size)
        self.palm_history = deque(maxlen=motion_history_size)
        self.fingertips_history = deque(maxlen=motion_history_size)  # for wave detection
        self.palm_base_history = deque(maxlen=motion_history_size)   # for wave detection
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0  # seconds between dynamic gestures
        self.is_pointing = False  # track pointing state
        self.current_dynamic_gesture = None  # track dynamic gestures separately
        
    def find_hands(self, frame: np.ndarray, draw=True) -> Tuple[np.ndarray, List]:
        """
        detect and track hands in frame
        
        args:
            frame (np.ndarray): input image/video frame
            draw (bool): whether to draw landmarks on frame
            
        returns:
            tuple: (processed frame, list of hand landmarks)
        """
        # convert to rgb for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        
        all_hands = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                # extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))
                all_hands.append(landmarks)
                
                # detect if we're pointing
                self.is_pointing = self._is_pointing(landmarks)
                
                # update histories based on current gesture
                if self.is_pointing:
                    if landmarks:
                        self.index_tip_history.append(landmarks[8])  # index fingertip
                        palm_center = np.mean([landmarks[0], landmarks[5], landmarks[17]], axis=0)
                        self.palm_history.append(palm_center)
                else:
                    # For wave detection, track all fingertips and palm base
                    fingertips = [landmarks[tip] for tip in [4, 8, 12, 16, 20]]  # thumb to pinky tips
                    palm_base = landmarks[0]  # wrist point
                    self.fingertips_history.append(fingertips)
                    self.palm_base_history.append(palm_base)
                    
                    # Clear pointing-related history
                    self.index_tip_history.clear()
                    self.palm_history.clear()
                    self.current_dynamic_gesture = None
                
        else:
            # no hands detected, clear all histories
            self.is_pointing = False
            self.index_tip_history.clear()
            self.palm_history.clear()
            self.fingertips_history.clear()
            self.palm_base_history.clear()
            self.current_dynamic_gesture = None
            
        return frame, all_hands
    
    def get_gesture(self, landmarks: List[Tuple[int, int]]) -> Optional[str]:
        """
        classify both static and dynamic hand gestures
        
        args:
            landmarks (List[Tuple[int, int]]): list of landmark coordinates
            
        returns:
            str: detected gesture name or None
        """
        if not landmarks:
            return None
            
        # detect static gesture
        static_gesture = self._detect_static_gesture(landmarks)
        
        # check for dynamic gestures based on current static gesture
        if self.is_pointing:
            self.current_dynamic_gesture = self._detect_dynamic_gesture()
            if self.current_dynamic_gesture:
                return f"point+{self.current_dynamic_gesture}"
        elif static_gesture == "open_palm":
            # check for wave when hand is open
            if self._detect_palm_wave():
                return "open_palm+wave"
        
        return static_gesture
    
    def _detect_palm_wave(self) -> bool:
        """Detect waving motion in open palm gesture"""
        if len(self.fingertips_history) < self.motion_history_size:
            return False
            
        # Check if palm base is relatively stable
        palm_positions = np.array(list(self.palm_base_history))
        palm_motion = np.std(palm_positions, axis=0)
        if np.mean(palm_motion) > 30:  # palm moving too much
            return False
            
        # Calculate fingertip motion
        total_motion = 0
        direction_changes = 0
        prev_direction = None
        
        # Convert history to numpy array for easier calculations
        fingertips_array = np.array(list(self.fingertips_history))
        
        # Look at vertical motion of each fingertip
        for finger_idx in range(5):  # 5 fingers
            y_positions = fingertips_array[:, finger_idx, 1]  # y coordinates for this finger
            
            # Calculate motion
            motion = np.diff(y_positions)
            total_motion += np.sum(np.abs(motion))
            
            # Count direction changes
            for i in range(1, len(motion)):
                current_direction = np.sign(motion[i])
                if prev_direction is not None and current_direction != prev_direction:
                    direction_changes += 1
                prev_direction = current_direction
                
        # Conditions for wave:
        # 1. Significant motion in fingertips
        # 2. Multiple direction changes (oscillation)
        # 3. Palm relatively stable (checked earlier)
        return total_motion > 500 and direction_changes > 6
    
    def _is_pointing(self, landmarks: List[Tuple[int, int]]) -> bool:
        """Check if the hand is in a pointing gesture (index extended, others closed)"""
        # get key finger states
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # get palm center for distance calculations
        palm_center = np.mean([landmarks[0], landmarks[5], landmarks[17]], axis=0)
        
        # compute distances
        distances = {
            'thumb': np.linalg.norm(np.array(thumb_tip) - palm_center),
            'index': np.linalg.norm(np.array(index_tip) - palm_center),
            'middle': np.linalg.norm(np.array(middle_tip) - palm_center),
            'ring': np.linalg.norm(np.array(ring_tip) - palm_center),
            'pinky': np.linalg.norm(np.array(pinky_tip) - palm_center)
        }
        
        # more forgiving thresholds:
        # - index only needs to be moderately extended (>120 instead of 150)
        # - other fingers can be slightly more open (<120 instead of 100)
        return (distances['index'] > 120 and 
                all(d < 120 for k, d in distances.items() if k != 'index'))
    
    def _detect_static_gesture(self, landmarks: List[Tuple[int, int]]) -> str:
        """detect static gestures"""
        # get key finger states
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # get palm center
        palm_center = np.mean([landmarks[0], landmarks[5], landmarks[17]], axis=0)
        
        # compute distances
        distances = {
            'thumb': np.linalg.norm(np.array(thumb_tip) - palm_center),
            'index': np.linalg.norm(np.array(index_tip) - palm_center),
            'middle': np.linalg.norm(np.array(middle_tip) - palm_center),
            'ring': np.linalg.norm(np.array(ring_tip) - palm_center),
            'pinky': np.linalg.norm(np.array(pinky_tip) - palm_center)
        }
        
        # adjusted thresholds to match _is_pointing
        if all(d < 120 for d in distances.values()):
            return "fist"
        elif all(d > 120 for d in distances.values()):
            return "open_palm"
        elif self.is_pointing:
            return "point"
        elif distances['thumb'] > 120 and distances['pinky'] > 120:
            return "hang_loose"
        else:
            return "unknown"
    
    def _detect_dynamic_gesture(self) -> Optional[str]:
        """detect dynamic gestures based on motion history"""
        # need enough history for dynamic gestures
        if len(self.index_tip_history) < self.motion_history_size:
            return None
            
        # check cooldown
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return None
            
        # convert history to numpy arrays for easier math
        index_positions = np.array(list(self.index_tip_history))
        palm_positions = np.array(list(self.palm_history))
        
        # compute motion metrics
        index_motion = np.diff(index_positions, axis=0)
        total_distance = np.sum(np.linalg.norm(index_motion, axis=1))
        
        # detect horizontal swipe
        x_motion = index_positions[-1][0] - index_positions[0][0]
        y_motion = index_positions[-1][1] - index_positions[0][1]
        
        if total_distance > 150:  # reduced from 200 - more sensitive to general motion
            # check for swipe gestures
            if abs(x_motion) > 100 and abs(y_motion) < 120:  # reduced horizontal requirement, increased vertical tolerance
                self.last_gesture_time = current_time
                return "swipe_right" if x_motion > 0 else "swipe_left"
                
            # check for circle gesture
            elif self._is_circular_motion(index_positions):
                self.last_gesture_time = current_time
                return "circle"
                
            # check for wave gesture
            elif self._is_wave_motion(index_positions):
                self.last_gesture_time = current_time
                return "wave"
                
        return None
    
    def _is_circular_motion(self, positions: np.ndarray) -> bool:
        """detect if motion forms a rough circle"""
        if len(positions) < self.motion_history_size:
            return False
            
        # center the positions
        centered = positions - np.mean(positions, axis=0)
        
        # fit a circle
        x, y = centered[:, 0], centered[:, 1]
        r = np.mean(np.sqrt(x**2 + y**2))
        
        # check if points roughly form a circle
        distances = np.abs(np.sqrt(x**2 + y**2) - r)
        # more forgiving circle detection
        return np.mean(distances) < 40 and np.std(distances) < 25  # increased from 30/20
    
    def _is_wave_motion(self, positions: np.ndarray) -> bool:
        """detect if motion forms a waving pattern"""
        if len(positions) < self.motion_history_size:
            return False
            
        # look at vertical motion
        y_positions = positions[:, 1]
        peaks = 0
        direction = 0  # 0=unknown, 1=up, -1=down
        
        for i in range(1, len(y_positions)):
            if y_positions[i] > y_positions[i-1] and direction <= 0:
                direction = 1
                peaks += 1
            elif y_positions[i] < y_positions[i-1] and direction >= 0:
                direction = -1
                peaks += 1
                
        return peaks >= 4  # at least 2 full waves
    
    def draw_info(self, frame: np.ndarray, gesture: Optional[str]) -> np.ndarray:
        """
        draw gesture information and motion trail on frame
        """
        if gesture:
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
        # draw motion trail whenever pointing, regardless of dynamic gestures
        if self.is_pointing and len(self.index_tip_history) > 1:
            points = np.array(list(self.index_tip_history))
            for i in range(1, len(points)):
                thickness = int((i / len(points)) * 4) + 1
                cv2.line(frame, 
                        tuple(points[i-1]), 
                        tuple(points[i]),
                        (0, 0, 255),
                        thickness)
                
        return frame 