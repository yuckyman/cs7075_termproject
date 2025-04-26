import numpy as np
import time
from typing import Tuple, Optional
from gestures.hand_tracker import HandTracker
from dataclasses import dataclass

@dataclass
class RobotCommand:
    """represents a command to be sent to the robot"""
    action: str  # move_xyz, move_xy, rotate, grip, etc.
    params: dict  # parameters for the action (coordinates, angles, etc.)

class GestureControl:
    def __init__(self, control_mode="arm"):
        self.hand_tracker = HandTracker()
        self.last_position: Optional[Tuple[float, float, float]] = None
        self.is_gripping = False
        self.control_mode = control_mode  # can be "arm" or "wheeled"
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # seconds between grip toggles
        
    def process_frame(self, frame) -> Optional[RobotCommand]:
        """process a frame and return a robot command if a gesture is detected"""
        frame, hands = self.hand_tracker.find_hands(frame)
        
        if not hands:
            return None
            
        landmarks = hands[0]  # we're only tracking one hand
        gesture = self.hand_tracker.get_gesture(landmarks)
        
        if not gesture:
            return None
            
        return self._gesture_to_command(gesture, landmarks)
        
    def _gesture_to_command(self, gesture: str, landmarks) -> Optional[RobotCommand]:
        """convert a gesture to a robot command"""
        
        if self.control_mode == "arm":
            return self._arm_gesture_to_command(gesture, landmarks)
        else:
            return self._wheeled_gesture_to_command(gesture, landmarks)
    
    def _arm_gesture_to_command(self, gesture: str, landmarks) -> Optional[RobotCommand]:
        """convert a gesture to a robot arm command"""
        
        # pointing gestures control xyz position
        if "point" in gesture:
            # use index finger tip (landmark 8) for position control
            index_tip = landmarks[8]
            # normalize coordinates to robot workspace
            x = self._normalize_coordinate(index_tip[0], 0, 640)  # adjust based on your camera
            y = self._normalize_coordinate(index_tip[1], 0, 480)
            z = 0.5  # default height, can be controlled by other gestures
            
            if "point+circle" in gesture:
                # circular motion could control z-axis
                z = self._calculate_z_from_circle()
            
            return RobotCommand(
                action="move_xyz",
                params={"position": (x, y, z)}
            )
            
        # open palm + wave could control gripping
        elif "open_palm+wave" in gesture:
            self.is_gripping = not self.is_gripping
            return RobotCommand(
                action="grip",
                params={"grip_state": self.is_gripping}
            )
            
        return None
    
    def _wheeled_gesture_to_command(self, gesture: str, landmarks) -> Optional[RobotCommand]:
        """convert a gesture to a wheeled robot command"""
        
        # pointing gestures control x,y position
        if "point" in gesture:
            # use index finger tip (landmark 8) for position control
            index_tip = landmarks[8]
            # normalize coordinates to robot workspace
            x = self._normalize_coordinate(index_tip[0], 0, 640)  # adjust based on your camera
            y = self._normalize_coordinate(index_tip[1], 0, 480)
            
            return RobotCommand(
                action="move_xy",
                params={"position": (x, y)}
            )
            
        # fist - rotate in place based on hand position
        elif gesture == "fist":
            wrist = landmarks[0]  # wrist landmark
            # Use x position to determine rotation direction and speed
            x_pos = wrist[0]
            # Map x position to a rotation angle (-1 to 1)
            rotation = (x_pos / 640) * 2 - 1
            
            return RobotCommand(
                action="rotate",
                params={"angle": rotation}
            )
            
        # open palm + wave - toggle stop/go
        elif "open_palm+wave" in gesture:
            current_time = time.time()
            if current_time - self.last_gesture_time > self.gesture_cooldown:
                self.last_gesture_time = current_time
                self.is_gripping = not self.is_gripping
                return RobotCommand(
                    action="grip",
                    params={"grip_state": self.is_gripping}
                )
            
        return None
        
    def _normalize_coordinate(self, value: float, min_val: float, max_val: float) -> float:
        """normalize a coordinate to the range [0, 1]"""
        return (value - min_val) / (max_val - min_val)
        
    def _calculate_z_from_circle(self) -> float:
        """calculate z coordinate based on circular motion"""
        # implement based on your specific needs
        return 0.5  # placeholder 