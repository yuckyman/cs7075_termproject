import numpy as np
import time
from typing import Tuple, Optional, List
from gestures.hand_tracker import HandTracker
from dataclasses import dataclass

@dataclass
class RobotCommand:
    """represents a command to be sent to the robot"""
    action: str  # move_xyz, move_xy, rotate, grip, etc.
    params: dict  # parameters for the action (coordinates, angles, etc.)

class GestureControl:
    def __init__(self, control_mode="arm"):
        self.hand_tracker = HandTracker(max_hands=2)  # enable two hand tracking
        self.last_position: Optional[Tuple[float, float, float]] = None
        self.is_gripping = False
        self.control_mode = control_mode  # can be "arm" or "wheeled"
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # seconds between grip toggles
        
    def process_frame(self, frame) -> List[Optional[RobotCommand]]:
        """process a frame and return robot commands for each detected hand"""
        frame, hands = self.hand_tracker.find_hands(frame)
        
        if not hands:
            return [None, None]
            
        commands = []
        for landmarks in hands:
            gesture = self.hand_tracker.get_gesture(landmarks)
            if not gesture:
                commands.append(None)
                continue
                
            # determine hand side based on pixel x coordinate relative to frame width
            h, w, _ = frame.shape
            hand_side = "left" if landmarks[0][0] < w / 2 else "right"
            
            if hand_side == "right" and gesture.startswith("velocity_"):
                # right hand controls velocity
                velocity = float(gesture.split("_")[1])
                commands.append(RobotCommand(
                    action="velocity",
                    params={"speed": velocity}
                ))
            elif hand_side == "left" and gesture.startswith("heading_"):
                # left hand controls steering based on pointer angle from gesture
                raw_angle = float(gesture.split("_")[1])
                # normalize to [-1, 1] range by dividing by pi
                normalized_angle = np.clip(-raw_angle / np.pi, -1, 1)  # negate to fix reversed rotation
                commands.append(RobotCommand(
                    action="rotate",
                    params={"angle": normalized_angle}
                ))
            elif hand_side == "left" and gesture == "fist":
                # fist means don't change angle
                commands.append(RobotCommand(
                    action="rotate",
                    params={"angle": 0.0}
                ))
            else:
                commands.append(None)
            
        return commands
        
    def _gesture_to_command(self, gesture: str, landmarks, hand_side: str) -> Optional[RobotCommand]:
        """convert a gesture to a robot command, considering which hand made it"""
        
        if self.control_mode == "arm":
            return self._arm_gesture_to_command(gesture, landmarks, hand_side)
        else:
            return self._wheeled_gesture_to_command(gesture, landmarks, hand_side)
    
    def _arm_gesture_to_command(self, gesture: str, landmarks, hand_side: str) -> Optional[RobotCommand]:
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
            
            # left hand controls x,y, right hand controls z
            if hand_side == "left":
                return RobotCommand(
                    action="move_xy",
                    params={"position": (x, y)}
                )
            else:
                return RobotCommand(
                    action="move_z",
                    params={"position": z}
                )
            
        # open palm + wave could control gripping
        elif "open_palm+wave" in gesture:
            self.is_gripping = not self.is_gripping
            return RobotCommand(
                action="grip",
                params={"grip_state": self.is_gripping}
            )
            
        return None
    
    def _wheeled_gesture_to_command(self, gesture: str, landmarks, hand_side: str) -> Optional[RobotCommand]:
        """convert a gesture to a wheeled robot command"""
        
        # pointing gestures control x,y position
        if "point" in gesture:
            # use index finger tip (landmark 8) for position control
            index_tip = landmarks[8]
            # normalize coordinates to robot workspace
            x = self._normalize_coordinate(index_tip[0], 0, 640)  # adjust based on your camera
            y = self._normalize_coordinate(index_tip[1], 0, 480)
            
            # left hand controls position, right hand controls rotation
            if hand_side == "left":
                return RobotCommand(
                    action="move_xy",
                    params={"position": (x, y)}
                )
            else:
                # use x position to determine rotation direction and speed
                x_pos = index_tip[0]
                rotation = (x_pos / 640) * 2 - 1
                return RobotCommand(
                    action="rotate",
                    params={"angle": rotation}
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