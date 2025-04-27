import numpy as np
import time
from typing import Tuple, Optional, List
from gestures.hand_tracker import HandTracker
from dataclasses import dataclass

@dataclass
class RobotCommand:
    """represents a command to be sent to the robot"""
    action: str  # velocity, rotate, grip
    params: dict  # parameters for the action

class GestureControl:
    def __init__(self):
        self.hand_tracker = HandTracker(max_hands=2)  # enable two hand tracking
        self.last_stop_toggle_time = 0
        self.stop_toggle_cooldown = 1.0  # seconds between stop/go toggles
        self.last_dual_pinch_time = 0
        self.dual_pinch_cooldown = 3.0  # seconds between dual-pinch toggles
        self.pinch_hands = set()  # track which hands made pinch gesture
        
    def process_frame(self, frame) -> List[Optional[RobotCommand]]:
        """process a frame and return robot commands for each detected hand"""
        frame, hands = self.hand_tracker.find_hands(frame)
        
        if not hands:
            return [None, None]
            
        # For dual-pinch detection
        current_pinch_hands = set()
        detected_hands = []
        commands = []
        
        # First pass - gather hand info and detect individual gestures
        for landmarks in hands:
            gesture = self.hand_tracker.get_gesture(landmarks)
            if not gesture:
                commands.append(None)
                continue
                
            # determine hand side based on pixel x coordinate relative to frame width
            h, w, _ = frame.shape
            hand_side = "left" if landmarks[0][0] < w / 2 else "right"
            detected_hands.append(hand_side)
            print(f"Detected {gesture} gesture with {hand_side} hand")  # Debug print
            
            # Track pinch gestures
            if gesture == "pinch":
                current_pinch_hands.add(hand_side)
                
            if hand_side == "right" and gesture.startswith("velocity_"):
                # right hand controls speed
                velocity = float(gesture.split("_")[1])
                commands.append(RobotCommand(
                    action="velocity",
                    params={"speed": velocity}
                ))
            elif hand_side == "left" and gesture.startswith("heading_"):
                # left hand controls steering
                raw_angle = float(gesture.split("_")[1])
                normalized_angle = np.clip(-raw_angle / np.pi, -1, 1)
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
            elif hand_side == "left" and gesture == "thumbs_down":
                # thumbs down puts robot in reverse
                commands.append(RobotCommand(
                    action="velocity",
                    params={"speed": -0.5}  # negative speed for reverse
                ))
            elif hand_side == "left" and gesture == "open_palm":
                # open palm means stop reversing
                commands.append(RobotCommand(
                    action="velocity",
                    params={"speed": 0.0}
                ))
            elif gesture == "open_palm" and hand_side in self.pinch_hands:
                # Remove hand from pinch tracking
                self.pinch_hands.discard(hand_side)
            else:
                commands.append(None)
        
        # Check for dual-pinch activation (both hands pinched)
        current_time = time.time()
        if ("left" in current_pinch_hands and "right" in current_pinch_hands and 
            len(current_pinch_hands) == 2 and 
            current_time - self.last_dual_pinch_time > self.dual_pinch_cooldown):
            
            if len(self.pinch_hands) < 2:  # If we weren't already in dual-pinch mode
                print("Both hands pinched - toggling manual override")
                self.last_dual_pinch_time = current_time
                commands.append(RobotCommand(
                    action="stop",
                    params={"stop": True, "dual_pinch": True, "toggle_override": True}
                ))
            self.pinch_hands = current_pinch_hands.copy()
        # Check if we were in dual-pinch mode but now one or both hands are released
        elif len(self.pinch_hands) == 2 and (len(current_pinch_hands) < 2 or len(detected_hands) < 2):
            print("Dual-pinch released")
            # don't send any command, just clear the pinch tracking
            self.pinch_hands.clear()
        # Track individual pinches for future reference
        else:
            self.pinch_hands = current_pinch_hands.copy()
            
        return commands 