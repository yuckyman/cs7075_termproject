import cv2
import numpy as np

class RobotSimulator:
    def __init__(self):
        # initialize robot state
        self.position = (0.5, 0.5, 0.5)  # normalized coordinates
        self.is_gripping = False
        
        # create visualization window
        self.window_size = (400, 400)  # pixels
        self.viz_window = np.zeros((self.window_size[0], self.window_size[1], 3), dtype=np.uint8)
        cv2.namedWindow('robot simulator')
        
        print("robot simulator initialized")
        print(f"starting position: {self.position}")
        
    def execute_command(self, command):
        """execute a robot command and update visualization"""
        if command.action == "move_xyz":
            self.position = command.params["position"]
            print(f"moving to position: {self.position}")
            
        elif command.action == "grip":
            self.is_gripping = command.params["grip_state"]
            state = "closed" if self.is_gripping else "open"
            print(f"gripper {state}")
            
        else:
            print(f"unknown command: {command.action}")
            
        self._update_visualization()
        
    def _update_visualization(self):
        """update the robot arm visualization"""
        # clear the window
        self.viz_window.fill(0)
        
        # convert normalized coordinates to pixel coordinates
        px = int(self.position[0] * self.window_size[0])
        py = int(self.position[1] * self.window_size[1])
        
        # draw base
        base_pos = (self.window_size[0]//2, self.window_size[1]-50)
        cv2.circle(self.viz_window, base_pos, 20, (0, 0, 255), -1)  # red circle for base
        
        # draw arm segments
        cv2.line(self.viz_window, base_pos, (px, py), (255, 255, 255), 3)  # white line for arm
        
        # draw end effector
        gripper_color = (0, 255, 0) if self.is_gripping else (0, 255, 255)  # green if gripping, yellow if not
        cv2.circle(self.viz_window, (px, py), 10, gripper_color, -1)
        
        # add z-height indicator (darker = lower, brighter = higher)
        z_brightness = int(self.position[2] * 255)
        cv2.putText(self.viz_window, f"z: {self.position[2]:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, z_brightness, 0), 2)
        
        # show the window
        cv2.imshow('robot simulator', self.viz_window)
        cv2.waitKey(1) 