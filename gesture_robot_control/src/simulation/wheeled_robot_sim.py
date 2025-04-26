import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import time
from matplotlib.animation import FuncAnimation

class WheeledRobotSimulator:
    def __init__(self):
        # Set up the simulation environment
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Wheeled Robot Gesture Control')
        self.ax.set_xlabel('X position')
        self.ax.set_ylabel('Y position')
        
        # Robot state
        self.robot_pos = np.array([0.0, 0.0])  # x, y position
        self.robot_heading = 0.0  # heading in radians
        self.robot_speed = 0.0  # current speed
        self.robot_angular_vel = 0.0  # current angular velocity
        self.robot_radius = 0.3  # robot radius for visualization
        self.trail_points = []  # store path history
        self.max_trail_length = 100  # maximum number of trail points
        
        # Draw the robot
        self.robot_body = Circle(
            (self.robot_pos[0], self.robot_pos[1]), 
            self.robot_radius, 
            color='blue', 
            alpha=0.7
        )
        
        # Add direction indicator (shows heading)
        self.direction_indicator, = self.ax.plot(
            [self.robot_pos[0], self.robot_pos[0] + self.robot_radius * np.cos(self.robot_heading)],
            [self.robot_pos[1], self.robot_pos[1] + self.robot_radius * np.sin(self.robot_heading)],
            color='red', 
            linewidth=2
        )
        
        # Add trail visualization
        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3)
        
        # Add objects to the environment
        self.add_environment_objects()
        
        # Add robot to plot
        self.ax.add_patch(self.robot_body)
        
        # Animation
        self.animation = FuncAnimation(
            self.fig, 
            self.update_animation, 
            interval=50,  # update every 50 ms 
            blit=False
        )
        
        # Show the plot without blocking
        plt.ion()
        plt.show()
        
    def add_environment_objects(self):
        """Add obstacles and targets to the environment"""
        # Add some obstacles (represented as rectangles)
        obstacles = [
            (-4, -3, 1, 1),  # x, y, width, height
            (2, 2, 1.5, 0.5),
            (-2, 1, 0.5, 2),
            (3, -2, 1, 1)
        ]
        
        for obs in obstacles:
            x, y, w, h = obs
            rect = Rectangle((x, y), w, h, color='gray', alpha=0.7)
            self.ax.add_patch(rect)
            
        # Add a target zone
        target = Circle((4, 4), 0.5, color='green', alpha=0.3)
        self.ax.add_patch(target)
        
    def execute_command(self, command):
        """
        Execute a robot command based on gesture input
        
        command format:
        RobotCommand(
            action="move_xy" | "rotate" | "grip",
            params={
                "position": (x, y)  # for move_xy
                "angle": float      # for rotate
                "grip_state": bool  # for grip (can be used for auxiliary actions)
            }
        )
        """
        if command.action == "move_xy":
            # Get target position
            target_x, target_y = command.params["position"]
            
            # Scale inputs to be in range [-5, 5]
            # Assuming input is in range [0, 1]
            target_x = (target_x * 10) - 5
            target_y = (target_y * 10) - 5
            
            # Calculate vector to target
            target_pos = np.array([target_x, target_y])
            dir_vector = target_pos - self.robot_pos
            
            # Calculate desired heading
            if np.linalg.norm(dir_vector) > 0.1:  # Only update if significant movement
                self.robot_speed = min(np.linalg.norm(dir_vector) * 0.1, 0.2)  # Scale speed
                desired_heading = np.arctan2(dir_vector[1], dir_vector[0])
                
                # Smoothly adjust heading
                heading_diff = self.normalize_angle(desired_heading - self.robot_heading)
                self.robot_angular_vel = heading_diff * 0.3  # Scale turning rate
                
                print(f"Moving towards: ({target_x:.2f}, {target_y:.2f}), " 
                      f"Speed: {self.robot_speed:.2f}, Angular Vel: {self.robot_angular_vel:.2f}")
            else:
                # If close to target, slow down
                self.robot_speed *= 0.9
                self.robot_angular_vel *= 0.9
                
        elif command.action == "rotate":
            # Direct control of rotation
            self.robot_angular_vel = command.params["angle"] * 2  # Scale angular velocity
            self.robot_speed = 0.05  # Maintain a small forward speed
            
        elif command.action == "grip":
            # For wheeled robot, we'll use this as a "stop/go" command
            stop = command.params["grip_state"]
            if stop:
                self.robot_speed = 0
                self.robot_angular_vel = 0
                print("Robot stopped")
            else:
                self.robot_speed = 0.1
                print("Robot moving")
    
    def update_animation(self, frame):
        """Update the robot position and orientation based on current state"""
        # Update robot position and heading
        self.robot_heading += self.robot_angular_vel
        self.robot_heading = self.normalize_angle(self.robot_heading)
        
        # Calculate new position based on heading and speed
        new_x = self.robot_pos[0] + self.robot_speed * np.cos(self.robot_heading)
        new_y = self.robot_pos[1] + self.robot_speed * np.sin(self.robot_heading)
        
        # Simple boundary checking
        new_x = np.clip(new_x, -5, 5)
        new_y = np.clip(new_y, -5, 5)
        
        # Update position
        self.robot_pos = np.array([new_x, new_y])
        
        # Add current position to trail
        self.trail_points.append((new_x, new_y))
        
        # Limit trail length
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points = self.trail_points[-self.max_trail_length:]
        
        # Update visualizations
        self.robot_body.center = (new_x, new_y)
        
        # Update direction indicator
        self.direction_indicator.set_data(
            [new_x, new_x + self.robot_radius * np.cos(self.robot_heading)],
            [new_y, new_y + self.robot_radius * np.sin(self.robot_heading)]
        )
        
        # Update trail
        if self.trail_points:
            xs, ys = zip(*self.trail_points)
            self.trail.set_data(xs, ys)
        
        # Simulate physics - gradually decrease speed and angular velocity
        self.robot_speed *= 0.98  # Gradual slowdown due to friction
        self.robot_angular_vel *= 0.95  # Gradual angular slowdown
        
        # Return artists that were updated
        return self.robot_body, self.direction_indicator, self.trail
    
    def normalize_angle(self, angle):
        """Normalize angle to be in range [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
        
    def close(self):
        """Clean up resources"""
        plt.ioff()
        plt.close(self.fig) 