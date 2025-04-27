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
        
        # Collision detection
        self.last_collision_time = 0
        self.collision_cooldown = 1.0  # seconds before allowing another collision
        self.is_resetting = False
        self.reset_start_time = 0
        
        # Celebration state
        self.is_celebrating = False
        self.celebration_start_time = 0
        self.celebration_duration = 2.0  # seconds
        self.celebration_radius = 0.5  # radius of celebration circle
        self.celebration_circle = None
        
        # Target state
        self.target_center = np.array([4, 4])  # initial target position
        self.target_radius = 0.5
        
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
        self.obstacles = [
            (-4, -3, 1, 1),  # x, y, width, height
            (2, 2, 1.5, 0.5),
            (-2, 1, 0.5, 2),
            (3, -2, 1, 1)
        ]
        
        for obs in self.obstacles:
            x, y, w, h = obs
            rect = Rectangle((x, y), w, h, color='gray', alpha=0.7)
            self.ax.add_patch(rect)
            
        # Add a target zone
        self.target = Circle(self.target_center, self.target_radius, color='green', alpha=0.3)
        self.ax.add_patch(self.target)
        
    def _sigmoid_velocity(self, raw_velocity: float) -> float:
        """apply sigmoid function to smooth velocity and ensure it goes to 0"""
        # shift and scale sigmoid to make it more sensitive around 0.5
        x = (raw_velocity - 0.5) * 10  # scale input to make transition sharper
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * 0.2  # scale output to max speed

    def _check_collision(self) -> bool:
        """check if robot has collided with any obstacles or boundaries"""
        # check boundary collisions
        if (abs(self.robot_pos[0]) + self.robot_radius > 5 or 
            abs(self.robot_pos[1]) + self.robot_radius > 5):
            return True
        
        # check obstacle collisions
        for obs in self.obstacles:
            # get the closest point on the obstacle to the robot
            closest_x = np.clip(self.robot_pos[0], obs[0], obs[0] + obs[2])
            closest_y = np.clip(self.robot_pos[1], obs[1], obs[1] + obs[3])
            
            # calculate distance from robot to closest point
            distance = np.sqrt((self.robot_pos[0] - closest_x)**2 + 
                              (self.robot_pos[1] - closest_y)**2)
            
            if distance < self.robot_radius:
                return True
            
        return False

    def execute_command(self, commands):
        """
        Execute robot commands from both hands
        
        commands format:
        List[RobotCommand] where each command can be:
        RobotCommand(
            action="velocity" | "rotate" | "grip",
            params={
                "speed": float      # for velocity (0-1)
                "angle": float      # for rotate
                "grip_state": bool  # for grip (can be used for auxiliary actions)
            }
        )
        """
        current_time = time.time()
        
        # check for collision and handle reset
        if self._check_collision() and not self.is_resetting:
            if current_time - self.last_collision_time > self.collision_cooldown:
                self.is_resetting = True
                self.reset_start_time = current_time
                self.robot_speed = 0
                self.robot_angular_vel = 0
                print("Collision detected! Resetting robot...")
        
        # process commands from both hands
        for command in commands:
            if command is None:
                continue
            
            if command.action == "velocity":
                # Set speed directly from right hand with sigmoid smoothing
                raw_velocity = command.params["speed"]
                self.robot_speed = self._sigmoid_velocity(raw_velocity)
                print(f"Setting speed to: {self.robot_speed:.2f}")
                
            elif command.action == "rotate":
                # Direct control of rotation from left hand
                # map angle from [-1, 1] to [-pi/4, pi/4] for smoother rotation
                angle = command.params["angle"]
                self.robot_angular_vel = angle * (np.pi / 4)  # max rotation speed of pi/4 rad/s
                print(f"Rotating with angular velocity: {self.robot_angular_vel:.2f}")
                
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
        current_time = time.time()
        
        # check if robot reached target
        distance_to_target = np.linalg.norm(self.robot_pos - self.target_center)
        if distance_to_target < self.target_radius and not self.is_celebrating:
            self.is_celebrating = True
            self.celebration_start_time = current_time
            self.celebration_circle = Circle(
                (self.robot_pos[0], self.robot_pos[1]),
                self.celebration_radius,
                color='yellow',
                alpha=0.5
            )
            self.ax.add_patch(self.celebration_circle)
            print("Target reached! Celebrating!")
            
            # Reset robot to origin and generate new target
            self.robot_pos = np.array([0.0, 0.0])
            self.robot_heading = 0.0
            self.robot_speed = 0.0
            self.robot_angular_vel = 0.0
            self.trail_points = []
            
            # Generate new target position
            self.target_center = self._generate_random_target()
            self.target.center = self.target_center
            print(f"New target at: {self.target_center}")
        
        # handle celebration animation
        if self.is_celebrating:
            elapsed = current_time - self.celebration_start_time
            if elapsed > self.celebration_duration:
                self.is_celebrating = False
                if self.celebration_circle:
                    self.celebration_circle.remove()
                    self.celebration_circle = None
            else:
                # pulse the celebration circle
                pulse_factor = 1.0 + 0.2 * np.sin(elapsed * 10)  # oscillate size
                self.celebration_circle.set_radius(self.celebration_radius * pulse_factor)
                self.celebration_circle.set_alpha(0.5 + 0.3 * np.sin(elapsed * 5))  # pulse opacity
        
        # handle reset if needed
        if self.is_resetting:
            if current_time - self.reset_start_time > 1.0:  # 1 second reset time
                self.robot_pos = np.array([0.0, 0.0])
                self.robot_heading = 0.0
                self.robot_speed = 0.0
                self.robot_angular_vel = 0.0
                self.is_resetting = False
                self.last_collision_time = current_time
                self.trail_points = []  # clear the trail
                print("Robot reset complete!")
            return self.robot_body, self.direction_indicator, self.trail
        
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

    def _generate_random_target(self):
        """Generate a new random target position that doesn't overlap with obstacles or boundaries"""
        while True:
            # Generate random position within bounds
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            
            # Check if it overlaps with any obstacles
            overlaps = False
            for obs in self.obstacles:
                obs_x, obs_y, w, h = obs
                if (abs(x - obs_x) < w/2 + self.target_radius and 
                    abs(y - obs_y) < h/2 + self.target_radius):
                    overlaps = True
                    break
            
            if not overlaps:
                return np.array([x, y]) 