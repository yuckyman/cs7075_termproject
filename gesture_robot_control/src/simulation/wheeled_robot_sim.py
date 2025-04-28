import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import time
from matplotlib.animation import FuncAnimation
import cv2
from metrics.gesture_latency_logger import GestureLatencyLogger
from metrics.system_metrics_logger import SystemMetricsLogger

class WheeledRobotSimulator:
    def __init__(self, roomba_mode=False):
        # set up the simulation environment
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Wheeled Robot Gesture Control')
        self.ax.set_xlabel('X position')
        self.ax.set_ylabel('Y position')
        
        # differential drive parameters
        self.wheelbase = 0.2  # distance between wheels in meters
        self.wheel_radius = 0.05  # radius of wheels in meters
        self.max_wheel_speed = 0.5  # maximum speed of each wheel in m/s
        
        # wheel speeds (left and right)
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        
        # robot state
        self.robot_pos = np.array([0.0, 0.0])  # x, y position
        self.robot_heading = 0.0  # heading in radians
        self.robot_radius = 0.3  # robot radius for visualization
        self.trail_points = []  # store path history
        self.max_trail_length = 100  # maximum number of trail points
        
        # collision detection
        self.last_collision_time = 0
        self.collision_cooldown = 1.0  # seconds before allowing another collision
        self.is_resetting = False
        self.reset_start_time = 0
        
        # celebration state
        self.is_celebrating = False
        self.celebration_start_time = 0
        self.celebration_duration = 2.0  # seconds
        self.celebration_radius = 0.5  # radius of celebration circle
        self.celebration_circle = None
        
        # target state
        self.target_center = np.array([4, 4])  # initial target position
        self.target_radius = 0.5
        
        # draw the robot
        self.robot_body = Circle(
            (self.robot_pos[0], self.robot_pos[1]), 
            self.robot_radius, 
            color='blue', 
            alpha=0.7
        )
        
        # add direction indicator (shows heading)
        self.direction_indicator, = self.ax.plot(
            [self.robot_pos[0], self.robot_pos[0] + self.robot_radius * np.cos(self.robot_heading)],
            [self.robot_pos[1], self.robot_pos[1] + self.robot_radius * np.sin(self.robot_heading)],
            color='red', 
            linewidth=2
        )
        
        # add trail visualization
        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3)
        
        # add objects to the environment
        self.add_environment_objects()
        
        # add robot to plot
        self.ax.add_patch(self.robot_body)
        
        # animation
        self.animation = FuncAnimation(
            self.fig, 
            self.update_animation, 
            interval=50,  # update every 50 ms 
            blit=False,
            save_count=1000  # limit cache to 1000 frames
        )
        
        # show the plot without blocking
        plt.ion()
        plt.show()
        
        # roomba mode settings
        self.roomba_mode = roomba_mode
        self.manual_override = False
        self.last_pinch_time = 0
        self.pinch_cooldown = 2.0  # reduced from 5.0 to make toggling more responsive 
        self.collision_response_state = None  # None, 'reversing', 'turning'
        self.collision_start_time = 0
        self.collision_duration = 1.0  # seconds to complete collision response
        self.turn_angle = np.pi / 6  # 30 degrees in radians
        self.reverse_distance = 0.1  # distance to reverse after collision
        self.roomba_speed = 0.12  # slower roomba speed
        
        # create visualization window
        self.window_size = (400, 400)
        self.viz_window = np.zeros((self.window_size[0], self.window_size[1], 3), dtype=np.uint8)
        cv2.namedWindow('robot simulator')
        
        # initialize metrics loggers
        self.metrics_logger = GestureLatencyLogger()
        self.system_metrics_logger = SystemMetricsLogger()
        
        print("robot simulator initialized")
        print(f"starting position: {self.robot_pos}")
        if roomba_mode:
            print("roomba mode enabled")
            # set initial wheel speeds for roomba mode
            self.left_wheel_speed = self.roomba_speed
            self.right_wheel_speed = self.roomba_speed

    def add_environment_objects(self):
        """Add obstacles and targets to the environment"""
        # add some obstacles (represented as rectangles)
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
            
        # add a target zone
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
        return self._check_collision_at_position(self.robot_pos[0], self.robot_pos[1])

    def _check_collision_at_position(self, x: float, y: float) -> bool:
        """check if a specific position would cause a collision with obstacles or boundaries"""
        # check boundary collisions
        if (abs(x) + self.robot_radius > 5 or 
            abs(y) + self.robot_radius > 5):
            return True
        
        # check obstacle collisions
        for obs in self.obstacles:
            # get the closest point on the obstacle to the robot
            closest_x = np.clip(x, obs[0], obs[0] + obs[2])
            closest_y = np.clip(y, obs[1], obs[1] + obs[3])
            
            # calculate distance from robot to closest point
            distance = np.sqrt((x - closest_x)**2 + 
                              (y - closest_y)**2)
            
            if distance < self.robot_radius:
                return True
            
        return False

    def _compute_differential_velocities(self):
        """Convert wheel speeds to robot velocities using differential drive kinematics"""
        # linear velocity = average of wheel speeds
        linear_vel = (self.left_wheel_speed + self.right_wheel_speed) / 2
        
        # angular velocity = difference in wheel speeds / wheelbase
        angular_vel = (self.right_wheel_speed - self.left_wheel_speed) / self.wheelbase
        
        return linear_vel, angular_vel

    def _compute_wheel_speeds(self, linear_vel, angular_vel):
        """Convert desired robot velocities to wheel speeds"""
        # left wheel speed = linear velocity - (angular velocity * wheelbase/2)
        left_speed = linear_vel - (angular_vel * self.wheelbase / 2)
        
        # right wheel speed = linear velocity + (angular velocity * wheelbase/2)
        right_speed = linear_vel + (angular_vel * self.wheelbase / 2)
        
        # clamp to max wheel speed
        left_speed = np.clip(left_speed, -self.max_wheel_speed, self.max_wheel_speed)
        right_speed = np.clip(right_speed, -self.max_wheel_speed, self.max_wheel_speed)
        
        return left_speed, right_speed

    def execute_command(self, commands):
        """
        Execute robot commands from both hands
        """
        # in roomba mode, check for dual-pinch manual override
        if self.roomba_mode:
            for command in commands:
                if command and command.action == "stop" and command.params.get("dual_pinch", False):
                    if command.params.get("toggle_override", False):
                        # log gesture recognition
                        gesture_data = self.metrics_logger.log_gesture(
                            "dual_pinch_toggle",
                            command.params
                        )
                        
                        self.manual_override = not self.manual_override
                        if self.manual_override:
                            print("Manual override enabled - stopping roomba behavior")
                            self.left_wheel_speed = 0
                            self.right_wheel_speed = 0
                        else:
                            print("Returning to roomba mode")
                            self.left_wheel_speed = self.roomba_speed
                            self.right_wheel_speed = self.roomba_speed
                        
                        # log control execution
                        latency = self.metrics_logger.log_control(gesture_data)
                        print(f"Gesture to control latency: {latency:.2f}ms")
            
            # if not in manual override, perform roomba behavior and return
            if not self.manual_override:
                self._update_roomba_mode()
                self._update_visualization()
                return
            # in manual override, continue to process other commands

        # process all commands in gesture mode or when manual override is enabled
        for command in commands:
            if command is None:
                continue
            
            # log gesture recognition
            gesture_data = self.metrics_logger.log_gesture(
                command.action,
                command.params
            )
            
            if command.action == "velocity":
                # right hand controls linear velocity
                raw_velocity = command.params["speed"]
                linear_vel = self._sigmoid_velocity(raw_velocity)
                
                # maintain current angular velocity
                _, current_angular_vel = self._compute_differential_velocities()
                
                # compute new wheel speeds
                self.left_wheel_speed, self.right_wheel_speed = self._compute_wheel_speeds(
                    linear_vel, current_angular_vel
                )
                print(f"Setting wheel speeds: left={self.left_wheel_speed:.2f}, right={self.right_wheel_speed:.2f}")
                
            elif command.action == "rotate":
                # left hand controls angular velocity
                angle = command.params["angle"]
                angular_vel = angle * (np.pi / 4)
                
                # maintain current linear velocity
                current_linear_vel, _ = self._compute_differential_velocities()
                
                # compute new wheel speeds
                self.left_wheel_speed, self.right_wheel_speed = self._compute_wheel_speeds(
                    current_linear_vel, angular_vel
                )
                print(f"Setting wheel speeds: left={self.left_wheel_speed:.2f}, right={self.right_wheel_speed:.2f}")
                
            elif command.action == "stop":
                # stop both wheels
                self.left_wheel_speed = 0
                self.right_wheel_speed = 0
                print("Robot stopped")
            
            # log control execution
            latency = self.metrics_logger.log_control(gesture_data)
            print(f"Gesture to control latency: {latency:.2f}ms")
        
        # update visualization
        self._update_visualization()

    def normalize_angle(self, angle):
        """Normalize angle to be in range [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
        
    def close(self):
        """Clean up resources"""
        plt.ioff()
        plt.close(self.fig)
        cv2.destroyAllWindows()
        self.system_metrics_logger.close()

    def _generate_random_target(self):
        """Generate a new random target position that doesn't overlap with obstacles or boundaries"""
        while True:
            # generate random position within bounds
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            
            # check if it overlaps with any obstacles
            overlaps = False
            for obs in self.obstacles:
                obs_x, obs_y, w, h = obs
                if (abs(x - obs_x) < w/2 + self.target_radius and 
                    abs(y - obs_y) < h/2 + self.target_radius):
                    overlaps = True
                    break
            
            if not overlaps:
                return np.array([x, y])

    def _update_roomba_mode(self):
        """update robot behavior in roomba mode"""
        if not self.roomba_mode or self.manual_override:
            return
            
        current_time = time.time()
        
        # handle collision response states
        if self.collision_response_state:
            elapsed = current_time - self.collision_start_time
            
            if self.collision_response_state == 'reversing':
                # reverse for half the collision duration
                if elapsed < self.collision_duration / 2:
                    self.left_wheel_speed = -self.roomba_speed
                    self.right_wheel_speed = -self.roomba_speed
                else:
                    # start turning
                    self.collision_response_state = 'turning'
                    # randomly choose turn direction
                    self.turn_direction = 1 if np.random.random() > 0.5 else -1
                    self.left_wheel_speed = -self.turn_direction * self.roomba_speed
                    self.right_wheel_speed = self.turn_direction * self.roomba_speed
                    
            elif self.collision_response_state == 'turning':
                # complete the turn
                if elapsed < self.collision_duration:
                    self.left_wheel_speed = -self.turn_direction * self.roomba_speed
                    self.right_wheel_speed = self.turn_direction * self.roomba_speed
                else:
                    # return to normal operation
                    self.collision_response_state = None
                    self.left_wheel_speed = self.roomba_speed
                    self.right_wheel_speed = self.roomba_speed
                    
        else:
            # normal operation - move forward
            self.left_wheel_speed = self.roomba_speed
            self.right_wheel_speed = self.roomba_speed
            
            # check for collision
            if self._check_collision():
                self.collision_response_state = 'reversing'
                self.collision_start_time = current_time
                print("Collision detected! Reversing and turning...")

    def update_animation(self, frame):
        """Update the robot position and orientation based on current state"""
        current_time = time.time()
        
        # log system metrics
        system_metrics = self.system_metrics_logger.log_metrics()
        
        # autonomously update roomba behavior if in roomba mode
        if self.roomba_mode and not self.manual_override:
            self._update_roomba_mode()
        
        # check for collisions
        if self._check_collision():
            if self.manual_override:
                # in manual override, just print collision but allow movement
                print("Collision detected in manual override mode!")
            else:
                # in roomba mode, handle collision response
                self.collision_response_state = 'reversing'
                self.collision_start_time = current_time
                print("Collision detected! Reversing and turning...")
        
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
            
            # reset robot to origin and generate new target
            self.robot_pos = np.array([0.0, 0.0])
            self.robot_heading = 0.0
            self.left_wheel_speed = 0.0
            self.right_wheel_speed = 0.0
            self.trail_points = []
            
            # generate new target position
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
                self.left_wheel_speed = 0.0
                self.right_wheel_speed = 0.0
                self.is_resetting = False
                self.last_collision_time = current_time
                self.trail_points = []  # clear the trail
                print("Robot reset complete!")
            return self.robot_body, self.direction_indicator, self.trail
        
        # compute robot velocities from wheel speeds
        linear_vel, angular_vel = self._compute_differential_velocities()
        
        # update heading
        self.robot_heading += angular_vel
        self.robot_heading = self.normalize_angle(self.robot_heading)
        
        # update position using differential drive kinematics
        new_x = self.robot_pos[0] + linear_vel * np.cos(self.robot_heading)
        new_y = self.robot_pos[1] + linear_vel * np.sin(self.robot_heading)
        
        # check for collisions with new position
        if self._check_collision_at_position(new_x, new_y):
            if self.manual_override:
                # in manual override, allow rotation but prevent moving through obstacles
                # keep the current position but allow rotation
                new_x = self.robot_pos[0]
                new_y = self.robot_pos[1]
                print("Collision detected in manual override mode - preventing movement through obstacle")
            else:
                # in roomba mode, handle collision response
                self.collision_response_state = 'reversing'
                self.collision_start_time = current_time
                print("Collision detected! Reversing and turning...")
                return self.robot_body, self.direction_indicator, self.trail
        
        # simple boundary checking
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
        
        # update cv2 window for visualization
        self._update_visualization()
        
        # Return artists that were updated
        return self.robot_body, self.direction_indicator, self.trail
    
    def _update_visualization(self):
        """update the robot visualization"""
        # clear the window
        self.viz_window.fill(0)
        
        # convert normalized coordinates to pixel coordinates
        px = int(self.robot_pos[0] * self.window_size[0])
        py = int(self.robot_pos[1] * self.window_size[1])
        
        # draw robot
        cv2.circle(self.viz_window, (px, py), 10, (0, 255, 0), -1)  # green circle for robot
        
        # draw wheels
        wheel_offset = int(self.wheelbase * self.window_size[0] / 2)  # convert wheelbase to pixels
        left_wheel_pos = (
            int(px - wheel_offset * np.sin(self.robot_heading)),
            int(py + wheel_offset * np.cos(self.robot_heading))
        )
        right_wheel_pos = (
            int(px + wheel_offset * np.sin(self.robot_heading)),
            int(py - wheel_offset * np.cos(self.robot_heading))
        )
        
        # draw wheels as circles
        cv2.circle(self.viz_window, left_wheel_pos, 5, (255, 0, 0), -1)  # blue for left wheel
        cv2.circle(self.viz_window, right_wheel_pos, 5, (0, 0, 255), -1)  # red for right wheel
        
        # draw direction indicator
        direction_length = 20
        dx = int(np.cos(self.robot_heading) * direction_length)
        dy = int(np.sin(self.robot_heading) * direction_length)
        cv2.line(self.viz_window, (px, py), (px + dx, py + dy), (0, 255, 0), 2)

        # show the window
        cv2.imshow('robot simulator', self.viz_window)
        cv2.waitKey(1) 