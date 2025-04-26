import time
import numpy as np

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        """
        initialize pid controller with gains.
        
        args:
            kp (float): proportional gain
            ki (float): integral gain
            kd (float): derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def reset(self):
        """reset internal state of the controller."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def update(self, setpoint, current_value):
        """
        compute pid control value based on current error.
        
        args:
            setpoint (float): desired value
            current_value (float): current measured value
            
        returns:
            float: control output
        """
        # get current time and compute dt
        current_time = time.time()
        dt = current_time - self.last_time
        
        # compute error
        error = setpoint - current_value
        
        # update integral term
        self.integral += error * dt
        
        # compute derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # compute pid output
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # update state
        self.prev_error = error
        self.last_time = current_time
        
        return output

    def tune(self, kp=None, ki=None, kd=None):
        """
        update pid gains.
        
        args:
            kp (float, optional): new proportional gain
            ki (float, optional): new integral gain
            kd (float, optional): new derivative gain
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd 