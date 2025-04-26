from .pid_controller import PIDController
import numpy as np
from typing import List, Tuple

class RobotController:
    def __init__(self, num_joints: int, kp=1.0, ki=0.1, kd=0.05):
        """
        controller for robot arm using pid control for each joint
        
        args:
            num_joints (int): number of joints to control
            kp (float): proportional gain
            ki (float): integral gain
            kd (float): derivative gain
        """
        # create a pid controller for each joint
        self.controllers = [
            PIDController(kp=kp, ki=ki, kd=kd)
            for _ in range(num_joints)
        ]
        
        self.num_joints = num_joints
        self.target_positions = np.zeros(num_joints)
        
    def set_target_positions(self, positions: List[float]):
        """
        set target positions for all joints
        
        args:
            positions (List[float]): target position for each joint
        """
        assert len(positions) == self.num_joints
        self.target_positions = np.array(positions)
        
    def compute_control(self, current_positions: List[float], 
                       current_velocities: List[float]) -> List[float]:
        """
        compute control outputs for all joints
        
        args:
            current_positions (List[float]): current joint positions
            current_velocities (List[float]): current joint velocities
            
        returns:
            List[float]: control outputs for each joint
        """
        control_outputs = []
        
        for i in range(self.num_joints):
            # get pid output for this joint
            output = self.controllers[i].update(
                setpoint=self.target_positions[i],
                current_value=current_positions[i]
            )
            
            # could add velocity damping here if needed
            control_outputs.append(output)
            
        return control_outputs
    
    def reset(self):
        """reset all pid controllers"""
        for controller in self.controllers:
            controller.reset()
            
    def tune(self, joint_idx: int, kp=None, ki=None, kd=None):
        """
        tune pid parameters for a specific joint
        
        args:
            joint_idx (int): index of joint to tune
            kp, ki, kd (float, optional): new gain values
        """
        self.controllers[joint_idx].tune(kp=kp, ki=ki, kd=kd) 