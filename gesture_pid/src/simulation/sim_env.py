import pybullet as p
import pybullet_data
import time
import numpy as np
from pathlib import Path

class RobotSimulation:
    def __init__(self, gui=True):
        """
        initialize robot simulation environment
        
        args:
            gui (bool): whether to use gui or direct mode
        """
        # connect to physics server
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # set up simulation parameters
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # we'll step manually
        
        # load ground plane
        p.loadURDF("plane.urdf")
        
        # load simple robot arm (we'll use kuka for now, can swap later)
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # get joint info
        self.joint_indices = range(self.num_joints)
        self.joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        
    def reset(self):
        """reset simulation to initial state"""
        p.resetSimulation()
        self.__init__(gui=(self.physics_client == p.GUI))
        
    def step(self):
        """step simulation forward"""
        p.stepSimulation()
        
    def get_joint_states(self):
        """get current joint positions and velocities"""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        return positions, velocities
    
    def set_joint_target(self, joint_idx, target_pos, max_force=100):
        """
        set target position for a specific joint
        
        args:
            joint_idx (int): index of joint to control
            target_pos (float): target position in radians
            max_force (float): maximum force to apply
        """
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=max_force
        )
    
    def get_end_effector_pos(self):
        """get current position of end effector"""
        state = p.getLinkState(self.robot_id, self.num_joints - 1)
        return state[0]  # (x, y, z) position
    
    def close(self):
        """disconnect from physics server"""
        p.disconnect() 