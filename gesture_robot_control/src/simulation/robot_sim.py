import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from roboticstoolbox.backends.PyPlot import PyPlot

class RobotSimulator:
    def __init__(self):
        # create panda robot
        self.robot = rtb.models.Panda()
        
        # set initial joint angles to ready position
        self.robot.q = self.robot.qr
        
        # create PyPlot environment
        self.env = PyPlot()
        self.env.launch()
        self.env.add(self.robot)
        self.env.step()
        
    def execute_command(self, command):
        """
        execute a robot command based on gesture input
        
        command format:
        RobotCommand(
            action="move_xyz" | "grip",
            params={
                "position": (x, y, z)  # for move_xyz
                "grip_state": bool     # for grip
            }
        )
        """
        if command.action == "move_xyz":
            # create desired end-effector pose
            Tep = sm.SE3(command.params["position"])
            
            # solve inverse kinematics
            sol = self.robot.ik_LM(Tep)
            
            # Handle the returned solution properly
            # Get the full solution object first
            solution = sol
            
            # Check if it's a tuple and has at least one element (the joint angles)
            if isinstance(solution, tuple) and len(solution) > 0:
                q = solution[0]  # First element is joint angles
                
                # Check if solution contains a success flag
                success = solution[1] if len(solution) > 1 else False
                
                if success:
                    # move robot to new position
                    self.robot.q = q
                    self.env.step()
                else:
                    print("IK solution not found")
            else:
                print("Invalid IK solution format")
                
        elif command.action == "grip":
            # TODO: implement gripper control
            # for now just print the command
            print(f"gripper command: {command.params['grip_state']}")
            
    def close(self):
        """clean up resources"""
        self.env.close() 