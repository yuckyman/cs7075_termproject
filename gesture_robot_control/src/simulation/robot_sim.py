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
        
        # track current position
        self.current_position = [0.5, 0.5, 0.5]  # x, y, z
        
    def execute_command(self, commands):
        """
        execute robot commands from both hands
        
        commands format:
        List[RobotCommand] where each command can be:
        RobotCommand(
            action="move_xy" | "move_z" | "grip",
            params={
                "position": (x, y)  # for move_xy
                "position": z       # for move_z
                "grip_state": bool  # for grip
            }
        )
        """
        # process commands from both hands
        for command in commands:
            if command is None:
                continue
                
            if command.action == "move_xy":
                # update x,y position
                x, y = command.params["position"]
                self.current_position[0] = x
                self.current_position[1] = y
                
            elif command.action == "move_z":
                # update z position
                self.current_position[2] = command.params["position"]
                
            elif command.action == "grip":
                # TODO: implement gripper control
                print(f"gripper command: {command.params['grip_state']}")
                
        # create desired end-effector pose with updated position
        Tep = sm.SE3(self.current_position)
        
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
            
    def close(self):
        """clean up resources"""
        self.env.close() 