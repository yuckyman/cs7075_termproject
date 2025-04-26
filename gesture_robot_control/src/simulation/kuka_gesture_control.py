#!/usr/bin/env python3
"""
kuka_gesture_control.py

Connects the MediaPipe-based gesture recognizer to the PyBullet simulated Kuka arm.
Maps point gestures to upward movements, swipe gestures to lateral movements,
and circle gestures to forward movements.
"""

import sys
import os
import time
import cv2
import pybullet as p

# adjust paths for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from config.settings import GESTURE_CONFIG, SIM_CONFIG
from multimodal.gesture_module import GestureRecognizer
from simulation.sim_env import RobotSimulation

def main():
    # initialize gesture recognizer
    gesture_recognizer = GestureRecognizer(GESTURE_CONFIG)

    # initialize simulation
    sim = RobotSimulation(gui=True)

    # initialize camera
    cap = cv2.VideoCapture(GESTURE_CONFIG['camera_id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, GESTURE_CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GESTURE_CONFIG['frame_height'])

    # get initial end effector position
    target_pos = list(sim.get_end_effector_pos())

    print("Starting control loop. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # detect gesture
            gesture = gesture_recognizer.process_frame(frame)
            print(f"Detected gesture: {gesture}")

            # map gestures to target position offsets
            if gesture == "<gesture_point>":
                target_pos[2] += 0.05  # raise
            elif gesture == "<gesture_swipe>":
                target_pos[0] += 0.05  # move right
            elif gesture == "<gesture_circle>":
                target_pos[1] += 0.05  # move forward

            # clamp within workspace [-0.5, 0.5]
            target_pos = [min(max(val, -0.5), 0.5) for val in target_pos]

            # compute inverse kinematics for desired target position
            quaternion = p.getQuaternionFromEuler([0, 0, 0])
            joint_angles = p.calculateInverseKinematics(
                sim.robot_id,
                sim.num_joints - 1,
                target_pos,
                targetOrientation=quaternion
            )

            # apply joint commands
            for idx, angle in enumerate(joint_angles):
                sim.set_joint_target(idx, angle, SIM_CONFIG['max_force'])

            # step simulation
            sim.step()
            time.sleep(SIM_CONFIG['timestep'])

            # display camera frame
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        gesture_recognizer.close()
        sim.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 