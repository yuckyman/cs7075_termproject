import cv2
import time
import sys
import argparse
from pathlib import Path

# add src directory to python path
src_dir = str(Path(__file__).parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from control.gesture_control import GestureControl
from simulation.wheeled_robot_sim import WheeledRobotSimulator

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Gesture-controlled wheeled robot')
    parser.add_argument('--mode', choices=['gesture', 'roomba'], default='gesture',
                      help='Control mode: gesture (default) or roomba')
    args = parser.parse_args()
    
    # initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # initialize gesture control and robot simulator
    gesture_control = GestureControl()
    robot_sim = WheeledRobotSimulator(roomba_mode=(args.mode == 'roomba'))
    
    print("Starting wheeled robot control system...")
    print(f"Mode: {args.mode}")
    print("Controls:")
    if args.mode == 'gesture':
        print("- point: drive robot towards pointed location")
        print("- fist: rotate the robot (position determines direction)")
        print("- open palm + wave: toggle stop/go")
    else:
        print("- pinch with BOTH hands simultaneously: override roomba mode with manual control")
        print("- release one or both pinches: return to roomba mode")
    print("Press 'q' to quit")
    
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # process frame and get commands for both hands
        commands = gesture_control.process_frame(frame)
        
        # execute commands if any were generated
        if any(commands):
            robot_sim.execute_command(commands)
            
        # display frame
        cv2.imshow('Wheeled Robot Control', frame)
        
        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    robot_sim.close()
        
if __name__ == "__main__":
    main() 