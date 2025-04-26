import cv2
import time
import sys
from pathlib import Path

# add src directory to python path
src_dir = str(Path(__file__).parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from multimodal.gesture_control import GestureControl
from simulation.wheeled_robot_sim import WheeledRobotSimulator

def main():
    # initialize components with wheeled robot mode
    gesture_control = GestureControl(control_mode="wheeled")
    robot_sim = WheeledRobotSimulator()
    
    # setup video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting wheeled robot gesture control system...")
    print("Controls:")
    print("- point: drive robot towards pointed location")
    print("- fist: rotate the robot (position determines direction)")
    print("- open palm + wave: toggle stop/go")
    print("Press 'q' to quit")
    
    try:
        while True:
            # capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # flip frame horizontally for more intuitive control
            frame = cv2.flip(frame, 1)
            
            # process frame and get robot command
            command = gesture_control.process_frame(frame)
            
            # execute command if one was generated
            if command:
                robot_sim.execute_command(command)
            
            # display frame
            cv2.imshow('Wheeled Robot Gesture Control', frame)
            
            # break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # maintain reasonable frame rate
            time.sleep(0.01)
            
    finally:
        # cleanup
        cap.release()
        cv2.destroyAllWindows()
        robot_sim.close()
        
if __name__ == "__main__":
    main() 