import cv2
from hand_tracker import HandTracker

def test_gesture_recognition():
    # initialize camera and hand tracker
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    print("\n=== gesture recognition test ===")
    print("press 'q' to quit\n")
    print("static gestures:")
    print("- open palm (spread fingers)")
    print("- fist (closed hand)")
    print("- point (index finger only)")
    print("- spiderman (thumb and pinky out)")
    print("\ndynamic gestures:")
    print("- swipe left/right (move hand horizontally)")
    print("- circle (draw circle with index finger)")
    print("- wave (wave index finger up and down)")
    print("\na red trail will show your finger movement!")
    
    while True:
        # read frame
        success, frame = cap.read()
        if not success:
            print("failed to get frame!")
            break
            
        # flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # detect hands and get gesture
        frame, hands = tracker.find_hands(frame)
        if hands:
            gesture = tracker.get_gesture(hands[0])
            frame = tracker.draw_info(frame, gesture)
            
        # show frame
        cv2.imshow("Gesture Test", frame)
        
        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_gesture_recognition() 