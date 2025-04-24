import cv2
import numpy as np
from deepface import DeepFace
import time

class EmotionRecognizer:
    def __init__(self, config):
        """Initializes the EmotionRecognizer."""
        self.config = config
        self.labels = config["labels"]["emotions"]
        self.last_analysis_time = 0
        # Analyze less frequently to save CPU, e.g., once every second
        self.analysis_interval = 1.0 
        self.last_emotion = self.labels[-1] # Default to neutral
        print("Emotion Recognizer initialized. DeepFace models might download on first use.")

    def process_frame(self, frame):
        """
        Analyzes a video frame for facial emotion using DeepFace.
        Returns the dominant emotion token (e.g., '<emotion_happy>')
        or the last known emotion if analysis is skipped or fails.
        
        Note: Runs analysis only once per `self.analysis_interval` seconds.
        """
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_interval:
            # Skip analysis if interval hasn't passed
            return f"<emotion_{self.last_emotion}>"
        
        self.last_analysis_time = current_time

        try:
            # DeepFace expects BGR images by default
            # Set detector_backend and enforce_detection=False to avoid errors if no face is found
            analysis = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True, # Suppress internal print statements
                detector_backend='opencv' # Use a faster backend like opencv or ssd
            )
            
            # DeepFace returns a list of results if multiple faces are found.
            # We'll just use the first detected face's result.
            if isinstance(analysis, list) and len(analysis) > 0:
                dominant_emotion = analysis[0]['dominant_emotion']
                if dominant_emotion in self.labels:
                    self.last_emotion = dominant_emotion
                else:
                     print(f"Warning: Detected emotion '{dominant_emotion}' not in configured labels.")
                     # Fallback to neutral if the detected label isn't expected
                     self.last_emotion = self.labels[-1] 
            elif isinstance(analysis, dict): # Handle case where only one face is returned as dict
                dominant_emotion = analysis.get('dominant_emotion')
                if dominant_emotion in self.labels:
                     self.last_emotion = dominant_emotion
                else:
                     print(f"Warning: Detected emotion '{dominant_emotion}' not in configured labels.")
                     self.last_emotion = self.labels[-1]
            # else: No face detected or analysis failed, keep last_emotion

        except Exception as e:
            # Avoid crashing the main loop if DeepFace fails
            # Common issue: Model files missing or download error
            print(f"Error during DeepFace analysis: {e}. Using last known emotion: {self.last_emotion}")
            # Fallback: keep the last known emotion
            pass 
            
        return f"<emotion_{self.last_emotion}>"

    def close(self):
        """Placeholder for any cleanup if needed."""
        print("EmotionRecognizer resources released (if any).")
        pass

# Example usage (for testing):
if __name__ == '__main__':
    print("Testing EmotionRecognizer...")
    # Need config for labels
    mock_config = {
        "labels": {
            "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        }
    }
    recognizer = EmotionRecognizer(mock_config)
    
    # Use a webcam feed for testing
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            emotion_token = recognizer.process_frame(frame)
            
            # Display the result
            cv2.putText(frame, emotion_token, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Emotion Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    recognizer.close() 