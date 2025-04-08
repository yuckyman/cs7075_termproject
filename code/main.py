# main.py - Placeholder 

import cv2
import time
import speech_recognition as sr # Import SpeechRecognition

# Import our modules
from multimodal.config import CONFIG
from multimodal.gesture_module import GestureRecognizer
from multimodal.audio_module import AudioToneRecognizer
from multimodal.emotion_module import EmotionRecognizer # Import EmotionRecognizer
from multimodal.llm_integration import LLMIntegrator

def main():
    print("Initializing modules...")
    try:
        gesture_recognizer = GestureRecognizer(CONFIG)
        audio_recognizer = AudioToneRecognizer(CONFIG) # Starts audio stream in background
        emotion_recognizer = EmotionRecognizer(CONFIG) # Initialize EmotionRecognizer
        llm = LLMIntegrator(CONFIG)
    except Exception as e:
        print(f"Error initializing modules: {e}")
        print("Please check your configuration and hardware (camera, microphone).")
        # Add cleanup for emotion recognizer if others failed
        if 'audio_recognizer' in locals() and audio_recognizer is not None: audio_recognizer.close()
        if 'gesture_recognizer' in locals() and gesture_recognizer is not None: gesture_recognizer.close()
        return

    print("Opening camera...")
    cap = cv2.VideoCapture(0) # Use 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        # Attempt to clean up audio if it started
        if 'audio_recognizer' in locals() and audio_recognizer is not None:
            audio_recognizer.close()
        return

    # --- Initialize Speech Recognition ---
    print("Initializing speech recognizer...")
    r = sr.Recognizer()
    # Set a higher energy threshold if needed (adjust based on your mic/noise)
    # r.energy_threshold = 4000 
    mic = sr.Microphone()
    try:
        with mic as source:
            print("Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source, duration=1.0) # Adjust for 1 second
            print("Ambient noise adjustment complete.")
    except Exception as e:
        print(f"Error initializing microphone: {e}. Please check mic permissions/settings.")
        # Cleanup other modules if necessary
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'audio_recognizer' in locals() and audio_recognizer is not None: audio_recognizer.close()
        if 'gesture_recognizer' in locals() and gesture_recognizer is not None: gesture_recognizer.close()
        return

    print("\n--- Real-Time Multimodal Interaction (Voice Input) --- \n")
    print("Speak clearly when you see 'Listening...'.")
    print("Press 'q' in the OpenCV window to quit.")
    print("------------------------------------------------------ \n")

    try:
        while True:
            # --- Video Processing ---
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from camera.")
                time.sleep(0.1) # Avoid busy-waiting if camera fails
                continue

            # Get gesture token
            gesture_token = gesture_recognizer.process_frame(frame)
            if gesture_token is None:
                gesture_token = f"<gesture_{CONFIG['labels']['gestures'][0]}>" # Default to neutral

            # --- Emotion Recognition ---
            emotion_token = emotion_recognizer.process_frame(frame) # Process frame for emotion
            # emotion_token will be the latest known emotion due to throttling

            # --- Audio Processing (Tone - runs in background) ---
            voice_tone_token = audio_recognizer.get_latest_tone()

            # --- Display Camera Feed (Optional) ---
            # Add detected gesture/tone to the frame for visualization
            display_text_line1 = f"Gesture: {gesture_token} | Tone: {voice_tone_token}"
            display_text_line2 = f"Emotion: {emotion_token}"
            cv2.putText(frame, display_text_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, display_text_line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Camera Feed", frame)

            # --- Check for Quit Key ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 print("\n'q' pressed, exiting...")
                 break
            
            # --- Speech Recognition & LLM Call (Replaces input()) ---
            user_input_text = None
            print("Listening for command...") # Indicate ready state
            try:
                with mic as source:
                    # Listen for speech - this blocks for a moment!
                    # timeout: max seconds to wait for phrase start
                    # phrase_time_limit: max seconds to allow phrase to continue
                    audio_data = r.listen(source, timeout=5.0, phrase_time_limit=5.0)
                
                print("Got audio, recognizing...")
                # Recognize speech using Google Web Speech API
                user_input_text = r.recognize_google(audio_data)
                print(f"[{gesture_token}][{voice_tone_token}] You said: {user_input_text}")

            except sr.WaitTimeoutError:
                # print("No speech detected within timeout.")
                pass # Simply loop again if no speech
            except sr.UnknownValueError:
                print("Assistant could not understand audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                 print(f"An unexpected error occurred during speech recognition: {e}")

            # If we successfully got text, send it to the LLM with all context
            if user_input_text:
                # Get cleaned tokens for psychological context logging
                cleaned_gesture = gesture_token.replace('<gesture_', '').replace('>', '')
                cleaned_tone = voice_tone_token.replace('<voice_tone=', '').replace('>', '')
                cleaned_emotion = emotion_token.replace('<emotion_', '').replace('>', '')
                
                # Build psychological context for logging
                psych_contexts = []
                
                # Add psychological contexts if available
                if cleaned_gesture in CONFIG.get("psychological_context", {}).get("gesture", {}):
                    gesture_psych = CONFIG["psychological_context"]["gesture"][cleaned_gesture]
                    psych_contexts.append(f"Gesture: {gesture_psych}")
                
                if cleaned_tone in CONFIG.get("psychological_context", {}).get("tone", {}):
                    tone_psych = CONFIG["psychological_context"]["tone"][cleaned_tone]
                    psych_contexts.append(f"Voice: {tone_psych}")
                
                if cleaned_emotion in CONFIG.get("psychological_context", {}).get("emotion", {}):
                    emotion_psych = CONFIG["psychological_context"]["emotion"][cleaned_emotion]
                    psych_contexts.append(f"Facial: {emotion_psych}")
                
                # Print psychological context if available
                if psych_contexts:
                    print("\nPsychological Context:")
                    for context in psych_contexts:
                        print(f"- {context}")
                
                print("\nAssistant thinking...")
                response = llm.get_response(user_input_text, gesture_token, voice_tone_token, emotion_token) # Add emotion_token
                print(f"\nAssistant: {response}")
                print("------------------------------------------------------ \n")

            # Removed the input() block entirely
            # Removed time.sleep() as listen() introduces pauses

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected during main loop, exiting...")
    finally:
        # --- Cleanup --- 
        print("\nReleasing resources...")
        if cap.isOpened():
            cap.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        # Safely close modules
        if 'audio_recognizer' in locals() and audio_recognizer is not None:
             audio_recognizer.close()
        if 'gesture_recognizer' in locals() and gesture_recognizer is not None:
             gesture_recognizer.close()
        if 'emotion_recognizer' in locals() and emotion_recognizer is not None:
             emotion_recognizer.close() # Close emotion recognizer
        print("Application finished.")

if __name__ == "__main__":
    main() 