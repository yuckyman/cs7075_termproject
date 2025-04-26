# llm_integration.py - Placeholder 

import requests
import json

# Import config dictionary
from .config import CONFIG

class LLMIntegrator:
    def __init__(self, config):
        """Initializes the LLMIntegrator with configuration."""
        self.config = config
        self.llm_endpoint = config.get("llm_endpoint", "http://localhost:11434/api/generate")
        self.llm_model = config.get("llm_model", "llama3") # Default model
        print(f"LLM Integrator configured for endpoint: {self.llm_endpoint} with model: {self.llm_model}")

    def get_response(self, user_input, gesture_token, tone_token, emotion_token):
        """
        Combines the recognized tokens (gesture, tone, emotion) with the user's text input
        to produce a context-aware response from the LLM via Ollama API.

        Args:
            user_input (str): The text input from the user.
            gesture_token (str): The recognized gesture token (e.g., '<gesture_neutral>').
            tone_token (str): The recognized voice tone token (e.g., '<voice_tone=neutral>').
            emotion_token (str): The recognized facial emotion token (e.g., '<emotion_happy>').

        Returns:
            str: The text response from the LLM, or an error message.
        """
        # --- Construct the prompt with multimodal context ---
        # Clean up tokens
        cleaned_gesture = gesture_token.replace('<gesture_', '').replace('>', '')
        cleaned_tone = tone_token.replace('<voice_tone=', '').replace('>', '')
        cleaned_emotion = emotion_token.replace('<emotion_', '').replace('>', '')

        # Get psychological context for each modality
        psych_contexts = []
        
        # Add gesture psychological context if available
        if cleaned_gesture in self.config.get("psychological_context", {}).get("gesture", {}):
            gesture_psych = self.config["psychological_context"]["gesture"][cleaned_gesture]
            psych_contexts.append(f"Gesture: {gesture_psych}")
        
        # Add tone psychological context if available
        if cleaned_tone in self.config.get("psychological_context", {}).get("tone", {}):
            tone_psych = self.config["psychological_context"]["tone"][cleaned_tone]
            psych_contexts.append(f"Voice: {tone_psych}")
        
        # Add emotion psychological context if available
        if cleaned_emotion in self.config.get("psychological_context", {}).get("emotion", {}):
            emotion_psych = self.config["psychological_context"]["emotion"][cleaned_emotion]
            psych_contexts.append(f"Facial: {emotion_psych}")
        
        # Build psychological context string
        psychological_context = ""
        if psych_contexts:
            psychological_context = "[Psychological Context: " + " | ".join(psych_contexts) + "]\n"

        # Integrate all context
        context_prefix = (
            f"[User Context: Gesture={cleaned_gesture}, VoiceTone={cleaned_tone}, FacialEmotion={cleaned_emotion}]\n"
            f"{psychological_context}"
            f"User: {user_input}\n"
            f"Assistant:"
        )
        
        # --- Prepare Ollama API Payload ---
        payload = {
            "model": self.llm_model,
            "prompt": context_prefix,
            "stream": False # Get the full response at once
        }

        # --- Call Ollama API ---
        try:
            response = requests.post(
                self.llm_endpoint,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=60 # Add a timeout (in seconds)
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the response
            response_data = response.json()
            llm_response = response_data.get("response", "").strip()
            
            if not llm_response:
                 print(f"Warning: LLM response was empty. Full response data: {response_data}")
                 return "(LLM response was empty)"
            
            return llm_response

        except requests.exceptions.ConnectionError:
            error_msg = f"Error: Could not connect to Ollama endpoint at {self.llm_endpoint}. Is Ollama running?"
            print(error_msg)
            return error_msg
        except requests.exceptions.Timeout:
            error_msg = f"Error: Request to Ollama timed out after 60 seconds."
            print(error_msg)
            return error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Error during Ollama API request: {e}\nResponse status: {e.response.status_code if e.response else 'N/A'}\nResponse text: {e.response.text if e.response else 'N/A'}"
            print(error_msg)
            return f"(Error communicating with LLM: {e.response.status_code if e.response else 'N/A'})"
        except json.JSONDecodeError:
            error_msg = f"Error: Could not decode JSON response from Ollama.\nResponse text: {response.text}"
            print(error_msg)
            return "(Error parsing LLM response)"
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            print(error_msg)
            return "(An unexpected error occurred)"

# Example usage (for testing):
if __name__ == '__main__':
    # Requires Ollama server to be running (e.g., `ollama serve` or Ollama Desktop)
    # Make sure the model specified in CONFIG (e.g., llama3) is pulled (`ollama pull llama3`)
    print("Testing LLMIntegrator context sensitivity...")
    
    # Use mock config if run directly, otherwise use imported CONFIG if available
    try: 
        cfg = CONFIG
    except NameError:
         print("Running with mock config as CONFIG not directly available.")
         cfg = {
            "llm_endpoint": "http://localhost:11434/api/generate",
            "llm_model": "llama3",
            "labels": {
                 "gestures": ["neutral", "angry", "open"],
                 "tones": ["neutral", "angry", "pleased"],
                 "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
             },
             "psychological_context": {
                "gesture": {
                    "neutral": "Baseline attentional state; no significant gestural signaling",
                    "angry": "Reactance theory; frustration-aggression hypothesis",
                    "open": "Approach behavior; pro-social signaling"
                },
                "tone": {
                    "neutral": "Baseline affective state; cognitive monitoring",
                    "angry": "Cognitive appraisal of threat; emotional contagion risk",
                    "pleased": "Broaden-and-build theory; positive affect" 
                },
                "emotion": {
                    "angry": "Fight response; goal blockage",
                    "happy": "Social reward signaling; hedonic well-being",
                    "neutral": "Cognitive equilibrium; emotion regulation"
                }
             }
        }

    integrator = LLMIntegrator(cfg)
    
    # --- Test Scenario --- 
    test_input = "What do you think of this situation?"
    print(f"\n--- Test Input: '{test_input}' ---")

    # 1. Neutral Context
    neutral_gesture = "<gesture_neutral>"
    neutral_tone = "<voice_tone=neutral>"
    neutral_emotion = "<emotion_neutral>"
    print(f"\n--- Scenario 1: NEUTRAL Context ---")
    print(f"Context: Gesture={neutral_gesture}, Tone={neutral_tone}, Emotion={neutral_emotion}")
    print("Sending prompt...")
    neutral_response = integrator.get_response(test_input, neutral_gesture, neutral_tone, neutral_emotion)
    print("\nLLM Response (Neutral Context):")
    print(neutral_response)
    print("-----------------------------------")

    # 2. Angry Context
    angry_gesture = "<gesture_angry>"
    angry_tone = "<voice_tone=angry>"
    angry_emotion = "<emotion_angry>"
    print(f"\n--- Scenario 2: ANGRY Context ---")
    print(f"Context: Gesture={angry_gesture}, Tone={angry_tone}, Emotion={angry_emotion}")
    print("Sending prompt...")
    angry_response = integrator.get_response(test_input, angry_gesture, angry_tone, angry_emotion)
    print("\nLLM Response (Angry Context):")
    print(angry_response)
    print("---------------------------------")
    
    # 3. Mixed Context with Psychological Theory
    mixed_gesture = "<gesture_open>"
    mixed_tone = "<voice_tone=pleased>"
    mixed_emotion = "<emotion_happy>"
    print(f"\n--- Scenario 3: MIXED POSITIVE Context with Psychological Theory ---")
    print(f"Context: Gesture={mixed_gesture}, Tone={mixed_tone}, Emotion={mixed_emotion}")
    print("Sending prompt...")
    mixed_response = integrator.get_response(test_input, mixed_gesture, mixed_tone, mixed_emotion)
    print("\nLLM Response (Mixed Positive Context):")
    print(mixed_response)
    print("---------------------------------")

    print("\nTest complete.") 