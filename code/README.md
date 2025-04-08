# Real-Time Multimodal Interaction Pipeline

This project aims to build a real-time pipeline that captures user gestures and voice tone, classifies them, and integrates this information into an interaction with a large language model (LLM).

## Project Structure

```
project_root/
 ┣━ requirements.txt           # list of dependencies
 ┣━ README.md                  # overview of project, how to run
 ┣━ data/                      # optional folder for any local datasets or sample files
 ┣━ models/                    # pretrained model weights or external resources if needed
 ┣━ multimodal/               # main python package (folder) containing your modules
 ┃   ┣━ __init__.py
 ┃   ┣━ gesture_module.py      # gesture detection & classification
 ┃   ┣━ audio_module.py        # prosodic (tone) detection
 ┃   ┣━ emotion_module.py      # facial emotion detection (NEW)
 ┃   ┣━ classifier_utils.py    # helper functions or any custom classification logic
 ┃   ┣━ llm_integration.py     # hooking the recognized tokens into an LLM
 ┃   ┗━ config.py              # any shared configs (model paths, labels, thresholds)
 ┗━ main.py                    # the main script to run the real-time pipeline
```

## Features

### Multimodal Detection
- **Gesture Detection**: Recognizes body language cues through MediaPipe
- **Voice Tone Analysis**: Processes audio in real-time to detect emotional tones
- **Facial Emotion Recognition**: Uses DeepFace to identify emotional states from facial expressions

### Psychological Context Integration
The system now enhances interactions by incorporating social psychology theories relevant to the detected emotional states. This provides the LLM with deeper context about the user's possible psychological state, leading to more nuanced responses.

Example of psychological theories mapped to emotional states:
- **Anger**:
  - Frustration-aggression hypothesis
  - Reactance theory
  - Cognitive appraisal of threat
- **Happiness/Pleasure**:
  - Broaden-and-build theory
  - Hedonic well-being
  - Pro-social behavior
- **Neutral**:
  - Emotion regulation
  - Cognitive monitoring
  - Attentional baseline

### Enhanced Prompt Example

```
[User Context: Gesture=angry, VoiceTone=angry, FacialEmotion=angry]
[Psychological Context: Gesture: Reactance theory; frustration-aggression hypothesis; territorial behavior | Voice: Cognitive appraisal of threat; emotional contagion risk; stress response | Facial: Fight response; goal blockage; cognitive dissonance]
User: What do you think of this situation?
Assistant:
```

This enriched prompt helps the LLM generate responses that are not only reactive to the user's emotional state but also informed by psychological frameworks that could explain the user's current state.

## TODO

Here's a rough checklist of the implementation steps:

-   [x] Populate `multimodal/config.py` with necessary paths, labels, and thresholds. (`2025-04-01` - Added emotion labels)
-   [x] Implement `multimodal/gesture_module.py`: (`2025-04-01` - Implemented basic rules)
    -   [x] Video capture setup (within class). (`2025-04-01`)
    -   [x] Real-time pose detection setup using MediaPipe. (`2025-04-01`)
    -   [x] Keypoint extraction and normalization logic (`2025-04-01` - Basic extraction).
    -   [x] Gesture classification logic (`2025-04-01` - Simple rule-based).
-   [x] Implement `multimodal/audio_module.py`: (`2025-04-01` - Implemented basic rules)
    -   [x] Real-time audio capture using PyAudio/Sounddevice (`2025-04-01` - Callback setup).
    -   [x] Audio segment processing (buffering) (`2025-04-01` - Deque buffer).
    -   [x] Feature extraction using Librosa (`2025-04-01` - MFCC, RMSE, Pitch).
    -   [x] Tone/emotion classification logic (`2025-04-01` - Simple rule-based).
-   [x] Implement `multimodal/emotion_module.py`: (`2025-04-01` - Basic DeepFace integration)
    -   [x] Initialize DeepFace. (`2025-04-01`)
    -   [x] Process frame for emotion (throttled). (`2025-04-01`)
    -   [x] Handle errors/no face detected. (`2025-04-01`)
-   [ ] Implement `multimodal/classifier_utils.py` (Optional):
    -   [ ] Helper functions for loading/saving classifiers (e.g., using `joblib`).
    -   [ ] Feature extraction utilities if shared between modules.
-   [x] Implement `multimodal/llm_integration.py`: (`2025-04-01`)
    -   [x] Define method to combine text input, gesture, tone, and emotion tokens. (`2025-04-01`)
    -   [x] Implement logic to call the target LLM (local or API). (`2025-04-01` - Ollama integration).
    -   [x] Add psychological context to enhance LLM understanding. (`2025-04-02`)
-   [x] Implement `main.py`: (`2025-04-01`)
    -   [x] Initialize all modules (`GestureRecognizer`, `AudioToneRecognizer`, `EmotionRecognizer`, `LLMIntegrator`). (`2025-04-01`)
    -   [x] Set up main loop for video, audio, and emotion processing. (`2025-04-01`)
    -   [/] Integrate user input mechanism (`2025-04-01` - Switched to SpeechRecognition).
    -   [x] Call LLM and display response. (`2025-04-01`)
    -   [x] Handle graceful shutdown (close camera, audio stream, etc.). (`2025-04-01`)
    -   [x] Display psychological context in console output. (`2025-04-02`)
-   [ ] Create `data/` directory if needed for sample data.
-   [x] Create `models/` directory if needed for storing trained classifier models. (`2025-04-01`)
-   [/] Refine `requirements.txt` (`2025-04-01` - Added SpeechRecognition, deepface).
-   [ ] Add detailed setup and run instructions to this README. 