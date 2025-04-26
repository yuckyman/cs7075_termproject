# config.py - Shared configuration

CONFIG = {
    # --- Classifier Paths ---
    "gesture_classifier_path": "models/gesture_clf.joblib",
    "tone_classifier_path": "models/tone_clf.joblib",

    # --- LLM Configuration ---
    "llm_endpoint": "http://localhost:11434/api/generate",  # Example for Ollama
    "llm_model": "llama3.2:latest", # Default model to use with Ollama, can be changed

    # --- MediaPipe Pose Configuration ---
    "pose_config": {
        "static_image_mode": False,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5
    },

    # --- Label Definitions ---
    "labels": {
        "gestures": ["neutral", "angry", "open"],
        "tones": ["neutral", "angry", "pleased"],
        "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"] # DeepFace defaults
    },

    # --- Audio Processing Configuration ---
    "audio_config": {
        "sample_rate": 16000,       # Sample rate for audio capture
        "chunk_size": 1024,         # Size of audio chunks to process
        "channels": 1,              # Number of audio channels (1 for mono)
        "format": "paInt16",        # PyAudio format code (16-bit integers)
        "analysis_window_sec": 1.0  # Duration (in seconds) of audio to analyze for tone
    },
    
    # --- Psychological Context Mappings ---
    "psychological_context": {
        # Mapping for gesture states
        "gesture": {
            "neutral": "Baseline attentional state; no significant gestural signaling",
            "angry": "Reactance theory; frustration-aggression hypothesis; territorial behavior",
            "open": "Approach behavior; pro-social signaling; self-disclosure readiness"
        },
        # Mapping for voice tone states
        "tone": {
            "neutral": "Baseline affective state; cognitive monitoring",
            "angry": "Cognitive appraisal of threat; emotional contagion risk; stress response",
            "pleased": "Broaden-and-build theory; positive affect; reciprocity principle"
        },
        # Mapping for facial emotion states
        "emotion": {
            "angry": "Fight response; goal blockage; cognitive dissonance",
            "disgust": "Behavioral immune system; moral judgment; avoidance motivation",
            "fear": "Flight response; risk aversion; uncertainty avoidance",
            "happy": "Social reward signaling; hedonic well-being; affiliative behavior",
            "sad": "Attachment theory implications; social support seeking; empathy elicitation",
            "surprise": "Schema violation; orienting response; information-seeking behavior",
            "neutral": "Cognitive equilibrium; emotion regulation; attentional baseline"
        }
    }
} 