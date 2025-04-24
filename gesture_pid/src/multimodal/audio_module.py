# audio_module.py - Placeholder 

import numpy as np
import librosa
import time
import pyaudio
import collections
import math

# Import config dictionary
from .config import CONFIG

class AudioToneRecognizer:
    def __init__(self, config):
        """Initializes PyAudio stream and stores configuration."""
        self.config = config
        self.audio_config = config["audio_config"]
        self.labels = config["labels"]["tones"]

        # Audio stream parameters
        self.sample_rate = self.audio_config["sample_rate"]
        self.chunk_size = self.audio_config["chunk_size"]
        self.channels = self.audio_config["channels"]
        self.format_str = self.audio_config["format"]
        self.format = getattr(pyaudio, self.format_str) # Convert string format to pyaudio constant
        self.analysis_window_sec = self.audio_config["analysis_window_sec"]

        # Calculate buffer size needed for the analysis window
        self.analysis_buffer_size = math.ceil(
            self.analysis_window_sec * self.sample_rate / self.chunk_size
        ) * self.chunk_size
        self.audio_buffer = collections.deque(maxlen=self.analysis_buffer_size)

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback # Use callback for non-blocking processing
            )
            self.stream.start_stream()
            print("Audio stream started.")
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.p.terminate()
            raise

        # TODO: Load or initialize the actual tone classifier if needed
        # self.classifier = load_classifier(config["tone_classifier_path"])

        # Store the latest classified tone
        self.last_tone_label = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function to process incoming audio chunks."""
        # Convert buffer to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=self._get_numpy_dtype())
        # Append to buffer
        self.audio_buffer.extend(audio_chunk)

        # Check if buffer has enough data for analysis
        if len(self.audio_buffer) >= self.analysis_buffer_size:
            # Process the segment from the buffer
            segment = np.array(self.audio_buffer)
            tone_label = self.process_audio_segment(segment)
            self.last_tone_label = tone_label # Store the latest label
            # print(f"Detected Tone: {tone_label}") # Debug print
        
        return (in_data, pyaudio.paContinue) # Indicate to continue streaming

    def _get_numpy_dtype(self):
        """Maps PyAudio format to NumPy dtype."""
        if self.format == pyaudio.paInt16:
            return np.int16
        elif self.format == pyaudio.paFloat32:
            return np.float32
        # Add other formats if needed
        else:
            raise ValueError(f"Unsupported PyAudio format: {self.format_str}")

    def get_latest_tone(self):
        """Returns the most recently classified tone label."""
        # Return default if no label yet
        if self.last_tone_label is None:
            return f"<voice_tone={self.labels[0]}>" # Default e.g., neutral
        return self.last_tone_label

    def process_audio_segment(self, segment):
        """
        Convert the raw audio segment to features and classify tone.
        Extracts MFCCs, RMSE (energy), and pitch (f0).
        """
        segment_float = segment.astype(np.float32)
        sr = self.sample_rate

        try:
            # Extract features
            mfccs = librosa.feature.mfcc(y=segment_float, sr=sr, n_mfcc=13)
            rmse = librosa.feature.rms(y=segment_float)[0] # Energy
            
            # Pitch (f0) - may require careful handling of unvoiced segments
            pitches, magnitudes = librosa.piptrack(y=segment_float, sr=sr)
            # Select pitches with high magnitude
            valid_pitches = pitches[magnitudes > np.median(magnitudes)]
            if len(valid_pitches) > 0:
                pitch_mean = np.mean(valid_pitches)
                pitch_std = np.std(valid_pitches)
            else:
                pitch_mean = 0
                pitch_std = 0

            # Aggregate features (using mean for simplicity)
            mfccs_mean = np.mean(mfccs, axis=1)
            rmse_mean = np.mean(rmse)

            # Combine features into a dictionary or single vector if needed later
            features = {
                'mfccs_mean': mfccs_mean,
                'rmse_mean': rmse_mean,
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std
            }
            # print(f"Features: RMSE={rmse_mean:.2f}, PitchMean={pitch_mean:.2f}, PitchStd={pitch_std:.2f}") # Debug print

            # Classify tone based on features
            tone_label = self._classify_tone(features)
            return tone_label
        except Exception as e:
            print(f"Error processing audio segment: {e}")
            return f"<voice_tone={self.labels[0]}>" # Default to neutral

    def _classify_tone(self, features):
        """
        Classifies tone based on extracted audio features using simple rules.
        Features expected: rmse_mean, pitch_mean, pitch_std.
        """
        # --- Define Rough Thresholds (These MUST be tuned) ---
        # Based on normalized values or typical ranges observed
        # Assuming features might not be normalized yet, these are guesses
        ENERGY_LOW_THRESH = 0.01  # Example threshold for low energy
        ENERGY_HIGH_THRESH = 0.05 # Example threshold for high energy
        PITCH_STD_LOW_THRESH = 10  # Example threshold for low pitch variation
        PITCH_STD_HIGH_THRESH = 50 # Example threshold for high pitch variation
        PITCH_MEAN_LOW_THRESH = 100 # Example threshold for low average pitch
        PITCH_MEAN_HIGH_THRESH = 200 # Example threshold for high average pitch

        rmse = features['rmse_mean']
        pitch_std = features['pitch_std']
        pitch_mean = features['pitch_mean']

        # --- Simple Rule-Based Logic ---
        tone = self.labels[0] # Default to neutral

        if rmse > ENERGY_HIGH_THRESH:
            # High energy: Could be angry or pleased
            if pitch_std > PITCH_STD_HIGH_THRESH and pitch_mean > PITCH_MEAN_HIGH_THRESH:
                tone = self.labels[2] # 'pleased' (high energy, high variation, high pitch)
            elif pitch_std < PITCH_STD_LOW_THRESH:
                 tone = self.labels[1] # 'angry' (high energy, low variation)
            # else remains ambiguous, maybe default based on pitch mean?
            elif pitch_mean > PITCH_MEAN_HIGH_THRESH:
                 tone = self.labels[2] # lean towards 'pleased'
            else:
                 tone = self.labels[1] # lean towards 'angry'
        
        elif rmse < ENERGY_LOW_THRESH:
             # Low energy: Likely neutral
             tone = self.labels[0] # 'neutral'
        
        else: # Medium energy
             # Could still be neutral, or mildly pleased/angry
             if pitch_std > PITCH_STD_HIGH_THRESH:
                 tone = self.labels[2] # 'pleased' (moderate energy, high variation)
             else:
                 tone = self.labels[0] # default to 'neutral'

        # print(f"Classified tone as {tone}") # Debug print
        return f"<voice_tone={tone}>"

    def close(self):
        """Stops and closes the audio stream and terminates PyAudio."""
        if self.stream is not None:
            if self.stream.is_active():
                self.stream.stop_stream()
                print("Audio stream stopped.")
            self.stream.close()
            print("Audio stream closed.")
        self.p.terminate()
        print("PyAudio terminated.") 