import librosa
import numpy as np
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        logger.info(f"AudioAnalyzer initialized with sample rate: {self.sample_rate}")

    def analyze_audio_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Analyzes a raw audio chunk (numpy array) for voice tonality features.
        Args:
            audio_chunk: A numpy array representing the audio chunk.
                         Expected to be mono, float32, at self.sample_rate.
        Returns:
            A dictionary containing extracted tonality features.
        """
        if audio_chunk.size == 0:
            return {
                "pitch_mean": None,
                "pitch_std": None,
                "energy_mean": None,
                "energy_std": None,
                "speech_rate": None, # Placeholder for future
                "voice_quality": None # Placeholder for future
            }

        try:
            # Pitch (F0) estimation using pYIN
            # F0 values are in Hz. NaN for unvoiced frames.
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=audio_chunk,
                sr=self.sample_rate,
                fmin=librosa.note_to_hz('C2'), # ~65 Hz
                fmax=librosa.note_to_hz('C5')  # ~523 Hz
            )
            
            # Filter out NaN values (unvoiced frames) for mean/std calculation
            f0_voiced = f0[~np.isnan(f0)]
            pitch_mean = float(np.mean(f0_voiced)) if f0_voiced.size > 0 else None
            pitch_std = float(np.std(f0_voiced)) if f0_voiced.size > 0 else None

            # Energy (RMS)
            # Frame length and hop length are important for meaningful energy
            frame_length = 2048 # Standard for librosa
            hop_length = 512    # Standard for librosa
            rms = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))

            # Placeholder for speech rate and voice quality
            speech_rate = None 
            voice_quality = None

            features = {
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "speech_rate": speech_rate,
                "voice_quality": voice_quality
            }
            # logger.debug(f"Extracted audio features: {features}")
            return features

        except Exception as e:
            logger.error(f"Error analyzing audio chunk: {e}")
            return {
                "pitch_mean": None,
                "pitch_std": None,
                "energy_mean": None,
                "energy_std": None,
                "speech_rate": None,
                "voice_quality": None
            }

if __name__ == '__main__':
    # Example Usage: Generate a dummy audio chunk and analyze
    logger.info("Running AudioAnalyzer example...")
    sample_rate = 16000
    duration = 2 # seconds
    frequency = 440 # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    dummy_audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    features = analyzer.analyze_audio_chunk(dummy_audio)
    logger.info(f"Analysis Result: {features}")

    # Example with a more complex signal (e.g., speech-like noise)
    logger.info("Running AudioAnalyzer example with noise...")
    noise_audio = np.random.randn(int(sample_rate * 1)).astype(np.float32) * 0.1
    features_noise = analyzer.analyze_audio_chunk(noise_audio)
    logger.info(f"Analysis Result (Noise): {features_noise}")

    # Example with silence
    logger.info("Running AudioAnalyzer example with silence...")
    silence_audio = np.zeros(int(sample_rate * 1)).astype(np.float32)
    features_silence = analyzer.analyze_audio_chunk(silence_audio)
    logger.info(f"Analysis Result (Silence): {features_silence}")
