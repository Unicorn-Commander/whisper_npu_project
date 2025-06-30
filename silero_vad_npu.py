#!/usr/bin/env python3
"""
Silero VAD NPU Integration
Continuous voice activity detection on NPU for always-listening capability
"""

import numpy as np
import torch
import librosa
import time
import tempfile
import logging
import os
import sys
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import onnxruntime as ort
import sounddevice as sd

# Add our modules to path
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')
from whisperx_npu_accelerator import NPUAccelerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SileroVADNPU:
    """Silero VAD with NPU acceleration for continuous voice activity detection"""
    
    def __init__(self):
        """Initialize Silero VAD + NPU system"""
        self.npu_accelerator = NPUAccelerator()
        self.vad_model = None
        self.is_ready = False
        self.is_listening = False
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 0.256  # 256ms chunks for real-time processing
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Silero VAD internal states
        self._h = np.zeros((2, 1, 128)).astype(np.float32) # Hidden state
        self._c = np.zeros((2, 1, 128)).astype(np.float32) # Cell state
        
        # VAD settings
        self.vad_threshold = 0.5
        self.min_speech_duration = 0.25  # Minimum 250ms of speech
        self.min_silence_duration = 0.5  # Minimum 500ms of silence to end speech
        
        # State tracking
        self.speech_start_time = None
        self.last_speech_time = None
        self.is_speech_active = False
        
        # Callback for speech events
        self.speech_start_callback = None
        self.speech_end_callback = None
        
        # Audio buffer
        self.audio_buffer = queue.Queue()
        self.audio_thread = None
        
    def initialize(self):
        """Initialize Silero VAD model for NPU"""
        try:
            logger.info("🎤 Initializing Silero VAD + NPU system...")
            
            # Check NPU
            if not self.npu_accelerator.is_available():
                logger.warning("⚠️ NPU not available, using CPU for VAD")
            else:
                logger.info("✅ NPU Phoenix detected for VAD processing")
            
            # Download and load Silero VAD model
            logger.info("📥 Loading Silero VAD model...")
            
            try:
                # Try to load from HuggingFace
                from huggingface_hub import hf_hub_download
                
                model_path = hf_hub_download(
                    repo_id="deepghs/silero-vad-onnx",
                    filename="silero_vad.onnx",
                    cache_dir="/home/ucadmin/Development/whisper_npu_project/vad_cache"
                )
                
                logger.info(f"📁 VAD model downloaded: {model_path}")
                
                # Load ONNX model with NPU providers if available
                providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider'] if self.npu_accelerator.is_available() else ['CPUExecutionProvider']
                
                self.vad_model = ort.InferenceSession(model_path, providers=providers)
                logger.info(f"✅ Silero VAD loaded with providers: {self.vad_model.get_providers()}")
                
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace download failed: {e}")
                
                # Fallback: Create simplified VAD using audio energy
                logger.info("🔄 Using simplified energy-based VAD as fallback")
                self.vad_model = "energy_based"
            
            self.is_ready = True
            logger.info("🎉 Silero VAD + NPU system ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ VAD initialization failed: {e}")
            return False
    
    def detect_voice_activity(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Detect voice activity in audio chunk"""
        try:
            if self.vad_model == "energy_based":
                # Simplified energy-based VAD
                energy = np.sqrt(np.mean(audio_chunk ** 2))
                is_speech = energy > 0.01  # Simple threshold
                confidence = min(energy * 10, 1.0)
                return is_speech, confidence
            
            elif self.vad_model is not None:
                # Use Silero VAD model
                # Prepare input (Silero expects specific format)
                if len(audio_chunk) != self.chunk_samples:
                    # Pad or truncate to expected size
                    if len(audio_chunk) < self.chunk_samples:
                        audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
                    else:
                        audio_chunk = audio_chunk[:self.chunk_samples]
                
                # Silero VAD input format
                input_feed = {
                    'input': input_tensor,
                    'sr': np.array(self.sample_rate, dtype=np.int64),
                    'state': np.concatenate((self._h, self._c), axis=1)
                }
                
                # Run VAD inference
                outputs = self.vad_model.run(None, input_feed)
                speech_prob = outputs[0][0][0]  # Extract speech probability
                self._h = outputs[1][:, :128, :]
                self._c = outputs[1][:, 128:, :]
                
                is_speech = speech_prob > self.vad_threshold
                return is_speech, speech_prob
            
            else:
                return False, 0.0
                
        except Exception as e:
            logger.warning(f"⚠️ VAD detection failed: {e}")
            return False, 0.0
    
    def process_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """Process audio chunk and manage speech state"""
        is_speech, confidence = self.detect_voice_activity(audio_chunk)
        
        current_time = time.time()
        
        if is_speech:
            self.last_speech_time = current_time
            
            if not self.is_speech_active:
                # Speech just started
                if self.speech_start_time is None:
                    self.speech_start_time = current_time
                elif current_time - self.speech_start_time >= self.min_speech_duration:
                    # Confirmed speech start
                    self.is_speech_active = True
                    logger.info(f"🗣️ Speech started (confidence: {confidence:.2f})")
                    
                    if self.speech_start_callback:
                        self.speech_start_callback(timestamp, confidence)
            
        else:
            # No speech detected
            if self.is_speech_active and self.last_speech_time:
                if current_time - self.last_speech_time >= self.min_silence_duration:
                    # Speech ended
                    self.is_speech_active = False
                    speech_duration = self.last_speech_time - self.speech_start_time
                    logger.info(f"🔇 Speech ended (duration: {speech_duration:.2f}s)")
                    
                    if self.speech_end_callback:
                        self.speech_end_callback(timestamp, speech_duration)
                    
                    self.speech_start_time = None
            
            elif not self.is_speech_active:
                self.speech_start_time = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback for real-time processing"""
        if status:
            logger.warning(f"⚠️ Audio status: {status}")
        
        # Convert to mono if needed
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]
        
        # Add to processing queue
        self.audio_buffer.put((audio_data.copy(), time_info.inputBufferAdcTime))
    
    def audio_processing_thread(self):
        """Background thread for audio processing"""
        logger.info("🔄 Audio processing thread started")
        
        while self.is_listening:
            try:
                # Get audio chunk from buffer
                audio_chunk, timestamp = self.audio_buffer.get(timeout=1.0)
                
                # Process the chunk
                self.process_audio_chunk(audio_chunk, timestamp)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Audio processing error: {e}")
        
        logger.info("🔄 Audio processing thread stopped")
    
    def start_listening(self, speech_start_callback: Callable = None, speech_end_callback: Callable = None):
        """Start continuous voice activity detection"""
        if not self.is_ready:
            logger.error("❌ VAD not initialized")
            return False
        
        if self.is_listening:
            logger.warning("⚠️ Already listening")
            return True
        
        try:
            logger.info("🎤 Starting continuous voice activity detection...")
            
            # Set callbacks
            self.speech_start_callback = speech_start_callback
            self.speech_end_callback = speech_end_callback
            
            # Start audio processing thread
            self.is_listening = True
            self.audio_thread = threading.Thread(target=self.audio_processing_thread, daemon=True)
            self.audio_thread.start()
            
            # Start audio input stream
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_samples,
                dtype=np.float32
            )
            
            self.audio_stream.start()
            logger.info("✅ VAD listening started (continuous mode)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start listening: {e}")
            self.is_listening = False
            return False
    
    def stop_listening(self):
        """Stop continuous voice activity detection"""
        if not self.is_listening:
            return
        
        logger.info("🔇 Stopping voice activity detection...")
        
        self.is_listening = False
        
        # Stop audio stream
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Wait for processing thread to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        logger.info("✅ VAD listening stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current VAD status"""
        return {
            "is_ready": self.is_ready,
            "is_listening": self.is_listening,
            "is_speech_active": self.is_speech_active,
            "npu_available": self.npu_accelerator.is_available(),
            "sample_rate": self.sample_rate,
            "vad_threshold": self.vad_threshold,
            "model_type": "silero" if self.vad_model != "energy_based" else "energy_based"
        }

def test_silero_vad():
    """Test Silero VAD system"""
    print("🧪 Testing Silero VAD + NPU system...")
    
    # Initialize VAD
    vad = SileroVADNPU()
    if not vad.initialize():
        print("❌ Failed to initialize VAD")
        return False
    
    # Get status
    status = vad.get_status()
    print(f"\n📊 VAD Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Define speech event handlers
    def on_speech_start(timestamp, confidence):
        print(f"🗣️ SPEECH STARTED at {timestamp:.2f}s (confidence: {confidence:.2f})")
    
    def on_speech_end(timestamp, duration):
        print(f"🔇 SPEECH ENDED at {timestamp:.2f}s (duration: {duration:.2f}s)")
    
    # Start listening for 10 seconds
    print(f"\n🎤 Starting VAD test - speak now for 10 seconds...")
    if vad.start_listening(on_speech_start, on_speech_end):
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        finally:
            vad.stop_listening()
    
    print("\n🎉 Silero VAD test completed!")
    return True

if __name__ == "__main__":
    test_silero_vad()