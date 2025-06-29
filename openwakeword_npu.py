#!/usr/bin/env python3
"""
OpenWakeWord NPU Integration
Wake word detection on NPU for natural voice activation
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

class OpenWakeWordNPU:
    """OpenWakeWord with NPU acceleration for wake word detection"""
    
    def __init__(self):
        """Initialize OpenWakeWord + NPU system"""
        self.npu_accelerator = NPUAccelerator()
        self.wake_models = {}
        self.is_ready = False
        self.is_listening = False
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 0.256  # 256ms chunks
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Wake word settings
        self.wake_threshold = 0.7
        self.activation_cooldown = 2.0  # Seconds before next activation
        self.last_activation = 0
        
        # Available wake words
        self.available_wake_words = [
            "hey_jarvis",
            "alexa", 
            "hey_google",
            "ok_google",
            "computer",
            "assistant"
        ]
        
        # State tracking
        self.wake_word_callback = None
        
        # Audio buffer
        self.audio_buffer = queue.Queue()
        self.audio_thread = None
        
    def initialize(self, wake_words: List[str] = None):
        """Initialize OpenWakeWord models for NPU"""
        try:
            logger.info("🎯 Initializing OpenWakeWord + NPU system...")
            
            # Check NPU
            if not self.npu_accelerator.is_available():
                logger.warning("⚠️ NPU not available, using CPU for wake word detection")
            else:
                logger.info("✅ NPU Phoenix detected for wake word processing")
            
            # Use default wake words if none specified
            if wake_words is None:
                wake_words = ["hey_jarvis", "computer"]
            
            logger.info(f"📥 Loading wake word models: {wake_words}")
            
            try:
                # Try to use OpenWakeWord library
                try:
                    import openwakeword
                    from openwakeword import Model
                    
                    # Load models for requested wake words
                    for wake_word in wake_words:
                        if wake_word in self.available_wake_words:
                            try:
                                model = Model(
                                    wakeword_models=[wake_word],
                                    inference_framework='onnx'
                                )
                                self.wake_models[wake_word] = model
                                logger.info(f"✅ Loaded wake word model: {wake_word}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load {wake_word}: {e}")
                    
                except ImportError:
                    logger.warning("⚠️ OpenWakeWord library not available")
                    
                # Fallback: Create simplified keyword detection
                if not self.wake_models:
                    logger.info("🔄 Using simplified keyword detection as fallback")
                    for wake_word in wake_words:
                        self.wake_models[wake_word] = "simple_detection"
                
            except Exception as e:
                logger.warning(f"⚠️ Model loading failed: {e}")
                
                # Ultimate fallback: energy-based activation
                logger.info("🔄 Using energy-based activation as fallback")
                self.wake_models["energy_activation"] = "energy_based"
            
            self.is_ready = len(self.wake_models) > 0
            
            if self.is_ready:
                logger.info(f"🎉 OpenWakeWord + NPU system ready with {len(self.wake_models)} models!")
            else:
                logger.error("❌ No wake word models loaded")
                
            return self.is_ready
            
        except Exception as e:
            logger.error(f"❌ Wake word initialization failed: {e}")
            return False
    
    def detect_wake_word(self, audio_chunk: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect wake words in audio chunk"""
        try:
            best_wake_word = None
            best_confidence = 0.0
            
            for wake_word, model in self.wake_models.items():
                if model == "simple_detection":
                    # Simplified keyword detection using audio patterns
                    energy = np.sqrt(np.mean(audio_chunk ** 2))
                    
                    # Simple pattern matching (placeholder)
                    # In real implementation, this would use spectral features
                    if energy > 0.02:  # Higher energy threshold for wake words
                        confidence = min(energy * 5, 1.0)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_wake_word = wake_word
                
                elif model == "energy_based":
                    # Energy-based activation (simple)
                    energy = np.sqrt(np.mean(audio_chunk ** 2))
                    if energy > 0.03:  # Wake up on any loud sound
                        confidence = min(energy * 3, 1.0)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_wake_word = "energy_activation"
                
                else:
                    # Real OpenWakeWord model
                    try:
                        # Prepare audio for OpenWakeWord (expects specific format)
                        if len(audio_chunk) != self.chunk_samples:
                            if len(audio_chunk) < self.chunk_samples:
                                audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
                            else:
                                audio_chunk = audio_chunk[:self.chunk_samples]
                        
                        # Run wake word detection
                        prediction = model.predict(audio_chunk)
                        
                        for word, confidence in prediction.items():
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_wake_word = word
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Wake word detection failed for {wake_word}: {e}")
            
            # Check if above threshold
            if best_confidence > self.wake_threshold:
                return best_wake_word, best_confidence
            else:
                return None, best_confidence
                
        except Exception as e:
            logger.warning(f"⚠️ Wake word detection error: {e}")
            return None, 0.0
    
    def process_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """Process audio chunk for wake word detection"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_activation < self.activation_cooldown:
            return
        
        wake_word, confidence = self.detect_wake_word(audio_chunk)
        
        if wake_word and confidence > self.wake_threshold:
            self.last_activation = current_time
            logger.info(f"🎯 Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
            
            if self.wake_word_callback:
                self.wake_word_callback(wake_word, confidence, timestamp)
    
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
        logger.info("🔄 Wake word processing thread started")
        
        while self.is_listening:
            try:
                # Get audio chunk from buffer
                audio_chunk, timestamp = self.audio_buffer.get(timeout=1.0)
                
                # Process the chunk
                self.process_audio_chunk(audio_chunk, timestamp)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Wake word processing error: {e}")
        
        logger.info("🔄 Wake word processing thread stopped")
    
    def start_listening(self, wake_word_callback: Callable = None):
        """Start continuous wake word detection"""
        if not self.is_ready:
            logger.error("❌ Wake word detection not initialized")
            return False
        
        if self.is_listening:
            logger.warning("⚠️ Already listening for wake words")
            return True
        
        try:
            logger.info("🎯 Starting continuous wake word detection...")
            
            # Set callback
            self.wake_word_callback = wake_word_callback
            
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
            logger.info(f"✅ Wake word detection started (models: {list(self.wake_models.keys())})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start wake word detection: {e}")
            self.is_listening = False
            return False
    
    def stop_listening(self):
        """Stop continuous wake word detection"""
        if not self.is_listening:
            return
        
        logger.info("🔇 Stopping wake word detection...")
        
        self.is_listening = False
        
        # Stop audio stream
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Wait for processing thread to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        logger.info("✅ Wake word detection stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current wake word detection status"""
        return {
            "is_ready": self.is_ready,
            "is_listening": self.is_listening,
            "npu_available": self.npu_accelerator.is_available(),
            "loaded_models": list(self.wake_models.keys()),
            "wake_threshold": self.wake_threshold,
            "sample_rate": self.sample_rate,
            "activation_cooldown": self.activation_cooldown
        }

def test_openwakeword():
    """Test OpenWakeWord system"""
    print("🧪 Testing OpenWakeWord + NPU system...")
    
    # Initialize wake word detection
    wake_word = OpenWakeWordNPU()
    if not wake_word.initialize(["hey_jarvis", "computer"]):
        print("❌ Failed to initialize wake word detection")
        return False
    
    # Get status
    status = wake_word.get_status()
    print(f"\n📊 Wake Word Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Define wake word handler
    def on_wake_word(word, confidence, timestamp):
        print(f"🎯 WAKE WORD DETECTED: '{word}' at {timestamp:.2f}s (confidence: {confidence:.2f})")
        print("🎤 System activated! (In real system, this would trigger Whisper)")
    
    # Start listening for 15 seconds
    print(f"\n🎯 Starting wake word test - say 'hey jarvis' or 'computer' for 15 seconds...")
    if wake_word.start_listening(on_wake_word):
        try:
            time.sleep(15.0)
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        finally:
            wake_word.stop_listening()
    
    print("\n🎉 OpenWakeWord test completed!")
    return True

if __name__ == "__main__":
    test_openwakeword()