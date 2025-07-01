#!/usr/bin/env python3
"""
Always Listening NPU System
Combines VAD, Wake Word Detection, and ONNX Whisper for complete NPU-powered voice assistant
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
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import onnxruntime as ort
import sounddevice as sd

# Add our modules to path
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')
from whisperx_npu_accelerator import NPUAccelerator
from onnx_whisper_npu import ONNXWhisperNPU
from silero_vad_npu import SileroVADNPU
from openwakeword_npu import OpenWakeWordNPU
from audio_analyzer import AudioAnalyzer # Import the new module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlwaysListeningNPU:
    """Complete NPU-powered always-listening voice assistant system"""
    
    def __init__(self):
        """Initialize the complete always-listening system"""
        self.npu_accelerator = NPUAccelerator()
        
        # Component systems
        self.whisper_npu = ONNXWhisperNPU()
        self.vad_npu = SileroVADNPU()
        self.wake_word_npu = OpenWakeWordNPU()
        self.audio_analyzer = AudioAnalyzer() # Instantiate AudioAnalyzer
        
        # System state
        self.is_ready = False
        self.is_listening = False
        self.is_processing = False
        
        # Audio settings
        self.sample_rate = 16000
        self.recording_buffer = []
        self.recording_active = False
        self.max_recording_duration = 30.0  # Maximum 30 seconds of recording
        
        # Activation modes
        self.activation_mode = "wake_word"  # "wake_word", "vad_only", "always_on"
        
        # Callbacks
        self.transcription_callback = None
        self.status_callback = None
        
        # Recording management
        self.recording_start_time = None
        self.silence_threshold = 0.01
        self.silence_duration = 0.0
        self.max_silence_duration = 2.0  # Stop recording after 2s of silence
        
    def initialize(self, whisper_model="base", wake_words=None, activation_mode="wake_word"):
        """Initialize all NPU components"""
        try:
            logger.info("🚀 Initializing Complete NPU Always-Listening System...")
            
            # Check NPU
            if not self.npu_accelerator.is_available():
                logger.warning("⚠️ NPU not available, system will use CPU fallbacks")
            else:
                logger.info("✅ NPU Phoenix detected for complete system acceleration")
            
            self.activation_mode = activation_mode
            
            # Initialize ONNX Whisper
            logger.info("🧠 Initializing ONNX Whisper + NPU...")
            if not self.whisper_npu.initialize(whisper_model):
                logger.error("❌ Failed to initialize ONNX Whisper")
                return False
            
            # Initialize VAD
            logger.info("🎤 Initializing Silero VAD + NPU...")
            if not self.vad_npu.initialize():
                logger.error("❌ Failed to initialize VAD")
                return False
            
            # Initialize Wake Word Detection (if needed)
            if activation_mode in ["wake_word", "hybrid"]:
                logger.info("🎯 Initializing Wake Word Detection + NPU...")
                if wake_words is None:
                    wake_words = ["hey_jarvis", "computer", "assistant"]
                
                if not self.wake_word_npu.initialize(wake_words):
                    logger.warning("⚠️ Wake word detection failed, falling back to VAD-only mode")
                    self.activation_mode = "vad_only"
            
            self.is_ready = True
            logger.info("🎉 Complete NPU Always-Listening System Ready!")
            logger.info(f"📊 System Configuration:")
            logger.info(f"  - Whisper Model: {whisper_model}")
            logger.info(f"  - Activation Mode: {self.activation_mode}")
            logger.info(f"  - NPU Acceleration: {self.npu_accelerator.is_available()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def on_speech_start(self, timestamp: float, confidence: float):
        """Handle speech start event from VAD"""
        if not self.recording_active:
            logger.info(f"🎙️ Speech detected, starting recording... (confidence: {confidence:.2f})")
            self.start_recording()
    
    def on_speech_end(self, timestamp: float, duration: float):
        """Handle speech end event from VAD"""
        if self.recording_active:
            logger.info(f"🔇 Speech ended (duration: {duration:.2f}s), processing recording...")
            self.process_recording()
    
    def on_wake_word(self, wake_word: str, confidence: float, timestamp: float):
        """Handle wake word detection"""
        logger.info(f"🎯 Wake word '{wake_word}' detected! (confidence: {confidence:.2f})")
        
        if not self.recording_active:
            logger.info("🎤 Wake word activated, starting recording...")
            self.start_recording()
            
            # Notify callback about activation
            if self.status_callback:
                self.status_callback("wake_word_detected", {
                    "wake_word": wake_word,
                    "confidence": confidence,
                    "timestamp": timestamp
                })
    
    def start_recording(self):
        """Start audio recording for transcription"""
        if self.recording_active:
            return
        
        self.recording_active = True
        self.recording_buffer = []
        self.recording_start_time = time.time()
        self.silence_duration = 0.0
        
        logger.info("🔴 Recording started...")
        
        if self.status_callback:
            self.status_callback("recording_started", {"timestamp": self.recording_start_time})
    
    def add_audio_to_recording(self, audio_chunk: np.ndarray):
        """Add audio chunk to recording buffer"""
        if not self.recording_active:
            return
        
        self.recording_buffer.append(audio_chunk.copy())
        
        # Check for silence to auto-stop recording
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        if energy < self.silence_threshold:
            self.silence_duration += len(audio_chunk) / self.sample_rate
        else:
            self.silence_duration = 0.0
        
        # Check recording limits
        current_duration = time.time() - self.recording_start_time
        
        if (self.silence_duration >= self.max_silence_duration or 
            current_duration >= self.max_recording_duration):
            logger.info(f"🔇 Auto-stopping recording (silence: {self.silence_duration:.1f}s, duration: {current_duration:.1f}s)")
            self.process_recording()
    
    def process_recording(self):
        """Process recorded audio with ONNX Whisper"""
        if not self.recording_active or not self.recording_buffer:
            return
        
        try:
            self.recording_active = False
            self.is_processing = True
            
            # Combine audio chunks
            full_audio = np.concatenate(self.recording_buffer)
            duration = len(full_audio) / self.sample_rate
            
            logger.info(f"🧠 Processing {duration:.1f}s of audio with ONNX Whisper + NPU...")
            
            if self.status_callback:
                self.status_callback("processing_started", {"duration": duration})
            
            # Perform audio analysis
            audio_features = self.audio_analyzer.analyze_audio_chunk(full_audio)
            logger.info(f"📊 Audio features extracted: {audio_features}")

            # Save to temporary file for Whisper processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, full_audio, self.sample_rate)
                temp_path = tmp_file.name
            
            # Process with ONNX Whisper + NPU
            result = self.whisper_npu.transcribe_audio(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Add system metadata and audio features
            result.update({
                "system": "AlwaysListeningNPU",
                "activation_mode": self.activation_mode,
                "recorded_duration": duration,
                "npu_accelerated": True,
                "audio_features": audio_features # Add audio features to the result
            })
            
            logger.info(f"✅ Transcription completed: '{result['text']}'")
            logger.info(f"⚡ Processing time: {result['processing_time']:.2f}s")
            logger.info(f"🚀 Real-time factor: {result['real_time_factor']:.3f}x")
            
            # Notify callback
            if self.transcription_callback:
                self.transcription_callback(result)
            
            if self.status_callback:
                self.status_callback("transcription_completed", result)
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            if self.status_callback:
                self.status_callback("processing_error", {"error": str(e)})
        
        finally:
            self.is_processing = False
            self.recording_buffer = []
    
    def start_always_listening(self, transcription_callback: Callable = None, status_callback: Callable = None):
        """Start the complete always-listening system"""
        if not self.is_ready:
            logger.error("❌ System not initialized")
            return False
        
        if self.is_listening:
            logger.warning("⚠️ Already listening")
            return True
        
        try:
            logger.info(f"🎤 Starting Always-Listening System (mode: {self.activation_mode})...")
            
            # Set callbacks
            self.transcription_callback = transcription_callback
            self.status_callback = status_callback
            
            # Start VAD
            logger.info("🎤 Starting VAD monitoring...")
            if not self.vad_npu.start_listening(self.on_speech_start, self.on_speech_end):
                logger.error("❌ Failed to start VAD")
                return False
            
            # Start Wake Word Detection (if enabled)
            if self.activation_mode in ["wake_word", "hybrid"]:
                logger.info("🎯 Starting wake word detection...")
                if not self.wake_word_npu.start_listening(self.on_wake_word):
                    logger.warning("⚠️ Wake word detection failed, continuing with VAD-only")
                    self.activation_mode = "vad_only"
            
            # Start audio monitoring for recording
            self.start_audio_monitoring()
            
            self.is_listening = True
            
            logger.info("🎉 Always-Listening System Active!")
            logger.info("🎤 NPU-powered VAD monitoring speech...")
            if self.activation_mode == "wake_word":
                logger.info("🎯 NPU-powered wake word detection active...")
            logger.info("🧠 ONNX Whisper + NPU ready for transcription...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start always-listening: {e}")
            return False
    
    def start_audio_monitoring(self):
        """Start audio monitoring for recording management"""
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"⚠️ Audio status: {status}")
            
            # Convert to mono
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            # Add to recording if active
            if self.recording_active:
                self.add_audio_to_recording(audio_data)
        
        # Start audio stream for recording
        self.recording_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            dtype=np.float32
        )
        
        self.recording_stream.start()
        logger.info("🎧 Audio monitoring started for recording management")
    
    def stop_always_listening(self):
        """Stop the complete always-listening system"""
        if not self.is_listening:
            return
        
        logger.info("🔇 Stopping Always-Listening System...")
        
        self.is_listening = False
        
        # Stop all components
        self.vad_npu.stop_listening()
        
        if self.activation_mode in ["wake_word", "hybrid"]:
            self.wake_word_npu.stop_listening()
        
        # Stop recording stream
        if hasattr(self, 'recording_stream'):
            self.recording_stream.stop()
            self.recording_stream.close()
        
        # Process any remaining recording
        if self.recording_active:
            self.process_recording()
        
        logger.info("✅ Always-Listening System stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            "is_ready": self.is_ready,
            "is_listening": self.is_listening,
            "is_processing": self.is_processing,
            "is_recording": self.recording_active,
            "activation_mode": self.activation_mode,
            "npu_available": self.npu_accelerator.is_available(),
            "whisper_ready": self.whisper_npu.is_ready,
            "vad_status": self.vad_npu.get_status(),
            "wake_word_status": self.wake_word_npu.get_status(),
            "sample_rate": self.sample_rate
        }

def test_always_listening():
    """Test the complete always-listening system"""
    print("🧪 Testing Complete NPU Always-Listening System...")
    
    # Initialize system
    system = AlwaysListeningNPU()
    if not system.initialize(
        whisper_model="base",
        wake_words=["hey_jarvis", "computer"], 
        activation_mode="wake_word"
    ):
        print("❌ Failed to initialize system")
        return False
    
    # Get status
    status = system.get_system_status()
    print(f"
📊 System Status:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Define callbacks
    def on_transcription(result):
        print(f"
🎤 TRANSCRIPTION RESULT:")
        print(f"  Text: '{result['text']}'")
        print(f"  Duration: {result['audio_duration']:.1f}s")
        print(f"  Processing: {result['processing_time']:.2f}s")
        print(f"  Real-time factor: {result['real_time_factor']:.3f}x")
        print(f"  NPU accelerated: {result['npu_accelerated']}")
        if 'audio_features' in result:
            print(f"  Audio Features: {result['audio_features']}")
    
    def on_status(event, data):
        print(f"📡 Status: {event} - {data}")
    
    # Start listening for 30 seconds
    print(f"
🎤 Starting always-listening test for 30 seconds...")
    print(f"🎯 Say 'hey jarvis' or 'computer' to activate, then speak...")
    
    if system.start_always_listening(on_transcription, on_status):
        try:
            time.sleep(30.0)
        except KeyboardInterrupt:
            print("
⏹️ Test interrupted by user")
        finally:
            system.stop_always_listening()
    
    print("
🎉 Always-Listening System test completed!")
    return True

if __name__ == "__main__":
    test_always_listening()