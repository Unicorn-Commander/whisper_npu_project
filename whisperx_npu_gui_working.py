#!/usr/bin/env python3
"""
WhisperX NPU GUI Application - Working Version

Working WhisperX integration with OpenAI Whisper fallback and NPU acceleration.
Uses OpenAI Whisper when WhisperX/ctranslate2 has issues.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import sys
import numpy as np
import wave
import subprocess
import sounddevice as sd
import soundfile as sf
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import queue
import json
import shutil

# Add our modules to path
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project/npu_kernels')

from whisperx_npu_accelerator import WhisperXNPUIntegration

# Try imports with fallbacks
WHISPERX_AVAILABLE = False
WHISPER_AVAILABLE = False

try:
    import whisperx
    import torch
    WHISPERX_AVAILABLE = True
    print("✅ WhisperX available")
except ImportError as e:
    print(f"⚠️ WhisperX not available: {e}")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ OpenAI Whisper available")
except ImportError as e:
    print(f"❌ OpenAI Whisper not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manage model downloads and caching for both WhisperX and OpenAI Whisper"""
    
    # Model information
    MODEL_INFO = {
        "tiny": {"size": "~39 MB", "description": "Fastest, lowest accuracy"},
        "base": {"size": "~74 MB", "description": "Good balance of speed and accuracy"},
        "small": {"size": "~244 MB", "description": "Better accuracy, slower"},
        "medium": {"size": "~769 MB", "description": "High accuracy, much slower"},
        "large": {"size": "~1550 MB", "description": "Highest accuracy, very slow"},
        "large-v2": {"size": "~1550 MB", "description": "Latest large model, best quality"}
    }
    
    def __init__(self):
        """Initialize model manager"""
        self.cache_dir = self.get_cache_directory()
        
    def get_cache_directory(self):
        """Get model cache directory"""
        # OpenAI Whisper uses different cache than WhisperX
        cache_home = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        return Path(cache_home) / "whisper"
    
    def is_model_cached(self, model_name):
        """Check if model is already downloaded"""
        try:
            # For OpenAI Whisper, models are stored as .pt files
            model_file = self.cache_dir / f"{model_name}.pt"
            return model_file.exists()
        except Exception as e:
            logger.error(f"Error checking model cache: {e}")
            return False
    
    def get_cached_models(self):
        """Get list of cached models"""
        cached = []
        for model_name in self.MODEL_INFO.keys():
            if self.is_model_cached(model_name):
                cached.append(model_name)
        return cached


class AudioCapture:
    """Real-time audio capture for live transcription"""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_duration=3.0):
        """Initialize audio capture"""
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        
    def list_audio_devices(self):
        """List available audio input devices"""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_sample_rate': device['default_samplerate']
                })
        return input_devices
    
    def start_recording(self, device_id=None):
        """Start recording audio"""
        if self.is_recording:
            return False
        
        try:
            self.stream = sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=self.chunk_size
            )
            self.stream.start()
            self.is_recording = True
            logger.info(f"Started recording: {self.sample_rate}Hz, {self.channels} channel(s)")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return False
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_recording = False
            logger.info("Stopped recording")
            return True
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False
    
    def _audio_callback(self, indata, frames, time, status):
        """Audio stream callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if needed
        if self.channels == 1 and indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0] if indata.shape[1] > 0 else indata
        
        # Add to queue for processing
        self.audio_queue.put(audio_data.copy())
    
    def get_audio_chunk(self, timeout=1.0):
        """Get next audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class WhisperProcessor:
    """Whisper processor with fallback support"""
    
    def __init__(self, npu_integration=None):
        """Initialize Whisper processor"""
        self.npu_integration = npu_integration
        self.model = None
        self.current_model_size = None
        self.using_whisperx = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, model_size="base", progress_callback=None):
        """Load model with fallback to OpenAI Whisper"""
        try:
            logger.info(f"Loading model: {model_size}")
            
            if progress_callback:
                progress_callback(10, f"Loading {model_size} model...")
            
            # Try WhisperX first
            if WHISPERX_AVAILABLE:
                try:
                    logger.info("Attempting to load WhisperX model...")
                    self.model = whisperx.load_model(
                        model_size, 
                        device=self.device, 
                        compute_type="float16"
                    )
                    self.using_whisperx = True
                    logger.info("✅ WhisperX model loaded successfully")
                    
                    if progress_callback:
                        progress_callback(70, "Applying NPU acceleration...")
                    
                    # Apply NPU acceleration if available
                    if self.npu_integration and self.npu_integration.acceleration_enabled:
                        logger.info("Applying NPU acceleration to model")
                        self.model = self.npu_integration.patch_whisperx_model(self.model)
                    
                except Exception as e:
                    logger.warning(f"WhisperX loading failed: {e}")
                    logger.info("Falling back to OpenAI Whisper...")
                    self.model = None
                    self.using_whisperx = False
            
            # Fallback to OpenAI Whisper
            if not self.model and WHISPER_AVAILABLE:
                try:
                    logger.info("Loading OpenAI Whisper model...")
                    import whisper
                    self.model = whisper.load_model(model_size)
                    self.using_whisperx = False
                    logger.info("✅ OpenAI Whisper model loaded successfully")
                except Exception as e:
                    logger.error(f"OpenAI Whisper loading failed: {e}")
                    self.model = None
            
            if not self.model:
                if progress_callback:
                    progress_callback(-1, "No working Whisper implementation found")
                return False
            
            self.current_model_size = model_size
            
            if progress_callback:
                progress_callback(100, f"Model {model_size} loaded successfully!")
            
            logger.info(f"Model loaded successfully using {'WhisperX' if self.using_whisperx else 'OpenAI Whisper'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if progress_callback:
                progress_callback(-1, f"Loading failed: {e}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            if self.using_whisperx:
                # WhisperX approach
                audio = whisperx.load_audio(audio_path)
                result = self.model.transcribe(audio, batch_size=16)
                
                # Convert to standard format
                if isinstance(result, dict) and 'segments' in result:
                    return result
                else:
                    # Handle different result formats
                    return {"text": str(result), "segments": []}
                    
            else:
                # OpenAI Whisper approach
                result = self.model.transcribe(audio_path)
                return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_audio_data(self, audio_data, sample_rate=16000):
        """Transcribe audio data array"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                tmp_path = tmp_file.name
            
            # Transcribe
            result = self.transcribe_audio(tmp_path)
            
            # Cleanup
            os.unlink(tmp_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Audio data transcription failed: {e}")
            raise


class WhisperXNPUGUIWorking:
    """Working GUI Application with real Whisper integration"""
    
    def __init__(self, root):
        """Initialize the working GUI application"""
        self.root = root
        self.npu_integration = None
        self.whisper_processor = None
        self.audio_capture = AudioCapture()
        self.model_manager = ModelManager()
        self.is_processing = False
        self.is_recording = False
        self.recording_thread = None
        
        self.setup_window()
        self.create_widgets()
        self.initialize_components()
        self.update_status()
        self.refresh_model_status()
    
    def setup_window(self):
        """Configure the main window"""
        self.root.title("WhisperX NPU Accelerator - Working")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Main tab
        self.main_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_frame, text="🎙️ Transcription")
        
        # Model management tab
        self.models_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.models_frame, text="🧠 Model Management")
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        self.models_frame.columnconfigure(0, weight=1)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        self.create_main_tab()
        self.create_models_tab()
    
    def create_main_tab(self):
        """Create main transcription tab"""
        # Title
        title_label = ttk.Label(
            self.main_frame, 
            text="🎙️ WhisperX NPU Accelerator - Working", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status Section
        self.create_status_section()
        
        # Model Loading Section
        self.create_model_section()
        
        # Audio Processing Section
        self.create_audio_section()
        
        # Real-time Section
        self.create_realtime_section()
        
        # Results Section
        self.create_results_section()
        
        # Performance Section
        self.create_performance_section()
    
    def create_models_tab(self):
        """Create model management tab"""
        # Title
        title_label = ttk.Label(
            self.models_frame, 
            text="🧠 Model Management", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Available models section
        available_frame = ttk.LabelFrame(self.models_frame, text="Available Models", padding="10")
        available_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        available_frame.columnconfigure(0, weight=1)
        
        # Model list with treeview
        columns = ('Model', 'Size', 'Status', 'Description')
        self.model_tree = ttk.Treeview(available_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.model_tree.heading(col, text=col)
            self.model_tree.column(col, width=150)
        
        self.model_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Scrollbar for model list
        model_scrollbar = ttk.Scrollbar(available_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        model_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.model_tree.configure(yscrollcommand=model_scrollbar.set)
        
        # Model actions frame
        actions_frame = ttk.Frame(available_frame)
        actions_frame.grid(row=1, column=0, pady=(10, 0))
        
        refresh_btn = ttk.Button(actions_frame, text="🔄 Refresh", 
                                command=self.refresh_model_status)
        refresh_btn.grid(row=0, column=0)
        
        # Configure grid weights
        self.models_frame.rowconfigure(1, weight=1)
        available_frame.rowconfigure(0, weight=1)
    
    def create_status_section(self):
        """Create status display section"""
        status_frame = ttk.LabelFrame(self.main_frame, text="🔧 System Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        # NPU Status
        ttk.Label(status_frame, text="NPU Available:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.npu_status_var = tk.StringVar(value="Initializing...")
        self.npu_status_label = ttk.Label(status_frame, textvariable=self.npu_status_var, font=('Arial', 10, 'bold'))
        self.npu_status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Whisper Status
        ttk.Label(status_frame, text="Whisper Available:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.whisper_status_var = tk.StringVar(value="Checking...")
        self.whisper_status_label = ttk.Label(status_frame, textvariable=self.whisper_status_var, font=('Arial', 10, 'bold'))
        self.whisper_status_label.grid(row=0, column=3, sticky=tk.W)
        
        # Model Status
        ttk.Label(status_frame, text="Current Model:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.model_status_var = tk.StringVar(value="Not loaded")
        ttk.Label(status_frame, textvariable=self.model_status_var).grid(row=1, column=1, sticky=tk.W)
        
        # Engine Type
        ttk.Label(status_frame, text="Engine:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.engine_var = tk.StringVar(value="Unknown")
        ttk.Label(status_frame, textvariable=self.engine_var).grid(row=1, column=3, sticky=tk.W)
    
    def create_model_section(self):
        """Create model loading section"""
        model_frame = ttk.LabelFrame(self.main_frame, text="🧠 Model Configuration", padding="10")
        model_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(2, weight=1)
        
        # Model selection
        ttk.Label(model_frame, text="Model Size:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.model_size_var = tk.StringVar(value="base")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_size_var, 
                                       values=list(ModelManager.MODEL_INFO.keys()),
                                       state="readonly", width=15)
        self.model_combo.grid(row=0, column=1, padx=(0, 20))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selection_changed)
        
        # Model info
        self.model_info_var = tk.StringVar(value="Select a model to see details")
        ttk.Label(model_frame, textvariable=self.model_info_var, font=('Arial', 9, 'italic')).grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        # Load model button
        self.load_model_btn = ttk.Button(model_frame, text="🔄 Load Model", 
                                        command=self.load_model, state=tk.DISABLED)
        self.load_model_btn.grid(row=0, column=3, padx=(20, 0))
        
        # Model loading progress
        self.model_progress_var = tk.DoubleVar()
        self.model_progress = ttk.Progressbar(model_frame, variable=self.model_progress_var, 
                                            maximum=100, length=500)
        self.model_progress.grid(row=1, column=0, columnspan=4, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Model status
        self.model_load_status_var = tk.StringVar(value="Ready")
        ttk.Label(model_frame, textvariable=self.model_load_status_var, 
                 font=('Arial', 9, 'italic')).grid(row=2, column=0, columnspan=4, pady=(5, 0))
        
        # Update model info on initial selection
        self.on_model_selection_changed()
    
    def create_audio_section(self):
        """Create audio file processing section"""
        audio_frame = ttk.LabelFrame(self.main_frame, text="🎵 Audio File Processing", padding="10")
        audio_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        audio_frame.columnconfigure(1, weight=1)
        
        # File selection
        ttk.Label(audio_frame, text="Audio File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.file_path_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(audio_frame, textvariable=self.file_path_var, relief=tk.SUNKEN, width=50)
        file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        browse_btn = ttk.Button(audio_frame, text="📁 Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=2)
        
        # Process button
        self.process_btn = ttk.Button(audio_frame, text="🚀 Process Audio", 
                                     command=self.process_audio, state=tk.DISABLED)
        self.process_btn.grid(row=1, column=0, columnspan=3, pady=(15, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(audio_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
    
    def create_realtime_section(self):
        """Create real-time transcription section"""
        realtime_frame = ttk.LabelFrame(self.main_frame, text="🎤 Real-time Transcription", padding="10")
        realtime_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        realtime_frame.columnconfigure(1, weight=1)
        
        # Audio device selection
        ttk.Label(realtime_frame, text="Audio Device:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.audio_device_var = tk.StringVar(value="Default")
        self.device_combo = ttk.Combobox(realtime_frame, textvariable=self.audio_device_var, 
                                        state="readonly", width=40)
        self.device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Refresh devices button
        refresh_devices_btn = ttk.Button(realtime_frame, text="🔄", command=self.refresh_audio_devices)
        refresh_devices_btn.grid(row=0, column=2)
        
        # Recording controls
        controls_frame = ttk.Frame(realtime_frame)
        controls_frame.grid(row=1, column=0, columnspan=3, pady=(15, 0))
        
        self.record_btn = ttk.Button(controls_frame, text="🎤 Start Recording", 
                                    command=self.toggle_recording, state=tk.DISABLED)
        self.record_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Recording status
        self.recording_status_var = tk.StringVar(value="Ready")
        ttk.Label(controls_frame, textvariable=self.recording_status_var, 
                 font=('Arial', 10, 'italic')).grid(row=0, column=1)
    
    def create_results_section(self):
        """Create transcription results section"""
        results_frame = ttk.LabelFrame(self.main_frame, text="📝 Transcription Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results toolbar
        toolbar_frame = ttk.Frame(results_frame)
        toolbar_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        clear_btn = ttk.Button(toolbar_frame, text="🗑️ Clear", command=self.clear_results)
        clear_btn.grid(row=0, column=0, padx=(0, 10))
        
        save_btn = ttk.Button(toolbar_frame, text="💾 Save", command=self.save_results)
        save_btn.grid(row=0, column=1, padx=(0, 10))
        
        copy_btn = ttk.Button(toolbar_frame, text="📋 Copy", command=self.copy_results)
        copy_btn.grid(row=0, column=2)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            wrap=tk.WORD, 
            width=100, 
            height=10,
            font=('Courier', 9)
        )
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main frame row weight for resizing
        self.main_frame.rowconfigure(5, weight=1)
    
    def create_performance_section(self):
        """Create performance metrics section"""
        perf_frame = ttk.LabelFrame(self.main_frame, text="⚡ Performance Metrics", padding="10")
        perf_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        perf_frame.columnconfigure(1, weight=1)
        perf_frame.columnconfigure(3, weight=1)
        
        # Processing time
        ttk.Label(perf_frame, text="Processing Time:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.proc_time_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.proc_time_var, font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky=tk.W)
        
        # NPU utilization
        ttk.Label(perf_frame, text="NPU Acceleration:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.npu_util_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.npu_util_var, font=('Arial', 10, 'bold')).grid(row=0, column=3, sticky=tk.W)
        
        # Real-time factor
        ttk.Label(perf_frame, text="Real-time Factor:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.rtf_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.rtf_var, font=('Arial', 10, 'bold')).grid(row=1, column=1, sticky=tk.W)
        
        # Audio duration
        ttk.Label(perf_frame, text="Audio Duration:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.duration_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.duration_var, font=('Arial', 10, 'bold')).grid(row=1, column=3, sticky=tk.W)
    
    def initialize_components(self):
        """Initialize NPU and Whisper components"""
        def init_worker():
            try:
                # Initialize NPU
                self.npu_integration = WhisperXNPUIntegration()
                
                # Initialize Whisper processor
                self.whisper_processor = WhisperProcessor(self.npu_integration)
                
                # Refresh audio devices
                self.root.after(0, self.refresh_audio_devices)
                
                self.root.after(0, self.on_components_initialized)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_components_error(str(e)))
        
        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()
    
    def on_components_initialized(self):
        """Called when components are initialized"""
        self.update_status()
        self.load_model_btn.config(state=tk.NORMAL)
        self.add_result("✅ WhisperX NPU Accelerator Working initialized successfully!\n", "success")
        
        if WHISPERX_AVAILABLE:
            self.whisper_status_var.set("✅ WhisperX Available")
        elif WHISPER_AVAILABLE:
            self.whisper_status_var.set("✅ OpenAI Whisper Available")
        else:
            self.whisper_status_var.set("❌ No Whisper Available")
    
    def on_components_error(self, error_msg):
        """Called when component initialization fails"""
        self.npu_status_var.set("❌ Error")
        self.whisper_status_var.set("❌ Error")
        self.add_result(f"❌ Initialization failed: {error_msg}\n", "error")
    
    def update_status(self):
        """Update status displays"""
        if self.npu_integration:
            status = self.npu_integration.get_acceleration_status()
            if status['npu_available']:
                self.npu_status_var.set("✅ Available")
            else:
                self.npu_status_var.set("❌ Unavailable")
    
    def refresh_model_status(self):
        """Refresh model status in the management tab"""
        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        # Add models with status
        cached_models = self.model_manager.get_cached_models()
        
        for model_name, info in ModelManager.MODEL_INFO.items():
            status = "✅ Cached" if model_name in cached_models else "📥 Will download on use"
            
            self.model_tree.insert('', 'end', values=(
                model_name,
                info['size'],
                status,
                info['description']
            ))
    
    def on_model_selection_changed(self, event=None):
        """Update model info when selection changes"""
        selected_model = self.model_size_var.get()
        if selected_model in ModelManager.MODEL_INFO:
            info = ModelManager.MODEL_INFO[selected_model]
            cached = self.model_manager.is_model_cached(selected_model)
            cache_status = "✅ Cached" if cached else "📥 Will download"
            self.model_info_var.set(f"{info['size']}, {info['description']} - {cache_status}")
        else:
            self.model_info_var.set("Select a model to see details")
    
    def refresh_audio_devices(self):
        """Refresh audio device list"""
        try:
            devices = self.audio_capture.list_audio_devices()
            device_names = ["Default"] + [f"{d['name']} ({d['id']})" for d in devices]
            self.device_combo['values'] = device_names
            if device_names:
                self.device_combo.set(device_names[0])
        except Exception as e:
            logger.error(f"Failed to refresh audio devices: {e}")
    
    def load_model(self):
        """Load Whisper model"""
        if not (WHISPERX_AVAILABLE or WHISPER_AVAILABLE):
            messagebox.showerror("Error", "No Whisper implementation available")
            return
        
        selected_model = self.model_size_var.get()
        
        self.load_model_btn.config(state=tk.DISABLED)
        self.model_progress_var.set(0)
        
        def load_worker():
            try:
                def progress_callback(percent, message):
                    self.root.after(0, lambda: self.model_progress_var.set(max(0, percent)))
                    self.root.after(0, lambda: self.model_load_status_var.set(message))
                
                # Load model
                success = self.whisper_processor.load_model(
                    selected_model, 
                    progress_callback=progress_callback
                )
                
                if not success:
                    self.root.after(0, lambda: self.on_model_load_error("Failed to load model"))
                    return
                
                self.root.after(0, self.on_model_loaded)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_model_load_error(str(e)))
        
        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()
    
    def on_model_loaded(self):
        """Called when model is loaded successfully"""
        engine_type = "WhisperX" if self.whisper_processor.using_whisperx else "OpenAI Whisper"
        self.model_status_var.set(f"Loaded: {self.model_size_var.get()}")
        self.engine_var.set(engine_type)
        self.process_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.NORMAL)
        self.load_model_btn.config(state=tk.NORMAL)
        self.model_progress_var.set(0)
        self.model_load_status_var.set("Ready")
        
        # Refresh model status to show as cached
        self.refresh_model_status()
        
        self.add_result(f"✅ Model loaded: {self.model_size_var.get()} using {engine_type}\n", "success")
        messagebox.showinfo("Success", f"Model loaded successfully using {engine_type}!")
    
    def on_model_load_error(self, error_msg):
        """Called when model loading fails"""
        self.model_status_var.set("Load failed")
        self.engine_var.set("None")
        self.load_model_btn.config(state=tk.NORMAL)
        self.model_progress_var.set(0)
        self.model_load_status_var.set("Load failed")
        
        self.add_result(f"❌ Model loading failed: {error_msg}\n", "error")
        messagebox.showerror("Error", f"Model loading failed: {error_msg}")
    
    def browse_file(self):
        """Browse for audio file"""
        file_types = [
            ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=file_types
        )
        
        if filename:
            self.file_path_var.set(os.path.basename(filename))
            self.selected_file = filename
            self.add_result(f"📁 Selected file: {os.path.basename(filename)}\n", "info")
            
            # Get audio duration
            try:
                duration = self.get_audio_duration(filename)
                self.duration_var.set(f"{duration:.1f}s")
            except:
                self.duration_var.set("Unknown")
    
    def get_audio_duration(self, filepath):
        """Get duration of audio file"""
        try:
            if filepath.endswith('.wav'):
                with wave.open(filepath, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / float(rate)
            else:
                # Use soundfile for other formats
                with sf.SoundFile(filepath) as f:
                    return len(f) / f.samplerate
        except:
            return 0.0
    
    def process_audio(self):
        """Process selected audio file with real Whisper"""
        if not hasattr(self, 'selected_file'):
            messagebox.showwarning("No File", "Please select an audio file first.")
            return
        
        if not self.whisper_processor or not self.whisper_processor.model:
            messagebox.showerror("No Model", "Please load a model first.")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        def process_worker():
            try:
                start_time = time.time()
                
                self.root.after(0, lambda: self.progress_var.set(10))
                self.root.after(0, lambda: self.add_result("🔄 Starting transcription...\n", "info"))
                
                # Transcribe with real Whisper
                result = self.whisper_processor.transcribe_audio(self.selected_file)
                
                self.root.after(0, lambda: self.progress_var.set(80))
                
                # Format results
                transcription = self.format_transcription_result(result, self.selected_file)
                
                # Calculate metrics
                processing_time = time.time() - start_time
                audio_duration = float(self.duration_var.get().replace('s', '')) if 's' in self.duration_var.get() else 0.0
                rtf = processing_time / audio_duration if audio_duration > 0 else 0
                
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.on_processing_complete(transcription, processing_time, rtf))
                
            except Exception as e:
                self.root.after(0, lambda: self.on_processing_error(str(e)))
        
        thread = threading.Thread(target=process_worker, daemon=True)
        thread.start()
    
    def format_transcription_result(self, result, filename):
        """Format transcription result for display"""
        engine_type = "WhisperX" if self.whisper_processor.using_whisperx else "OpenAI Whisper"
        
        output = f"""🎙️ TRANSCRIPTION RESULTS

File: {os.path.basename(filename)}
Model: {self.model_size_var.get()}
Engine: {engine_type}
NPU Acceleration: {'Enabled' if self.npu_integration and self.npu_integration.acceleration_enabled else 'Disabled'}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Handle different result formats
        if isinstance(result, dict):
            if 'text' in result and result['text'].strip():
                output += f"FULL TEXT:\n{result['text'].strip()}\n\n"
            
            if 'segments' in result and result['segments']:
                output += "SEGMENTS:\n"
                for i, segment in enumerate(result['segments']):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    
                    if text:
                        output += f"[{start:06.2f} → {end:06.2f}] {text}\n"
            elif 'text' in result and not result['text'].strip():
                output += "No speech detected in audio file.\n"
        else:
            output += f"Result: {str(result)}\n"
        
        output += f"\n✅ Transcription completed successfully using {engine_type}!\n"
        return output
    
    def toggle_recording(self):
        """Toggle real-time recording"""
        if not self.whisper_processor or not self.whisper_processor.model:
            messagebox.showerror("No Model", "Please load a model first.")
            return
        
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start real-time recording and transcription"""
        try:
            # Get selected device
            device_id = None
            if self.audio_device_var.get() != "Default":
                device_name = self.audio_device_var.get()
                if '(' in device_name and ')' in device_name:
                    device_id = int(device_name.split('(')[-1].split(')')[0])
            
            # Start audio capture
            if not self.audio_capture.start_recording(device_id):
                messagebox.showerror("Error", "Failed to start audio recording")
                return
            
            self.is_recording = True
            self.record_btn.config(text="⏹️ Stop Recording")
            self.recording_status_var.set("🎤 Recording...")
            
            # Start processing thread
            self.recording_thread = threading.Thread(target=self.recording_worker, daemon=True)
            self.recording_thread.start()
            
            self.add_result("🎤 Started real-time recording...\n", "info")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")
    
    def stop_recording(self):
        """Stop real-time recording"""
        try:
            self.is_recording = False
            self.audio_capture.stop_recording()
            
            self.record_btn.config(text="🎤 Start Recording")
            self.recording_status_var.set("Ready")
            
            self.add_result("⏹️ Stopped real-time recording\n", "info")
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
    
    def recording_worker(self):
        """Background worker for real-time transcription"""
        while self.is_recording:
            try:
                # Get audio chunk
                audio_chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                if audio_chunk is None:
                    continue
                
                # Only process if chunk has sufficient audio
                if len(audio_chunk) < self.audio_capture.sample_rate * 1.0:  # Less than 1 second
                    continue
                
                # Check for actual audio content (not just silence)
                if np.max(np.abs(audio_chunk)) < 0.01:  # Very quiet
                    continue
                
                # Transcribe chunk
                try:
                    result = self.whisper_processor.transcribe_audio_data(
                        audio_chunk, 
                        self.audio_capture.sample_rate
                    )
                    
                    # Format and display result
                    text_segments = []
                    
                    if isinstance(result, dict):
                        if 'segments' in result and result['segments']:
                            for segment in result['segments']:
                                text = segment.get('text', '').strip()
                                if text:
                                    text_segments.append(text)
                        elif 'text' in result and result['text'].strip():
                            text_segments.append(result['text'].strip())
                    
                    if text_segments:
                        timestamp = time.strftime('%H:%M:%S')
                        live_text = f"[{timestamp}] {' '.join(text_segments)}\n"
                        self.root.after(0, lambda t=live_text: self.add_result(t, "result"))
                
                except Exception as e:
                    logger.error(f"Transcription chunk failed: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Recording worker error: {e}")
                break
    
    def on_processing_complete(self, transcription, processing_time, rtf):
        """Called when file processing is complete"""
        self.add_result(transcription, "result")
        
        # Update metrics
        self.proc_time_var.set(f"{processing_time:.2f}s")
        self.rtf_var.set(f"{rtf:.2f}x")
        self.npu_util_var.set("Active" if self.npu_integration and self.npu_integration.acceleration_enabled else "Disabled")
        
        # Reset UI
        self.progress_var.set(0)
        self.process_btn.config(state=tk.NORMAL)
        self.is_processing = False
        
        messagebox.showinfo("Complete", "Audio processing completed successfully!")
    
    def on_processing_error(self, error_msg):
        """Called when processing fails"""
        self.add_result(f"❌ Processing failed: {error_msg}\n", "error")
        self.progress_var.set(0)
        self.process_btn.config(state=tk.NORMAL)
        self.is_processing = False
        
        messagebox.showerror("Error", f"Processing failed: {error_msg}")
    
    def add_result(self, text, result_type="normal"):
        """Add text to results area with formatting"""
        self.results_text.config(state=tk.NORMAL)
        
        # Configure tags for different result types
        if not hasattr(self, '_tags_configured'):
            self.results_text.tag_configure("success", foreground="green")
            self.results_text.tag_configure("error", foreground="red")
            self.results_text.tag_configure("info", foreground="blue")
            self.results_text.tag_configure("result", foreground="black", font=('Courier', 9))
            self._tags_configured = True
        
        # Insert text with appropriate tag
        self.results_text.insert(tk.END, text, result_type)
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear results text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def save_results(self):
        """Save results to file"""
        if not self.results_text.get(1.0, tk.END).strip():
            messagebox.showwarning("No Results", "No results to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Saved", f"Results saved to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def copy_results(self):
        """Copy results to clipboard"""
        results = self.results_text.get(1.0, tk.END).strip()
        if results:
            self.root.clipboard_clear()
            self.root.clipboard_append(results)
            messagebox.showinfo("Copied", "Results copied to clipboard!")
        else:
            messagebox.showwarning("No Results", "No results to copy.")


def main():
    """Main function to run the working GUI application"""
    root = tk.Tk()
    app = WhisperXNPUGUIWorking(root)
    
    # Handle window closing
    def on_closing():
        if app.is_processing or app.is_recording:
            if messagebox.askokcancel("Quit", "Processing/recording is in progress. Do you want to quit?"):
                if app.is_recording:
                    app.stop_recording()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()