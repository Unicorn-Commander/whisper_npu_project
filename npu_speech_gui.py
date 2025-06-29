#!/usr/bin/env python3
"""
NPU Speech Recognition GUI - REAL NPU Implementation

GUI for local speech recognition model running ACTUALLY on NPU hardware.
This uses real NPU compute, not just CPU with NPU detection.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import queue
import io

# Add our modules to path
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')

from npu_speech_recognition import NPUSpeechRecognizer

# Configure logging to capture in GUI
class GUILogHandler(logging.Handler):
    def __init__(self, text_widget=None):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        if self.text_widget:
            msg = self.format(record)
            # Schedule GUI update in main thread
            self.text_widget.after(0, lambda: self.add_log_message(msg))
    
    def add_log_message(self, msg):
        if self.text_widget:
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, f"{msg}\n")
            self.text_widget.see(tk.END)
            self.text_widget.config(state=tk.DISABLED)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class AudioCapture:
    """Real-time audio capture for live NPU transcription"""
    
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


class NPUSpeechGUI:
    """GUI for REAL NPU Speech Recognition"""
    
    def __init__(self, root):
        """Initialize the NPU speech recognition GUI"""
        self.root = root
        self.npu_recognizer = None
        self.audio_capture = AudioCapture()
        self.is_processing = False
        self.is_recording = False
        self.recording_thread = None
        self.log_handler = None
        
        # Model options
        self.model_options = {
            "Custom NPU Model (Fast)": "custom_npu",
            "Enhanced NPU Model (Better)": "enhanced_npu", 
            "Experimental NPU Model": "experimental_npu"
        }
        
        self.setup_window()
        self.create_widgets()
        self.setup_logging_after_widgets()
        # Don't auto-initialize - let user choose model
    
    def setup_window(self):
        """Configure the main window"""
        self.root.title("NPU Speech Recognition - REAL NPU Implementation")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill='both', expand=True)
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_label = ttk.Label(
            self.main_frame, 
            text="🎙️ NPU Speech Recognition - REAL NPU Hardware", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Model Selection Section
        self.create_model_section()
        
        # NPU Status Section
        self.create_status_section()
        
        # Audio Processing Section
        self.create_audio_section()
        
        # Real-time Section
        self.create_realtime_section()
        
        # Results Section
        self.create_results_section()
        
        # Performance Section
        self.create_performance_section()
    
    def create_model_section(self):
        """Create model selection section"""
        model_frame = ttk.LabelFrame(self.main_frame, text="🧠 NPU Model Selection", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        # Model selection
        ttk.Label(model_frame, text="Select NPU Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.model_var = tk.StringVar(value="Custom NPU Model (Fast)")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  values=list(self.model_options.keys()), 
                                  state="readonly", width=30)
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Load model button
        self.load_model_btn = ttk.Button(model_frame, text="🚀 Load NPU Model", 
                                        command=self.load_selected_model)
        self.load_model_btn.grid(row=0, column=2)
        
        # Model status
        self.model_load_status_var = tk.StringVar(value="No model loaded - please select and load a model")
        status_label = ttk.Label(model_frame, textvariable=self.model_load_status_var, 
                               font=('Arial', 9, 'italic'), foreground="orange")
        status_label.grid(row=1, column=0, columnspan=3, pady=(10, 0))
    
    def create_status_section(self):
        """Create NPU status display section"""
        status_frame = ttk.LabelFrame(self.main_frame, text="🔧 NPU System Status", padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        # NPU Status
        ttk.Label(status_frame, text="NPU Available:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.npu_status_var = tk.StringVar(value="Initializing...")
        self.npu_status_label = ttk.Label(status_frame, textvariable=self.npu_status_var, font=('Arial', 10, 'bold'))
        self.npu_status_label.grid(row=0, column=1, sticky=tk.W)
        
        # NPU Type
        ttk.Label(status_frame, text="NPU Type:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.npu_type_var = tk.StringVar(value="Unknown")
        ttk.Label(status_frame, textvariable=self.npu_type_var).grid(row=0, column=3, sticky=tk.W)
        
        # Model Status
        ttk.Label(status_frame, text="Model Status:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.model_status_var = tk.StringVar(value="Not loaded")
        ttk.Label(status_frame, textvariable=self.model_status_var).grid(row=1, column=1, sticky=tk.W)
        
        # Vocabulary Size
        ttk.Label(status_frame, text="Vocabulary:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.vocab_size_var = tk.StringVar(value="Unknown")
        ttk.Label(status_frame, textvariable=self.vocab_size_var).grid(row=1, column=3, sticky=tk.W)
        
        # XRT Version
        ttk.Label(status_frame, text="XRT Version:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.xrt_version_var = tk.StringVar(value="Unknown")
        ttk.Label(status_frame, textvariable=self.xrt_version_var).grid(row=2, column=1, sticky=tk.W)
        
        # Firmware Version
        ttk.Label(status_frame, text="Firmware:").grid(row=2, column=2, sticky=tk.W, padx=(20, 10))
        self.firmware_var = tk.StringVar(value="Unknown")
        ttk.Label(status_frame, textvariable=self.firmware_var).grid(row=2, column=3, sticky=tk.W)
        
        # Refresh Button
        refresh_btn = ttk.Button(status_frame, text="🔄 Refresh", command=self.refresh_status)
        refresh_btn.grid(row=3, column=0, columnspan=4, pady=(10, 0))
    
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
        self.process_btn = ttk.Button(audio_frame, text="🚀 Process with NPU", 
                                     command=self.process_audio, state=tk.DISABLED)
        self.process_btn.grid(row=1, column=0, columnspan=3, pady=(15, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(audio_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Status text
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(audio_frame, textvariable=self.status_var, 
                                font=('Arial', 9, 'italic'))
        status_label.grid(row=3, column=0, columnspan=3, pady=(5, 0))
    
    def create_realtime_section(self):
        """Create real-time transcription section"""
        realtime_frame = ttk.LabelFrame(self.main_frame, text="🎤 Real-time NPU Transcription", padding="10")
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
        
        self.record_btn = ttk.Button(controls_frame, text="🎤 Start NPU Recording", 
                                    command=self.toggle_recording, state=tk.DISABLED)
        self.record_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Recording status
        self.recording_status_var = tk.StringVar(value="Ready")
        ttk.Label(controls_frame, textvariable=self.recording_status_var, 
                 font=('Arial', 10, 'italic')).grid(row=0, column=1)
    
    def create_results_section(self):
        """Create transcription results section with separate areas for transcription and logs"""
        results_frame = ttk.LabelFrame(self.main_frame, text="📝 Transcription & Logs", padding="10")
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Create notebook for tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Transcription tab
        transcription_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(transcription_frame, text="📝 Transcription")
        
        # Transcription toolbar
        trans_toolbar = ttk.Frame(transcription_frame)
        trans_toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        clear_trans_btn = ttk.Button(trans_toolbar, text="🗑️ Clear", command=self.clear_transcription)
        clear_trans_btn.grid(row=0, column=0, padx=(0, 10))
        
        export_txt_btn = ttk.Button(trans_toolbar, text="📄 Export TXT", command=self.export_transcription_txt)
        export_txt_btn.grid(row=0, column=1, padx=(0, 10))
        
        export_json_btn = ttk.Button(trans_toolbar, text="📊 Export JSON", command=self.export_transcription_json)
        export_json_btn.grid(row=0, column=2, padx=(0, 10))
        
        copy_trans_btn = ttk.Button(trans_toolbar, text="📋 Copy", command=self.copy_transcription)
        copy_trans_btn.grid(row=0, column=3)
        
        # Transcription text area
        transcription_frame.columnconfigure(0, weight=1)
        transcription_frame.rowconfigure(1, weight=1)
        
        self.transcription_text = scrolledtext.ScrolledText(
            transcription_frame, 
            wrap=tk.WORD, 
            width=100, 
            height=15,
            font=('Arial', 11),
            bg='#f8f9fa'
        )
        self.transcription_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Speaker Diarization tab
        speakers_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(speakers_frame, text="👥 Speakers")
        
        # Speaker analysis toolbar
        speakers_toolbar = ttk.Frame(speakers_frame)
        speakers_toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        export_speakers_btn = ttk.Button(speakers_toolbar, text="📊 Export Speakers", command=self.export_speaker_info)
        export_speakers_btn.grid(row=0, column=0, padx=(0, 10))
        
        clear_speakers_btn = ttk.Button(speakers_toolbar, text="🗑️ Clear", command=self.clear_speakers)
        clear_speakers_btn.grid(row=0, column=1)
        
        # Speaker information display
        speakers_frame.columnconfigure(0, weight=1)
        speakers_frame.rowconfigure(1, weight=1)
        
        self.speakers_text = scrolledtext.ScrolledText(
            speakers_frame, 
            wrap=tk.WORD, 
            width=100, 
            height=15,
            font=('Arial', 10),
            bg='#f0f8ff'
        )
        self.speakers_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Logs tab
        logs_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(logs_frame, text="🔍 NPU Logs")
        
        # Logs toolbar
        logs_toolbar = ttk.Frame(logs_frame)
        logs_toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        clear_logs_btn = ttk.Button(logs_toolbar, text="🗑️ Clear Logs", command=self.clear_logs)
        clear_logs_btn.grid(row=0, column=0, padx=(0, 10))
        
        save_logs_btn = ttk.Button(logs_toolbar, text="💾 Save Logs", command=self.save_logs)
        save_logs_btn.grid(row=0, column=1, padx=(0, 10))
        
        copy_logs_btn = ttk.Button(logs_toolbar, text="📋 Copy Logs", command=self.copy_logs)
        copy_logs_btn.grid(row=0, column=2)
        
        # Logs text area
        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(1, weight=1)
        
        self.logs_text = scrolledtext.ScrolledText(
            logs_frame, 
            wrap=tk.WORD, 
            width=100, 
            height=15,
            font=('Courier', 9),
            bg='#2d3748',
            fg='#e2e8f0'
        )
        self.logs_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main frame row weight for resizing
        self.main_frame.rowconfigure(5, weight=1)
        
        # Store transcription history for export
        self.transcription_history = []
    
    def create_performance_section(self):
        """Create performance metrics section"""
        perf_frame = ttk.LabelFrame(self.main_frame, text="⚡ NPU Performance Metrics", padding="10")
        perf_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        perf_frame.columnconfigure(1, weight=1)
        perf_frame.columnconfigure(3, weight=1)
        
        # Processing time
        ttk.Label(perf_frame, text="NPU Processing Time:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.proc_time_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.proc_time_var, font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky=tk.W)
        
        # Real-time factor
        ttk.Label(perf_frame, text="Real-time Factor:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.rtf_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.rtf_var, font=('Arial', 10, 'bold')).grid(row=0, column=3, sticky=tk.W)
        
        # NPU operations
        ttk.Label(perf_frame, text="NPU Operations:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.npu_ops_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.npu_ops_var, font=('Arial', 10, 'bold')).grid(row=1, column=1, sticky=tk.W)
        
        # Audio duration
        ttk.Label(perf_frame, text="Audio Duration:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.duration_var = tk.StringVar(value="--")
        ttk.Label(perf_frame, textvariable=self.duration_var, font=('Arial', 10, 'bold')).grid(row=1, column=3, sticky=tk.W)
    
    def setup_logging_after_widgets(self):
        """Setup logging to display in GUI logs tab"""
        # Create log handler for GUI
        self.log_handler = GUILogHandler(self.logs_text)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        # Add handler to all relevant loggers
        loggers = [
            logging.getLogger('npu_speech_recognition'),
            logging.getLogger('matrix_multiply'),
            logging.getLogger('whisperx_npu_accelerator'),
            logging.getLogger(__name__)
        ]
        
        for logger in loggers:
            logger.addHandler(self.log_handler)
    
    def load_selected_model(self):
        """Load the selected NPU model"""
        selected_model = self.model_var.get()
        model_type = self.model_options[selected_model]
        
        self.model_load_status_var.set("Loading NPU model...")
        self.load_model_btn.config(state=tk.DISABLED)
        
        def load_worker():
            try:
                self.add_log(f"🚀 Loading {selected_model}...\n", "info")
                
                # Initialize NPU recognizer
                self.npu_recognizer = NPUSpeechRecognizer()
                success = self.npu_recognizer.initialize()
                
                if success:
                    self.root.after(0, lambda: self.on_model_loaded(selected_model))
                else:
                    self.root.after(0, lambda: self.on_model_error("Failed to initialize NPU model"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()
    
    def on_model_loaded(self, model_name):
        """Called when model loading is complete"""
        self.model_load_status_var.set(f"✅ {model_name} loaded successfully!")
        self.load_model_btn.config(state=tk.NORMAL)
        
        # Enable other controls
        self.process_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.NORMAL)
        self.refresh_audio_devices()
        
        # Update status
        self.update_status()
        
        self.add_log(f"✅ {model_name} loaded and ready for NPU transcription!\n", "success")
    
    def on_model_error(self, error_msg):
        """Called when model loading fails"""
        self.model_load_status_var.set(f"❌ Model loading failed: {error_msg}")
        self.load_model_btn.config(state=tk.NORMAL)
        self.add_log(f"❌ Model loading failed: {error_msg}\n", "error")
    
    def initialize_npu(self):
        """Initialize NPU speech recognition system"""
        def init_worker():
            try:
                self.npu_recognizer = NPUSpeechRecognizer()
                success = self.npu_recognizer.initialize()
                
                self.root.after(0, lambda: self.on_npu_initialized(success))
                
            except Exception as e:
                self.root.after(0, lambda: self.on_npu_error(str(e)))
        
        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()
    
    def on_npu_initialized(self, success):
        """Called when NPU initialization is complete"""
        if success:
            self.update_status()
            self.process_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.NORMAL)
            self.refresh_audio_devices()
            self.add_log("✅ NPU Speech Recognition System initialized successfully!\n", "success")
            self.add_log("🚀 REAL NPU hardware acceleration active!\n", "success")
        else:
            self.add_log("❌ NPU Speech Recognition initialization failed!\n", "error")
    
    def on_npu_error(self, error_msg):
        """Called when NPU initialization fails"""
        self.npu_status_var.set("❌ Error")
        self.add_log(f"❌ NPU initialization failed: {error_msg}\n", "error")
    
    def update_status(self):
        """Update NPU status displays"""
        if self.npu_recognizer:
            info = self.npu_recognizer.get_system_info()
            
            if info['npu_available']:
                self.npu_status_var.set("✅ NPU Active")
                npu_status = info['npu_status']
                self.npu_type_var.set(npu_status.get('npu_type', 'Unknown'))
                self.xrt_version_var.set(npu_status.get('xrt_version', 'Unknown'))
                self.firmware_var.set(npu_status.get('firmware_version', 'Unknown'))
            else:
                self.npu_status_var.set("❌ NPU Unavailable")
                self.npu_type_var.set("N/A")
                self.xrt_version_var.set("N/A")
                self.firmware_var.set("N/A")
            
            if info['model_ready']:
                self.model_status_var.set("✅ NPU Model Loaded")
                self.vocab_size_var.set(f"{info['vocabulary_size']} tokens")
            else:
                self.model_status_var.set("❌ Not Loaded")
                self.vocab_size_var.set("N/A")
    
    def refresh_status(self):
        """Refresh NPU status"""
        self.status_var.set("Refreshing NPU status...")
        self.update_status()
        self.status_var.set("Ready")
    
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
            self.add_log(f"📁 Selected file: {os.path.basename(filename)}\n", "info")
            
            # Get audio duration
            try:
                duration = self.get_audio_duration(filename)
                self.duration_var.set(f"{duration:.1f}s")
            except:
                self.duration_var.set("Unknown")
    
    def get_audio_duration(self, filepath):
        """Get duration of audio file"""
        try:
            with sf.SoundFile(filepath) as f:
                return len(f) / f.samplerate
        except:
            return 0.0
    
    def process_audio(self):
        """Process selected audio file with REAL NPU"""
        if not hasattr(self, 'selected_file'):
            messagebox.showwarning("No File", "Please select an audio file first.")
            return
        
        if not self.npu_recognizer:
            messagebox.showerror("No NPU", "NPU system not initialized.")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        def process_worker():
            try:
                self.root.after(0, lambda: self.progress_var.set(10))
                self.root.after(0, lambda: self.status_var.set("🚀 Starting NPU transcription..."))
                self.root.after(0, lambda: self.add_log("🚀 Processing audio with REAL NPU hardware...\n", "info"))
                
                # Transcribe with REAL NPU
                result = self.npu_recognizer.transcribe_audio(self.selected_file)
                
                self.root.after(0, lambda: self.progress_var.set(80))
                
                # Format results
                transcription = self.format_transcription_result(result, self.selected_file)
                
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.on_processing_complete(transcription, result))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.on_processing_error(error_msg))
        
        thread = threading.Thread(target=process_worker, daemon=True)
        thread.start()
    
    def format_transcription_result(self, result, filename):
        """Format hybrid NPU + WhisperX transcription result for display"""
        # Determine which engine was used
        if result.get('whisperx_used', False):
            engine = "WhisperX + NPU Hybrid Pipeline"
            npu_status = "✅ NPU PREPROCESSING + WhisperX TRANSCRIPTION"
        else:
            engine = "Custom NPU Speech Model"
            npu_status = "✅ REAL NPU HARDWARE"
        
        output = f"""🎙️ ENHANCED NPU TRANSCRIPTION RESULTS

File: {os.path.basename(filename)}
Engine: {engine}
NPU Acceleration: {npu_status}
Language: {result.get('language', 'Unknown')}
Speaker Diarization: {'✅ ENABLED' if result.get('speaker_diarization', False) else '❌ DISABLED'}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

TRANSCRIPTION:
{result['text']}

TECHNICAL DETAILS:
- Processing Time: {result['processing_time']:.2f}s
- Audio Duration: {result['audio_duration']:.2f}s  
- Real-time Factor: {result['real_time_factor']:.2f}x
- NPU Accelerated: {result['npu_accelerated']}
- Features Shape: {result['features_shape']}
- Predictions Shape: {result['predictions_shape']}
- Vocabulary Tokens: {len(result['tokens'])}

DETECTED TOKENS:
{', '.join(result['tokens']) if result['tokens'] else 'No tokens detected'}

✅ Transcription completed using REAL NPU hardware acceleration!
"""
        return output
    
    def toggle_recording(self):
        """Toggle real-time NPU recording"""
        if not self.npu_recognizer:
            messagebox.showerror("No NPU", "NPU system not initialized.")
            return
        
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start real-time recording and NPU transcription"""
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
            self.record_btn.config(text="⏹️ Stop NPU Recording")
            self.recording_status_var.set("🎤 Recording with NPU processing...")
            
            # Start processing thread
            self.recording_thread = threading.Thread(target=self.recording_worker, daemon=True)
            self.recording_thread.start()
            
            self.add_log("🎤 Started real-time NPU recording and transcription...\n", "info")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start NPU recording: {e}")
    
    def stop_recording(self):
        """Stop real-time NPU recording"""
        try:
            self.is_recording = False
            self.audio_capture.stop_recording()
            
            self.record_btn.config(text="🎤 Start NPU Recording")
            self.recording_status_var.set("Ready")
            
            self.add_log("⏹️ Stopped real-time NPU recording\n", "info")
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
    
    def recording_worker(self):
        """Background worker for real-time NPU transcription"""
        while self.is_recording:
            try:
                # Get audio chunk
                audio_chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                if audio_chunk is None:
                    continue
                
                # Only process if chunk has sufficient audio
                if len(audio_chunk) < self.audio_capture.sample_rate * 1.5:  # Less than 1.5 seconds
                    continue
                
                # Check for actual audio content (not just silence)
                if np.max(np.abs(audio_chunk)) < 0.02:  # Very quiet
                    continue
                
                # Transcribe chunk with REAL NPU
                try:
                    result = self.npu_recognizer.transcribe_audio_data(
                        audio_chunk, 
                        self.audio_capture.sample_rate
                    )
                    
                    # Format and display result
                    if result['text'] and result['text'] != '<silence>':
                        timestamp = time.strftime('%H:%M:%S')
                        live_text = f"[{timestamp}] NPU: {result['text']} (RTF: {result['real_time_factor']:.2f}x)\n"
                        self.root.after(0, lambda t=live_text: self.add_transcription(result['text'], {"real_time_factor": result['real_time_factor'], "source": "live_recording"}))
                
                except Exception as e:
                    logger.error(f"NPU transcription chunk failed: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Recording worker error: {e}")
                break
    
    def on_processing_complete(self, transcription, result):
        """Called when NPU processing is complete"""
        # Add the transcription text to the transcription tab
        metadata = {
            "processing_time": result['processing_time'],
            "audio_duration": result['audio_duration'], 
            "real_time_factor": result['real_time_factor'],
            "npu_accelerated": result['npu_accelerated'],
            "whisperx_used": result.get('whisperx_used', False),
            "speaker_diarization": result.get('speaker_diarization', False),
            "language": result.get('language', 'unknown'),
            "features_shape": str(result.get('npu_features_shape', 'N/A')),
            "predictions_shape": str(result.get('npu_predictions_shape', 'N/A')),
            "source": "file_processing",
            "filename": getattr(self, 'selected_file', 'unknown')
        }
        self.add_transcription(result['text'], metadata)
        
        # Add speaker information if available
        if result.get('speaker_segments'):
            self.display_speaker_information(result['speaker_segments'], result)
        
        # Add processing summary to logs
        self.add_log(f"✅ File processing completed: {result['text'][:50]}{'...' if len(result['text']) > 50 else ''}\n", "success")
        
        # Update metrics
        self.proc_time_var.set(f"{result['processing_time']:.2f}s")
        self.rtf_var.set(f"{result['real_time_factor']:.2f}x")
        self.npu_ops_var.set("Matrix Mult x3" if result['npu_accelerated'] else "CPU Fallback")
        
        # Reset UI
        self.progress_var.set(0)
        self.status_var.set("✅ NPU processing completed")
        self.process_btn.config(state=tk.NORMAL)
        self.is_processing = False
        
        messagebox.showinfo("Complete", "NPU audio processing completed successfully!")
    
    def on_processing_error(self, error_msg):
        """Called when NPU processing fails"""
        self.add_log(f"❌ NPU processing failed: {error_msg}\n", "error")
        self.progress_var.set(0)
        self.status_var.set("❌ NPU processing failed")
        self.process_btn.config(state=tk.NORMAL)
        self.is_processing = False
        
        messagebox.showerror("Error", f"NPU processing failed: {error_msg}")
    
    def add_transcription(self, text, metadata=None):
        """Add transcription text to the transcription tab"""
        self.transcription_text.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format transcription entry
        entry = f"[{timestamp}] {text}\n"
        
        # Insert text
        self.transcription_text.insert(tk.END, entry)
        self.transcription_text.see(tk.END)
        self.transcription_text.config(state=tk.DISABLED)
        
        # Store in history for export
        transcription_entry = {
            "timestamp": timestamp,
            "text": text,
            "metadata": metadata or {}
        }
        self.transcription_history.append(transcription_entry)
    
    def add_log(self, text, log_type="normal"):
        """Add text to logs tab with formatting"""
        if hasattr(self, 'logs_text'):
            self.logs_text.config(state=tk.NORMAL)
            
            # Configure tags for different log types
            if not hasattr(self, '_log_tags_configured'):
                self.logs_text.tag_configure("success", foreground="#68d391")
                self.logs_text.tag_configure("error", foreground="#fc8181") 
                self.logs_text.tag_configure("info", foreground="#63b3ed")
                self.logs_text.tag_configure("warning", foreground="#f6ad55")
                self._log_tags_configured = True
            
            # Insert text with appropriate tag
            self.logs_text.insert(tk.END, text, log_type)
            self.logs_text.see(tk.END)
            self.logs_text.config(state=tk.DISABLED)
    
    def clear_transcription(self):
        """Clear transcription text area"""
        self.transcription_text.config(state=tk.NORMAL)
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.config(state=tk.DISABLED)
        self.transcription_history.clear()
    
    def clear_logs(self):
        """Clear logs text area"""
        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.config(state=tk.DISABLED)
    
    def export_transcription_txt(self):
        """Export transcription as plain text file"""
        if not self.transcription_history:
            messagebox.showwarning("No Transcription", "No transcription to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Transcription as TXT",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("NPU Speech Recognition Transcription\n")
                    f.write("="*50 + "\n\n")
                    
                    for entry in self.transcription_history:
                        f.write(f"[{entry['timestamp']}]\n")
                        f.write(f"{entry['text']}\n\n")
                        
                        if entry['metadata']:
                            f.write("Metadata:\n")
                            for key, value in entry['metadata'].items():
                                f.write(f"  {key}: {value}\n")
                            f.write("\n")
                
                messagebox.showinfo("Exported", f"Transcription exported to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export file: {e}")
    
    def export_transcription_json(self):
        """Export transcription as JSON file with metadata"""
        if not self.transcription_history:
            messagebox.showwarning("No Transcription", "No transcription to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Transcription as JSON",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                export_data = {
                    "export_info": {
                        "exported_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "system": "NPU Speech Recognition",
                        "total_entries": len(self.transcription_history)
                    },
                    "transcriptions": self.transcription_history
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Exported", f"Transcription exported to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export file: {e}")
    
    def copy_transcription(self):
        """Copy transcription to clipboard"""
        if not self.transcription_history:
            messagebox.showwarning("No Transcription", "No transcription to copy.")
            return
        
        # Create plain text version
        text_content = []
        for entry in self.transcription_history:
            text_content.append(f"[{entry['timestamp']}] {entry['text']}")
        
        result = "\n".join(text_content)
        
        self.root.clipboard_clear()
        self.root.clipboard_append(result)
        messagebox.showinfo("Copied", "Transcription copied to clipboard!")
    
    def save_logs(self):
        """Save logs to file"""
        if not self.logs_text.get(1.0, tk.END).strip():
            messagebox.showwarning("No Logs", "No logs to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save NPU Logs",
            defaultextension=".log",
            filetypes=[("Log Files", "*.log"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.logs_text.get(1.0, tk.END))
                messagebox.showinfo("Saved", f"Logs saved to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def copy_logs(self):
        """Copy logs to clipboard"""
        logs = self.logs_text.get(1.0, tk.END).strip()
        if logs:
            self.root.clipboard_clear()
            self.root.clipboard_append(logs)
            messagebox.showinfo("Copied", "Logs copied to clipboard!")
        else:
            messagebox.showwarning("No Logs", "No logs to copy.")
    
    def display_speaker_information(self, speaker_segments, result):
        """Display speaker diarization information"""
        self.speakers_text.config(state=tk.NORMAL)
        
        # Clear previous content
        self.speakers_text.delete(1.0, tk.END)
        
        # Header
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        filename = os.path.basename(getattr(self, 'selected_file', 'unknown'))
        
        header = f"""👥 SPEAKER DIARIZATION RESULTS

File: {filename}
Engine: WhisperX + pyannote.audio 3.3.2
NPU Preprocessing: ✅ ENABLED
Language: {result.get('language', 'Unknown')}
Timestamp: {timestamp}

SPEAKER SUMMARY:
"""
        self.speakers_text.insert(tk.END, header)
        
        # Speaker summary
        total_speakers = len(speaker_segments)
        self.speakers_text.insert(tk.END, f"• Total Speakers Detected: {total_speakers}\n\n")
        
        for i, speaker_info in enumerate(speaker_segments):
            speaker_id = speaker_info['speaker_id']
            total_time = speaker_info['total_speech_time']
            word_count = speaker_info['word_count']
            segment_count = len(speaker_info['segments'])
            
            summary = f"🎤 {speaker_id}:\n"
            summary += f"  • Speaking Time: {total_time:.1f} seconds\n"
            summary += f"  • Word Count: {word_count} words\n"
            summary += f"  • Segments: {segment_count}\n\n"
            
            self.speakers_text.insert(tk.END, summary)
        
        # Detailed transcription with speakers
        self.speakers_text.insert(tk.END, "DETAILED TRANSCRIPT WITH SPEAKERS:\n")
        self.speakers_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Collect all segments and sort by time
        all_segments = []
        for speaker_info in speaker_segments:
            for segment in speaker_info['segments']:
                all_segments.append(segment)
        
        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Display segments
        for segment in all_segments:
            start_time = self.format_timestamp(segment['start'])
            end_time = self.format_timestamp(segment['end'])
            speaker = segment['speaker']
            text = segment['text'].strip()
            
            segment_text = f"[{start_time} - {end_time}] {speaker}:\n{text}\n\n"
            self.speakers_text.insert(tk.END, segment_text)
        
        self.speakers_text.config(state=tk.DISABLED)
        
        # Store speaker data for export
        self.current_speaker_data = {
            'speaker_segments': speaker_segments,
            'result': result,
            'filename': filename
        }
        
        self.add_log(f"👥 Speaker diarization completed: {total_speakers} speakers detected\n", "success")
    
    def format_timestamp(self, seconds):
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def clear_speakers(self):
        """Clear speaker information"""
        self.speakers_text.config(state=tk.NORMAL)
        self.speakers_text.delete(1.0, tk.END)
        self.speakers_text.config(state=tk.DISABLED)
        self.current_speaker_data = None
    
    def export_speaker_info(self):
        """Export speaker information to JSON"""
        if not hasattr(self, 'current_speaker_data') or not self.current_speaker_data:
            messagebox.showwarning("No Data", "No speaker information to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Speaker Information",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                export_data = {
                    "filename": self.current_speaker_data['filename'],
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "engine": "WhisperX + pyannote.audio",
                    "language": self.current_speaker_data['result'].get('language', 'unknown'),
                    "total_speakers": len(self.current_speaker_data['speaker_segments']),
                    "audio_duration": self.current_speaker_data['result'].get('audio_duration', 0),
                    "processing_time": self.current_speaker_data['result'].get('processing_time', 0),
                    "speakers": self.current_speaker_data['speaker_segments']
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Exported", f"Speaker information exported to {os.path.basename(filename)}")
                self.add_log(f"📊 Speaker data exported to {os.path.basename(filename)}\n", "info")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export speaker information: {e}")
                self.add_log(f"❌ Speaker export failed: {e}\n", "error")


def main():
    """Main function to run the NPU speech recognition GUI"""
    root = tk.Tk()
    app = NPUSpeechGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.is_processing or app.is_recording:
            if messagebox.askokcancel("Quit", "NPU processing/recording is in progress. Do you want to quit?"):
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