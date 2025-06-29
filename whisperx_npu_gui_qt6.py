#!/usr/bin/env python3
"""
NPU Always-Listening Voice Assistant GUI - Qt6/PySide6 Version
Compatible with KDE6/Qt6/Wayland environment
Complete GUI for testing, configuration, and real-time transcription
"""

import sys
import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project path
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')

# Qt6/PySide6 imports
try:
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *
    QT6_AVAILABLE = True
    print("✅ Using PySide6/Qt6 for KDE6/Wayland compatibility")
except ImportError:
    print("❌ PySide6 not available")
    sys.exit(1)

# Import our NPU systems
try:
    from always_listening_npu import AlwaysListeningNPU
    from onnx_whisper_npu import ONNXWhisperNPU
    from whisperx_npu_accelerator import NPUAccelerator
    print("✅ NPU modules loaded")
except ImportError as e:
    print(f"⚠️ Some NPU modules not available: {e}")

class AlwaysListeningGUI(QMainWindow):
    """Qt6 GUI for NPU Always-Listening Voice Assistant"""
    
    # Qt6 Signals
    transcription_received = Signal(dict)
    status_updated = Signal(str)
    initialization_completed = Signal(bool)
    
    def __init__(self):
        """Initialize the Qt6 GUI application"""
        super().__init__()
        
        # Initialize NPU systems
        try:
            self.always_listening_system = AlwaysListeningNPU()
            self.onnx_whisper = ONNXWhisperNPU()
            self.npu_accelerator = NPUAccelerator()
        except Exception as e:
            print(f"⚠️ NPU system initialization warning: {e}")
            self.always_listening_system = None
            self.onnx_whisper = None
            self.npu_accelerator = None
        
        # GUI state
        self.is_always_listening = False
        self.is_processing = False
        self.current_results = []
        
        # Connect signals
        self.transcription_received.connect(self.on_transcription_result)
        self.status_updated.connect(self.update_status_display)
        self.initialization_completed.connect(self.on_initialization_complete)
        
        # Initialize GUI
        self.init_gui()
        
    def init_gui(self):
        """Initialize the Qt6 GUI"""
        self.setWindowTitle("NPU Always-Listening Voice Assistant - Qt6/KDE6")
        self.setGeometry(100, 100, 1200, 900)
        
        # Apply Qt6/KDE6 compatible styling
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #2b2b2b; 
                color: #ffffff; 
            }
            QTabWidget::pane { 
                border: 1px solid #555; 
                background-color: #2b2b2b; 
            }
            QTabBar::tab { 
                background-color: #444; 
                color: #fff; 
                padding: 12px 20px; 
                margin: 2px; 
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected { 
                background-color: #666; 
                font-weight: bold;
            }
            QPushButton { 
                background-color: #4a90e2; 
                color: white; 
                border: none; 
                padding: 12px 20px; 
                border-radius: 6px; 
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { 
                background-color: #357abd; 
            }
            QPushButton:pressed {
                background-color: #2a5a8d;
            }
            QPushButton:disabled { 
                background-color: #666; 
                color: #999; 
            }
            QTextEdit { 
                background-color: #1e1e1e; 
                color: #ffffff; 
                border: 2px solid #555; 
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            QLabel { 
                color: #ffffff; 
                font-size: 13px;
            }
            QComboBox { 
                background-color: #444; 
                color: #fff; 
                border: 2px solid #555; 
                padding: 8px; 
                border-radius: 4px;
            }
            QLineEdit {
                background-color: #444;
                color: #fff;
                border: 2px solid #555;
                padding: 8px;
                border-radius: 4px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_always_listening_tab()
        self.create_single_file_tab()
        self.create_configuration_tab()
        self.create_system_status_tab()
        
    def create_always_listening_tab(self):
        """Create the always-listening interface tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("🎤 NPU Always-Listening Voice Assistant")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #4a90e2; padding: 15px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Quick configuration
        config_group = QGroupBox("Quick Configuration")
        config_layout = QGridLayout(config_group)
        
        # Activation mode
        config_layout.addWidget(QLabel("Activation Mode:"), 0, 0)
        self.activation_mode = QComboBox()
        self.activation_mode.addItems(["wake_word", "vad_only", "always_on"])
        self.activation_mode.setCurrentText("wake_word")
        config_layout.addWidget(self.activation_mode, 0, 1)
        
        # Wake words
        config_layout.addWidget(QLabel("Wake Words:"), 1, 0)
        self.wake_words_input = QLineEdit("hey_jarvis, computer, assistant")
        config_layout.addWidget(self.wake_words_input, 1, 1)
        
        # Whisper model
        config_layout.addWidget(QLabel("Whisper Model:"), 2, 0)
        self.whisper_model = QComboBox()
        self.whisper_model.addItems(["base", "tiny", "small"])
        config_layout.addWidget(self.whisper_model, 2, 1)
        
        layout.addWidget(config_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.init_button = QPushButton("🚀 Initialize System")
        self.init_button.clicked.connect(self.initialize_always_listening)
        control_layout.addWidget(self.init_button)
        
        self.start_button = QPushButton("🎤 Start Always Listening")
        self.start_button.clicked.connect(self.start_always_listening)
        self.start_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("🔇 Stop Listening")
        self.stop_button.clicked.connect(self.stop_always_listening)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        layout.addLayout(control_layout)
        
        # Status indicators
        status_group = QGroupBox("Live Status Indicators")
        status_layout = QGridLayout(status_group)
        
        self.vad_status = QLabel("VAD: ⭕ Inactive")
        self.wake_word_status = QLabel("Wake Word: ⭕ Inactive")
        self.recording_status = QLabel("Recording: ⭕ Not Recording")
        self.processing_status = QLabel("Processing: ⭕ Idle")
        
        # Style status labels
        status_style = "font-size: 14px; font-weight: bold; padding: 8px; border-radius: 4px; background-color: #333;"
        for label in [self.vad_status, self.wake_word_status, self.recording_status, self.processing_status]:
            label.setStyleSheet(status_style)
        
        status_layout.addWidget(self.vad_status, 0, 0)
        status_layout.addWidget(self.wake_word_status, 0, 1)
        status_layout.addWidget(self.recording_status, 1, 0)
        status_layout.addWidget(self.processing_status, 1, 1)
        
        layout.addWidget(status_group)
        
        # Live transcription results
        results_group = QGroupBox("Live Transcription Results")
        results_layout = QVBoxLayout(results_group)
        
        self.live_results = QTextEdit()
        self.live_results.setPlainText("""🎤 NPU ALWAYS-LISTENING VOICE ASSISTANT - Qt6/KDE6 VERSION

🚀 READY FOR OPERATION!

Instructions:
1. Click 'Initialize System' to set up all NPU components
2. Configure your preferences (activation mode, wake words, model)  
3. Click 'Start Always Listening' to begin continuous monitoring
4. Say your wake word (e.g., 'hey jarvis') to activate transcription
5. Speak your message - recording will auto-start and stop
6. View transcription results here in real-time

🧠 NPU Components Available:
- 🎤 Silero VAD: Continuous voice activity detection (<1W power)
- 🎯 OpenWakeWord: Natural wake word activation on NPU
- 🧠 ONNX Whisper: High-speed NPU transcription (10-45x real-time)
- 🤖 Conversation Intelligence: Smart activation without wake words

⚡ Features:
- Real-time transcription with live status
- Multiple activation modes and configuration options
- Export capabilities (TXT, JSON with metadata)
- System diagnostics and performance monitoring
- KDE6/Qt6/Wayland optimized interface

Ready to begin! Initialize the system when you're ready to test.""")
        
        self.live_results.setMinimumHeight(300)
        results_layout.addWidget(self.live_results)
        
        # Results control buttons
        results_control_layout = QHBoxLayout()
        
        clear_button = QPushButton("🗑️ Clear Results")
        clear_button.clicked.connect(lambda: self.live_results.clear())
        results_control_layout.addWidget(clear_button)
        
        export_txt_button = QPushButton("📄 Export TXT")
        export_txt_button.clicked.connect(self.export_results_txt)
        results_control_layout.addWidget(export_txt_button)
        
        export_json_button = QPushButton("📊 Export JSON")
        export_json_button.clicked.connect(self.export_results_json)
        results_control_layout.addWidget(export_json_button)
        
        results_layout.addLayout(results_control_layout)
        layout.addWidget(results_group)
        
        self.tab_widget.addTab(tab, "🎤 Always Listening")
    
    def create_single_file_tab(self):
        """Create single file processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("📁 Single File Processing & Testing")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a90e2; padding: 15px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # File selection
        file_group = QGroupBox("Audio File Selection")
        file_layout = QVBoxLayout(file_group)
        
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("background-color: #333; padding: 10px; border-radius: 4px;")
        file_select_layout.addWidget(self.file_path_label)
        
        browse_button = QPushButton("📂 Browse Audio File")
        browse_button.clicked.connect(self.browse_audio_file)
        file_select_layout.addWidget(browse_button)
        
        file_layout.addLayout(file_select_layout)
        layout.addWidget(file_group)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout(options_group)
        
        options_layout.addWidget(QLabel("Backend:"), 0, 0)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["ONNX Whisper + NPU", "WhisperX (if available)"])
        options_layout.addWidget(self.backend_combo, 0, 1)
        
        options_layout.addWidget(QLabel("Model Size:"), 1, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["base", "tiny", "small"])
        options_layout.addWidget(self.model_size_combo, 1, 1)
        
        layout.addWidget(options_group)
        
        # Process button
        self.process_button = QPushButton("🧠 Process with ONNX Whisper + NPU")
        self.process_button.clicked.connect(self.process_single_file)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)
        
        # Results display
        results_group = QGroupBox("Processing Results")
        results_layout = QVBoxLayout(results_group)
        
        self.single_file_results = QTextEdit()
        self.single_file_results.setPlainText("""📁 SINGLE FILE PROCESSING READY

Select an audio file using the Browse button above, then click Process to transcribe with NPU acceleration.

Supported formats: WAV, MP3, M4A, FLAC, OGG
Recommended: WAV files for best compatibility

Results will include:
- Complete transcription text
- Processing time and performance metrics  
- Technical details (encoder shapes, mel features)
- NPU acceleration status
- Export options for TXT and JSON formats""")
        
        results_layout.addWidget(self.single_file_results)
        layout.addWidget(results_group)
        
        self.tab_widget.addTab(tab, "📁 Single File")
    
    def create_configuration_tab(self):
        """Create advanced configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("⚙️ Advanced Configuration & Settings")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a90e2; padding: 15px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # VAD Settings
        vad_group = QGroupBox("Voice Activity Detection (VAD) Settings")
        vad_layout = QGridLayout(vad_group)
        
        vad_layout.addWidget(QLabel("VAD Threshold:"), 0, 0)
        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.1, 1.0)
        self.vad_threshold.setSingleStep(0.1)
        self.vad_threshold.setValue(0.5)
        vad_layout.addWidget(self.vad_threshold, 0, 1)
        
        vad_layout.addWidget(QLabel("Min Speech Duration (s):"), 1, 0)
        self.min_speech_duration = QDoubleSpinBox()
        self.min_speech_duration.setRange(0.1, 2.0)
        self.min_speech_duration.setSingleStep(0.1)
        self.min_speech_duration.setValue(0.25)
        vad_layout.addWidget(self.min_speech_duration, 1, 1)
        
        layout.addWidget(vad_group)
        
        # Wake Word Settings
        wake_group = QGroupBox("Wake Word Detection Settings")
        wake_layout = QGridLayout(wake_group)
        
        wake_layout.addWidget(QLabel("Wake Threshold:"), 0, 0)
        self.wake_threshold = QDoubleSpinBox()
        self.wake_threshold.setRange(0.1, 1.0)
        self.wake_threshold.setSingleStep(0.1)
        self.wake_threshold.setValue(0.7)
        wake_layout.addWidget(self.wake_threshold, 0, 1)
        
        wake_layout.addWidget(QLabel("Activation Cooldown (s):"), 1, 0)
        self.activation_cooldown = QDoubleSpinBox()
        self.activation_cooldown.setRange(0.5, 10.0)
        self.activation_cooldown.setSingleStep(0.5)
        self.activation_cooldown.setValue(2.0)
        wake_layout.addWidget(self.activation_cooldown, 1, 1)
        
        layout.addWidget(wake_group)
        
        # Recording Settings
        recording_group = QGroupBox("Recording Settings")
        recording_layout = QGridLayout(recording_group)
        
        recording_layout.addWidget(QLabel("Max Recording Duration (s):"), 0, 0)
        self.max_recording_duration = QSpinBox()
        self.max_recording_duration.setRange(5, 60)
        self.max_recording_duration.setValue(30)
        recording_layout.addWidget(self.max_recording_duration, 0, 1)
        
        recording_layout.addWidget(QLabel("Max Silence Duration (s):"), 1, 0)
        self.max_silence_duration = QDoubleSpinBox()
        self.max_silence_duration.setRange(0.5, 5.0)
        self.max_silence_duration.setSingleStep(0.5)
        self.max_silence_duration.setValue(2.0)
        recording_layout.addWidget(self.max_silence_duration, 1, 1)
        
        layout.addWidget(recording_group)
        
        # Apply settings button
        apply_button = QPushButton("💾 Apply Configuration")
        apply_button.clicked.connect(self.apply_configuration)
        layout.addWidget(apply_button)
        
        # Add spacer
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "⚙️ Configuration")
    
    def create_system_status_tab(self):
        """Create system status monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("📊 System Status & Diagnostics")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a90e2; padding: 15px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        refresh_button = QPushButton("🔄 Refresh Status")
        refresh_button.clicked.connect(self.refresh_system_status)
        button_layout.addWidget(refresh_button)
        
        test_audio_button = QPushButton("🎤 Test Audio System")
        test_audio_button.clicked.connect(self.test_audio_system)
        button_layout.addWidget(test_audio_button)
        
        test_npu_button = QPushButton("🧠 Test NPU System")
        test_npu_button.clicked.connect(self.test_npu_system)
        button_layout.addWidget(test_npu_button)
        
        layout.addLayout(button_layout)
        
        # Status display
        self.system_status_display = QTextEdit()
        self.system_status_display.setReadOnly(True)
        layout.addWidget(self.system_status_display)
        
        self.tab_widget.addTab(tab, "📊 System Status")
        
        # Initial status refresh
        QTimer.singleShot(1000, self.refresh_system_status)  # Delay 1 second for startup
    
    def initialize_always_listening(self):
        """Initialize the always-listening system"""
        try:
            self.update_status_display("🚀 Initializing NPU Always-Listening System...")
            
            if not self.always_listening_system:
                self.update_status_display("❌ Always-listening system not available")
                return
            
            # Get configuration
            activation_mode = self.activation_mode.currentText()
            wake_words = [w.strip() for w in self.wake_words_input.text().split(',')]
            whisper_model = self.whisper_model.currentText()
            
            # Initialize in background thread
            def init_thread():
                try:
                    success = self.always_listening_system.initialize(
                        whisper_model=whisper_model,
                        wake_words=wake_words,
                        activation_mode=activation_mode
                    )
                    
                    self.initialization_completed.emit(success)
                    
                except Exception as e:
                    self.status_updated.emit(f"❌ Initialization error: {e}")
            
            threading.Thread(target=init_thread, daemon=True).start()
            
            # Disable init button
            self.init_button.setEnabled(False)
            self.init_button.setText("🔄 Initializing...")
            
        except Exception as e:
            self.update_status_display(f"❌ Initialization failed: {e}")
    
    def on_initialization_complete(self, success: bool):
        """Handle initialization completion"""
        if success:
            self.update_status_display("✅ NPU Always-Listening System initialized successfully!")
            self.start_button.setEnabled(True)
            self.init_button.setText("✅ System Ready")
        else:
            self.update_status_display("❌ System initialization failed")
            self.init_button.setEnabled(True)
            self.init_button.setText("🚀 Initialize System")
    
    def start_always_listening(self):
        """Start the always-listening system"""
        try:
            self.update_status_display("🎤 Starting always-listening mode...")
            
            if not self.always_listening_system:
                self.update_status_display("❌ System not available")
                return
            
            success = self.always_listening_system.start_always_listening(
                transcription_callback=self.on_transcription_callback,
                status_callback=self.on_system_status_callback
            )
            
            if success:
                self.is_always_listening = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.update_status_display("✅ Always-listening active! NPU monitoring for speech...")
                self.update_status_indicators()
            else:
                self.update_status_display("❌ Failed to start always-listening mode")
                
        except Exception as e:
            self.update_status_display(f"❌ Start error: {e}")
    
    def stop_always_listening(self):
        """Stop the always-listening system"""
        try:
            self.update_status_display("🔇 Stopping always-listening mode...")
            
            if self.always_listening_system:
                self.always_listening_system.stop_always_listening()
            
            self.is_always_listening = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            self.update_status_display("✅ Always-listening stopped")
            self.reset_status_indicators()
            
        except Exception as e:
            self.update_status_display(f"❌ Stop error: {e}")
    
    def on_transcription_callback(self, result: Dict[str, Any]):
        """Callback for transcription results"""
        self.transcription_received.emit(result)
    
    def on_system_status_callback(self, event: str, data: Dict[str, Any]):
        """Callback for system status updates"""
        self.status_updated.emit(f"[{event}] {data}")
    
    def on_transcription_result(self, result: Dict[str, Any]):
        """Handle transcription results from always-listening system"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            transcription_text = f"""
🎤 TRANSCRIPTION RESULT [{timestamp}]
{'='*60}

📝 Text: "{result['text']}"

⏱️ Performance Metrics:
   Duration: {result['audio_duration']:.1f}s
   Processing: {result['processing_time']:.2f}s  
   Real-time factor: {result['real_time_factor']:.3f}x
   NPU Accelerated: {result['npu_accelerated']}
   Activation Mode: {result['activation_mode']}

📊 Technical Details:
   Encoder Output: {result.get('encoder_output_shape', 'N/A')}
   Mel Features: {result.get('mel_features_shape', 'N/A')}

{'='*60}
"""
            
            # Update live results
            current_text = self.live_results.toPlainText()
            self.live_results.setPlainText(transcription_text + current_text)
            
            # Store result
            self.current_results.append(result)
            
        except Exception as e:
            self.update_status_display(f"❌ Result processing error: {e}")
    
    def update_status_indicators(self):
        """Update status indicators for active listening"""
        if self.is_always_listening:
            self.vad_status.setText("VAD: 🎤 Monitoring")
            self.vad_status.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px; border-radius: 4px; background-color: #2d5a2d; color: #90EE90;")
            
            activation_mode = self.activation_mode.currentText()
            if activation_mode in ["wake_word", "hybrid"]:
                self.wake_word_status.setText("Wake Word: 🎯 Monitoring")
                self.wake_word_status.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px; border-radius: 4px; background-color: #2d4a5a; color: #87CEEB;")
            else:
                self.wake_word_status.setText("Wake Word: ⭕ Disabled")
                self.wake_word_status.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px; border-radius: 4px; background-color: #333; color: #888;")
                
            self.recording_status.setText("Recording: ⭕ Not Recording")
            self.processing_status.setText("Processing: ⭕ Idle")
    
    def reset_status_indicators(self):
        """Reset all status indicators"""
        status_style = "font-size: 14px; font-weight: bold; padding: 8px; border-radius: 4px; background-color: #333; color: #888;"
        
        self.vad_status.setText("VAD: ⭕ Inactive")
        self.vad_status.setStyleSheet(status_style)
        
        self.wake_word_status.setText("Wake Word: ⭕ Inactive")
        self.wake_word_status.setStyleSheet(status_style)
        
        self.recording_status.setText("Recording: ⭕ Not Recording")
        self.recording_status.setStyleSheet(status_style)
        
        self.processing_status.setText("Processing: ⭕ Idle")
        self.processing_status.setStyleSheet(status_style)
    
    def browse_audio_file(self):
        """Browse for audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            self.process_button.setEnabled(True)
    
    def process_single_file(self):
        """Process single audio file"""
        try:
            file_path = self.file_path_label.text()
            if file_path == "No file selected":
                return
            
            self.process_button.setEnabled(False)
            self.process_button.setText("🔄 Processing...")
            
            # Initialize ONNX Whisper if needed
            if not self.onnx_whisper:
                self.single_file_results.setPlainText("❌ ONNX Whisper not available")
                self.process_button.setEnabled(True)
                self.process_button.setText("🧠 Process with ONNX Whisper + NPU")
                return
            
            if not self.onnx_whisper.is_ready:
                self.onnx_whisper.initialize()
            
            # Process in background thread
            def process_thread():
                try:
                    result = self.onnx_whisper.transcribe_audio(file_path)
                    QTimer.singleShot(0, lambda: self.on_single_file_complete(result))
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_single_file_error(str(e)))
            
            threading.Thread(target=process_thread, daemon=True).start()
            
        except Exception as e:
            self.single_file_results.setPlainText(f"❌ Processing error: {e}")
            self.process_button.setEnabled(True)
            self.process_button.setText("🧠 Process with ONNX Whisper + NPU")
    
    def on_single_file_complete(self, result: Dict[str, Any]):
        """Handle single file processing completion"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result_text = f"""
🎙️ SINGLE FILE TRANSCRIPTION RESULTS
{'='*80}

📁 File: {self.file_path_label.text()}
🎯 Model: {self.whisper_model.currentText()}
🚀 Backend: ONNX Whisper + NPU ⭐
🌍 Language: {result['language']}
🧠 NPU Acceleration: {'✅ Enabled' if result['npu_accelerated'] else '❌ Disabled'}

⏱️ PERFORMANCE METRICS:
   Processing Time: {result['processing_time']:.2f}s
   Real-time Factor: {result['real_time_factor']:.3f}x
   Audio Duration: {result['audio_duration']:.1f}s

📝 TRANSCRIPTION:
{result['text']}

📊 TECHNICAL DETAILS:
   Encoder Output Shape: {result.get('encoder_output_shape', 'N/A')}
   Mel Features Shape: {result.get('mel_features_shape', 'N/A')}

🕒 Completed: {timestamp}

✅ Transcription completed successfully with ONNX Whisper + NPU!
"""
        
        self.single_file_results.setPlainText(result_text)
        self.process_button.setEnabled(True)
        self.process_button.setText("🧠 Process with ONNX Whisper + NPU")
    
    def on_single_file_error(self, error: str):
        """Handle single file processing error"""
        self.single_file_results.setPlainText(f"❌ Processing failed: {error}")
        self.process_button.setEnabled(True)
        self.process_button.setText("🧠 Process with ONNX Whisper + NPU")
    
    def apply_configuration(self):
        """Apply advanced configuration settings"""
        try:
            # Apply VAD settings
            if self.always_listening_system and hasattr(self.always_listening_system, 'vad_npu'):
                self.always_listening_system.vad_npu.vad_threshold = self.vad_threshold.value()
                self.always_listening_system.vad_npu.min_speech_duration = self.min_speech_duration.value()
            
            # Apply wake word settings  
            if self.always_listening_system and hasattr(self.always_listening_system, 'wake_word_npu'):
                self.always_listening_system.wake_word_npu.wake_threshold = self.wake_threshold.value()
                self.always_listening_system.wake_word_npu.activation_cooldown = self.activation_cooldown.value()
            
            # Apply recording settings
            if self.always_listening_system:
                self.always_listening_system.max_recording_duration = self.max_recording_duration.value()
                self.always_listening_system.max_silence_duration = self.max_silence_duration.value()
            
            self.update_status_display("✅ Configuration applied successfully!")
            
        except Exception as e:
            self.update_status_display(f"❌ Configuration error: {e}")
    
    def export_results_txt(self):
        """Export results to TXT file"""
        if not self.current_results:
            self.update_status_display("⚠️ No results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results as TXT", 
            f"transcription_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("NPU Always-Listening Voice Assistant - Transcription Results\n")
                    f.write("=" * 70 + "\n\n")
                    
                    for i, result in enumerate(self.current_results, 1):
                        f.write(f"Result {i}:\n")
                        f.write(f"Text: {result['text']}\n")
                        f.write(f"Duration: {result['audio_duration']:.1f}s\n")
                        f.write(f"Processing: {result['processing_time']:.2f}s\n")
                        f.write(f"NPU Accelerated: {result['npu_accelerated']}\n")
                        f.write("-" * 50 + "\n\n")
                
                self.update_status_display(f"✅ Results exported to {file_path}")
                
            except Exception as e:
                self.update_status_display(f"❌ Export error: {e}")
    
    def export_results_json(self):
        """Export results to JSON file"""
        if not self.current_results:
            self.update_status_display("⚠️ No results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results as JSON",
            f"transcription_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                export_data = {
                    "export_info": {
                        "timestamp": datetime.now().isoformat(),
                        "system": "NPU Always-Listening Voice Assistant",
                        "total_results": len(self.current_results)
                    },
                    "results": self.current_results
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                self.update_status_display(f"✅ Results exported to {file_path}")
                
            except Exception as e:
                self.update_status_display(f"❌ Export error: {e}")
    
    def test_audio_system(self):
        """Test audio system"""
        try:
            import sounddevice as sd
            
            self.update_status_display("🎤 Testing audio system...")
            
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if input_devices:
                self.update_status_display(f"✅ Found {len(input_devices)} audio input device(s)")
                for i, device in enumerate(input_devices[:3]):  # Show first 3
                    self.update_status_display(f"  [{i}] {device['name']} - {device['max_input_channels']} channels")
            else:
                self.update_status_display("❌ No audio input devices found")
                
        except Exception as e:
            self.update_status_display(f"❌ Audio test failed: {e}")
    
    def test_npu_system(self):
        """Test NPU system"""
        try:
            self.update_status_display("🧠 Testing NPU system...")
            
            if self.npu_accelerator and self.npu_accelerator.is_available():
                self.update_status_display("✅ NPU Phoenix available")
                device_info = self.npu_accelerator.get_device_info()
                self.update_status_display(f"  Firmware: {device_info.get('NPU Firmware Version', 'Unknown')}")
                self.update_status_display("  Performance: 16 TOPS (INT8)")
            else:
                self.update_status_display("⚠️ NPU not available, will use CPU fallbacks")
                
        except Exception as e:
            self.update_status_display(f"❌ NPU test failed: {e}")
    
    def refresh_system_status(self):
        """Refresh system status display"""
        try:
            status_text = "📊 NPU ALWAYS-LISTENING SYSTEM STATUS - Qt6/KDE6\n"
            status_text += "=" * 80 + "\n\n"
            
            # Environment info
            status_text += "🖥️ ENVIRONMENT:\n"
            status_text += f"  Desktop: KDE6/Qt6/Wayland\n"
            status_text += f"  GUI Framework: PySide6 (Qt6)\n"
            status_text += f"  Python: {sys.version.split()[0]}\n\n"
            
            # NPU Accelerator Status
            status_text += "🧠 NPU ACCELERATOR:\n"
            if self.npu_accelerator and self.npu_accelerator.is_available():
                status_text += "  ✅ NPU Phoenix Available\n"
                status_text += f"  Device: [0000:c7:00.1] NPU Phoenix\n"
                status_text += f"  Performance: 16 TOPS (INT8)\n"
                status_text += f"  Power: <1W idle, 2-5W active\n"
            else:
                status_text += "  ❌ NPU Not Available\n"
            
            status_text += "\n"
            
            # Always-Listening System Status
            if self.always_listening_system:
                try:
                    system_status = self.always_listening_system.get_system_status()
                    status_text += "🎤 ALWAYS-LISTENING SYSTEM:\n"
                    status_text += f"  Ready: {'✅' if system_status['is_ready'] else '❌'}\n"
                    status_text += f"  Listening: {'✅' if system_status['is_listening'] else '❌'}\n"
                    status_text += f"  Processing: {'✅' if system_status['is_processing'] else '❌'}\n"
                    status_text += f"  Recording: {'✅' if system_status['is_recording'] else '❌'}\n"
                    status_text += f"  Activation Mode: {system_status['activation_mode']}\n"
                    status_text += f"  Sample Rate: {system_status['sample_rate']}Hz\n"
                except:
                    status_text += "🎤 ALWAYS-LISTENING SYSTEM:\n"
                    status_text += "  Status: Not initialized\n"
            
            status_text += "\n"
            
            # ONNX Whisper Status
            if self.onnx_whisper:
                try:
                    onnx_info = self.onnx_whisper.get_system_info()
                    status_text += "🧠 ONNX WHISPER:\n"
                    status_text += f"  Ready: {'✅' if onnx_info['onnx_whisper_ready'] else '❌'}\n"
                    status_text += f"  NPU Available: {'✅' if onnx_info['npu_available'] else '❌'}\n"
                    status_text += f"  ONNX Providers: {onnx_info['onnx_providers']}\n"
                    status_text += f"  Model Path: {onnx_info.get('model_path', 'Not loaded')}\n"
                except:
                    status_text += "🧠 ONNX WHISPER:\n"
                    status_text += "  Status: Not initialized\n"
            
            status_text += "\n"
            
            # Audio System
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_input_channels'] > 0]
                status_text += "🎤 AUDIO SYSTEM:\n"
                status_text += f"  Input Devices: {len(input_devices)} available\n"
                if input_devices:
                    for device in input_devices[:2]:  # Show first 2
                        status_text += f"    - {device['name'][:40]} ({device['max_input_channels']} ch)\n"
            except Exception as e:
                status_text += f"🎤 AUDIO SYSTEM:\n  ❌ Error: {e}\n"
            
            status_text += "\n"
            status_text += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            self.system_status_display.setPlainText(status_text)
            
        except Exception as e:
            self.system_status_display.setPlainText(f"❌ Status refresh error: {e}")
    
    def update_status_display(self, message: str):
        """Update status in live results"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_message = f"[{timestamp}] {message}\n"
        
        current_text = self.live_results.toPlainText()
        self.live_results.setPlainText(status_message + current_text)

def main():
    """Main entry point for Qt6 GUI"""
    print("🚀 Starting NPU Always-Listening Voice Assistant - Qt6/KDE6 Version...")
    
    app = QApplication(sys.argv)
    app.setApplicationName("NPU Always-Listening Voice Assistant")
    app.setApplicationVersion("2.0-Qt6")
    
    # Set Qt6 application properties for KDE6/Wayland
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    window = AlwaysListeningGUI()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())