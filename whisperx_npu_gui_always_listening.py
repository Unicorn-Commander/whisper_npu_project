#!/usr/bin/env python3
"""
Enhanced WhisperX NPU GUI with Always-Listening Capabilities
Complete NPU-powered voice assistant interface with VAD, Wake Words, and ONNX Whisper
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

# GUI imports
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    PYQT_AVAILABLE = True
except ImportError:
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
        PYQT_AVAILABLE = False
    except ImportError:
        print("❌ No GUI framework available. Please install PyQt5 or tkinter")
        sys.exit(1)

# Import our NPU systems
from always_listening_npu import AlwaysListeningNPU
from onnx_whisper_npu import ONNXWhisperNPU
from whisperx_npu_accelerator import NPUAccelerator

class AlwaysListeningGUI:
    """Enhanced GUI for Always-Listening NPU System"""
    
    def __init__(self):
        """Initialize the GUI application"""
        self.always_listening_system = AlwaysListeningNPU()
        self.onnx_whisper = ONNXWhisperNPU()
        self.npu_accelerator = NPUAccelerator()
        
        # GUI state
        self.is_always_listening = False
        self.is_processing = False
        self.current_results = []
        
        # Initialize GUI
        if PYQT_AVAILABLE:
            self.init_pyqt_gui()
        else:
            self.init_tkinter_gui()
    
    def init_pyqt_gui(self):
        """Initialize PyQt5 GUI"""
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("NPU Always-Listening Voice Assistant")
        self.window.setGeometry(100, 100, 1000, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Always-Listening Tab
        self.create_always_listening_tab()
        
        # Single File Tab
        self.create_single_file_tab()
        
        # System Status Tab
        self.create_system_status_tab()
        
        # Set stylesheet
        self.window.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #555; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #444; color: #fff; padding: 8px 16px; margin: 2px; }
            QTabBar::tab:selected { background-color: #666; }
            QPushButton { 
                background-color: #4a90e2; color: white; border: none; 
                padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:disabled { background-color: #666; color: #999; }
            QTextEdit { background-color: #1e1e1e; color: #ffffff; border: 1px solid #555; }
            QLabel { color: #ffffff; }
            QComboBox { background-color: #444; color: #fff; border: 1px solid #555; padding: 5px; }
        """)
    
    def create_always_listening_tab(self):
        """Create the always-listening interface tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("🎤 NPU Always-Listening Voice Assistant")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a90e2; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # System configuration
        config_group = QGroupBox("System Configuration")
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
        status_group = QGroupBox("Live Status")
        status_layout = QGridLayout(status_group)
        
        self.vad_status = QLabel("VAD: ⭕ Inactive")
        self.wake_word_status = QLabel("Wake Word: ⭕ Inactive")
        self.recording_status = QLabel("Recording: ⭕ Not Recording")
        self.processing_status = QLabel("Processing: ⭕ Idle")
        
        status_layout.addWidget(self.vad_status, 0, 0)
        status_layout.addWidget(self.wake_word_status, 0, 1)
        status_layout.addWidget(self.recording_status, 1, 0)
        status_layout.addWidget(self.processing_status, 1, 1)
        
        layout.addWidget(status_group)
        
        # Live transcription results
        results_group = QGroupBox("Live Transcription Results")
        results_layout = QVBoxLayout(results_group)
        
        self.live_results = QTextEdit()
        self.live_results.setPlainText("🎤 Always-listening system ready...\n\n"
                                      "Instructions:\n"
                                      "1. Click 'Initialize System' to set up NPU components\n"
                                      "2. Click 'Start Always Listening' to begin monitoring\n"
                                      "3. Say your wake word (e.g., 'hey jarvis') to activate\n"
                                      "4. Speak your message after wake word detection\n"
                                      "5. Transcription will appear here automatically\n\n"
                                      "NPU Components:\n"
                                      "- 🎤 Silero VAD: Continuous voice activity detection\n"
                                      "- 🎯 Wake Word: OpenWakeWord for natural activation\n"
                                      "- 🧠 ONNX Whisper: High-speed NPU transcription")
        results_layout.addWidget(self.live_results)
        
        # Clear button
        clear_button = QPushButton("🗑️ Clear Results")
        clear_button.clicked.connect(lambda: self.live_results.clear())
        results_layout.addWidget(clear_button)
        
        layout.addWidget(results_group)
        
        self.tab_widget.addTab(tab, "🎤 Always Listening")
    
    def create_single_file_tab(self):
        """Create single file processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("📁 Single File Processing")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a90e2; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        file_layout.addWidget(self.file_path_label)
        
        browse_button = QPushButton("📂 Browse Audio File")
        browse_button.clicked.connect(self.browse_audio_file)
        file_layout.addWidget(browse_button)
        
        layout.addLayout(file_layout)
        
        # Process button
        self.process_button = QPushButton("🧠 Process with ONNX Whisper + NPU")
        self.process_button.clicked.connect(self.process_single_file)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)
        
        # Results
        self.single_file_results = QTextEdit()
        layout.addWidget(self.single_file_results)
        
        self.tab_widget.addTab(tab, "📁 Single File")
    
    def create_system_status_tab(self):
        """Create system status monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("📊 System Status & Diagnostics")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a90e2; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Refresh button
        refresh_button = QPushButton("🔄 Refresh Status")
        refresh_button.clicked.connect(self.refresh_system_status)
        layout.addWidget(refresh_button)
        
        # Status display
        self.system_status_display = QTextEdit()
        self.system_status_display.setReadOnly(True)
        layout.addWidget(self.system_status_display)
        
        self.tab_widget.addTab(tab, "📊 System Status")
        
        # Initial status refresh
        self.refresh_system_status()
    
    def initialize_always_listening(self):
        """Initialize the always-listening system"""
        try:
            self.update_status("🚀 Initializing NPU Always-Listening System...")
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
                    
                    QTimer.singleShot(0, lambda: self.on_initialization_complete(success))
                    
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_initialization_error(str(e)))
            
            threading.Thread(target=init_thread, daemon=True).start()
            
            # Disable init button
            self.init_button.setEnabled(False)
            self.init_button.setText("🔄 Initializing...")
            
        except Exception as e:
            self.update_status(f"❌ Initialization failed: {e}")
    
    def on_initialization_complete(self, success: bool):
        """Handle initialization completion"""
        if success:
            self.update_status("✅ NPU Always-Listening System initialized successfully!")
            self.start_button.setEnabled(True)
            self.init_button.setText("✅ System Ready")
        else:
            self.update_status("❌ System initialization failed")
            self.init_button.setEnabled(True)
            self.init_button.setText("🚀 Initialize System")
    
    def on_initialization_error(self, error: str):
        """Handle initialization error"""
        self.update_status(f"❌ Initialization error: {error}")
        self.init_button.setEnabled(True)
        self.init_button.setText("🚀 Initialize System")
    
    def start_always_listening(self):
        """Start the always-listening system"""
        try:
            self.update_status("🎤 Starting always-listening mode...")
            
            success = self.always_listening_system.start_always_listening(
                transcription_callback=self.on_transcription_result,
                status_callback=self.on_system_status_update
            )
            
            if success:
                self.is_always_listening = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.update_status("✅ Always-listening active! NPU monitoring for speech...")
                self.update_status_indicators()
            else:
                self.update_status("❌ Failed to start always-listening mode")
                
        except Exception as e:
            self.update_status(f"❌ Start error: {e}")
    
    def stop_always_listening(self):
        """Stop the always-listening system"""
        try:
            self.update_status("🔇 Stopping always-listening mode...")
            
            self.always_listening_system.stop_always_listening()
            
            self.is_always_listening = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            self.update_status("✅ Always-listening stopped")
            self.reset_status_indicators()
            
        except Exception as e:
            self.update_status(f"❌ Stop error: {e}")
    
    def on_transcription_result(self, result: Dict[str, Any]):
        """Handle transcription results from always-listening system"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            transcription_text = f"""
🎤 TRANSCRIPTION RESULT [{timestamp}]

Text: "{result['text']}"
Duration: {result['audio_duration']:.1f}s
Processing: {result['processing_time']:.2f}s
Real-time factor: {result['real_time_factor']:.3f}x
NPU Accelerated: {result['npu_accelerated']}
Activation Mode: {result['activation_mode']}

{'='*50}
"""
            
            # Update live results
            current_text = self.live_results.toPlainText()
            self.live_results.setPlainText(transcription_text + current_text)
            
            # Store result
            self.current_results.append(result)
            
        except Exception as e:
            self.update_status(f"❌ Result processing error: {e}")
    
    def on_system_status_update(self, event: str, data: Dict[str, Any]):
        """Handle system status updates"""
        try:
            if event == "wake_word_detected":
                self.wake_word_status.setText(f"Wake Word: ✅ Detected '{data['wake_word']}'")
                self.update_status(f"🎯 Wake word detected: {data['wake_word']}")
                
            elif event == "recording_started":
                self.recording_status.setText("Recording: 🔴 Active")
                self.update_status("🔴 Recording started...")
                
            elif event == "processing_started":
                self.processing_status.setText("Processing: 🧠 NPU Active")
                self.update_status(f"🧠 Processing {data['duration']:.1f}s with ONNX Whisper + NPU...")
                
            elif event == "transcription_completed":
                self.recording_status.setText("Recording: ⭕ Not Recording")
                self.processing_status.setText("Processing: ⭕ Idle")
                self.wake_word_status.setText("Wake Word: 🎯 Monitoring")
                
            elif event == "processing_error":
                self.processing_status.setText("Processing: ❌ Error")
                self.update_status(f"❌ Processing error: {data['error']}")
                
        except Exception as e:
            print(f"Status update error: {e}")
    
    def update_status_indicators(self):
        """Update status indicators for active listening"""
        if self.is_always_listening:
            self.vad_status.setText("VAD: 🎤 Monitoring")
            
            activation_mode = self.activation_mode.currentText()
            if activation_mode in ["wake_word", "hybrid"]:
                self.wake_word_status.setText("Wake Word: 🎯 Monitoring")
            else:
                self.wake_word_status.setText("Wake Word: ⭕ Disabled")
                
            self.recording_status.setText("Recording: ⭕ Not Recording")
            self.processing_status.setText("Processing: ⭕ Idle")
    
    def reset_status_indicators(self):
        """Reset all status indicators"""
        self.vad_status.setText("VAD: ⭕ Inactive")
        self.wake_word_status.setText("Wake Word: ⭕ Inactive")
        self.recording_status.setText("Recording: ⭕ Not Recording")
        self.processing_status.setText("Processing: ⭕ Idle")
    
    def browse_audio_file(self):
        """Browse for audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
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
🎙️ TRANSCRIPTION RESULTS

File: {self.file_path_label.text()}
Model: {self.whisper_model.currentText()}
Backend: ONNX Whisper + NPU ⭐
Language: {result['language']}
NPU Acceleration: {'✅ Enabled' if result['npu_accelerated'] else '❌ Disabled'}
Processing Time: {result['processing_time']:.2f}s
Real-time Factor: {result['real_time_factor']:.3f}x
Timestamp: {timestamp}

SEGMENTS:
{result['text']}

📊 ONNX TECHNICAL DETAILS:
Encoder Output: {result['encoder_output_shape']}
Mel Features: {result['mel_features_shape']}

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
    
    def refresh_system_status(self):
        """Refresh system status display"""
        try:
            status_text = "📊 NPU ALWAYS-LISTENING SYSTEM STATUS\n"
            status_text += "=" * 60 + "\n\n"
            
            # NPU Accelerator Status
            status_text += "🧠 NPU ACCELERATOR:\n"
            if self.npu_accelerator.is_available():
                status_text += "  ✅ NPU Phoenix Available\n"
                status_text += f"  Device: [0000:c7:00.1] NPU Phoenix\n"
                status_text += f"  Performance: 16 TOPS (INT8)\n"
                status_text += f"  Power: <1W idle, 2-5W active\n"
            else:
                status_text += "  ❌ NPU Not Available\n"
            
            status_text += "\n"
            
            # Always-Listening System Status
            if hasattr(self, 'always_listening_system'):
                system_status = self.always_listening_system.get_system_status()
                status_text += "🎤 ALWAYS-LISTENING SYSTEM:\n"
                status_text += f"  Ready: {'✅' if system_status['is_ready'] else '❌'}\n"
                status_text += f"  Listening: {'✅' if system_status['is_listening'] else '❌'}\n"
                status_text += f"  Processing: {'✅' if system_status['is_processing'] else '❌'}\n"
                status_text += f"  Recording: {'✅' if system_status['is_recording'] else '❌'}\n"
                status_text += f"  Activation Mode: {system_status['activation_mode']}\n"
                status_text += f"  Sample Rate: {system_status['sample_rate']}Hz\n"
                
                # VAD Status
                vad_status = system_status.get('vad_status', {})
                status_text += f"\n🎤 VAD (Voice Activity Detection):\n"
                status_text += f"  Ready: {'✅' if vad_status.get('is_ready', False) else '❌'}\n"
                status_text += f"  Listening: {'✅' if vad_status.get('is_listening', False) else '❌'}\n"
                status_text += f"  Model Type: {vad_status.get('model_type', 'Unknown')}\n"
                status_text += f"  Threshold: {vad_status.get('vad_threshold', 'N/A')}\n"
                
                # Wake Word Status
                wake_status = system_status.get('wake_word_status', {})
                status_text += f"\n🎯 WAKE WORD DETECTION:\n"
                status_text += f"  Ready: {'✅' if wake_status.get('is_ready', False) else '❌'}\n"
                status_text += f"  Listening: {'✅' if wake_status.get('is_listening', False) else '❌'}\n"
                status_text += f"  Loaded Models: {wake_status.get('loaded_models', [])}\n"
                status_text += f"  Threshold: {wake_status.get('wake_threshold', 'N/A')}\n"
                status_text += f"  Cooldown: {wake_status.get('activation_cooldown', 'N/A')}s\n"
            
            status_text += "\n"
            
            # ONNX Whisper Status
            if hasattr(self, 'onnx_whisper'):
                onnx_info = self.onnx_whisper.get_system_info()
                status_text += "🧠 ONNX WHISPER:\n"
                status_text += f"  Ready: {'✅' if onnx_info['onnx_whisper_ready'] else '❌'}\n"
                status_text += f"  NPU Available: {'✅' if onnx_info['npu_available'] else '❌'}\n"
                status_text += f"  ONNX Providers: {onnx_info['onnx_providers']}\n"
                status_text += f"  Model Path: {onnx_info.get('model_path', 'Not loaded')}\n"
            
            status_text += "\n"
            status_text += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            self.system_status_display.setPlainText(status_text)
            
        except Exception as e:
            self.system_status_display.setPlainText(f"❌ Status refresh error: {e}")
    
    def update_status(self, message: str):
        """Update status in live results"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_message = f"[{timestamp}] {message}\n"
        
        current_text = self.live_results.toPlainText()
        self.live_results.setPlainText(status_message + current_text)
    
    def run(self):
        """Run the GUI application"""
        if PYQT_AVAILABLE:
            self.window.show()
            return self.app.exec_()
        else:
            self.root.mainloop()

def main():
    """Main entry point"""
    print("🚀 Starting NPU Always-Listening Voice Assistant GUI...")
    
    app = AlwaysListeningGUI()
    return app.run()

if __name__ == "__main__":
    sys.exit(main())