#!/bin/bash
"""
WhisperX NPU GUI Launcher Script

Quick launcher for the WhisperX NPU GUI application.
Automatically activates the correct Python environment.
"""

echo "🚀 Starting WhisperX NPU GUI..."
echo "Project: /home/ucadmin/Development/whisper_npu_project"
echo

cd /home/ucadmin/Development/whisper_npu_project

# Activate WhisperX environment
echo "📦 Activating WhisperX environment..."
source whisperx_env/bin/activate

# Check if GUI dependencies are available
echo "🔍 Checking dependencies..."
python3 -c "import tkinter; print('✅ tkinter available')" 2>/dev/null || {
    echo "❌ tkinter not available. Installing..."
    sudo apt-get update && sudo apt-get install -y python3-tk
}

# Set environment variable to work around ctranslate2 executable stack issue
export LD_PRELOAD=""
export PYTHONPATH="${PYTHONPATH}:/home/ucadmin/Development/whisper_npu_project"

# Launch GUI
echo "🎙️ Launching WhisperX NPU GUI..."
echo "Close the terminal window to stop the application."
echo "Note: If you see ctranslate2 warnings, they can be safely ignored."
echo

# Try different approaches to work around ctranslate2 issue
if command -v setarch >/dev/null 2>&1; then
    # Use setarch to disable security features if available
    setarch x86_64 -R python3 whisperx_npu_gui_working.py
else
    # Fallback to normal execution
    python3 whisperx_npu_gui_working.py
fi