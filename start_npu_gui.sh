#!/bin/bash
echo "🚀 Starting NPU Speech Recognition GUI..."
echo "====================================="
echo ""
echo "🔧 NPU System Detection:"
echo "  - AMD NPU Phoenix with firmware 1.5.5.391"
echo "  - XRT Version: 2.20.0"
echo "  - Real NPU matrix operations enabled"
echo ""
echo "🧠 Features:"
echo "  - Model selection and loading"
echo "  - File processing with NPU acceleration" 
echo "  - Real-time microphone transcription"
echo "  - Performance metrics and logging"
echo ""
echo "📋 Instructions:"
echo "  1. Select an NPU model from the dropdown"
echo "  2. Click 'Load NPU Model' to initialize"
echo "  3. Browse and process audio files OR start live recording"
echo "  4. View transcription in the 'Transcription' tab"
echo "  5. Monitor NPU operations in the 'NPU Logs' tab"
echo "  6. Export transcription as TXT or JSON with metadata"
echo ""
echo "Starting GUI..."
echo ""

cd /home/ucadmin/Development/whisper_npu_project

# Activate WhisperX environment for full features
if [ -d "whisperx_env" ]; then
    echo "🧠 Activating WhisperX environment..."
    source whisperx_env/bin/activate
    echo "✅ Enhanced features enabled:"
    echo "  - WhisperX real transcription"
    echo "  - Speaker diarization with pyannote.audio"
    echo "  - Multi-language support"
else
    echo "⚠️ WhisperX environment not found, using NPU-only mode"
fi

python3 npu_speech_gui.py