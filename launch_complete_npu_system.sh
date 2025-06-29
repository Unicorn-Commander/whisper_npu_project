#!/bin/bash
"""
Complete NPU Always-Listening Voice Assistant Launcher
Launches the full NPU-powered system with all components
"""

# Set up environment
export PYTHONPATH="/home/ucadmin/Development/whisper_npu_project:$PYTHONPATH"

# NPU environment
if [ -f "/opt/xilinx/xrt/setup.sh" ]; then
    source /opt/xilinx/xrt/setup.sh
    echo "✅ XRT environment loaded"
else
    echo "⚠️ XRT not found, continuing without NPU driver setup"
fi

# Check system requirements
echo "🧪 Checking system requirements..."

# Check Python dependencies
python3 -c "
import sys
required_packages = [
    'numpy', 'torch', 'librosa', 'sounddevice', 'soundfile',
    'onnxruntime', 'transformers', 'huggingface_hub'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        print(f'❌ {package} - MISSING')
        missing_packages.append(package)

if missing_packages:
    print(f'\\n❌ Missing packages: {missing_packages}')
    print('Install with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('\\n✅ All Python dependencies available')
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed"
    exit 1
fi

# Check audio system
echo "🎤 Checking audio system..."
python3 -c "
import sounddevice as sd
try:
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    if input_devices:
        print(f'✅ Found {len(input_devices)} audio input device(s)')
        for i, device in enumerate(input_devices):
            print(f'  [{i}] {device[\"name\"]} - {device[\"max_input_channels\"]} channels')
    else:
        print('❌ No audio input devices found')
        exit(1)
except Exception as e:
    print(f'❌ Audio system check failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Audio system check failed"
    exit 1
fi

# Check NPU availability
echo "🧠 Checking NPU availability..."
python3 -c "
import sys
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')

try:
    from whisperx_npu_accelerator import NPUAccelerator
    npu = NPUAccelerator()
    if npu.is_available():
        print('✅ NPU Phoenix available')
        device_info = npu.get_device_info()
        print(f'  Device: {device_info.get(\"NPU Firmware Version\", \"Unknown\")}')
        print(f'  Performance: 16 TOPS (INT8)')
    else:
        print('⚠️ NPU not available, will use CPU fallbacks')
except Exception as e:
    print(f'⚠️ NPU check failed: {e}')
"

# Launch options
echo ""
echo "🚀 NPU Always-Listening Voice Assistant"
echo "========================================="
echo ""
echo "Select launch option:"
echo "1) 🎤 Complete Always-Listening GUI (Recommended)"
echo "2) 📁 Enhanced Single-File GUI"
echo "3) 🧪 Test Individual Components"
echo "4) 🔧 System Diagnostics"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🎤 Launching Complete Always-Listening GUI..."
        python3 /home/ucadmin/Development/whisper_npu_project/whisperx_npu_gui_always_listening.py
        ;;
    2)
        echo "📁 Launching Enhanced Single-File GUI..."
        python3 /home/ucadmin/Development/whisper_npu_project/whisperx_npu_gui_final.py
        ;;
    3)
        echo "🧪 Component Testing Menu"
        echo "========================"
        echo "a) Test ONNX Whisper + NPU"
        echo "b) Test Silero VAD"
        echo "c) Test Wake Word Detection"
        echo "d) Test Always-Listening System"
        echo "e) Test NPU Optimization"
        echo "f) Test Conversation State Manager"
        echo ""
        read -p "Enter test choice (a-f): " test_choice
        
        case $test_choice in
            a)
                echo "🧠 Testing ONNX Whisper + NPU..."
                python3 /home/ucladmin/Development/whisper_npu_project/onnx_whisper_npu.py
                ;;
            b)
                echo "🎤 Testing Silero VAD..."
                python3 /home/ucadmin/Development/whisper_npu_project/silero_vad_npu.py
                ;;
            c)
                echo "🎯 Testing Wake Word Detection..."
                python3 /home/ucadmin/Development/whisper_npu_project/openwakeword_npu.py
                ;;
            d)
                echo "🎤 Testing Complete Always-Listening System..."
                python3 /home/ucadmin/Development/whisper_npu_project/always_listening_npu.py
                ;;
            e)
                echo "⚡ Testing NPU Optimization..."
                python3 /home/ucadmin/Development/whisper_npu_project/npu_optimization.py
                ;;
            f)
                echo "🧠 Testing Conversation State Manager..."
                python3 /home/ucadmin/Development/whisper_npu_project/conversation_state_manager.py
                ;;
            *)
                echo "❌ Invalid test choice"
                exit 1
                ;;
        esac
        ;;
    4)
        echo "🔧 Running System Diagnostics..."
        python3 -c "
import sys
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')

print('🔧 COMPLETE SYSTEM DIAGNOSTICS')
print('=' * 50)

# NPU Status
try:
    from whisperx_npu_accelerator import NPUAccelerator
    npu = NPUAccelerator()
    print('\\n🧠 NPU STATUS:')
    if npu.is_available():
        print('✅ NPU Available')
        device_info = npu.get_device_info()
        for key, value in device_info.items():
            if 'NPU' in key or 'Firmware' in key:
                print(f'  {key}: {value}')
    else:
        print('❌ NPU Not Available')
except Exception as e:
    print(f'❌ NPU Check Failed: {e}')

# ONNX Whisper Status
try:
    from onnx_whisper_npu import ONNXWhisperNPU
    whisper = ONNXWhisperNPU()
    print('\\n🧠 ONNX WHISPER STATUS:')
    if whisper.initialize():
        print('✅ ONNX Whisper Available')
        info = whisper.get_system_info()
        for key, value in info.items():
            print(f'  {key}: {value}')
    else:
        print('❌ ONNX Whisper Initialization Failed')
except Exception as e:
    print(f'❌ ONNX Whisper Check Failed: {e}')

# Audio System
try:
    import sounddevice as sd
    print('\\n🎤 AUDIO SYSTEM STATUS:')
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    print(f'✅ {len(input_devices)} input device(s) available')
    for device in input_devices[:3]:  # Show first 3
        print(f'  - {device[\"name\"]} ({device[\"max_input_channels\"]} ch)')
except Exception as e:
    print(f'❌ Audio System Check Failed: {e}')

# Model Availability
try:
    import os
    print('\\n📂 MODEL AVAILABILITY:')
    
    # ONNX Whisper models
    onnx_cache = '/home/ucadmin/Development/whisper_npu_project/whisper_onnx_cache'
    if os.path.exists(onnx_cache):
        print('✅ ONNX Whisper cache exists')
        models = os.listdir(onnx_cache)
        print(f'  Models: {models}')
    else:
        print('❌ ONNX Whisper cache not found')
    
    # VAD cache
    vad_cache = '/home/ucadmin/Development/whisper_npu_project/vad_cache'
    if os.path.exists(vad_cache):
        print('✅ VAD cache exists')
    else:
        print('⚠️ VAD cache not found (will download on first use)')
        
except Exception as e:
    print(f'❌ Model Check Failed: {e}')

print('\\n✅ System Diagnostics Complete')
"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 NPU Voice Assistant session completed!"
echo "Thank you for using the NPU Always-Listening System!"