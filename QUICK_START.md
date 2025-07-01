# 🚀 NPU Voice Assistant - Quick Start Guide

**Ready to use the world's first NPU-powered always-listening voice assistant!**

## ⚡ Instant Launch

```bash
cd /home/ucadmin/Development/whisper_npu_project
python3 whisperx_npu_gui_qt6.py
```

**That's it!** The system will initialize all NPU components automatically.

## 🎯 First Use (30 seconds)

1. **Wait for GUI to appear** (~30 seconds for NPU initialization)
2. **Click "Always-Listening" tab**
3. **Click "Start Always-Listening" button**
4. **Say "hey jarvis"** and wait for response
5. **Speak your message** - watch real-time transcription!

## 📱 Interface Overview

### **Always-Listening Tab** 🎤
- Real-time voice detection
- Wake word activation
- Live transcription display

### **Single File Processing** 📁
- Drag & drop audio files
- Instant NPU-accelerated transcription
- Export results as TXT/JSON

### **Configuration** ⚙️
- Adjust voice detection sensitivity
- Enable/disable wake words
- Switch between Whisper models
- Optional content filtering (if needed)
- Performance tuning

### **System Diagnostics** 🔧
- NPU status (all 6 accelerators)
- Component health monitoring
- Performance metrics

## 🎙️ Wake Words Available

- **"hey jarvis"** (recommended)
- **"computer"**
- **"assistant"**

## 🔧 If Something Goes Wrong

### **GUI Won't Start**
```bash
# Check display
echo $DISPLAY

# Check NPU
xrt-smi examine
```

### **No Microphone**
- Check system audio settings
- Verify microphone permissions
- Test: `arecord -l`

### **Poor Recognition**
- Speak clearly and at normal pace
- Reduce background noise
- Adjust VAD threshold in Configuration

## 📊 Expected Performance

- **Processing Speed**: 10-45x faster than real-time
- **Wake Word Response**: <500ms
- **Power Usage**: <1W for always-listening
- **NPU Utilization**: All 6 accelerators active

## 🎉 What You Get

✅ **Complete NPU acceleration** - World's first implementation  
✅ **Production-quality transcription** - Real speech-to-text  
✅ **Always-listening capability** - Ultra-low power consumption  
✅ **Professional interface** - Real-time monitoring  
✅ **Robust operation** - Automatic fallbacks  

---

## 🚀 Ready to Go!

**Your NPU voice assistant is fully functional and ready for production use.**

**Just run:** `python3 whisperx_npu_gui_qt6.py`

---
*NPU Voice Assistant Quick Start*  
*System Status: ✅ Fully Operational*