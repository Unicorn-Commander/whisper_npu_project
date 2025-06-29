# WhisperX NPU Project Usage Guide

This guide covers how to use the WhisperX NPU project for speech recognition and NPU development.

## WhisperX Usage (Production Ready)

WhisperX provides enhanced Whisper functionality with word-level timestamps, speaker diarization, and optimized inference.

### Basic Usage

#### Environment Activation
```bash
# Quick activation
source /home/ucadmin/Development/whisper_npu_project/activate_whisperx.sh

# Manual activation
source /home/ucadmin/Development/whisper_npu_project/whisperx_env/bin/activate
```

#### Simple Transcription
```bash
# Basic transcription
whisperx audio_file.wav

# Specify model size
whisperx audio_file.wav --model large-v2

# Specify language (faster and more accurate)
whisperx audio_file.wav --language en
```

#### Advanced Features

**Speaker Diarization** (Who spoke when):
```bash
whisperx audio_file.wav --diarize --min_speakers 2 --max_speakers 4
```

**Word-level Timestamps**:
```bash
whisperx audio_file.wav --model large-v2 --return_char_alignments
```

**Multiple Output Formats**:
```bash
# Generate all formats (SRT, VTT, TXT, JSON)
whisperx audio_file.wav --output_format all

# Specific format only
whisperx audio_file.wav --output_format srt
```

**Batch Processing**:
```bash
# Process multiple files
whisperx file1.wav file2.mp3 file3.m4a

# With output directory
whisperx *.wav --output_dir ./transcriptions/
```

### WhisperX Configuration Options

#### Model Selection
- `tiny`: Fastest, least accurate
- `base`: Good balance
- `small`: Default, good accuracy
- `medium`: Better accuracy
- `large`: Best accuracy (default for diarization)
- `large-v2`: Latest large model
- `large-v3`: Newest model (if available)

#### Performance Options
```bash
# GPU acceleration (if available)
whisperx audio_file.wav --device cuda

# CPU threads
whisperx audio_file.wav --threads 8

# Batch size optimization
whisperx audio_file.wav --batch_size 16
```

#### Quality Settings
```bash
# High precision
whisperx audio_file.wav --compute_type float32

# Balanced (default)
whisperx audio_file.wav --compute_type float16

# Fast/low memory
whisperx audio_file.wav --compute_type int8
```

### Example Workflows

#### Podcast Transcription with Speakers
```bash
# Full featured podcast processing
whisperx podcast.mp3 \
  --model large-v2 \
  --language en \
  --diarize \
  --min_speakers 2 \
  --max_speakers 4 \
  --output_format all \
  --output_dir ./podcast_transcription/
```

#### Meeting Notes
```bash
# Meeting transcription with timestamps
whisperx meeting.wav \
  --model medium \
  --diarize \
  --highlight_words \
  --segment_resolution sentence \
  --output_format srt
```

#### Multilingual Content
```bash
# Auto-detect language
whisperx multilingual_audio.wav --model large-v2

# Process multiple languages
whisperx french_audio.wav --language fr
whisperx spanish_audio.wav --language es
```

## NPU Development (Development Phase)

The NPU development environment is set up for creating hardware-accelerated speech recognition applications.

### Environment Activation
```bash
# Quick activation
source /home/ucadmin/Development/whisper_npu_project/activate_iron.sh

# Manual activation
source /opt/xilinx/xrt/setup.sh
source /home/ucadmin/Development/whisper_npu_project/mlir-aie/ironenv/bin/activate
source /home/ucadmin/Development/whisper_npu_project/mlir-aie/utils/env_setup.sh
```

### NPU Hardware Monitoring

#### Check NPU Status
```bash
# Detailed NPU information
xrt-smi examine

# Monitor NPU usage
xrt-smi examine --report thermal power
```

#### Device Information
```bash
# List all devices
xrt-smi list

# NPU firmware version
xrt-smi examine | grep "NPU Firmware Version"
```

### MLIR-AIE Development

#### Environment Verification
```bash
# Check environment variables
echo "MLIR-AIE Install: $MLIR_AIE_INSTALL_DIR"
echo "Peano Install: $PEANO_INSTALL_DIR"
echo "Python Path: $PYTHONPATH"

# Test MLIR-AIE import
python3 -c "import aie; print('MLIR-AIE ready')"
```

#### Basic NPU Programming

**Vector Operations Example**:
```python
# Simple NPU vector addition (Python API)
import numpy as np
from aie.dialects import aie, aiex
from aie.extras.context import mlir_mod_ctx

# Create simple vector operation
with mlir_mod_ctx() as ctx:
    # Define AIE array structure
    # Map operations to NPU tiles
    # Configure data movement
    pass
```

**Accessing Programming Examples**:
```bash
# Navigate to examples
cd /home/ucadmin/Development/whisper_npu_project/mlir-aie/programming_examples/basic/

# Available examples:
ls -la
# - vector_scalar_add/
# - matrix_multiplication/
# - vector_vector_add/
# - passthrough_kernel/
```

### Development Workflow

#### 1. Profile WhisperX Operations
```bash
# Run WhisperX with profiling
python3 -m cProfile -o whisperx_profile.prof \
  -c "import whisperx; model = whisperx.load_model('base'); whisperx.transcribe(model, 'test.wav')"

# Analyze profile
python3 -c "import pstats; p = pstats.Stats('whisperx_profile.prof'); p.sort_stats('cumulative').print_stats(20)"
```

#### 2. Identify NPU-Suitable Operations
Focus on:
- **Matrix Multiplications** (transformer attention)
- **Convolution Operations** (if present)
- **Vector Operations** (element-wise operations)
- **Reduction Operations** (softmax, normalization)

#### 3. NPU Kernel Development
```bash
# Access MLIR-AIE tutorials
cd /home/ucadmin/Development/whisper_npu_project/mlir-aie/mlir_tutorials/

# Study examples:
ls -la
# - tutorial-1/ (Basic AIE structure)
# - tutorial-2/ (Data movement)
# - tutorial-3/ (Multiple cores)
```

## Integration Development (Future)

### Planned Integration Points

#### 1. Transformer Layer Acceleration
- **Attention Mechanism**: NPU matrix multiplication
- **Feed-Forward Networks**: NPU vector operations
- **Layer Normalization**: NPU reduction operations

#### 2. Audio Preprocessing
- **Mel Spectrogram**: NPU signal processing
- **Feature Extraction**: NPU convolution operations

#### 3. Post-processing
- **Beam Search**: NPU parallel operations
- **Token Alignment**: NPU sequence operations

## Performance Monitoring

### CPU Baseline Measurement
```bash
# Time WhisperX CPU execution
time whisperx test_audio.wav --model large-v2

# Memory usage monitoring
/usr/bin/time -v whisperx test_audio.wav --model large-v2
```

### NPU Performance Profiling
```bash
# NPU utilization monitoring (when available)
xrt-smi examine --report thermal power --format json

# Trace NPU operations (development)
# (Commands will be available once NPU integration is complete)
```

## File Organization

### Project Structure
```
/home/ucadmin/Development/whisper_npu_project/
├── whisperx_env/          # WhisperX Python environment
├── mlir-aie/              # IRON/MLIR-AIE development
├── xdna-driver/           # NPU driver and XRT
├── activate_whisperx.sh   # WhisperX activation script
├── activate_iron.sh       # NPU development activation script
├── PROJECT_STATUS.md      # Current project status
├── SETUP.md               # Setup instructions
├── USAGE.md               # This file
├── TROUBLESHOOTING.md     # Problem resolution
└── ROADMAP.md             # Development roadmap
```

### Output Organization
```bash
# Recommended output structure
mkdir -p ~/whisperx_projects/{audio,transcriptions,logs}

# Use with WhisperX
whisperx ~/whisperx_projects/audio/*.wav \
  --output_dir ~/whisperx_projects/transcriptions/
```

## Best Practices

### Audio Input
- **Supported Formats**: WAV, MP3, M4A, FLAC, MP4
- **Quality**: 16kHz+ sample rate recommended
- **Length**: Files up to several hours supported
- **Preprocessing**: Clean audio improves accuracy

### Performance Optimization
- **Model Selection**: Balance accuracy vs speed
- **Batch Processing**: Process multiple files together
- **Hardware**: Use GPU if available for CPU inference
- **Memory**: Large models require significant RAM

### Output Formats
- **SRT**: Subtitles with timestamps
- **VTT**: Web video text tracks
- **JSON**: Programmatic processing
- **TXT**: Plain text transcription

## Common Use Cases

### Research and Development
```bash
# Research workflow
whisperx research_audio.wav \
  --model large-v3 \
  --diarize \
  --return_char_alignments \
  --output_format json
```

### Production Transcription
```bash
# Production pipeline
whisperx production_audio.wav \
  --model large-v2 \
  --language en \
  --batch_size 16 \
  --output_format srt
```

### Real-time Processing
```bash
# Low-latency configuration
whisperx live_audio.wav \
  --model base \
  --compute_type int8 \
  --chunk_size 15
```

---
*Usage guide for WhisperX NPU Project*
*Last updated: 2025-06-27*