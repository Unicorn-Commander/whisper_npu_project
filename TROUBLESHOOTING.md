# WhisperX NPU Project Troubleshooting Guide

This guide helps resolve common issues encountered when setting up and using the WhisperX NPU project.

## Quick Diagnostics

### System Health Check
```bash
# Run comprehensive system check
echo "=== System Information ==="
lsb_release -a
uname -r

echo "=== NPU Hardware Detection ==="
lspci | grep -i "signal processing\|npu\|ipu"

echo "=== NPU Driver Status ==="
lsmod | grep amdxdna

echo "=== XRT Status ==="
source /opt/xilinx/xrt/setup.sh 2>/dev/null && xrt-smi examine

echo "=== Python Environments ==="
ls -la /home/ucadmin/Development/whisper_npu_project/*/bin/python*
```

## NPU Hardware Issues

### Problem: NPU Not Detected
**Symptoms**: `lspci` doesn't show NPU device, or shows with different name

**Solutions**:
1. **Check BIOS Settings**:
   ```bash
   # NPU must be enabled in BIOS
   # BIOS → Advanced → CPU Configuration → IPU → Enabled
   ```

2. **Verify Processor Compatibility**:
   ```bash
   # Check CPU model
   cat /proc/cpuinfo | grep "model name"
   
   # Must be AMD Ryzen AI processor (Phoenix, Hawk Point, Strix)
   ```

3. **Check Kernel Version**:
   ```bash
   uname -r
   # Should be 6.11+ (6.14+ recommended for Ubuntu 25.04)
   ```

### Problem: NPU Driver Not Loading
**Symptoms**: `lsmod | grep amdxdna` returns nothing

**Solutions**:
1. **Manual Driver Loading**:
   ```bash
   sudo modprobe amdxdna
   
   # Check driver messages
   sudo dmesg | grep -i amdxdna
   ```

2. **Driver Installation**:
   ```bash
   # If driver missing, may need to install
   sudo apt update
   sudo apt install --install-recommends linux-generic-hwe-24.04
   sudo reboot
   ```

3. **Check Driver Blacklisting**:
   ```bash
   # Ensure driver isn't blacklisted
   grep -r amdxdna /etc/modprobe.d/
   
   # Should return nothing or show driver is allowed
   ```

## XRT Runtime Issues

### Problem: XRT Not Found
**Symptoms**: `xrt-smi` command not found

**Solutions**:
1. **Install XRT**:
   ```bash
   # Check if XRT directory exists
   ls -la /opt/xilinx/
   
   # If missing, build from source
   cd /home/ucadmin/Development/whisper_npu_project/xdna-driver
   bash ./tools/amdxdna_deps.sh
   cd build && ./build.sh -release
   ```

2. **Environment Setup**:
   ```bash
   # Add to shell profile
   echo 'source /opt/xilinx/xrt/setup.sh' >> ~/.bashrc
   source ~/.bashrc
   ```

### Problem: NPU Not Detected by XRT
**Symptoms**: `xrt-smi examine` shows no devices

**Solutions**:
1. **Check Device Permissions**:
   ```bash
   ls -la /dev/accel/
   # Should show NPU device files
   
   # Check user permissions
   groups
   # User should be in 'render' group
   sudo usermod -a -G render $USER
   ```

2. **Restart Services**:
   ```bash
   sudo systemctl restart udev
   sudo rmmod amdxdna
   sudo modprobe amdxdna
   ```

3. **Check Device Files**:
   ```bash
   # NPU should appear in device files
   ls /sys/class/accel/
   ls /dev/accel/
   ```

## WhisperX Issues

### Problem: WhisperX Environment Not Activating
**Symptoms**: `source whisperx_env/bin/activate` fails

**Solutions**:
1. **Recreate Environment**:
   ```bash
   cd /home/ucadmin/Development/whisper_npu_project
   rm -rf whisperx_env/
   python3 -m venv whisperx_env
   source whisperx_env/bin/activate
   pip install --upgrade pip
   pip install whisperx
   ```

2. **Check Python Version**:
   ```bash
   python3 --version
   # Should be Python 3.8+ (3.10+ recommended)
   ```

### Problem: WhisperX Import Errors
**Symptoms**: `ModuleNotFoundError` when running WhisperX

**Solutions**:
1. **Install Missing Dependencies**:
   ```bash
   source whisperx_env/bin/activate
   pip install faster-whisper torch transformers librosa pyannote-audio
   ```

2. **Fix PyTorch Installation**:
   ```bash
   # If PyTorch issues
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Check Audio Libraries**:
   ```bash
   # Install system audio libraries
   sudo apt install ffmpeg libsndfile1 libsox-fmt-all
   ```

### Problem: WhisperX Model Download Fails
**Symptoms**: Network errors when downloading models

**Solutions**:
1. **Manual Model Download**:
   ```bash
   # Pre-download models
   python3 -c "
   import whisperx
   model = whisperx.load_model('base')  # Downloads model
   model = whisperx.load_model('large-v2')  # Download large model
   "
   ```

2. **Check Disk Space**:
   ```bash
   df -h ~
   # Ensure sufficient space (models can be 1-3GB each)
   ```

### Problem: WhisperX Audio Processing Errors
**Symptoms**: Audio files not recognized or processing fails

**Solutions**:
1. **Check Audio Format**:
   ```bash
   # Install additional codecs
   sudo apt install ubuntu-restricted-extras
   
   # Convert problematic audio
   ffmpeg -i input.m4a -ar 16000 output.wav
   ```

2. **Test with Simple Audio**:
   ```bash
   # Create test audio
   ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" test.wav
   whisperx test.wav
   ```

## MLIR-AIE/IRON Issues

### Problem: MLIR-AIE Import Fails
**Symptoms**: `ModuleNotFoundError: No module named 'aie'`

**Solutions**:
1. **Reinstall MLIR-AIE**:
   ```bash
   source mlir-aie/ironenv/bin/activate
   pip uninstall mlir_aie
   pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels --force-reinstall
   ```

2. **Check Environment Variables**:
   ```bash
   source mlir-aie/utils/env_setup.sh
   echo $MLIR_AIE_INSTALL_DIR
   echo $PYTHONPATH
   ```

### Problem: IRON Compilation Errors
**Symptoms**: `aie.extras.runtime` module not found

**Known Issue**: The current MLIR-AIE wheel packages have incomplete dependencies.

**Workarounds**:
1. **Install Additional Dependencies**:
   ```bash
   source mlir-aie/ironenv/bin/activate
   pip install -r mlir-aie/python/requirements_extras.txt
   ```

2. **Alternative: Build from Source** (Advanced):
   ```bash
   # This requires significant setup - see MLIR-AIE docs
   git clone https://github.com/Xilinx/mlir-aie.git mlir-aie-source
   # Follow build instructions in their README
   ```

### Problem: NPU Example Compilation Fails
**Symptoms**: Makefile errors in programming examples

**Solutions**:
1. **Check Dependencies**:
   ```bash
   # Ensure all tools are in PATH
   which aiecc.py
   which python3
   ```

2. **Manual Compilation**:
   ```bash
   # Try compiling step by step
   cd mlir-aie/programming_examples/basic/vector_scalar_add
   mkdir -p build
   python3 vector_scalar_add.py npu > build/aie.mlir
   # Check if MLIR file is generated correctly
   ```

## Performance Issues

### Problem: WhisperX Running Slowly
**Solutions**:
1. **Optimize Model Selection**:
   ```bash
   # Use smaller model for faster processing
   whisperx audio.wav --model base  # Instead of large-v2
   ```

2. **Increase CPU Threads**:
   ```bash
   whisperx audio.wav --threads $(nproc)
   ```

3. **Check System Resources**:
   ```bash
   htop  # Monitor CPU and memory usage
   free -h  # Check available memory
   ```

### Problem: Memory Issues
**Symptoms**: Out of memory errors or system freezing

**Solutions**:
1. **Reduce Batch Size**:
   ```bash
   whisperx audio.wav --batch_size 1  # Default is 8
   ```

2. **Use Smaller Models**:
   ```bash
   whisperx audio.wav --model tiny  # Smallest model
   ```

3. **Close Other Applications**:
   ```bash
   # Free up memory
   sudo systemctl stop mysql
   sudo systemctl stop apache2
   # Stop any unnecessary services
   ```

## Environment Issues

### Problem: Multiple Python Environments Conflict
**Symptoms**: Wrong packages loaded, import errors

**Solutions**:
1. **Use Full Path Activation**:
   ```bash
   # Always use full paths
   source /home/ucadmin/Development/whisper_npu_project/whisperx_env/bin/activate
   source /home/ucadmin/Development/whisper_npu_project/mlir-aie/ironenv/bin/activate
   ```

2. **Check Active Environment**:
   ```bash
   which python3
   pip list | head
   ```

3. **Clean Environment**:
   ```bash
   deactivate  # Exit current environment
   unset PYTHONPATH  # Clear Python path
   # Then reactivate correct environment
   ```

### Problem: Permission Errors
**Symptoms**: Cannot write files, permission denied

**Solutions**:
1. **Fix Ownership**:
   ```bash
   sudo chown -R $USER:$USER /home/ucadmin/Development/whisper_npu_project/
   ```

2. **Check Directory Permissions**:
   ```bash
   chmod -R 755 /home/ucadmin/Development/whisper_npu_project/
   ```

## Audio-Specific Issues

### Problem: Audio File Not Supported
**Solutions**:
1. **Convert Audio Format**:
   ```bash
   # Convert to supported format
   ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
   ```

2. **Check File Integrity**:
   ```bash
   ffprobe audio_file.wav  # Check file information
   ```

### Problem: Poor Transcription Quality
**Solutions**:
1. **Improve Audio Quality**:
   ```bash
   # Denoise audio
   ffmpeg -i noisy.wav -af "highpass=f=200,lowpass=f=3000" clean.wav
   ```

2. **Use Appropriate Model**:
   ```bash
   # Use larger model for better accuracy
   whisperx audio.wav --model large-v2
   ```

3. **Specify Language**:
   ```bash
   # Specify language for better accuracy
   whisperx audio.wav --language en
   ```

## Getting Help

### Collect Diagnostic Information
```bash
# Create diagnostic report
cat > diagnostic_report.txt << EOF
=== System Information ===
$(lsb_release -a 2>/dev/null)
$(uname -a)

=== Hardware ===
$(lspci | grep -i "signal processing\|npu\|ipu")

=== Drivers ===
$(lsmod | grep amdxdna)

=== XRT Status ===
$(source /opt/xilinx/xrt/setup.sh 2>/dev/null && xrt-smi examine 2>&1)

=== Python Environments ===
$(ls -la /home/ucadmin/Development/whisper_npu_project/*/bin/python* 2>/dev/null)

=== Disk Space ===
$(df -h /home/ucadmin/Development/whisper_npu_project/)
EOF

echo "Diagnostic report saved to diagnostic_report.txt"
```

### Resources for Further Help
- **MLIR-AIE Documentation**: https://xilinx.github.io/mlir-aie/
- **WhisperX Issues**: https://github.com/m-bain/whisperX/issues
- **AMD NPU Documentation**: https://github.com/amd/xdna-driver
- **Ubuntu NPU Support**: Ubuntu 25.04 release notes

### Community Support
- **AMD Developer Community**: https://community.amd.com/
- **MLIR Forums**: https://llvm.discourse.group/c/mlir/
- **PyTorch Forums**: https://discuss.pytorch.org/

---
*Troubleshooting guide for WhisperX NPU Project*
*Last updated: 2025-06-27*