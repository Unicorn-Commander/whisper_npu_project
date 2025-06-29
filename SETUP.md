# WhisperX NPU Project Setup Guide

This guide provides step-by-step instructions to set up the WhisperX NPU development environment from scratch.

## Prerequisites

### Hardware Requirements
- **AMD Ryzen™ AI Processor** with NPU (Phoenix, Hawk Point, or Strix architectures)
- **8GB+ RAM** (16GB recommended for large models)
- **20GB+ free disk space**

### Software Requirements
- **Ubuntu 25.04** (recommended - includes native NPU support)
  - Alternative: Ubuntu 24.04+ with kernel 6.11+
- **KDE6 Desktop** (optional but recommended)
- **Internet connection** for package downloads

### BIOS Configuration
1. **Enable NPU/IPU**:
   ```
   BIOS → Advanced → CPU Configuration → IPU → Enabled
   ```
2. **Secure Boot**: Can remain enabled (Ubuntu 25.04 has signed NPU drivers)

## Installation Steps

### Step 1: Verify NPU Hardware Detection

```bash
# Check for NPU device
lspci | grep -i "signal processing"

# Should show something like:
# c7:00.1 Signal processing controller: Advanced Micro Devices, Inc. [AMD] AMD IPU Device

# Verify driver is loaded
lsmod | grep amdxdna
```

### Step 2: Install XRT Runtime

The XRT (Xilinx Runtime) should already be installed on Ubuntu 25.04. Verify installation:

```bash
# Check XRT installation
ls -la /opt/xilinx/xrt/

# Test NPU detection
source /opt/xilinx/xrt/setup.sh
xrt-smi examine

# Expected output should show:
# Device(s) Present
# |BDF             |Name         |
# |----------------|-------------|
# |[0000:c7:00.1]  |NPU Phoenix  |
```

If XRT is not installed, you may need to build it from the project sources in `/home/ucadmin/Development/whisper_npu_project/xdna-driver/`.

### Step 3: Set Up WhisperX Environment

#### Create Python Virtual Environment
```bash
cd /home/ucadmin/Development/whisper_npu_project

# Create WhisperX environment (if not already present)
python3 -m venv whisperx_env
source whisperx_env/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip
```

#### Install WhisperX and Dependencies
```bash
# Install WhisperX
pip install whisperx

# Install additional dependencies
pip install faster-whisper torch transformers librosa pyannote-audio
```

#### Verify WhisperX Installation
```bash
# Test WhisperX CLI
whisperx --help

# Should display the full help menu with all options
```

### Step 4: Set Up MLIR-AIE (IRON) Environment

#### Create IRON Environment
```bash
cd /home/ucadmin/Development/whisper_npu_project/mlir-aie

# Create IRON environment (if not already present)
python3 -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip
```

#### Install MLIR-AIE and Dependencies
```bash
# Install basic requirements
pip install -r python/requirements.txt

# Install MLIR-AIE from wheels
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels

# Install Peano (LLVM-AIE)
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# Install extras
pip install -r python/requirements_extras.txt
```

#### Configure IRON Environment
```bash
# Set up environment variables
source utils/env_setup.sh

# Verify MLIR-AIE installation
python3 -c "import aie; print('MLIR-AIE imported successfully')"
```

### Step 5: Verify Complete Setup

#### Test NPU Hardware
```bash
# Source XRT environment
source /opt/xilinx/xrt/setup.sh

# Check NPU status
xrt-smi examine

# Should show NPU Phoenix detected and firmware version
```

#### Test WhisperX
```bash
# Activate WhisperX environment
source /home/ucadmin/Development/whisper_npu_project/whisperx_env/bin/activate

# Test with a simple command (will show help if no audio file)
whisperx --version

# Create a test audio file or download one to test transcription
```

#### Test IRON/MLIR-AIE
```bash
# Activate IRON environment
source /home/ucadmin/Development/whisper_npu_project/mlir-aie/ironenv/bin/activate
source /home/ucadmin/Development/whisper_npu_project/mlir-aie/utils/env_setup.sh

# Verify environment
echo $MLIR_AIE_INSTALL_DIR
echo $PYTHONPATH
```

## Environment Activation Scripts

For convenience, create activation scripts:

### WhisperX Activation Script
```bash
# Create activation script
cat > /home/ucadmin/Development/whisper_npu_project/activate_whisperx.sh << 'EOF'
#!/bin/bash
echo "Activating WhisperX environment..."
source /home/ucadmin/Development/whisper_npu_project/whisperx_env/bin/activate
echo "WhisperX environment activated. Use 'whisperx --help' for usage."
EOF

chmod +x /home/ucadmin/Development/whisper_npu_project/activate_whisperx.sh
```

### IRON Development Activation Script
```bash
# Create activation script
cat > /home/ucadmin/Development/whisper_npu_project/activate_iron.sh << 'EOF'
#!/bin/bash
echo "Activating IRON development environment..."
source /opt/xilinx/xrt/setup.sh
source /home/ucadmin/Development/whisper_npu_project/mlir-aie/ironenv/bin/activate
source /home/ucadmin/Development/whisper_npu_project/mlir-aie/utils/env_setup.sh
echo "IRON environment activated. NPU development ready."
echo "NPU Status:"
xrt-smi examine | grep -A 5 "Device(s) Present"
EOF

chmod +x /home/ucadmin/Development/whisper_npu_project/activate_iron.sh
```

## Verification Checklist

- [ ] NPU device detected in `lspci`
- [ ] `amdxdna` driver loaded (`lsmod | grep amdxdna`)
- [ ] XRT installed and NPU detected (`xrt-smi examine`)
- [ ] WhisperX environment activated and functional
- [ ] IRON environment activated and MLIR-AIE imports successfully
- [ ] Both activation scripts created and executable

## Quick Test Commands

### Test NPU Hardware
```bash
source /opt/xilinx/xrt/setup.sh && xrt-smi examine
```

### Test WhisperX
```bash
source /home/ucadmin/Development/whisper_npu_project/activate_whisperx.sh
whisperx --help
```

### Test IRON Development
```bash
source /home/ucadmin/Development/whisper_npu_project/activate_iron.sh
python3 -c "import aie; print('IRON ready for NPU development')"
```

## Next Steps

After successful setup:

1. **Review [USAGE.md](USAGE.md)** for operating instructions
2. **Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** if you encounter issues
3. **See [ROADMAP.md](ROADMAP.md)** for development priorities
4. **Read [PROJECT_STATUS.md](PROJECT_STATUS.md)** for current project status

## Support and Resources

- **MLIR-AIE Documentation**: https://xilinx.github.io/mlir-aie/
- **WhisperX GitHub**: https://github.com/m-bain/whisperX
- **AMD NPU Resources**: https://github.com/amd/xdna-driver
- **Project Issues**: See TROUBLESHOOTING.md

---
*Setup guide for WhisperX NPU Project*
*Last updated: 2025-06-27*