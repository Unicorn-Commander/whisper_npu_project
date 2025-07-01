#!/bin/bash

# Unicorn Commander - NPU Voice Assistant Pro Launcher
echo "🦄 Launching Unicorn Commander - NPU Voice Assistant Pro..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Set environment variables for better Qt6/Wayland support
export QT_QPA_PLATFORM=wayland
export QT_WAYLAND_DECORATION=adwaita

# Navigate to project directory
cd "$(dirname "$0")"

# Launch the professional GUI
python3 whisperx_npu_gui_qt6.py

echo "🦄 Unicorn Commander session ended."