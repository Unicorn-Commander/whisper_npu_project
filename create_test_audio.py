#!/usr/bin/env python3
"""
Create test audio file for WhisperX testing
"""

import numpy as np
import soundfile as sf
import sys

def create_test_audio():
    """Create a simple test audio file with spoken content simulation"""
    
    # Audio parameters
    sample_rate = 16000
    duration = 10.0  # seconds
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a more complex audio signal that might resemble speech
    # Mix of frequencies that could represent human speech patterns
    
    # Fundamental frequency (similar to human voice)
    f1 = 200  # Base frequency
    f2 = 400  # First harmonic
    f3 = 800  # Second harmonic
    
    # Create speech-like signal with envelope
    signal = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add envelope to simulate speech patterns (pauses and emphasis)
    envelope = np.ones_like(t)
    
    # Add some speech-like pauses
    pause_times = [2.0, 2.5, 5.0, 5.3, 7.5, 8.0]
    for pause_start in pause_times:
        pause_end = pause_start + 0.3
        mask = (t >= pause_start) & (t <= pause_end)
        envelope[mask] = 0.1
    
    # Add emphasis at certain points
    emphasis_times = [1.0, 3.5, 6.0, 9.0]
    for emphasis_time in emphasis_times:
        mask = (t >= emphasis_time) & (t <= emphasis_time + 0.5)
        envelope[mask] = envelope[mask] * 1.5
    
    # Apply envelope
    signal = signal * envelope
    
    # Add some noise to make it more realistic
    noise = 0.02 * np.random.randn(len(signal))
    signal = signal + noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save as WAV file
    output_file = "test_audio.wav"
    sf.write(output_file, signal, sample_rate)
    
    print(f"✅ Created test audio file: {output_file}")
    print(f"   Duration: {duration} seconds")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   File size: {len(signal)} samples")
    
    return output_file

if __name__ == "__main__":
    create_test_audio()