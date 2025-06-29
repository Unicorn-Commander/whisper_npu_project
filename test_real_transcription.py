#!/usr/bin/env python3
"""
Test script for real NPU + Whisper transcription
"""

import sys
import os
sys.path.insert(0, '/home/ucadmin/Development/whisper_npu_project')

from npu_speech_recognition import NPUSpeechRecognizer

def test_real_transcription():
    """Test real transcription with OpenAI Whisper"""
    print("🧪 Testing NPU + Real Transcription...")
    
    # Initialize recognizer
    recognizer = NPUSpeechRecognizer()
    
    # Try to initialize with OpenAI Whisper
    success = recognizer.initialize(whisper_model='base', use_whisperx=True)
    print(f"Initialization: {'✅' if success else '❌'}")
    
    if not success:
        print("❌ Failed to initialize")
        return
    
    # Check available models
    print(f"WhisperX available: {recognizer.whisperx_model is not None}")
    print(f"OpenAI Whisper available: {hasattr(recognizer, 'openai_whisper_model')}")
    
    # Force-load OpenAI Whisper if not available
    if not hasattr(recognizer, 'openai_whisper_model'):
        print("\n🔄 Force-loading OpenAI Whisper...")
        try:
            import whisper
            recognizer.openai_whisper_model = whisper.load_model('base')
            print("✅ OpenAI Whisper force-loaded")
        except Exception as e:
            print(f"❌ OpenAI Whisper force-load failed: {e}")
            return
    
    # Test audio file
    audio_file = "/home/ucadmin/Music/Untitled.m4a"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"\n🎙️ Testing with: {os.path.basename(audio_file)}")
    
    try:
        # Force real transcription
        print("🔧 Forcing real transcription...")
        result = recognizer.transcribe_audio(audio_file, use_real_transcription=True)
        
        print(f"\n✅ TRANSCRIPTION RESULTS:")
        print(f"Text: '{result['text']}'")
        print(f"Language: {result['language']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Audio duration: {result['audio_duration']:.2f}s")
        print(f"Real-time factor: {result['real_time_factor']:.3f}x")
        print(f"WhisperX used: {result.get('whisperx_used', False)}")
        print(f"NPU accelerated: {result['npu_accelerated']}")
        
        if result.get('segments'):
            print(f"\nSegments: {len(result['segments'])}")
            for i, seg in enumerate(result['segments'][:3]):  # Show first 3 segments
                print(f"  {i+1}: {seg}")
                
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_transcription()