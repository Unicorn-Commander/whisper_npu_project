#!/usr/bin/env python3
"""
Conversation State Manager
Smart activation logic and conversation flow analysis for NPU voice assistant
Following Open Interpreter 01's "no wake word" approach
"""

import numpy as np
import time
import logging
import os
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Conversation states for smart activation"""
    IDLE = "idle"
    LISTENING = "listening" 
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_FOR_RESPONSE = "waiting_for_response"
    CONVERSATION_ACTIVE = "conversation_active"

@dataclass
class AudioEvent:
    """Audio event data structure"""
    timestamp: float
    event_type: str  # "speech_start", "speech_end", "wake_word", "silence"
    confidence: float
    duration: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class ConversationContext:
    """Conversation context tracking"""
    last_activation: float = 0.0
    last_speech: float = 0.0
    last_response: float = 0.0
    conversation_start: float = 0.0
    total_interactions: int = 0
    avg_response_time: float = 0.0
    user_engagement_score: float = 0.0

class ConversationStateManager:
    """Intelligent conversation state management for natural voice interactions"""
    
    def __init__(self):
        """Initialize conversation state manager"""
        self.current_state = ConversationState.IDLE
        self.context = ConversationContext()
        
        # Audio event history
        self.audio_events = deque(maxlen=100)  # Keep last 100 events
        self.speech_patterns = deque(maxlen=50)  # Keep speech pattern history
        
        # Activation logic parameters
        self.activation_threshold = 0.7
        self.conversation_timeout = 30.0  # 30 seconds of silence ends conversation
        self.quick_response_window = 5.0   # 5 seconds for quick follow-up
        self.engagement_decay_rate = 0.95  # Engagement score decay
        
        # Decision factors
        self.decision_weights = {
            "speech_duration": 0.3,
            "silence_gaps": 0.2,
            "conversation_context": 0.25,
            "user_engagement": 0.15,
            "audio_patterns": 0.1
        }
        
        # Smart activation callbacks
        self.activation_callback = None
        self.state_change_callback = None
        
        # Threading
        self.analysis_thread = None
        self.is_running = False
        self.state_lock = threading.Lock()
        
    def initialize(self):
        """Initialize conversation state management"""
        try:
            logger.info("🧠 Initializing Conversation State Manager...")
            
            # Start background analysis
            self.is_running = True
            self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.analysis_thread.start()
            
            logger.info("✅ Conversation State Manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Conversation State Manager initialization failed: {e}")
            return False
    
    def process_audio_event(self, event: AudioEvent):
        """Process incoming audio event and update conversation state"""
        try:
            with self.state_lock:
                # Add to event history
                self.audio_events.append(event)
                
                # Update context based on event
                self._update_context(event)
                
                # Analyze and potentially change state
                self._analyze_conversation_flow()
                
                # Check for smart activation
                if self._should_activate(event):
                    self._trigger_activation(event)
                
        except Exception as e:
            logger.error(f"❌ Audio event processing failed: {e}")
    
    def _update_context(self, event: AudioEvent):
        """Update conversation context based on audio event"""
        current_time = time.time()
        
        if event.event_type == "speech_start":
            self.context.last_speech = current_time
            
        elif event.event_type == "speech_end":
            # Track speech patterns
            speech_duration = event.duration
            self.speech_patterns.append({
                "duration": speech_duration,
                "timestamp": current_time,
                "confidence": event.confidence
            })
            
        elif event.event_type == "wake_word":
            self.context.last_activation = current_time
            if self.context.conversation_start == 0:
                self.context.conversation_start = current_time
        
        # Update engagement score
        self._update_engagement_score(event)
    
    def _update_engagement_score(self, event: AudioEvent):
        """Update user engagement score based on interaction patterns"""
        current_time = time.time()
        
        # Decay existing engagement
        time_decay = (current_time - self.context.last_speech) / 60.0  # Minutes
        self.context.user_engagement_score *= (self.engagement_decay_rate ** time_decay)
        
        # Boost engagement based on event type
        if event.event_type == "speech_start":
            self.context.user_engagement_score += 0.1
        elif event.event_type == "wake_word":
            self.context.user_engagement_score += 0.3
        elif event.event_type == "speech_end" and event.duration > 2.0:
            # Longer speech indicates higher engagement
            self.context.user_engagement_score += min(event.duration * 0.05, 0.2)
        
        # Cap engagement score
        self.context.user_engagement_score = min(self.context.user_engagement_score, 1.0)
    
    def _analyze_conversation_flow(self):
        """Analyze conversation flow and update state"""
        current_time = time.time()
        
        # Check for conversation timeout
        if (self.current_state == ConversationState.CONVERSATION_ACTIVE and
            current_time - self.context.last_speech > self.conversation_timeout):
            self._change_state(ConversationState.IDLE)
            self._reset_conversation_context()
            return
        
        # Analyze recent events for state transitions
        recent_events = [e for e in self.audio_events if current_time - e.timestamp < 10.0]
        
        if not recent_events:
            return
        
        # State transition logic
        if self.current_state == ConversationState.IDLE:
            if any(e.event_type == "wake_word" for e in recent_events):
                self._change_state(ConversationState.LISTENING)
                
        elif self.current_state == ConversationState.LISTENING:
            if any(e.event_type == "speech_end" for e in recent_events):
                self._change_state(ConversationState.PROCESSING)
                
        elif self.current_state == ConversationState.PROCESSING:
            # This would be set externally when processing completes
            pass
            
        elif self.current_state == ConversationState.RESPONDING:
            # This would be set externally when response starts
            self._change_state(ConversationState.WAITING_FOR_RESPONSE)
            
        elif self.current_state == ConversationState.WAITING_FOR_RESPONSE:
            # Quick follow-up detection
            if (current_time - self.context.last_response < self.quick_response_window and
                any(e.event_type == "speech_start" for e in recent_events)):
                self._change_state(ConversationState.CONVERSATION_ACTIVE)
    
    def _should_activate(self, event: AudioEvent) -> bool:
        """Determine if system should activate based on conversation analysis"""
        
        # Always activate on explicit wake word
        if event.event_type == "wake_word":
            return True
        
        # Smart activation analysis
        activation_score = self._calculate_activation_score(event)
        
        logger.debug(f"Activation score: {activation_score:.3f} (threshold: {self.activation_threshold})")
        
        return activation_score > self.activation_threshold
    
    def _calculate_activation_score(self, event: AudioEvent) -> float:
        """Calculate activation score based on multiple factors"""
        score = 0.0
        current_time = time.time()
        
        # Factor 1: Speech duration patterns
        if event.event_type == "speech_end":
            duration_score = min(event.duration / 5.0, 1.0)  # Normalize to 5 seconds
            score += duration_score * self.decision_weights["speech_duration"]
        
        # Factor 2: Silence gap analysis
        if len(self.audio_events) >= 2:
            last_silence = current_time - self.context.last_speech
            if 1.0 < last_silence < 10.0:  # Natural pause duration
                silence_score = 1.0 - min(last_silence / 10.0, 1.0)
                score += silence_score * self.decision_weights["silence_gaps"]
        
        # Factor 3: Conversation context
        context_score = 0.0
        if self.current_state in [ConversationState.CONVERSATION_ACTIVE, ConversationState.WAITING_FOR_RESPONSE]:
            context_score = 0.8  # High score if conversation is active
        elif current_time - self.context.last_activation < self.quick_response_window:
            context_score = 0.6  # Medium score for quick follow-up
        
        score += context_score * self.decision_weights["conversation_context"]
        
        # Factor 4: User engagement
        engagement_score = self.context.user_engagement_score
        score += engagement_score * self.decision_weights["user_engagement"]
        
        # Factor 5: Audio patterns
        pattern_score = self._analyze_speech_patterns()
        score += pattern_score * self.decision_weights["audio_patterns"]
        
        return min(score, 1.0)
    
    def _analyze_speech_patterns(self) -> float:
        """Analyze speech patterns for activation hints"""
        if len(self.speech_patterns) < 3:
            return 0.0
        
        recent_patterns = list(self.speech_patterns)[-5:]  # Last 5 speech events
        
        # Look for patterns indicating directed speech
        total_duration = sum(p["duration"] for p in recent_patterns)
        avg_confidence = sum(p["confidence"] for p in recent_patterns) / len(recent_patterns)
        
        # Pattern indicators
        pattern_score = 0.0
        
        # Longer total speech time indicates more intentional communication
        if total_duration > 10.0:
            pattern_score += 0.3
        
        # High confidence across multiple utterances
        if avg_confidence > 0.7:
            pattern_score += 0.2
        
        # Consistent speech intervals (conversation-like)
        intervals = []
        for i in range(1, len(recent_patterns)):
            interval = recent_patterns[i]["timestamp"] - recent_patterns[i-1]["timestamp"]
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if 2.0 < avg_interval < 8.0:  # Natural conversation timing
                pattern_score += 0.3
        
        return min(pattern_score, 1.0)
    
    def _trigger_activation(self, event: AudioEvent):
        """Trigger system activation"""
        logger.info(f"🎯 Smart activation triggered by {event.event_type} (confidence: {event.confidence:.2f})")
        
        self.context.total_interactions += 1
        
        if self.activation_callback:
            self.activation_callback(event, self.current_state, self.context)
    
    def _change_state(self, new_state: ConversationState):
        """Change conversation state"""
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state
            
            logger.info(f"🔄 State changed: {old_state.value} → {new_state.value}")
            
            if self.state_change_callback:
                self.state_change_callback(old_state, new_state, self.context)
    
    def _reset_conversation_context(self):
        """Reset conversation context after timeout"""
        logger.info("🔄 Conversation context reset")
        self.context.conversation_start = 0.0
        self.context.user_engagement_score *= 0.5  # Gradual decay
    
    def _analysis_worker(self):
        """Background worker for continuous conversation analysis"""
        logger.info("🔄 Conversation analysis worker started")
        
        while self.is_running:
            try:
                with self.state_lock:
                    self._periodic_analysis()
                time.sleep(1.0)  # Analyze every second
                
            except Exception as e:
                logger.error(f"❌ Analysis worker error: {e}")
        
        logger.info("🔄 Conversation analysis worker stopped")
    
    def _periodic_analysis(self):
        """Periodic conversation analysis"""
        current_time = time.time()
        
        # Clean old events
        cutoff_time = current_time - 300.0  # Keep 5 minutes of history
        while self.audio_events and self.audio_events[0].timestamp < cutoff_time:
            self.audio_events.popleft()
        
        # Update average response time
        if self.context.total_interactions > 0:
            # Simple approximation - in production, track actual response times
            estimated_response_time = 2.0  # Placeholder
            self.context.avg_response_time = (
                (self.context.avg_response_time * (self.context.total_interactions - 1) + 
                 estimated_response_time) / self.context.total_interactions
            )
    
    def set_activation_callback(self, callback: Callable):
        """Set callback for smart activation events"""
        self.activation_callback = callback
    
    def set_state_change_callback(self, callback: Callable):
        """Set callback for state change events"""
        self.state_change_callback = callback
    
    def update_processing_state(self, processing: bool):
        """Update processing state from external system"""
        if processing and self.current_state == ConversationState.LISTENING:
            self._change_state(ConversationState.PROCESSING)
        elif not processing and self.current_state == ConversationState.PROCESSING:
            self._change_state(ConversationState.RESPONDING)
    
    def update_response_state(self, responding: bool):
        """Update response state from external system"""
        if responding and self.current_state == ConversationState.RESPONDING:
            self.context.last_response = time.time()
        elif not responding and self.current_state == ConversationState.RESPONDING:
            self._change_state(ConversationState.WAITING_FOR_RESPONSE)
    
    def get_conversation_status(self) -> Dict[str, Any]:
        """Get current conversation status"""
        return {
            "current_state": self.current_state.value,
            "context": {
                "last_activation": self.context.last_activation,
                "last_speech": self.context.last_speech,
                "last_response": self.context.last_response,
                "conversation_active": self.context.conversation_start > 0,
                "total_interactions": self.context.total_interactions,
                "user_engagement_score": self.context.user_engagement_score,
                "avg_response_time": self.context.avg_response_time
            },
            "recent_events": len(self.audio_events),
            "speech_patterns": len(self.speech_patterns),
            "activation_threshold": self.activation_threshold
        }
    
    def shutdown(self):
        """Shutdown conversation state manager"""
        logger.info("🔇 Shutting down Conversation State Manager...")
        
        self.is_running = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        logger.info("✅ Conversation State Manager shutdown complete")

def test_conversation_state_manager():
    """Test conversation state management"""
    print("🧪 Testing Conversation State Manager...")
    
    # Initialize
    manager = ConversationStateManager()
    if not manager.initialize():
        print("❌ Failed to initialize")
        return False
    
    # Test callbacks
    def on_activation(event, state, context):
        print(f"🎯 ACTIVATION: {event.event_type} in state {state.value}")
    
    def on_state_change(old_state, new_state, context):
        print(f"🔄 STATE CHANGE: {old_state.value} → {new_state.value}")
    
    manager.set_activation_callback(on_activation)
    manager.set_state_change_callback(on_state_change)
    
    # Simulate conversation flow
    print("\\n🎭 Simulating conversation flow...")
    
    # Wake word activation
    wake_event = AudioEvent(
        timestamp=time.time(),
        event_type="wake_word",
        confidence=0.9,
        metadata={"word": "hey_jarvis"}
    )
    manager.process_audio_event(wake_event)
    
    time.sleep(1)
    
    # Speech events
    speech_start = AudioEvent(
        timestamp=time.time(),
        event_type="speech_start",
        confidence=0.8
    )
    manager.process_audio_event(speech_start)
    
    time.sleep(2)
    
    speech_end = AudioEvent(
        timestamp=time.time(),
        event_type="speech_end",
        confidence=0.8,
        duration=2.0
    )
    manager.process_audio_event(speech_end)
    
    # Get status
    status = manager.get_conversation_status()
    print(f"\\n📊 Conversation Status:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Cleanup
    manager.shutdown()
    
    print("\\n🎉 Conversation State Manager test completed!")
    return True

if __name__ == "__main__":
    test_conversation_state_manager()