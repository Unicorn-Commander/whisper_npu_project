#!/usr/bin/env python3
"""
Semantic Emotion Analyzer for Unicorn Commander
Advanced emotion detection using semantic understanding and context

Features:
- Contextual emotion analysis beyond keywords
- Sarcasm detection
- Emotion intensity scaling
- Multi-dimensional emotional state
- Semantic similarity for emotional patterns
"""

import re
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Try to import advanced NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EmotionalDimension(Enum):
    """Multi-dimensional emotional analysis"""
    VALENCE = "valence"        # Positive vs Negative
    AROUSAL = "arousal"        # High energy vs Low energy  
    DOMINANCE = "dominance"    # Control vs Submission


@dataclass
class SemanticEmotionResult:
    """Result of semantic emotion analysis"""
    primary_emotion: str
    emotion_confidence: float
    emotion_dimensions: Dict[EmotionalDimension, float]  # -1 to 1 scale
    contextual_indicators: List[str]
    sarcasm_probability: float
    emotional_progression: List[Tuple[str, str, float]]  # (text, emotion, confidence)
    semantic_similarity_scores: Dict[str, float]


class SemanticEmotionAnalyzer:
    """Advanced semantic emotion analysis beyond keyword matching"""
    
    def __init__(self, use_transformers: bool = True):
        self.use_transformers = use_transformers and SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Initialize transformer model if available
        if self.use_transformers:
            try:
                # Use a lightweight but effective model
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic emotion analysis with transformers enabled")
            except Exception as e:
                print(f"⚠️ Transformer model failed to load: {e}")
                self.use_transformers = False
                self.sentence_model = None
        else:
            self.sentence_model = None
            print("📝 Using fallback semantic analysis (no transformers)")
        
        # Emotional reference patterns for semantic comparison
        self.emotional_reference_patterns = {
            "joy": [
                "I am absolutely thrilled and delighted with this outcome",
                "This brings me immense happiness and satisfaction", 
                "I feel wonderful and grateful for this experience",
                "Everything is going perfectly and I couldn't be happier"
            ],
            "sadness": [
                "I feel deeply disappointed and let down by this situation",
                "This outcome leaves me feeling empty and dejected",
                "I'm experiencing profound grief and sorrow",
                "Nothing seems to be going right and I feel hopeless"
            ],
            "anger": [
                "I am furious and outraged by this unacceptable behavior",
                "This incompetence is absolutely infuriating to me",
                "I demand immediate action to resolve this disaster",
                "I will not tolerate this disrespectful treatment any longer"
            ],
            "fear": [
                "I am deeply worried and anxious about potential consequences",
                "This uncertainty fills me with dread and apprehension",
                "I feel vulnerable and concerned about what might happen",
                "The unknown possibilities make me nervous and scared"
            ],
            "surprise": [
                "I am completely taken aback by this unexpected development",
                "This sudden change has caught me off guard entirely",
                "I never anticipated this surprising turn of events",
                "This revelation has left me stunned and amazed"
            ],
            "disgust": [
                "This behavior is absolutely repulsive and unacceptable to me",
                "I find this situation thoroughly offensive and disturbing",
                "The poor quality of this is nauseating and revolting",
                "This appalling conduct makes me sick to my stomach"
            ],
            "neutral": [
                "I acknowledge the information you have provided to me",
                "This appears to be a standard operational procedure",
                "The data indicates normal performance within expected parameters",
                "I understand the current status of this situation"
            ]
        }
        
        # Contextual patterns that modify emotional interpretation
        self.contextual_modifiers = {
            "sarcasm_indicators": [
                "oh great", "just perfect", "wonderful", "fantastic", "exactly what I needed",
                "that's just great", "how lovely", "well that's nice", "just brilliant"
            ],
            "intensity_amplifiers": [
                "absolutely", "completely", "totally", "utterly", "extremely", 
                "incredibly", "tremendously", "exceptionally", "remarkably"
            ],
            "uncertainty_markers": [
                "maybe", "perhaps", "possibly", "might", "could be", 
                "not sure", "uncertain", "unclear", "confused"
            ],
            "negation_patterns": [
                r"not (\w+)", r"never (\w+)", r"don't (\w+)", r"doesn't (\w+)", 
                r"can't (\w+)", r"won't (\w+)", r"shouldn't (\w+)"
            ]
        }
        
        # Pre-compute reference embeddings if using transformers
        if self.use_transformers:
            self._compute_reference_embeddings()
    
    def _compute_reference_embeddings(self):
        """Pre-compute embeddings for emotional reference patterns"""
        try:
            self.reference_embeddings = {}
            for emotion, patterns in self.emotional_reference_patterns.items():
                # Compute average embedding for each emotion
                embeddings = self.sentence_model.encode(patterns)
                avg_embedding = np.mean(embeddings, axis=0)
                self.reference_embeddings[emotion] = avg_embedding
        except Exception as e:
            print(f"⚠️ Failed to compute reference embeddings: {e}")
            self.use_transformers = False
    
    def analyze_emotion(self, text: str, context: Dict[str, Any] = None) -> SemanticEmotionResult:
        """Perform comprehensive semantic emotion analysis"""
        
        # Step 1: Analyze emotional dimensions
        dimensions = self._analyze_emotional_dimensions(text)
        
        # Step 2: Detect primary emotion using semantic similarity
        if self.use_transformers:
            emotion, confidence, similarity_scores = self._analyze_with_transformers(text)
        else:
            emotion, confidence, similarity_scores = self._analyze_with_fallback(text)
        
        # Step 3: Detect contextual indicators
        contextual_indicators = self._extract_contextual_indicators(text)
        
        # Step 4: Analyze sarcasm probability
        sarcasm_prob = self._analyze_sarcasm(text, emotion, dimensions)
        
        # Step 5: Analyze emotional progression through text
        progression = self._analyze_emotional_progression(text)
        
        return SemanticEmotionResult(
            primary_emotion=emotion,
            emotion_confidence=confidence,
            emotion_dimensions=dimensions,
            contextual_indicators=contextual_indicators,
            sarcasm_probability=sarcasm_prob,
            emotional_progression=progression,
            semantic_similarity_scores=similarity_scores
        )
    
    def _analyze_emotional_dimensions(self, text: str) -> Dict[EmotionalDimension, float]:
        """Analyze text on valence, arousal, and dominance dimensions"""
        text_lower = text.lower()
        
        # Valence analysis (positive vs negative)
        positive_indicators = [
            "great", "excellent", "wonderful", "amazing", "perfect", "love", "happy",
            "pleased", "satisfied", "delighted", "thrilled", "grateful", "thank"
        ]
        negative_indicators = [
            "terrible", "awful", "horrible", "hate", "disgusting", "furious", "angry",
            "disappointed", "frustrated", "upset", "annoyed", "disgusted", "revolting"
        ]
        
        pos_score = sum(1 for word in positive_indicators if word in text_lower)
        neg_score = sum(1 for word in negative_indicators if word in text_lower)
        valence = (pos_score - neg_score) / max(pos_score + neg_score, 1)
        
        # Arousal analysis (high energy vs low energy)
        high_arousal = [
            "excited", "thrilled", "furious", "outraged", "amazing", "incredible",
            "shocking", "unbelievable", "urgent", "emergency", "immediately"
        ]
        low_arousal = [
            "calm", "peaceful", "tired", "exhausted", "bored", "indifferent",
            "whatever", "okay", "fine", "meh", "slowly", "gradually"
        ]
        
        high_score = sum(1 for word in high_arousal if word in text_lower)
        low_score = sum(1 for word in low_arousal if word in text_lower)
        arousal = (high_score - low_score) / max(high_score + low_score, 1)
        
        # Dominance analysis (control vs submission)
        dominant_indicators = [
            "demand", "insist", "require", "must", "will", "need", "expect",
            "unacceptable", "immediately", "now", "right now", "manager"
        ]
        submissive_indicators = [
            "please", "could you", "would you", "if possible", "sorry", "excuse me",
            "help me", "I don't know", "uncertain", "confused", "lost"
        ]
        
        dom_score = sum(1 for word in dominant_indicators if word in text_lower)
        sub_score = sum(1 for word in submissive_indicators if word in text_lower)
        dominance = (dom_score - sub_score) / max(dom_score + sub_score, 1)
        
        return {
            EmotionalDimension.VALENCE: max(-1.0, min(1.0, valence)),
            EmotionalDimension.AROUSAL: max(-1.0, min(1.0, arousal)),
            EmotionalDimension.DOMINANCE: max(-1.0, min(1.0, dominance))
        }
    
    def _analyze_with_transformers(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Analyze emotion using sentence transformers"""
        try:
            # Encode the input text
            text_embedding = self.sentence_model.encode([text])[0]
            
            # Calculate similarity to each emotional reference
            similarities = {}
            for emotion, ref_embedding in self.reference_embeddings.items():
                # Cosine similarity
                similarity = np.dot(text_embedding, ref_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(ref_embedding)
                )
                similarities[emotion] = float(similarity)
            
            # Find primary emotion
            primary_emotion = max(similarities.keys(), key=lambda k: similarities[k])
            confidence = similarities[primary_emotion]
            
            # Normalize confidence to 0-1 range
            confidence = (confidence + 1) / 2  # Convert from [-1,1] to [0,1]
            
            return primary_emotion, confidence, similarities
            
        except Exception as e:
            print(f"⚠️ Transformer analysis failed: {e}")
            return self._analyze_with_fallback(text)
    
    def _analyze_with_fallback(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Fallback emotion analysis using keyword patterns"""
        text_lower = text.lower()
        
        # Basic emotion patterns
        emotion_patterns = {
            "joy": ["happy", "great", "excellent", "wonderful", "amazing", "love", "perfect"],
            "sadness": ["sad", "disappointed", "upset", "depressed", "crying", "tears"],
            "anger": ["angry", "furious", "mad", "outraged", "hate", "disgusting"],
            "fear": ["scared", "afraid", "worried", "anxious", "nervous", "concerned"],
            "surprise": ["surprised", "shocked", "unexpected", "sudden", "amazing"],
            "disgust": ["disgusting", "revolting", "gross", "awful", "terrible"],
            "neutral": ["okay", "fine", "normal", "standard", "regular"]
        }
        
        scores = {}
        for emotion, keywords in emotion_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[emotion] = score / len(keywords)  # Normalize
        
        # Find primary emotion
        if max(scores.values()) > 0:
            primary_emotion = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[primary_emotion]
        else:
            primary_emotion = "neutral"
            confidence = 0.7
        
        return primary_emotion, confidence, scores
    
    def _extract_contextual_indicators(self, text: str) -> List[str]:
        """Extract contextual indicators that modify emotional interpretation"""
        text_lower = text.lower()
        indicators = []
        
        # Check for sarcasm
        for indicator in self.contextual_modifiers["sarcasm_indicators"]:
            if indicator in text_lower:
                indicators.append(f"sarcasm: {indicator}")
        
        # Check for intensity amplifiers
        for amplifier in self.contextual_modifiers["intensity_amplifiers"]:
            if amplifier in text_lower:
                indicators.append(f"intensity: {amplifier}")
        
        # Check for uncertainty
        for marker in self.contextual_modifiers["uncertainty_markers"]:
            if marker in text_lower:
                indicators.append(f"uncertainty: {marker}")
        
        # Check for negation patterns
        for pattern in self.contextual_modifiers["negation_patterns"]:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                indicators.append(f"negation: not {match}")
        
        return indicators
    
    def _analyze_sarcasm(self, text: str, detected_emotion: str, dimensions: Dict) -> float:
        """Analyze probability of sarcasm in the text"""
        text_lower = text.lower()
        sarcasm_score = 0.0
        
        # Check for sarcasm indicators
        sarcasm_count = sum(1 for indicator in self.contextual_modifiers["sarcasm_indicators"] 
                          if indicator in text_lower)
        if sarcasm_count > 0:
            sarcasm_score += 0.4
        
        # Check for positive words with negative context
        positive_words = ["great", "wonderful", "perfect", "excellent", "fantastic"]
        negative_context = ["broken", "failed", "wrong", "terrible", "awful"]
        
        has_positive = any(word in text_lower for word in positive_words)
        has_negative_context = any(word in text_lower for word in negative_context)
        
        if has_positive and has_negative_context:
            sarcasm_score += 0.3
        
        # Check emotional dimension conflicts (positive valence with negative context)
        if dimensions[EmotionalDimension.VALENCE] > 0.3 and has_negative_context:
            sarcasm_score += 0.2
        
        # Exclamation marks with potentially sarcastic phrases
        if "!" in text and any(indicator in text_lower for indicator in self.contextual_modifiers["sarcasm_indicators"]):
            sarcasm_score += 0.1
        
        return min(1.0, sarcasm_score)
    
    def _analyze_emotional_progression(self, text: str) -> List[Tuple[str, str, float]]:
        """Analyze how emotions progress through the text"""
        sentences = re.split(r'[.!?]+', text)
        progression = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # Skip very short fragments
                # Analyze each sentence
                if self.use_transformers:
                    emotion, confidence, _ = self._analyze_with_transformers(sentence)
                else:
                    emotion, confidence, _ = self._analyze_with_fallback(sentence)
                
                progression.append((sentence, emotion, confidence))
        
        return progression


def main():
    """Test the semantic emotion analyzer"""
    print("🧠 Testing Semantic Emotion Analyzer")
    
    analyzer = SemanticEmotionAnalyzer(use_transformers=SENTENCE_TRANSFORMERS_AVAILABLE)
    
    test_cases = [
        "I am absolutely thrilled with this amazing product! It works perfectly!",
        "Oh great, just what I needed - another broken feature that doesn't work.",
        "I'm really confused about how this is supposed to function. Can you help?",
        "This is completely unacceptable! I demand to speak with a manager immediately!",
        "The system appears to be functioning within normal operational parameters."
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {test_text}")
        
        result = analyzer.analyze_emotion(test_text)
        
        print(f"Primary Emotion: {result.primary_emotion} (confidence: {result.emotion_confidence:.2f})")
        print(f"Valence: {result.emotion_dimensions[EmotionalDimension.VALENCE]:.2f}")
        print(f"Arousal: {result.emotion_dimensions[EmotionalDimension.AROUSAL]:.2f}")
        print(f"Dominance: {result.emotion_dimensions[EmotionalDimension.DOMINANCE]:.2f}")
        print(f"Sarcasm Probability: {result.sarcasm_probability:.2f}")
        
        if result.contextual_indicators:
            print(f"Context Indicators: {', '.join(result.contextual_indicators)}")
    
    print("\n🎉 Semantic emotion analysis tests completed!")


if __name__ == "__main__":
    main()