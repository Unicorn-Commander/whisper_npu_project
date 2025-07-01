#!/usr/bin/env python3
"""
Enhanced Topical Filtering Framework for Unicorn Commander
Advanced transcription filtering with emotional recognition and custom analysis

Features:
- Emotional sentiment analysis
- Custom complaint detection
- Healthcare conversation enhancement  
- Business meeting intelligence
- Real-time confidence scoring
- Multi-dimensional filtering
"""

import re
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Import base classes from existing framework
try:
    from topical_filtering_framework import (
        FilterMode, FilterResult, TopicalFilter, TopicalFilterManager,
        MedicalConversationFilter, BusinessMeetingFilter
    )
    BASE_FRAMEWORK_AVAILABLE = True
except ImportError:
    # Define base classes if not available
    BASE_FRAMEWORK_AVAILABLE = False
    
    class FilterMode(Enum):
        INCLUDE = "include"
        EXCLUDE = "exclude" 
        CATEGORIZE = "categorize"
        EXTRACT = "extract"

    @dataclass
    class FilterResult:
        original_text: str
        filtered_text: str
        relevance_score: float
        categories: List[str]
        extracted_info: Dict[str, Any]
        confidence: float
        filter_applied: str

    class TopicalFilter(ABC):
        def __init__(self, name: str, mode: FilterMode):
            self.name = name
            self.mode = mode
            self.enabled = True
            
        @abstractmethod
        def filter(self, text: str, context: Dict[str, Any] = None) -> FilterResult:
            pass
            
        @abstractmethod
        def get_keywords(self) -> List[str]:
            pass


class EmotionalState(Enum):
    """Detected emotional states in conversation"""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    SATISFIED = "satisfied"
    ANGRY = "angry"
    CONFUSED = "confused"
    CONCERNED = "concerned"
    RELIEVED = "relieved"


class ComplaintType(Enum):
    """Types of complaints detected"""
    PRODUCT_QUALITY = "product_quality"
    SERVICE_ISSUE = "service_issue"
    BILLING_PROBLEM = "billing_problem"
    TECHNICAL_SUPPORT = "technical_support"
    DELIVERY_DELAY = "delivery_delay"
    ACCESSIBILITY = "accessibility"
    POLICY_CONCERN = "policy_concern"
    GENERAL_FEEDBACK = "general_feedback"


@dataclass
class EmotionalAnalysis:
    """Result of emotional analysis"""
    primary_emotion: EmotionalState
    emotion_confidence: float
    emotion_intensity: float  # 0.0 to 1.0
    emotional_progression: List[Tuple[str, EmotionalState]]  # (text_segment, emotion)
    sentiment_score: float  # -1.0 (negative) to 1.0 (positive)
    emotional_keywords: List[str]


@dataclass
class ComplaintAnalysis:
    """Result of complaint detection"""
    is_complaint: bool
    complaint_types: List[ComplaintType]
    severity_level: float  # 0.0 to 1.0
    complaint_confidence: float
    key_issues: List[str]
    suggested_actions: List[str]
    urgency_score: float


class EmotionalRecognitionFilter(TopicalFilter):
    """Advanced emotional recognition and sentiment analysis"""
    
    def __init__(self):
        super().__init__("Emotional Recognition", FilterMode.EXTRACT)
        
        # Emotional keyword patterns
        self.emotion_patterns = {
            EmotionalState.POSITIVE: [
                r"(great|excellent|amazing|wonderful|fantastic|perfect|love|happy|satisfied|pleased)",
                r"(thank you|appreciate|grateful|helpful|wonderful experience)",
                r"(works (great|well|perfectly)|exactly what I needed|very happy)"
            ],
            EmotionalState.NEGATIVE: [
                r"(terrible|awful|horrible|worst|hate|disappointed|unsatisfied)",
                r"(doesn't work|broken|failed|error|problem|issue|wrong)",
                r"(waste of (time|money)|regret|mistake|poor quality)"
            ],
            EmotionalState.FRUSTRATED: [
                r"(frustrated|annoyed|irritated|fed up|sick of)",
                r"(this is ridiculous|can't believe|unacceptable)",
                r"(why (is|does|can't)|how hard can it be|simple request)"
            ],
            EmotionalState.ANXIOUS: [
                r"(worried|concerned|nervous|anxious|stressed|scared)",
                r"(what if|hoping|really need|urgent|emergency)",
                r"(can you help|need assistance|don't know what to do)"
            ],
            EmotionalState.CONFUSED: [
                r"(confused|don't understand|unclear|how do I|what does)",
                r"(can you explain|not sure|which one|what's the difference)",
                r"(help me understand|lost|complicated|doesn't make sense)"
            ],
            EmotionalState.ANGRY: [
                r"(angry|furious|outraged|disgusted|livid)",
                r"(this is (bull|ridiculous|unacceptable)|demand|insist)",
                r"(speak to (manager|supervisor)|escalate|complaint|lawsuit)"
            ]
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            "very": 1.3, "extremely": 1.5, "absolutely": 1.4, "totally": 1.3,
            "completely": 1.4, "really": 1.2, "quite": 1.1, "somewhat": 0.8,
            "a little": 0.7, "slightly": 0.6, "kind of": 0.8
        }
        
        # Sentiment scoring words
        self.positive_words = [
            "good", "great", "excellent", "amazing", "perfect", "love", "like",
            "helpful", "useful", "easy", "fast", "quick", "smooth", "clear"
        ]
        
        self.negative_words = [
            "bad", "terrible", "awful", "slow", "difficult", "hard", "confusing",
            "broken", "failed", "error", "problem", "issue", "wrong", "poor"
        ]

    def get_keywords(self) -> List[str]:
        """Get all emotional keywords"""
        keywords = []
        for patterns in self.emotion_patterns.values():
            for pattern in patterns:
                # Extract basic words from regex patterns
                words = re.findall(r'\w+', pattern)
                keywords.extend(words)
        return list(set(keywords))

    def filter(self, text: str, context: Dict[str, Any] = None) -> FilterResult:
        """Perform emotional analysis on text"""
        
        # Perform emotional analysis
        emotional_analysis = self._analyze_emotions(text)
        
        # Calculate overall relevance (how emotionally charged the text is)
        relevance_score = emotional_analysis.emotion_intensity
        
        # Create filtered text with emotional annotations
        filtered_text = self._annotate_emotional_content(text, emotional_analysis)
        
        # Determine categories
        categories = self._categorize_emotional_content(emotional_analysis)
        
        # Package extracted information
        extracted_info = {
            "primary_emotion": emotional_analysis.primary_emotion.value,
            "emotion_confidence": emotional_analysis.emotion_confidence,
            "emotion_intensity": emotional_analysis.emotion_intensity,
            "sentiment_score": emotional_analysis.sentiment_score,
            "emotional_keywords": emotional_analysis.emotional_keywords,
            "emotional_progression": [
                {"text": seg, "emotion": emo.value} 
                for seg, emo in emotional_analysis.emotional_progression
            ]
        }
        
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            relevance_score=relevance_score,
            categories=categories,
            extracted_info=extracted_info,
            confidence=emotional_analysis.emotion_confidence,
            filter_applied=self.name
        )

    def _analyze_emotions(self, text: str) -> EmotionalAnalysis:
        """Analyze emotions in the text"""
        text_lower = text.lower()
        
        # Detect emotions with confidence scores
        emotion_scores = {}
        emotional_keywords = []
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            matches = []
            
            for pattern in patterns:
                pattern_matches = re.findall(pattern, text_lower)
                if pattern_matches:
                    matches.extend(pattern_matches)
                    score += len(pattern_matches)
            
            if matches:
                emotional_keywords.extend(matches)
                # Apply intensity modifiers
                intensity = self._calculate_intensity(text_lower, matches)
                emotion_scores[emotion] = score * intensity
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            emotion_confidence = min(emotion_scores[primary_emotion] / 5.0, 1.0)  # Normalize
            emotion_intensity = min(sum(emotion_scores.values()) / 10.0, 1.0)
        else:
            primary_emotion = EmotionalState.NEUTRAL
            emotion_confidence = 0.8  # High confidence in neutral
            emotion_intensity = 0.1
        
        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment(text_lower)
        
        # Analyze emotional progression through text
        emotional_progression = self._analyze_emotional_progression(text)
        
        return EmotionalAnalysis(
            primary_emotion=primary_emotion,
            emotion_confidence=emotion_confidence,
            emotion_intensity=emotion_intensity,
            emotional_progression=emotional_progression,
            sentiment_score=sentiment_score,
            emotional_keywords=list(set(emotional_keywords))
        )

    def _calculate_intensity(self, text: str, matches: List[str]) -> float:
        """Calculate emotional intensity based on modifiers"""
        base_intensity = 1.0
        
        for modifier, multiplier in self.intensity_modifiers.items():
            if modifier in text:
                base_intensity *= multiplier
        
        # Check for repetition (increases intensity)
        unique_matches = len(set(matches))
        total_matches = len(matches)
        if total_matches > unique_matches:
            base_intensity *= (1 + (total_matches - unique_matches) * 0.2)
        
        return min(base_intensity, 2.0)  # Cap at 2x

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score (-1 to 1)"""
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        # Calculate final sentiment
        sentiment = positive_ratio - negative_ratio
        return max(-1.0, min(1.0, sentiment * 10))  # Scale and clamp

    def _analyze_emotional_progression(self, text: str) -> List[Tuple[str, EmotionalState]]:
        """Analyze how emotions change throughout the text"""
        sentences = re.split(r'[.!?]+', text)
        progression = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Simple emotion detection for each sentence (avoid recursion)
            sentence_emotion = self._detect_sentence_emotion(sentence.lower())
            progression.append((sentence, sentence_emotion))
        
        return progression

    def _detect_sentence_emotion(self, sentence: str) -> EmotionalState:
        """Detect emotion in a single sentence without recursion"""
        emotion_scores = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, sentence):
                    score += 1
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        else:
            return EmotionalState.NEUTRAL

    def _annotate_emotional_content(self, text: str, analysis: EmotionalAnalysis) -> str:
        """Add emotional annotations to text"""
        if analysis.emotion_intensity < 0.3:
            return text  # Not emotionally significant
        
        emotion_tag = f"[{analysis.primary_emotion.value.upper()}]"
        intensity_indicator = "!" * min(int(analysis.emotion_intensity * 3), 3)
        
        return f"{emotion_tag}{intensity_indicator} {text}"

    def _categorize_emotional_content(self, analysis: EmotionalAnalysis) -> List[str]:
        """Categorize content based on emotional analysis"""
        categories = []
        
        if analysis.emotion_intensity > 0.6:
            categories.append("high_emotional_intensity")
        
        if analysis.sentiment_score > 0.5:
            categories.append("positive_sentiment")
        elif analysis.sentiment_score < -0.5:
            categories.append("negative_sentiment")
        
        if analysis.primary_emotion in [EmotionalState.FRUSTRATED, EmotionalState.ANGRY]:
            categories.append("escalation_risk")
        
        if analysis.primary_emotion == EmotionalState.CONFUSED:
            categories.append("needs_clarification")
        
        if analysis.primary_emotion == EmotionalState.ANXIOUS:
            categories.append("requires_reassurance")
        
        return categories


class ComplaintDetectionFilter(TopicalFilter):
    """Advanced complaint detection and analysis"""
    
    def __init__(self):
        super().__init__("Complaint Detection", FilterMode.EXTRACT)
        
        # Complaint indicators
        self.complaint_keywords = [
            "complaint", "complain", "issue", "problem", "trouble", "difficult",
            "wrong", "error", "mistake", "failed", "broken", "doesn't work",
            "poor", "bad", "terrible", "awful", "disappointed", "unsatisfied"
        ]
        
        # Complaint type patterns
        self.complaint_patterns = {
            ComplaintType.PRODUCT_QUALITY: [
                r"(defective|broken|poor quality|cheaply made|falls apart)",
                r"(doesn't work|stopped working|not working|faulty)",
                r"(product (problem|issue|defect)|manufacturing (error|defect))"
            ],
            ComplaintType.SERVICE_ISSUE: [
                r"(poor service|bad service|rude|unhelpful|unprofessional)",
                r"(customer service|support (team|staff)|representative)",
                r"(waited (too long|forever)|slow response|ignored)"
            ],
            ComplaintType.BILLING_PROBLEM: [
                r"(overcharged|wrong amount|billing (error|mistake|issue))",
                r"(refund|money back|charge(d)?.*wrong|payment (problem|issue))",
                r"(invoice|bill.*incorrect|unexpected (charge|fee))"
            ],
            ComplaintType.TECHNICAL_SUPPORT: [
                r"(technical (issue|problem|support)|software (bug|error|glitch))",
                r"(app (crashes|freezes|doesn't work)|website (down|slow|broken))",
                r"(connection (problem|issue)|login (problem|issue))"
            ],
            ComplaintType.DELIVERY_DELAY: [
                r"(late delivery|delayed|never arrived|still waiting)",
                r"(shipping (problem|delay|issue)|package (lost|missing))",
                r"(delivery.*late|expected.*yesterday|where is my order)"
            ]
        }
        
        # Severity indicators
        self.severity_indicators = {
            "mild": ["minor", "small", "little", "slightly"],
            "moderate": ["issue", "problem", "trouble", "concern"],
            "severe": ["major", "serious", "terrible", "awful", "horrible", "disaster"],
            "critical": ["emergency", "urgent", "critical", "lawsuit", "legal action"]
        }
        
        # Urgency indicators
        self.urgency_keywords = [
            "urgent", "emergency", "asap", "immediately", "right now", "critical",
            "deadline", "time sensitive", "can't wait", "need now"
        ]

    def get_keywords(self) -> List[str]:
        """Get complaint detection keywords"""
        return self.complaint_keywords

    def filter(self, text: str, context: Dict[str, Any] = None) -> FilterResult:
        """Detect and analyze complaints in text"""
        
        complaint_analysis = self._analyze_complaint(text)
        
        # Calculate relevance based on complaint likelihood
        relevance_score = complaint_analysis.complaint_confidence if complaint_analysis.is_complaint else 0.0
        
        # Create filtered text with complaint annotations
        filtered_text = self._annotate_complaint_content(text, complaint_analysis)
        
        # Determine categories
        categories = self._categorize_complaint_content(complaint_analysis)
        
        # Package extracted information
        extracted_info = {
            "is_complaint": complaint_analysis.is_complaint,
            "complaint_types": [ct.value for ct in complaint_analysis.complaint_types],
            "severity_level": complaint_analysis.severity_level,
            "key_issues": complaint_analysis.key_issues,
            "suggested_actions": complaint_analysis.suggested_actions,
            "urgency_score": complaint_analysis.urgency_score
        }
        
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            relevance_score=relevance_score,
            categories=categories,
            extracted_info=extracted_info,
            confidence=complaint_analysis.complaint_confidence,
            filter_applied=self.name
        )

    def _analyze_complaint(self, text: str) -> ComplaintAnalysis:
        """Analyze text for complaint indicators"""
        text_lower = text.lower()
        
        # Check for complaint indicators
        complaint_score = 0
        for keyword in self.complaint_keywords:
            if keyword in text_lower:
                complaint_score += 1
        
        # Detect complaint types
        complaint_types = []
        type_scores = {}
        
        for complaint_type, patterns in self.complaint_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    complaint_types.append(complaint_type)
                    type_scores[complaint_type] = type_scores.get(complaint_type, 0) + 1
        
        # Remove duplicates while preserving order
        complaint_types = list(dict.fromkeys(complaint_types))
        
        # Determine if this is actually a complaint
        is_complaint = complaint_score > 0 or len(complaint_types) > 0
        complaint_confidence = min(complaint_score / 3.0, 1.0) if is_complaint else 0.0
        
        # Calculate severity
        severity_level = self._calculate_severity(text_lower)
        
        # Extract key issues
        key_issues = self._extract_key_issues(text, complaint_types)
        
        # Calculate urgency
        urgency_score = self._calculate_urgency(text_lower)
        
        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(complaint_types, severity_level, urgency_score)
        
        return ComplaintAnalysis(
            is_complaint=is_complaint,
            complaint_types=complaint_types,
            severity_level=severity_level,
            complaint_confidence=complaint_confidence,
            key_issues=key_issues,
            suggested_actions=suggested_actions,
            urgency_score=urgency_score
        )

    def _calculate_severity(self, text: str) -> float:
        """Calculate complaint severity (0.0 to 1.0)"""
        severity_scores = {
            "mild": 0.25,
            "moderate": 0.5,
            "severe": 0.75,
            "critical": 1.0
        }
        
        max_severity = 0.0
        for level, keywords in self.severity_indicators.items():
            for keyword in keywords:
                if keyword in text:
                    max_severity = max(max_severity, severity_scores[level])
        
        return max_severity if max_severity > 0 else 0.4  # Default moderate

    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency score (0.0 to 1.0)"""
        urgency_count = sum(1 for keyword in self.urgency_keywords if keyword in text)
        return min(urgency_count / 3.0, 1.0)

    def _extract_key_issues(self, text: str, complaint_types: List[ComplaintType]) -> List[str]:
        """Extract specific issues mentioned in the text"""
        sentences = re.split(r'[.!?]+', text)
        key_issues = []
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(keyword in sentence for keyword in self.complaint_keywords):
                # Clean up and add significant sentences
                if len(sentence) > 10:  # Filter out very short fragments
                    key_issues.append(sentence.capitalize())
        
        return key_issues[:3]  # Limit to top 3 issues

    def _generate_suggested_actions(self, complaint_types: List[ComplaintType], 
                                   severity: float, urgency: float) -> List[str]:
        """Generate suggested actions based on complaint analysis"""
        actions = []
        
        if urgency > 0.7:
            actions.append("Prioritize for immediate response")
        
        if severity > 0.7:
            actions.append("Escalate to supervisor/manager")
        
        for complaint_type in complaint_types:
            if complaint_type == ComplaintType.BILLING_PROBLEM:
                actions.append("Review billing/payment records")
            elif complaint_type == ComplaintType.TECHNICAL_SUPPORT:
                actions.append("Assign to technical support team")
            elif complaint_type == ComplaintType.PRODUCT_QUALITY:
                actions.append("Initiate quality control review")
            elif complaint_type == ComplaintType.SERVICE_ISSUE:
                actions.append("Schedule service recovery follow-up")
            elif complaint_type == ComplaintType.DELIVERY_DELAY:
                actions.append("Check shipping/delivery status")
        
        if not actions:
            actions.append("Acknowledge complaint and gather more details")
        
        return list(set(actions))  # Remove duplicates

    def _annotate_complaint_content(self, text: str, analysis: ComplaintAnalysis) -> str:
        """Add complaint annotations to text"""
        if not analysis.is_complaint:
            return text
        
        severity_tag = ""
        if analysis.severity_level > 0.7:
            severity_tag = "[HIGH SEVERITY] "
        elif analysis.severity_level > 0.4:
            severity_tag = "[MODERATE] "
        
        urgency_tag = ""
        if analysis.urgency_score > 0.5:
            urgency_tag = "[URGENT] "
        
        complaint_types = ", ".join([ct.value.replace("_", " ").title() for ct in analysis.complaint_types])
        type_tag = f"[{complaint_types}] " if complaint_types else "[COMPLAINT] "
        
        return f"{urgency_tag}{severity_tag}{type_tag}{text}"

    def _categorize_complaint_content(self, analysis: ComplaintAnalysis) -> List[str]:
        """Categorize complaint content"""
        categories = []
        
        if analysis.is_complaint:
            categories.append("complaint_detected")
        
        if analysis.urgency_score > 0.5:
            categories.append("urgent_response_needed")
        
        if analysis.severity_level > 0.7:
            categories.append("high_severity")
        
        for complaint_type in analysis.complaint_types:
            categories.append(f"complaint_{complaint_type.value}")
        
        return categories


class EnhancedTopicalFilterManager:
    """Enhanced filter manager with emotional and complaint detection"""
    
    def __init__(self):
        self.filters: Dict[str, TopicalFilter] = {}
        self.active_filters: List[str] = []
        self.global_confidence_threshold = 0.3
        
        # Register enhanced filters
        self._register_default_filters()
    
    def _register_default_filters(self):
        """Register default enhanced filters"""
        try:
            # Try to import and register base filters
            if BASE_FRAMEWORK_AVAILABLE:
                self.register_filter(MedicalConversationFilter())
                self.register_filter(BusinessMeetingFilter())
        except:
            pass
        
        # Register enhanced filters
        self.register_filter(EmotionalRecognitionFilter())
        self.register_filter(ComplaintDetectionFilter())
    
    def register_filter(self, filter_instance: TopicalFilter):
        """Register a new filter"""
        self.filters[filter_instance.name] = filter_instance
    
    def set_active_filters(self, filter_names: List[str]):
        """Set multiple active filters"""
        valid_filters = [name for name in filter_names if name in self.filters]
        self.active_filters = valid_filters
    
    def filter_text(self, text: str, context: Dict[str, Any] = None) -> Dict[str, FilterResult]:
        """Apply all active filters to text"""
        results = {}
        
        for filter_name in self.active_filters:
            if filter_name in self.filters:
                filter_instance = self.filters[filter_name]
                if filter_instance.enabled:
                    try:
                        result = filter_instance.filter(text, context)
                        if result.confidence >= self.global_confidence_threshold:
                            results[filter_name] = result
                    except Exception as e:
                        print(f"Error in filter {filter_name}: {e}")
        
        return results
    
    def get_comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Get comprehensive analysis from all filters"""
        results = self.filter_text(text)
        
        # Aggregate insights
        analysis = {
            "emotional_state": "neutral",
            "sentiment_score": 0.0,
            "contains_complaint": False,
            "urgency_level": "normal",
            "recommended_actions": [],
            "filter_results": results
        }
        
        # Extract emotional insights
        if "Emotional Recognition" in results:
            emotional_result = results["Emotional Recognition"]
            analysis["emotional_state"] = emotional_result.extracted_info.get("primary_emotion", "neutral")
            analysis["sentiment_score"] = emotional_result.extracted_info.get("sentiment_score", 0.0)
        
        # Extract complaint insights
        if "Complaint Detection" in results:
            complaint_result = results["Complaint Detection"]
            analysis["contains_complaint"] = complaint_result.extracted_info.get("is_complaint", False)
            urgency = complaint_result.extracted_info.get("urgency_score", 0.0)
            if urgency > 0.7:
                analysis["urgency_level"] = "high"
            elif urgency > 0.4:
                analysis["urgency_level"] = "medium"
            
            analysis["recommended_actions"] = complaint_result.extracted_info.get("suggested_actions", [])
        
        return analysis


def main():
    """Test the enhanced topical filtering"""
    print("🧠 Testing Enhanced Topical Filtering")
    
    # Initialize enhanced filter manager
    manager = EnhancedTopicalFilterManager()
    manager.set_active_filters(["Emotional Recognition", "Complaint Detection"])
    
    # Test cases
    test_cases = [
        "I'm really frustrated with this product. It doesn't work at all and I want my money back!",
        "Thank you so much for your help! You've been absolutely wonderful and I really appreciate it.",
        "I'm confused about how to use this feature. Can you help me understand it better?",
        "This is terrible! The delivery was late, the product is broken, and customer service was rude. I demand to speak to a manager!",
        "I have a small issue with my billing. There seems to be an extra charge that I don't understand."
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {test_text}")
        
        # Get comprehensive analysis
        analysis = manager.get_comprehensive_analysis(test_text)
        
        print(f"Emotional State: {analysis['emotional_state']}")
        print(f"Sentiment Score: {analysis['sentiment_score']:.2f}")
        print(f"Contains Complaint: {analysis['contains_complaint']}")
        print(f"Urgency Level: {analysis['urgency_level']}")
        
        if analysis['recommended_actions']:
            print(f"Recommended Actions: {', '.join(analysis['recommended_actions'])}")
    
    print("\n🎉 Enhanced filtering tests completed!")


if __name__ == "__main__":
    main()