#!/usr/bin/env python3
"""
Topical Filtering Framework for NPU Voice Assistant
Provides domain-specific transcription filtering and categorization
"""

import re
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class FilterMode(Enum):
    """Filter operation modes"""
    INCLUDE = "include"  # Only include relevant content
    EXCLUDE = "exclude"  # Exclude non-relevant content  
    CATEGORIZE = "categorize"  # Categorize content by topic
    EXTRACT = "extract"  # Extract specific information

@dataclass
class FilterResult:
    """Result of topic filtering operation"""
    original_text: str
    filtered_text: str
    relevance_score: float  # 0.0 to 1.0
    categories: List[str]
    extracted_info: Dict[str, Any]
    confidence: float
    filter_applied: str

class TopicalFilter(ABC):
    """Abstract base class for topical filters"""
    
    def __init__(self, name: str, mode: FilterMode):
        self.name = name
        self.mode = mode
        self.enabled = True
        
    @abstractmethod
    def filter(self, text: str, context: Dict[str, Any] = None) -> FilterResult:
        """Apply filter to text and return result"""
        pass
    
    @abstractmethod
    def get_keywords(self) -> List[str]:
        """Get keywords this filter looks for"""
        pass
    
    def is_relevant(self, text: str) -> bool:
        """Quick relevance check"""
        keywords = self.get_keywords()
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

class MedicalConversationFilter(TopicalFilter):
    """Filter for medical conversations - extracts patient-relevant information"""
    
    def __init__(self):
        super().__init__("Medical Conversation", FilterMode.EXTRACT)
        
        # Medical keywords and phrases
        self.medical_keywords = [
            # Symptoms
            "pain", "ache", "discomfort", "symptom", "feels like", "experiencing",
            "headache", "nausea", "fever", "cough", "shortness of breath",
            "dizzy", "fatigue", "tired", "weakness", "swelling",
            
            # Medical terms
            "diagnosis", "condition", "treatment", "medication", "prescription",
            "test results", "blood pressure", "heart rate", "temperature",
            "allergies", "allergy", "allergic", "medical history",
            
            # Actions and instructions
            "take medication", "follow up", "appointment", "schedule",
            "monitor", "check", "measure", "record", "daily", "twice daily",
            "dosage", "mg", "milligrams", "tablets", "pills",
            
            # Body parts and systems
            "chest", "heart", "lungs", "stomach", "abdomen", "head", "back",
            "joints", "muscles", "skin", "eyes", "throat", "ears",
        ]
        
        # Non-medical (social) patterns to exclude
        self.social_patterns = [
            r"how are you\s*(today|doing)?",
            r"nice to see you",
            r"how's the weather",
            r"have a good day",
            r"take care",
            r"see you (later|soon|next time)",
            r"thank you",
            r"you're welcome",
            r"good morning|good afternoon|good evening",
            r"how's the family",
            r"weekend plans"
        ]
        
    def get_keywords(self) -> List[str]:
        return self.medical_keywords
    
    def filter(self, text: str, context: Dict[str, Any] = None) -> FilterResult:
        """Extract medical information from conversation"""
        
        # Split into sentences for better analysis
        sentences = re.split(r'[.!?]+', text)
        
        medical_content = []
        extracted_info = {
            "symptoms": [],
            "medications": [],
            "instructions": [],
            "measurements": [],
            "appointments": [],
            "concerns": []
        }
        
        total_sentences = len(sentences)
        relevant_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Skip social pleasantries
            if self._is_social_pleasantry(sentence):
                continue
                
            # Check for medical relevance
            if self._is_medically_relevant(sentence):
                relevant_sentences += 1
                medical_content.append(sentence)
                
                # Extract specific information
                self._extract_medical_info(sentence, extracted_info)
        
        # Calculate relevance score
        relevance_score = relevant_sentences / max(total_sentences, 1)
        
        # Create filtered text
        filtered_text = ". ".join(medical_content)
        if filtered_text:
            filtered_text += "."
            
        # Determine categories
        categories = self._categorize_content(extracted_info)
        
        # Calculate confidence
        confidence = self._calculate_confidence(text, extracted_info, relevance_score)
        
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            relevance_score=relevance_score,
            categories=categories,
            extracted_info=extracted_info,
            confidence=confidence,
            filter_applied=self.name
        )
    
    def _is_social_pleasantry(self, text: str) -> bool:
        """Check if text is social pleasantry"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.social_patterns)
    
    def _is_medically_relevant(self, text: str) -> bool:
        """Check if text contains medical content"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.medical_keywords)
    
    def _extract_medical_info(self, text: str, info: Dict[str, List]):
        """Extract specific medical information from text"""
        text_lower = text.lower()
        
        # Extract symptoms
        symptom_patterns = [
            r"(pain|ache|hurt|discomfort) in (my|the) (\w+)",
            r"i (feel|have|am experiencing) (\w+)",
            r"(headache|nausea|fever|cough|dizzy|tired)"
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                symptom = " ".join(match) if isinstance(match, tuple) else match
                info["symptoms"].append(symptom.strip())
        
        # Extract medications
        medication_patterns = [
            r"(take|taking|prescribed) (\w+)",
            r"(\w+) (mg|milligrams|tablets|pills)",
            r"medication (\w+)"
        ]
        
        for pattern in medication_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                med = " ".join(match) if isinstance(match, tuple) else match
                info["medications"].append(med.strip())
        
        # Extract instructions
        instruction_patterns = [
            r"(take|follow|monitor|check|measure) (\w+.*?)(?:\.|$)",
            r"(daily|twice daily|once daily|as needed)",
            r"(follow up|appointment|schedule)"
        ]
        
        for pattern in instruction_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                instruction = " ".join(match) if isinstance(match, tuple) else match
                info["instructions"].append(instruction.strip())
        
        # Extract measurements
        measurement_patterns = [
            r"(\d+/\d+) (blood pressure|bp)",
            r"(\d+) (beats per minute|bpm|heart rate)",
            r"(\d+\.?\d*) (degrees|temperature)",
            r"(\d+) (mg|milligrams)"
        ]
        
        for pattern in measurement_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                measurement = " ".join(match)
                info["measurements"].append(measurement.strip())
    
    def _categorize_content(self, extracted_info: Dict[str, List]) -> List[str]:
        """Categorize extracted medical content"""
        categories = []
        
        if extracted_info["symptoms"]:
            categories.append("symptom_assessment")
        if extracted_info["medications"]:
            categories.append("medication_management")
        if extracted_info["instructions"]:
            categories.append("care_instructions")
        if extracted_info["measurements"]:
            categories.append("vital_signs")
        if extracted_info["appointments"]:
            categories.append("scheduling")
            
        return categories if categories else ["general_medical"]
    
    def _calculate_confidence(self, text: str, extracted_info: Dict, relevance_score: float) -> float:
        """Calculate confidence in filtering accuracy"""
        # Base confidence on relevance score
        confidence = relevance_score
        
        # Boost confidence if we extracted specific information
        extraction_count = sum(len(items) for items in extracted_info.values())
        if extraction_count > 0:
            confidence = min(1.0, confidence + (extraction_count * 0.1))
        
        # Reduce confidence for very short texts
        if len(text.split()) < 5:
            confidence *= 0.7
            
        return confidence

class BusinessMeetingFilter(TopicalFilter):
    """Filter for business meetings - extracts action items and decisions"""
    
    def __init__(self):
        super().__init__("Business Meeting", FilterMode.EXTRACT)
        
        self.business_keywords = [
            "action item", "todo", "follow up", "deadline", "due date",
            "decision", "agreed", "approved", "rejected", "postponed",
            "budget", "cost", "expense", "revenue", "profit",
            "project", "milestone", "deliverable", "timeline",
            "meeting", "conference", "call", "presentation",
            "client", "customer", "vendor", "stakeholder"
        ]
    
    def get_keywords(self) -> List[str]:
        return self.business_keywords
    
    def filter(self, text: str, context: Dict[str, Any] = None) -> FilterResult:
        """Extract business-relevant information"""
        
        # Split into sentences for analysis
        sentences = re.split(r'[.!?]+', text)
        
        business_content = []
        extracted_info = {
            "action_items": [],
            "decisions": [],
            "deadlines": [],
            "participants": [],
            "budget_items": [],
            "projects": []
        }
        
        total_sentences = len(sentences)
        relevant_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for business relevance
            if self._is_business_relevant(sentence):
                relevant_sentences += 1
                business_content.append(sentence)
                
                # Extract specific business information
                self._extract_business_info(sentence, extracted_info)
        
        # Calculate relevance score
        relevance_score = relevant_sentences / max(total_sentences, 1)
        
        # Create filtered text
        filtered_text = ". ".join(business_content)
        if filtered_text:
            filtered_text += "."
            
        # Determine categories
        categories = self._categorize_business_content(extracted_info)
        
        # Calculate confidence
        confidence = self._calculate_business_confidence(text, extracted_info, relevance_score)
        
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            relevance_score=relevance_score,
            categories=categories,
            extracted_info=extracted_info,
            confidence=confidence,
            filter_applied=self.name
        )
    
    def _is_business_relevant(self, text: str) -> bool:
        """Check if text contains business content"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.business_keywords)
    
    def _extract_business_info(self, text: str, info: Dict[str, List]):
        """Extract specific business information from text"""
        text_lower = text.lower()
        
        # Extract action items
        action_patterns = [
            r"(need to|should|must|will) (\w+.*?)(?:\.|$)",
            r"action item:?\s*(.+?)(?:\.|$)",
            r"(follow up|followup) (?:on|with) (.+?)(?:\.|$)",
            r"(\w+) will (do|handle|complete|work on) (.+?)(?:\.|$)"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                action = " ".join(match) if isinstance(match, tuple) else match
                info["action_items"].append(action.strip())
        
        # Extract decisions
        decision_patterns = [
            r"(decided|agreed|approved|rejected) (?:to|that) (.+?)(?:\.|$)",
            r"decision:?\s*(.+?)(?:\.|$)",
            r"we (will|won't|should|shouldn't) (.+?)(?:\.|$)"
        ]
        
        for pattern in decision_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                decision = " ".join(match) if isinstance(match, tuple) else match
                info["decisions"].append(decision.strip())
        
        # Extract deadlines
        deadline_patterns = [
            r"(due|deadline|by) (monday|tuesday|wednesday|thursday|friday|saturday|sunday|\w+day)",
            r"(due|deadline|by) (\w+ \d+)",
            r"(by|before) (end of|eod|next) (\w+)",
            r"deadline:?\s*(.+?)(?:\.|$)"
        ]
        
        for pattern in deadline_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                deadline = " ".join(match) if isinstance(match, tuple) else match
                info["deadlines"].append(deadline.strip())
        
        # Extract budget items
        budget_patterns = [
            r"\$(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(\d+(?:,\d+)*(?:\.\d+)?) dollars",
            r"budget.*?(\$?\d+(?:,\d+)*(?:\.\d+)?)",
            r"cost.*?(\$?\d+(?:,\d+)*(?:\.\d+)?)"
        ]
        
        for pattern in budget_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                budget_item = match if isinstance(match, str) else " ".join(match)
                info["budget_items"].append(budget_item.strip())
        
        # Extract project names
        project_patterns = [
            r"project (\w+)",
            r"(\w+) project",
            r"working on (\w+.*?)(?:\.|$)"
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                project = match if isinstance(match, str) else " ".join(match)
                if len(project.strip()) > 2:  # Filter out very short matches
                    info["projects"].append(project.strip())
    
    def _categorize_business_content(self, extracted_info: Dict[str, List]) -> List[str]:
        """Categorize extracted business content"""
        categories = []
        
        if extracted_info["action_items"]:
            categories.append("action_items")
        if extracted_info["decisions"]:
            categories.append("decisions")
        if extracted_info["deadlines"]:
            categories.append("deadlines")
        if extracted_info["budget_items"]:
            categories.append("budget_discussion")
        if extracted_info["projects"]:
            categories.append("project_updates")
            
        return categories if categories else ["general_business"]
    
    def _calculate_business_confidence(self, text: str, extracted_info: Dict, relevance_score: float) -> float:
        """Calculate confidence in business filtering accuracy"""
        # Base confidence on relevance score
        confidence = relevance_score
        
        # Boost confidence if we extracted specific information
        extraction_count = sum(len(items) for items in extracted_info.values())
        if extraction_count > 0:
            confidence = min(1.0, confidence + (extraction_count * 0.15))
        
        # Reduce confidence for very short texts
        if len(text.split()) < 5:
            confidence *= 0.7
            
        return confidence

class TopicalFilterManager:
    """Manages multiple topical filters"""
    
    def __init__(self):
        self.filters: Dict[str, TopicalFilter] = {}
        self.active_filter: Optional[str] = None
        
    def register_filter(self, filter_instance: TopicalFilter):
        """Register a new filter"""
        self.filters[filter_instance.name] = filter_instance
    
    def set_active_filter(self, filter_name: str):
        """Set the active filter"""
        if filter_name in self.filters:
            self.active_filter = filter_name
        else:
            raise ValueError(f"Filter '{filter_name}' not found")
    
    def filter_text(self, text: str, context: Dict[str, Any] = None) -> Optional[FilterResult]:
        """Apply active filter to text"""
        if not self.active_filter or self.active_filter not in self.filters:
            return None
            
        filter_instance = self.filters[self.active_filter]
        if not filter_instance.enabled:
            return None
            
        return filter_instance.filter(text, context)
    
    def get_available_filters(self) -> List[str]:
        """Get list of available filter names"""
        return list(self.filters.keys())
    
    def disable_filtering(self):
        """Disable all filtering"""
        self.active_filter = None

# Example usage and testing
def demo_medical_filter():
    """Demonstrate medical conversation filtering"""
    
    # Sample medical conversation
    sample_conversation = """
    Good morning, Dr. Smith. How are you today? The weather is nice, isn't it?
    
    Well, I've been experiencing some chest pain for the past two days. 
    It feels like a sharp pain when I breathe deeply. I also have been feeling 
    quite tired and short of breath when I walk upstairs.
    
    Have you taken any medication for this? I took some ibuprofen yesterday 
    but it didn't help much. My blood pressure was 140 over 90 when I checked 
    it this morning.
    
    That's concerning. I want you to take this prescription for lisinopril, 
    5 mg daily. Also schedule a follow-up appointment in two weeks and 
    monitor your blood pressure daily.
    
    Thank you, doctor. Have a great day!
    """
    
    # Apply medical filter
    medical_filter = MedicalConversationFilter()
    result = medical_filter.filter(sample_conversation)
    
    print("=== MEDICAL CONVERSATION FILTER DEMO ===")
    print(f"Original length: {len(sample_conversation)} characters")
    print(f"Filtered length: {len(result.filtered_text)} characters")
    print(f"Relevance Score: {result.relevance_score:.2f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Categories: {', '.join(result.categories)}")
    
    print("\n--- FILTERED TEXT ---")
    print(result.filtered_text)
    
    print("\n--- EXTRACTED INFORMATION ---")
    for category, items in result.extracted_info.items():
        if items:
            print(f"{category.upper()}: {', '.join(items)}")

if __name__ == "__main__":
    demo_medical_filter()