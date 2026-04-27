"""
Communication Inhibition System - Reasons not to communicate.

This module computes factors that suppress the urge to speak:
low-value content, bad timing, redundancy, respect for silence,
still processing, uncertainty, and recent output frequency.

It provides the counterbalancing force to the Communication Drive System,
enabling genuine communication agency through selective silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Constants for keyword extraction
PUNCTUATION = '.,!?;:()[]{}"\'"'
MIN_KEYWORD_LENGTH = 3


class InhibitionType(Enum):
    """Types of communication inhibitions."""
    LOW_VALUE = "low_value"               # Content isn't valuable enough to share
    BAD_TIMING = "bad_timing"             # Just spoke, need spacing between outputs
    REDUNDANCY = "redundancy"             # Already said this or something similar
    RESPECT_SILENCE = "respect_silence"   # Silence is the appropriate response
    STILL_PROCESSING = "still_processing" # Not ready to respond yet
    UNCERTAINTY = "uncertainty"           # Too uncertain to commit to a response
    RECENT_OUTPUT = "recent_output"       # High output frequency, give space
    SYSTEM_OVERLOAD = "system_overload"   # Cognitive system is bottlenecked


@dataclass
class InhibitionFactor:
    """
    Represents a specific inhibition against communication.
    
    Attributes:
        inhibition_type: The type of inhibition
        strength: How strong the inhibition is (0.0 to 1.0)
        reason: Why this inhibition exists
        created_at: When this inhibition arose
        duration: How long this inhibition lasts (None = indefinite)
        priority: Relative priority among inhibitions
    """
    inhibition_type: InhibitionType
    strength: float
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    duration: Optional[timedelta] = None
    priority: float = 0.5
    
    def get_current_strength(self) -> float:
        """
        Get strength after checking expiration.
        
        Returns 0.0 if expired, otherwise returns full strength.
        Duration-based inhibitions don't decay, they expire.
        """
        if self.is_expired():
            return 0.0
        return max(0.0, min(1.0, self.strength))
    
    def is_expired(self) -> bool:
        """Check if inhibition has expired based on duration."""
        if self.duration is None:
            return False
        elapsed = datetime.now() - self.created_at
        return elapsed >= self.duration


class CommunicationInhibitionSystem:
    """
    Computes reasons not to communicate based on state, timing, and context.
    Provides counterbalancing force to the Communication Drive System.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, rhythm_model: Optional[Any] = None):
        """
        Initialize with optional configuration and rhythm model.
        
        Args:
            config: Optional configuration dict
            rhythm_model: Optional ConversationalRhythmModel for timing inhibition
        """
        self.active_inhibitions: List[InhibitionFactor] = []
        self.recent_outputs: List[Dict[str, Any]] = []
        self.last_output_time: Optional[datetime] = None
        self.rhythm_model = rhythm_model
        self._load_config(config or {})
    
    def _load_config(self, config: Dict[str, Any]) -> None:
        """Load and validate configuration parameters."""
        self.low_value_threshold = self._clamp(config.get("low_value_threshold", 0.3), 0.0, 1.0)
        self.min_output_spacing_seconds = max(0.0, config.get("min_output_spacing_seconds", 5.0))
        self.redundancy_similarity_threshold = self._clamp(config.get("redundancy_similarity_threshold", 0.8), 0.0, 1.0)
        self.uncertainty_threshold = self._clamp(config.get("uncertainty_threshold", 0.7), 0.0, 1.0)
        self.max_output_frequency_per_minute = max(1, config.get("max_output_frequency_per_minute", 6))
        self.recent_output_window_minutes = max(1, config.get("recent_output_window_minutes", 5))
        self.max_inhibitions = max(1, config.get("max_inhibitions", 10))
        self.max_recent_outputs = max(1, config.get("max_recent_outputs", 20))
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))
    
    def compute_inhibitions(
        self,
        workspace_state: Any,
        urges: List[Any],
        confidence: float,
        content_value: float,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> List[InhibitionFactor]:
        """
        Compute all inhibitions from current state.
        
        Returns:
            List of newly generated inhibitions
        """
        emotional_state = emotional_state or {}
        
        # Compute all inhibition types (now 8 with timing)
        new_inhibitions = [
            *self._compute_low_value_inhibition(content_value),
            *self._compute_bad_timing_inhibition(),
            *self._compute_redundancy_inhibition(workspace_state),
            *self._compute_respect_silence_inhibition(emotional_state, urges),
            *self._compute_still_processing_inhibition(workspace_state),
            *self._compute_uncertainty_inhibition(confidence),
            *self._compute_recent_output_inhibition(),
            *self._compute_timing_inhibition()
        ]
        
        # Maintain active inhibitions list
        self.active_inhibitions.extend(new_inhibitions)
        self._cleanup_expired_inhibitions()
        self._limit_active_inhibitions()
        
        return new_inhibitions
    
    def _limit_active_inhibitions(self) -> None:
        """Keep only the strongest inhibitions up to max_inhibitions limit."""
        if len(self.active_inhibitions) > self.max_inhibitions:
            self.active_inhibitions.sort(
                key=lambda i: i.get_current_strength() * i.priority,
                reverse=True
            )
            self.active_inhibitions = self.active_inhibitions[:self.max_inhibitions]
    
    def _compute_low_value_inhibition(self, content_value: float) -> List[InhibitionFactor]:
        """Compute inhibition from low-value content."""
        if content_value >= self.low_value_threshold:
            return []
        
        strength = 1.0 - (content_value / self.low_value_threshold)
        return [InhibitionFactor(
            inhibition_type=InhibitionType.LOW_VALUE,
            strength=strength,
            reason=f"Content value too low ({content_value:.2f})",
            priority=0.7
        )]
    
    def _compute_bad_timing_inhibition(self) -> List[InhibitionFactor]:
        """Compute inhibition from recent output (spacing constraint)."""
        if self.last_output_time is None:
            return []
        
        elapsed = (datetime.now() - self.last_output_time).total_seconds()
        if elapsed >= self.min_output_spacing_seconds:
            return []
        
        strength = 1.0 - (elapsed / self.min_output_spacing_seconds)
        return [InhibitionFactor(
            inhibition_type=InhibitionType.BAD_TIMING,
            strength=strength,
            reason=f"Only {elapsed:.1f}s since last output",
            priority=0.8,
            duration=timedelta(seconds=self.min_output_spacing_seconds - elapsed)
        )]
    
    def _compute_redundancy_inhibition(self, workspace_state: Any) -> List[InhibitionFactor]:
        """Compute inhibition from redundant content using keyword similarity."""
        if not hasattr(workspace_state, 'percepts') or not self.recent_outputs:
            return []
        
        current_keywords = self._extract_keywords(workspace_state.percepts)
        if not current_keywords:
            return []
        
        # Check last 5 outputs for similarity
        for recent_output in self.recent_outputs[-5:]:
            recent_keywords = recent_output.get('keywords', set())
            if not recent_keywords:
                continue
            
            similarity = self._calculate_similarity(current_keywords, recent_keywords)
            if similarity >= self.redundancy_similarity_threshold:
                return [InhibitionFactor(
                    inhibition_type=InhibitionType.REDUNDANCY,
                    strength=similarity,
                    reason=f"Content {similarity:.0%} similar to recent output",
                    priority=0.75,
                    duration=timedelta(minutes=2)
                )]
        
        return []
    
    def _extract_keywords(self, percepts: Dict) -> set:
        """Extract keywords from percepts for redundancy detection."""
        keywords = set()
        for percept in percepts.values():
            content = str(getattr(percept, 'raw', ''))
            if content:
                words = [w.strip(PUNCTUATION).lower() 
                        for w in content.split()
                        if len(w.strip(PUNCTUATION)) > MIN_KEYWORD_LENGTH]
                keywords.update(words)
        return keywords
    
    def _calculate_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _compute_respect_silence_inhibition(
        self,
        emotional_state: Dict[str, float],
        urges: List[Any]
    ) -> List[InhibitionFactor]:
        """Compute inhibition from respecting silence (low arousal, weak urges)."""
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.0)
        
        # Low arousal + neutral valence = contemplative silence
        if abs(arousal) >= 0.3 or abs(valence) >= 0.3:
            return []
        
        # Check urge strength
        total_urge = sum(
            getattr(u, 'intensity', 0) * getattr(u, 'priority', 0.5)
            for u in urges
        ) / max(1, len(urges))
        
        if total_urge < 0.4:
            return [InhibitionFactor(
                inhibition_type=InhibitionType.RESPECT_SILENCE,
                strength=0.6,
                reason="Low arousal, weak urges - silence appropriate",
                priority=0.5,
                duration=timedelta(seconds=30)
            )]
        
        return []
    
    def _compute_still_processing_inhibition(self, workspace_state: Any) -> List[InhibitionFactor]:
        """Compute inhibition from active processing indicators."""
        if not hasattr(workspace_state, 'percepts'):
            return []
        
        processing_percepts = [
            p for p in workspace_state.percepts.values()
            if getattr(p, 'source', '').lower() in ['introspection', 'processing']
        ]
        
        if processing_percepts:
            return [InhibitionFactor(
                inhibition_type=InhibitionType.STILL_PROCESSING,
                strength=0.7,
                reason=f"{len(processing_percepts)} active processing percepts",
                priority=0.65,
                duration=timedelta(seconds=10)
            )]
        
        return []
    
    def _compute_uncertainty_inhibition(self, confidence: float) -> List[InhibitionFactor]:
        """Compute inhibition from low confidence."""
        if confidence >= self.uncertainty_threshold:
            return []
        
        strength = 1.0 - (confidence / self.uncertainty_threshold)
        return [InhibitionFactor(
            inhibition_type=InhibitionType.UNCERTAINTY,
            strength=strength,
            reason=f"Confidence too low ({confidence:.2f})",
            priority=0.7
        )]
    
    def _compute_recent_output_inhibition(self) -> List[InhibitionFactor]:
        """Compute inhibition from high output frequency."""
        if not self.recent_outputs:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=self.recent_output_window_minutes)
        recent_count = sum(
            1 for output in self.recent_outputs
            if output.get('timestamp', datetime.min) > cutoff_time
        )
        
        outputs_per_minute = recent_count / self.recent_output_window_minutes
        if outputs_per_minute <= self.max_output_frequency_per_minute:
            return []
        
        excess_ratio = outputs_per_minute / self.max_output_frequency_per_minute
        strength = min(1.0, (excess_ratio - 1.0) / 2.0 + 0.5)
        
        return [InhibitionFactor(
            inhibition_type=InhibitionType.RECENT_OUTPUT,
            strength=strength,
            reason=f"Output frequency ({outputs_per_minute:.1f}/min) too high",
            priority=0.6,
            duration=timedelta(minutes=1)
        )]
    
    def _compute_timing_inhibition(self) -> List[InhibitionFactor]:
        """
        Compute inhibition from conversational rhythm/timing using rhythm model.
        
        Uses the ConversationalRhythmModel to determine if now is an appropriate
        time to speak based on conversation flow, turn-taking, and pauses.
        """
        if self.rhythm_model is None:
            return []
        
        appropriateness = self.rhythm_model.get_timing_appropriateness()
        
        # Only create inhibition if timing is poor (< 0.3)
        if appropriateness >= 0.3:
            return []
        
        # Stronger inhibition for worse timing
        strength = 1.0 - appropriateness
        
        return [InhibitionFactor(
            inhibition_type=InhibitionType.BAD_TIMING,
            strength=strength,
            reason=f"Not a natural pause point (appropriateness: {appropriateness:.2f})",
            priority=0.85,  # High priority for conversational rhythm
            duration=timedelta(seconds=self.rhythm_model.get_suggested_wait_time())
        )]
    
    def _cleanup_expired_inhibitions(self) -> None:
        """Remove expired inhibitions."""
        self.active_inhibitions = [i for i in self.active_inhibitions if not i.is_expired()]
    
    def get_total_inhibition(self) -> float:
        """
        Get combined inhibition using diminishing returns (1/n weighting).
        Returns value in range [0.0, 1.0].
        """
        if not self.active_inhibitions:
            return 0.0
        
        sorted_inhibitions = sorted(
            self.active_inhibitions,
            key=lambda i: i.get_current_strength() * i.priority,
            reverse=True
        )
        
        total = sum(
            inhibition.get_current_strength() * inhibition.priority / (i + 1)
            for i, inhibition in enumerate(sorted_inhibitions)
        )
        
        return min(1.0, total)
    
    def should_inhibit(self, urges: List[Any], threshold: float = 0.5) -> bool:
        """
        Decide whether to inhibit communication.
        
        Args:
            urges: Active communication urges
            threshold: Inhibition multiplier (default: 0.5, higher = more powerful)
            
        Returns:
            True if should inhibit (stay silent), False otherwise
        """
        # Validate inputs
        if threshold <= 0:
            return False
        
        total_inhibition = self.get_total_inhibition()
        
        # Calculate average urge strength
        total_urge = sum(
            getattr(u, 'get_current_intensity', lambda: getattr(u, 'intensity', 0))() *
            getattr(u, 'priority', 0.5)
            for u in urges
        ) / max(1, len(urges)) if urges else 0.0
        
        # Inhibit if effective inhibition exceeds urge
        return (total_inhibition * threshold) > total_urge
    
    def record_output(self, content: Optional[str] = None) -> None:
        """Record output timestamp and content for tracking."""
        now = datetime.now()
        self.last_output_time = now
        
        # Extract keywords if content provided
        keywords = set()
        if content:
            words = [w.strip(PUNCTUATION).lower() 
                    for w in content.split()
                    if len(w.strip(PUNCTUATION)) > MIN_KEYWORD_LENGTH]
            keywords = set(words)
        
        self.recent_outputs.append({
            'timestamp': now,
            'keywords': keywords,
            'content_preview': content[:100] if content else None
        })
        
        # Limit history size
        if len(self.recent_outputs) > self.max_recent_outputs:
            self.recent_outputs = self.recent_outputs[-self.max_recent_outputs:]
    
    def get_strongest_inhibition(self) -> Optional[InhibitionFactor]:
        """Get strongest inhibition by weighted strength."""
        if not self.active_inhibitions:
            return None
        
        return max(
            self.active_inhibitions,
            key=lambda i: i.get_current_strength() * i.priority
        )
    
    def get_inhibition_summary(self) -> Dict[str, Any]:
        """Get summary of inhibition state."""
        return {
            "total_inhibition": self.get_total_inhibition(),
            "active_inhibitions": len(self.active_inhibitions),
            "strongest_inhibition": self.get_strongest_inhibition(),
            "inhibitions_by_type": {
                it.value: len([i for i in self.active_inhibitions if i.inhibition_type == it])
                for it in InhibitionType
            },
            "time_since_output": (
                (datetime.now() - self.last_output_time).total_seconds()
                if self.last_output_time else None
            ),
            "recent_output_count": len([
                o for o in self.recent_outputs
                if o.get('timestamp', datetime.min) > 
                   datetime.now() - timedelta(minutes=self.recent_output_window_minutes)
            ])
        }
