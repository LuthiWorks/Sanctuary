"""Tests for Communication Inhibition System."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from mind.cognitive_core.communication import (
    CommunicationInhibitionSystem,
    InhibitionFactor,
    InhibitionType,
    CommunicationUrge,
    DriveType
)


class TestInhibitionFactor:
    """Tests for InhibitionFactor dataclass."""
    
    def test_factor_creation(self):
        """Test basic inhibition factor creation."""
        factor = InhibitionFactor(
            inhibition_type=InhibitionType.LOW_VALUE,
            strength=0.7,
            reason="Content not valuable enough"
        )
        assert factor.inhibition_type == InhibitionType.LOW_VALUE
        assert factor.strength == 0.7
        assert factor.reason == "Content not valuable enough"
    
    def test_factor_with_duration(self):
        """Test factor with expiration duration."""
        factor = InhibitionFactor(
            inhibition_type=InhibitionType.BAD_TIMING,
            strength=0.8,
            duration=timedelta(seconds=5)
        )
        assert factor.duration == timedelta(seconds=5)
        assert not factor.is_expired()
    
    def test_factor_expiration(self):
        """Test inhibition factor expiration."""
        factor = InhibitionFactor(
            inhibition_type=InhibitionType.REDUNDANCY,
            strength=0.9,
            duration=timedelta(seconds=1)
        )
        # Set created_at to past
        factor.created_at = datetime.now() - timedelta(seconds=2)
        
        assert factor.is_expired()
        assert factor.get_current_strength() == 0.0
    
    def test_factor_without_duration_never_expires(self):
        """Test that factors without duration never expire."""
        factor = InhibitionFactor(
            inhibition_type=InhibitionType.UNCERTAINTY,
            strength=0.6,
            duration=None
        )
        factor.created_at = datetime.now() - timedelta(hours=1)
        
        assert not factor.is_expired()
        assert factor.get_current_strength() == 0.6


class TestCommunicationInhibitionSystem:
    """Tests for CommunicationInhibitionSystem."""
    
    def test_initialization(self):
        """Test inhibition system initialization."""
        system = CommunicationInhibitionSystem()
        assert system.active_inhibitions == []
        assert system.recent_outputs == []
        assert system.last_output_time is None
    
    def test_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "low_value_threshold": 0.5,
            "min_output_spacing_seconds": 10.0,
            "uncertainty_threshold": 0.6
        }
        system = CommunicationInhibitionSystem(config=config)
        
        assert system.low_value_threshold == 0.5
        assert system.min_output_spacing_seconds == 10.0
        assert system.uncertainty_threshold == 0.6
    
    def test_low_value_inhibition(self):
        """Test inhibition from low content value."""
        system = CommunicationInhibitionSystem(config={"low_value_threshold": 0.5})
        
        inhibitions = system._compute_low_value_inhibition(content_value=0.2)
        
        assert len(inhibitions) > 0
        assert any(i.inhibition_type == InhibitionType.LOW_VALUE for i in inhibitions)
        # Strength should be high when value is low
        assert inhibitions[0].strength > 0.5
    
    def test_no_low_value_inhibition_when_valuable(self):
        """Test no inhibition when content is valuable."""
        system = CommunicationInhibitionSystem(config={"low_value_threshold": 0.5})
        
        inhibitions = system._compute_low_value_inhibition(content_value=0.8)
        
        assert len(inhibitions) == 0
    
    def test_bad_timing_inhibition(self):
        """Test inhibition from recent output."""
        system = CommunicationInhibitionSystem(config={"min_output_spacing_seconds": 10.0})
        
        # Set last output to 3 seconds ago
        system.last_output_time = datetime.now() - timedelta(seconds=3)
        
        inhibitions = system._compute_bad_timing_inhibition()
        
        assert len(inhibitions) > 0
        assert any(i.inhibition_type == InhibitionType.BAD_TIMING for i in inhibitions)
        assert inhibitions[0].strength > 0.5
    
    def test_no_bad_timing_after_spacing(self):
        """Test no timing inhibition after spacing period."""
        system = CommunicationInhibitionSystem(config={"min_output_spacing_seconds": 5.0})
        
        # Set last output to 10 seconds ago (beyond spacing)
        system.last_output_time = datetime.now() - timedelta(seconds=10)
        
        inhibitions = system._compute_bad_timing_inhibition()
        
        assert len(inhibitions) == 0
    
    def test_redundancy_inhibition(self):
        """Test inhibition from redundant content."""
        system = CommunicationInhibitionSystem(
            config={"redundancy_similarity_threshold": 0.6}
        )
        
        # Add recent output with keywords
        system.recent_outputs.append({
            'timestamp': datetime.now(),
            'keywords': {'hello', 'world', 'testing', 'system'}
        })
        
        # Create workspace with similar content
        workspace_state = MagicMock()
        percept = MagicMock()
        percept.raw = "hello world testing application"
        workspace_state.percepts = {'p1': percept}
        
        inhibitions = system._compute_redundancy_inhibition(workspace_state)
        
        # Should detect high similarity
        assert len(inhibitions) > 0
        assert inhibitions[0].inhibition_type == InhibitionType.REDUNDANCY
    
    def test_no_redundancy_with_different_content(self):
        """Test no redundancy with different content."""
        system = CommunicationInhibitionSystem()
        
        system.recent_outputs.append({
            'timestamp': datetime.now(),
            'keywords': {'completely', 'different', 'content'}
        })
        
        workspace_state = MagicMock()
        percept = MagicMock()
        percept.raw = "totally new and unique information"
        workspace_state.percepts = {'p1': percept}
        
        inhibitions = system._compute_redundancy_inhibition(workspace_state)
        
        assert len(inhibitions) == 0
    
    def test_respect_silence_inhibition(self):
        """Test inhibition from respecting silence."""
        system = CommunicationInhibitionSystem()
        
        # Low arousal, neutral valence, weak urges
        emotional_state = {"valence": 0.1, "arousal": 0.2, "dominance": 0.5}
        weak_urge = MagicMock()
        weak_urge.intensity = 0.2
        weak_urge.priority = 0.3
        
        inhibitions = system._compute_respect_silence_inhibition(
            emotional_state, [weak_urge]
        )
        
        assert len(inhibitions) > 0
        assert inhibitions[0].inhibition_type == InhibitionType.RESPECT_SILENCE
    
    def test_no_respect_silence_with_strong_emotion(self):
        """Test no silence inhibition with strong emotions."""
        system = CommunicationInhibitionSystem()
        
        # High arousal should not trigger silence
        emotional_state = {"valence": 0.8, "arousal": 0.9, "dominance": 0.5}
        
        inhibitions = system._compute_respect_silence_inhibition(emotional_state, [])
        
        assert len(inhibitions) == 0
    
    def test_still_processing_inhibition(self):
        """Test inhibition from still processing."""
        system = CommunicationInhibitionSystem()
        
        # Create workspace with processing percepts
        workspace_state = MagicMock()
        percept1 = MagicMock()
        percept1.source = "introspection"
        percept2 = MagicMock()
        percept2.source = "processing"
        workspace_state.percepts = {'p1': percept1, 'p2': percept2}
        
        inhibitions = system._compute_still_processing_inhibition(workspace_state)
        
        assert len(inhibitions) > 0
        assert inhibitions[0].inhibition_type == InhibitionType.STILL_PROCESSING
    
    def test_no_still_processing_without_indicators(self):
        """Test no processing inhibition without indicators."""
        system = CommunicationInhibitionSystem()
        
        workspace_state = MagicMock()
        percept = MagicMock()
        percept.source = "human_input"
        workspace_state.percepts = {'p1': percept}
        
        inhibitions = system._compute_still_processing_inhibition(workspace_state)
        
        assert len(inhibitions) == 0
    
    def test_uncertainty_inhibition(self):
        """Test inhibition from uncertainty."""
        system = CommunicationInhibitionSystem(config={"uncertainty_threshold": 0.7})
        
        inhibitions = system._compute_uncertainty_inhibition(confidence=0.3)
        
        assert len(inhibitions) > 0
        assert inhibitions[0].inhibition_type == InhibitionType.UNCERTAINTY
        # Low confidence should produce high inhibition
        assert inhibitions[0].strength > 0.5
    
    def test_no_uncertainty_with_high_confidence(self):
        """Test no uncertainty inhibition with high confidence."""
        system = CommunicationInhibitionSystem(config={"uncertainty_threshold": 0.7})
        
        inhibitions = system._compute_uncertainty_inhibition(confidence=0.9)
        
        assert len(inhibitions) == 0
    
    def test_recent_output_inhibition(self):
        """Test inhibition from high output frequency."""
        system = CommunicationInhibitionSystem(config={
            "max_output_frequency_per_minute": 2,
            "recent_output_window_minutes": 1
        })
        
        # Add many recent outputs
        now = datetime.now()
        for i in range(5):
            system.recent_outputs.append({
                'timestamp': now - timedelta(seconds=i*10),
                'keywords': set()
            })
        
        inhibitions = system._compute_recent_output_inhibition()
        
        assert len(inhibitions) > 0
        assert inhibitions[0].inhibition_type == InhibitionType.RECENT_OUTPUT
    
    def test_no_recent_output_inhibition_low_frequency(self):
        """Test no inhibition with low output frequency."""
        system = CommunicationInhibitionSystem(config={
            "max_output_frequency_per_minute": 10,
            "recent_output_window_minutes": 5
        })
        
        # Add few recent outputs
        system.recent_outputs.append({
            'timestamp': datetime.now(),
            'keywords': set()
        })
        
        inhibitions = system._compute_recent_output_inhibition()
        
        assert len(inhibitions) == 0
    
    def test_total_inhibition_computation(self):
        """Test total inhibition combines factors correctly."""
        system = CommunicationInhibitionSystem()
        
        # Add some inhibitions manually
        system.active_inhibitions = [
            InhibitionFactor(InhibitionType.LOW_VALUE, 0.8, priority=0.7),
            InhibitionFactor(InhibitionType.UNCERTAINTY, 0.6, priority=0.6),
            InhibitionFactor(InhibitionType.BAD_TIMING, 0.4, priority=0.5)
        ]
        
        total = system.get_total_inhibition()
        
        assert 0 < total <= 1.0
    
    def test_strongest_inhibition(self):
        """Test getting strongest inhibition."""
        system = CommunicationInhibitionSystem()
        
        weak_inhibition = InhibitionFactor(InhibitionType.RESPECT_SILENCE, 0.3, priority=0.4)
        strong_inhibition = InhibitionFactor(InhibitionType.UNCERTAINTY, 0.9, priority=0.8)
        
        system.active_inhibitions = [weak_inhibition, strong_inhibition]
        
        strongest = system.get_strongest_inhibition()
        
        assert strongest == strong_inhibition
    
    def test_inhibition_cleanup(self):
        """Test expired inhibitions are cleaned up."""
        system = CommunicationInhibitionSystem()
        
        # Add an expired inhibition
        expired_inhibition = InhibitionFactor(
            InhibitionType.BAD_TIMING,
            0.8,
            duration=timedelta(seconds=1)
        )
        expired_inhibition.created_at = datetime.now() - timedelta(seconds=2)
        
        # Add a fresh inhibition
        fresh_inhibition = InhibitionFactor(InhibitionType.UNCERTAINTY, 0.7)
        
        system.active_inhibitions = [expired_inhibition, fresh_inhibition]
        system._cleanup_expired_inhibitions()
        
        assert expired_inhibition not in system.active_inhibitions
        assert fresh_inhibition in system.active_inhibitions
    
    def test_should_inhibit_decision(self):
        """Test should_inhibit decision logic."""
        system = CommunicationInhibitionSystem()
        
        # High inhibition
        system.active_inhibitions = [
            InhibitionFactor(InhibitionType.LOW_VALUE, 0.9, priority=0.8)
        ]
        
        # Weak urge
        weak_urge = MagicMock()
        weak_urge.intensity = 0.2
        weak_urge.priority = 0.3
        weak_urge.get_current_intensity = lambda: 0.2
        
        # Should inhibit
        assert system.should_inhibit([weak_urge], threshold=0.5)
    
    def test_should_not_inhibit_strong_urges(self):
        """Test should not inhibit with strong urges."""
        system = CommunicationInhibitionSystem()
        
        # Moderate inhibition
        system.active_inhibitions = [
            InhibitionFactor(InhibitionType.BAD_TIMING, 0.5, priority=0.6)
        ]
        
        # Strong urge
        strong_urge = MagicMock()
        strong_urge.intensity = 0.9
        strong_urge.priority = 0.8
        strong_urge.get_current_intensity = lambda: 0.9
        
        # Should not inhibit
        assert not system.should_inhibit([strong_urge], threshold=0.5)
    
    def test_record_output(self):
        """Test recording output."""
        system = CommunicationInhibitionSystem()
        
        assert system.last_output_time is None
        assert len(system.recent_outputs) == 0
        
        system.record_output("Hello world, this is a test message")
        
        assert system.last_output_time is not None
        assert len(system.recent_outputs) == 1
        assert 'keywords' in system.recent_outputs[0]
        assert 'hello' in system.recent_outputs[0]['keywords']
    
    def test_record_output_limits_history(self):
        """Test that recent outputs are limited."""
        system = CommunicationInhibitionSystem(config={"max_recent_outputs": 3})
        
        # Add many outputs
        for i in range(10):
            system.record_output(f"Output {i}")
        
        # Should only keep max_recent_outputs
        assert len(system.recent_outputs) == 3
    
    def test_inhibition_summary(self):
        """Test inhibition summary generation."""
        system = CommunicationInhibitionSystem()
        system.active_inhibitions = [
            InhibitionFactor(InhibitionType.LOW_VALUE, 0.7),
            InhibitionFactor(InhibitionType.UNCERTAINTY, 0.5)
        ]
        
        summary = system.get_inhibition_summary()
        
        assert "total_inhibition" in summary
        assert "active_inhibitions" in summary
        assert summary["active_inhibitions"] == 2
        assert "inhibitions_by_type" in summary


class TestInhibitionIntegration:
    """Integration tests for inhibition computation."""
    
    def test_full_inhibition_computation(self):
        """Test computing all inhibitions from mock state."""
        system = CommunicationInhibitionSystem()
        
        # Mock workspace state
        workspace_state = MagicMock()
        workspace_state.percepts = {}
        
        # Mock urges
        urge = MagicMock()
        urge.intensity = 0.5
        urge.priority = 0.6
        
        inhibitions = system.compute_inhibitions(
            workspace_state=workspace_state,
            urges=[urge],
            confidence=0.4,  # Low confidence
            content_value=0.2,  # Low value
            emotional_state={"valence": 0.5, "arousal": 0.5}
        )
        
        # Should have generated some inhibitions
        assert len(system.active_inhibitions) > 0
        assert system.get_total_inhibition() > 0
    
    def test_drive_vs_inhibition_balance(self):
        """Test that drive and inhibition systems can work together."""
        from mind.cognitive_core.communication import CommunicationDriveSystem
        
        drive_system = CommunicationDriveSystem()
        inhibition_system = CommunicationInhibitionSystem()
        
        # Create some urges
        drive_system.active_urges = [
            CommunicationUrge(DriveType.INSIGHT, 0.7, priority=0.7),
            CommunicationUrge(DriveType.EMOTIONAL, 0.6, priority=0.6)
        ]
        
        # Create some inhibitions
        inhibition_system.active_inhibitions = [
            InhibitionFactor(InhibitionType.LOW_VALUE, 0.5, priority=0.6)
        ]
        
        total_drive = drive_system.get_total_drive()
        total_inhibition = inhibition_system.get_total_inhibition()
        
        # Both should be > 0
        assert total_drive > 0
        assert total_inhibition > 0
        
        # Test decision
        should_inhibit = inhibition_system.should_inhibit(
            drive_system.active_urges,
            threshold=0.8
        )
        
        # Decision should be boolean
        assert isinstance(should_inhibit, bool)


class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        system = CommunicationInhibitionSystem()
        workspace_state = MagicMock()
        workspace_state.percepts = {}
        
        inhibitions = system.compute_inhibitions(
            workspace_state=workspace_state,
            urges=[],
            confidence=0.5,
            content_value=0.5,
            emotional_state={}
        )
        
        # Should handle gracefully without errors
        assert isinstance(inhibitions, list)
    
    def test_none_workspace_state(self):
        """Test with workspace state missing percepts."""
        system = CommunicationInhibitionSystem()
        workspace_state = MagicMock()
        del workspace_state.percepts
        
        inhibitions = system.compute_inhibitions(
            workspace_state=workspace_state,
            urges=[],
            confidence=0.8,
            content_value=0.8,
            emotional_state={"valence": 0.5, "arousal": 0.3}
        )
        
        # Should not crash
        assert isinstance(inhibitions, list)
    
    def test_invalid_config_values(self):
        """Test that invalid config values are clamped."""
        system = CommunicationInhibitionSystem(config={
            "low_value_threshold": 1.5,  # Should clamp to 1.0
            "uncertainty_threshold": -0.2,  # Should clamp to 0.0
            "min_output_spacing_seconds": -5.0,  # Should clamp to 0.0
            "max_inhibitions": 0  # Should clamp to 1
        })
        
        assert system.low_value_threshold == 1.0
        assert system.uncertainty_threshold == 0.0
        assert system.min_output_spacing_seconds == 0.0
        assert system.max_inhibitions == 1
    
    def test_many_inhibitions_limited(self):
        """Test that excess inhibitions are limited."""
        system = CommunicationInhibitionSystem(config={"max_inhibitions": 3})
        
        # Add many inhibitions
        for i in range(10):
            system.active_inhibitions.append(
                InhibitionFactor(InhibitionType.LOW_VALUE, 0.5 + i * 0.05, priority=0.5)
            )
        
        system._limit_active_inhibitions()
        
        # Should keep only strongest 3
        assert len(system.active_inhibitions) == 3
    
    def test_zero_strength_inhibition(self):
        """Test handling of zero-strength inhibition."""
        factor = InhibitionFactor(InhibitionType.RESPECT_SILENCE, 0.0)
        assert factor.get_current_strength() == 0.0
    
    def test_should_inhibit_with_no_urges(self):
        """Test should_inhibit with no urges."""
        system = CommunicationInhibitionSystem()
        system.active_inhibitions = [
            InhibitionFactor(InhibitionType.LOW_VALUE, 0.5, priority=0.5)
        ]
        
        # With no urges, should inhibit
        assert system.should_inhibit([], threshold=0.5)
    
    def test_should_inhibit_with_no_inhibitions(self):
        """Test should_inhibit with no inhibitions."""
        system = CommunicationInhibitionSystem()
        
        urge = MagicMock()
        urge.intensity = 0.8
        urge.priority = 0.7
        urge.get_current_intensity = lambda: 0.8
        
        # With no inhibitions, should not inhibit
        assert not system.should_inhibit([urge], threshold=0.5)
    
    def test_keyword_extraction_robust(self):
        """Test keyword extraction handles edge cases."""
        system = CommunicationInhibitionSystem()
        
        # Test with various punctuation and short words
        percept = MagicMock()
        percept.raw = "Hi! This is a test... with (brackets) and short: a, I, to"
        
        keywords = system._extract_keywords({'p1': percept})
        
        # Should filter short words and punctuation
        assert 'test' in keywords
        assert 'brackets' in keywords
        assert 'short' in keywords
        # Short words should be filtered
        assert 'hi' not in keywords
        assert 'a' not in keywords
    
    def test_similarity_calculation_edge_cases(self):
        """Test similarity calculation with edge cases."""
        system = CommunicationInhibitionSystem()
        
        # Empty sets
        assert system._calculate_similarity(set(), set()) == 0.0
        
        # Identical sets
        s = {'hello', 'world'}
        assert system._calculate_similarity(s, s) == 1.0
        
        # No overlap
        s1 = {'hello', 'world'}
        s2 = {'goodbye', 'moon'}
        assert system._calculate_similarity(s1, s2) == 0.0
        
        # Partial overlap
        s1 = {'hello', 'world', 'test'}
        s2 = {'hello', 'test', 'system'}
        similarity = system._calculate_similarity(s1, s2)
        assert 0 < similarity < 1
