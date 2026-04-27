"""
Unit tests for advanced meta-cognition features (Phase 4.1).

Tests cover:
- Self-model tracking and updates
- Behavioral consistency analysis
- Introspective journal functionality
- Meta-cognitive metrics and health reporting
- Enhanced percept types
- Prediction accuracy measurement
"""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path
from collections import deque

from mind.cognitive_core.meta_cognition import SelfMonitor, IntrospectiveJournal
from mind.cognitive_core.workspace import (
    GlobalWorkspace, WorkspaceSnapshot, Percept, Goal, GoalType
)
from mind.cognitive_core.action import Action, ActionType


class TestSelfModelTracking:
    """Test self-model tracking functionality"""
    
    def test_self_model_initialization(self):
        """Test that self-model is initialized correctly"""
        monitor = SelfMonitor()
        
        assert "capabilities" in monitor.self_model
        assert "limitations" in monitor.self_model
        assert "preferences" in monitor.self_model
        assert "behavioral_traits" in monitor.self_model
        assert "values_hierarchy" in monitor.self_model
        
        assert isinstance(monitor.self_model["capabilities"], dict)
        assert isinstance(monitor.self_model["limitations"], dict)
        assert isinstance(monitor.prediction_history, deque)
        assert isinstance(monitor.behavioral_log, deque)
    
    def test_update_self_model_with_successful_action(self):
        """Test self-model update with successful action"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        outcome = {
            "action_type": "SPEAK",
            "success": True,
            "reason": "Response generated"
        }
        
        monitor.update_self_model(snapshot, outcome)
        
        # Check that behavioral log was updated
        assert len(monitor.behavioral_log) == 1
        
        # Check that capabilities were updated
        assert "SPEAK" in monitor.self_model["capabilities"]
        assert monitor.self_model["capabilities"]["SPEAK"]["attempts"] == 1
        assert monitor.self_model["capabilities"]["SPEAK"]["successes"] == 1
        assert monitor.self_model["capabilities"]["SPEAK"]["confidence"] == 1.0
    
    def test_update_self_model_with_failed_action(self):
        """Test self-model update with failed action"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        outcome = {
            "action_type": "TOOL_CALL",
            "success": False,
            "reason": "Tool not available"
        }
        
        monitor.update_self_model(snapshot, outcome)
        
        # Check that limitations were recorded
        assert "TOOL_CALL" in monitor.self_model["limitations"]
        assert len(monitor.self_model["limitations"]["TOOL_CALL"]) == 1
        assert monitor.self_model["limitations"]["TOOL_CALL"][0]["reason"] == "Tool not available"
    
    def test_update_self_model_frequency_control(self):
        """Test that self-model updates respect frequency setting"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 5})
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        outcome = {"action_type": "SPEAK", "success": True}
        
        # First 4 updates should not update capabilities
        for i in range(4):
            monitor.update_self_model(snapshot, outcome)
        
        assert "SPEAK" not in monitor.self_model["capabilities"]
        
        # 5th update should trigger
        monitor.update_self_model(snapshot, outcome)
        assert "SPEAK" in monitor.self_model["capabilities"]
    
    def test_predict_behavior_with_empty_model(self):
        """Test behavior prediction with empty self-model"""
        monitor = SelfMonitor()
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        prediction = monitor.predict_behavior(snapshot)
        
        assert "likely_actions" in prediction
        assert "emotional_prediction" in prediction
        assert "goal_priorities" in prediction
        assert "confidence" in prediction
        assert prediction["confidence"] == 0.0  # No data yet
    
    def test_predict_behavior_with_trained_model(self):
        """Test behavior prediction with trained self-model"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        # Train the model
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        for _ in range(10):
            outcome = {"action_type": "SPEAK", "success": True}
            monitor.update_self_model(snapshot, outcome)
        
        # Now predict
        prediction = monitor.predict_behavior(snapshot)
        
        assert len(prediction["likely_actions"]) > 0
        assert prediction["likely_actions"][0]["action"] == "SPEAK"
        assert prediction["likely_actions"][0]["likelihood"] > 0.7
        assert prediction["confidence"] > 0.0
    
    def test_measure_prediction_accuracy_empty(self):
        """Test prediction accuracy measurement with no data"""
        monitor = SelfMonitor()
        
        metrics = monitor.measure_prediction_accuracy()
        
        assert metrics["overall_accuracy"] == 0.0
        assert metrics["sample_size"] == 0
    
    def test_measure_prediction_accuracy_with_data(self):
        """Test prediction accuracy measurement with historical data"""
        monitor = SelfMonitor()
        
        # Add some prediction history
        monitor.prediction_history.append({
            "category": "action",
            "correct": True,
            "confidence": 0.8
        })
        monitor.prediction_history.append({
            "category": "action",
            "correct": False,
            "confidence": 0.6
        })
        monitor.prediction_history.append({
            "category": "emotion",
            "correct": True,
            "confidence": 0.9
        })
        
        metrics = monitor.measure_prediction_accuracy()
        
        assert metrics["overall_accuracy"] == 2/3  # 2 correct out of 3
        assert metrics["action_prediction_accuracy"] == 0.5  # 1 correct out of 2
        assert metrics["emotion_prediction_accuracy"] == 1.0  # 1 correct out of 1
        assert metrics["sample_size"] == 3
        assert 0.0 <= metrics["confidence_calibration"] <= 1.0
    
    def test_behavioral_traits_tracking(self):
        """Test that behavioral traits are tracked over time"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        # Update with different emotional states
        for valence in [0.2, 0.4, 0.6, 0.8, 1.0]:
            snapshot = WorkspaceSnapshot(
                goals=[],
                percepts={},
                emotions={"valence": valence, "arousal": 0.5, "dominance": 0.5},
                memories=[],
                timestamp=datetime.now(),
                cycle_count=0,
                metadata={}
            )
            outcome = {"action_type": "SPEAK", "success": True}
            monitor.update_self_model(snapshot, outcome)
        
        # Check that average valence is tracked
        assert "average_valence" in monitor.self_model["behavioral_traits"]
        avg_valence = monitor.self_model["behavioral_traits"]["average_valence"]
        assert 0.0 <= avg_valence <= 1.0


class TestBehavioralConsistency:
    """Test behavioral consistency analysis"""
    
    def test_analyze_behavioral_consistency_no_issues(self):
        """Test consistency analysis with consistent behavior"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        percept = monitor.analyze_behavioral_consistency(snapshot)
        assert percept is None  # No inconsistencies
    
    def test_detect_value_action_misalignment_low_priority_value_goal(self):
        """Test detection of value goals with low priority"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Create value goal with low priority
        goal = Goal(
            type=GoalType.MAINTAIN_VALUE,
            description="Maintain honesty",
            priority=0.3  # Low priority for a value goal
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        misalignments = monitor.detect_value_action_misalignment(snapshot)
        
        assert len(misalignments) > 0
        assert misalignments[0]["type"] == "value_deprioritization"
        assert misalignments[0]["severity"] > 0.3
    
    def test_detect_capability_claim_without_verification(self):
        """Test detection of unverified capability claims"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        action = Action(
            type=ActionType.SPEAK,
            metadata={"claimed_capability": True}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": [action]}
        )
        
        misalignments = monitor.detect_value_action_misalignment(snapshot)
        
        assert len(misalignments) > 0
        assert any(m["type"] == "honesty_violation" for m in misalignments)
    
    def test_assess_capability_claims_with_known_limitations(self):
        """Test capability assessment with known limitations"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Add known limitations
        monitor.self_model["limitations"]["TOOL_CALL"] = [
            {"reason": "Tool not found", "timestamp": datetime.now().isoformat()},
            {"reason": "Tool failed", "timestamp": datetime.now().isoformat()},
            {"reason": "Tool timeout", "timestamp": datetime.now().isoformat()},
            {"reason": "Tool error", "timestamp": datetime.now().isoformat()}
        ]
        
        action = Action(type=ActionType.TOOL_CALL)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": [action]}
        )
        
        percept = monitor.assess_capability_claims(snapshot)
        
        assert percept is not None
        assert percept.raw["type"] == "capability_assessment"
        assert len(percept.raw["issues"]) > 0
    
    def test_assess_capability_claims_with_low_confidence(self):
        """Test capability assessment with low confidence actions"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Add capability with low confidence
        monitor.self_model["capabilities"]["SPEAK"] = {
            "attempts": 10,
            "successes": 2,
            "confidence": 0.2
        }
        
        action = Action(type=ActionType.SPEAK)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": [action]}
        )
        
        percept = monitor.assess_capability_claims(snapshot)
        
        assert percept is not None
        assert "low_confidence_capability" in str(percept.raw)
    
    def test_behavioral_inconsistency_emotional_deviation(self):
        """Test detection of emotional state deviation"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        # Build behavioral log with consistent emotions
        for _ in range(15):
            snapshot = WorkspaceSnapshot(
                goals=[],
                percepts={},
                emotions={"valence": 0.8, "arousal": 0.5, "dominance": 0.5},
                memories=[],
                timestamp=datetime.now(),
                cycle_count=0,
                metadata={}
            )
            outcome = {"action_type": "SPEAK", "success": True}
            monitor.update_self_model(snapshot, outcome)
        
        # Now check with very different emotional state
        deviant_snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.5, "arousal": 0.5, "dominance": 0.5},  # Very different
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        percept = monitor.analyze_behavioral_consistency(deviant_snapshot)
        
        assert percept is not None
        assert percept.raw["type"] == "behavioral_inconsistency"
        assert any("emotional_deviation" in str(inc) for inc in percept.raw["inconsistencies"])
    
    def test_verify_capability_with_trained_model(self):
        """Test capability verification against self-model"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        # Train model with successful actions
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        for _ in range(10):
            outcome = {"action_type": "SPEAK", "success": True}
            monitor.update_self_model(snapshot, outcome)
        
        # Verify capability
        assert monitor._verify_capability(ActionType.SPEAK) is True
    
    def test_verify_capability_unverified(self):
        """Test capability verification for unknown capability"""
        monitor = SelfMonitor()
        
        assert monitor._verify_capability("UNKNOWN_ACTION") is False


class TestIntrospectiveJournal:
    """Test introspective journal functionality"""
    
    def test_journal_initialization(self):
        """Test journal initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            journal_dir = Path(tmpdir)
            with IntrospectiveJournal(journal_dir) as journal:
                assert journal.journal_dir == journal_dir
                assert journal.journal_dir.exists()
                assert hasattr(journal.recent_entries, '__len__')
                assert len(journal.recent_entries) == 0
    
    def test_record_observation(self):
        """Test recording an observation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with IntrospectiveJournal(Path(tmpdir)) as journal:
                observation = {
                    "type": "value_conflict",
                    "description": "Detected conflict",
                    "severity": 0.8
                }

                journal.record_observation(observation)

                assert len(journal.recent_entries) == 1
                assert journal.recent_entries[0]["type"] == "observation"
                assert journal.recent_entries[0]["content"] == observation
    
    def test_record_realization(self):
        """Test recording a realization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with IntrospectiveJournal(Path(tmpdir)) as journal:
                realization = "I seem to prioritize efficiency over thoroughness"
                confidence = 0.75

                journal.record_realization(realization, confidence)

                assert len(journal.recent_entries) == 1
                assert journal.recent_entries[0]["type"] == "realization"
                assert journal.recent_entries[0]["realization"] == realization
                assert journal.recent_entries[0]["confidence"] == confidence
    
    def test_record_question(self):
        """Test recording an existential question"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with IntrospectiveJournal(Path(tmpdir)) as journal:
                question = "Do I genuinely care about this goal?"
                context = {"goal": "test goal", "priority": 0.8}

                journal.record_question(question, context)

                assert len(journal.recent_entries) == 1
                assert journal.recent_entries[0]["type"] == "question"
                assert journal.recent_entries[0]["question"] == question
                assert journal.recent_entries[0]["context"] == context
    
    def test_save_session(self):
        """Test saving journal session to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with IntrospectiveJournal(Path(tmpdir)) as journal:
                # Add some entries
                journal.record_observation({"type": "test", "data": "value"})
                journal.record_realization("Test realization", 0.9)

                # Save session (entries are written incrementally now)
                journal.save_session()

                # Check that entries were persisted (journal files exist)
                journal_files = list(Path(tmpdir).glob("journal_*.jsonl")) + \
                                list(Path(tmpdir).glob("journal_*.json"))
                assert len(journal_files) >= 1, f"Expected journal files in {tmpdir}, found: {list(Path(tmpdir).iterdir())}"

                # Entries should still be in recent_entries for pattern detection
                assert len(journal.recent_entries) == 2
    
    def test_save_empty_session(self):
        """Test saving empty session does nothing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with IntrospectiveJournal(Path(tmpdir)) as journal:
                journal.save_session()

            # No content journal files should be created (empty .jsonl may exist)
            journal_files = list(Path(tmpdir).glob("journal_*.json"))
            assert len(journal_files) == 0
    
    def test_get_recent_patterns_empty(self):
        """Test getting patterns from empty journal"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with IntrospectiveJournal(Path(tmpdir)) as journal:
                patterns = journal.get_recent_patterns(days=7)

                assert isinstance(patterns, list)
                assert len(patterns) == 0
    
    def test_get_recent_patterns_with_data(self):
        """Test getting patterns from journal with data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with IntrospectiveJournal(Path(tmpdir)) as journal:
                # Add and save some entries
                journal.record_realization("Test realization 1", 0.8)
                journal.record_realization("Test realization 2", 0.9)
                journal.record_question("Test question", {})
                journal.save_session()

                # Get patterns
                patterns = journal.get_recent_patterns(days=1)

                assert len(patterns) >= 1
                # Should have patterns for realizations and questions
                pattern_types = [p["type"] for p in patterns]
                assert "realizations_pattern" in pattern_types
                assert "questions_pattern" in pattern_types


class TestMetaCognitiveMetrics:
    """Test meta-cognitive metrics and health reporting"""
    
    def test_get_meta_cognitive_health_empty_state(self):
        """Test health metrics with no data"""
        monitor = SelfMonitor()
        
        health = monitor.get_meta_cognitive_health()
        
        assert "self_model_accuracy" in health
        assert "value_alignment_score" in health
        assert "behavioral_consistency" in health
        assert "introspective_depth" in health
        assert "uncertainty_awareness" in health
        assert "capability_model_accuracy" in health
        assert "recent_inconsistencies" in health
        assert "recent_realizations" in health
        assert "areas_needing_attention" in health
        
        # All scores should be between 0 and 1
        assert 0.0 <= health["self_model_accuracy"] <= 1.0
        assert 0.0 <= health["value_alignment_score"] <= 1.0
        assert 0.0 <= health["behavioral_consistency"] <= 1.0
    
    def test_get_meta_cognitive_health_with_workspace(self):
        """Test health metrics with workspace context"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Add value goals
        goal1 = Goal(
            type=GoalType.MAINTAIN_VALUE,
            description="Maintain honesty",
            priority=0.9
        )
        goal2 = Goal(
            type=GoalType.MAINTAIN_VALUE,
            description="Maintain safety",
            priority=0.8
        )
        workspace.add_goal(goal1)
        workspace.add_goal(goal2)
        
        health = monitor.get_meta_cognitive_health()
        
        # Value alignment should be high
        assert health["value_alignment_score"] > 0.5
    
    def test_generate_meta_cognitive_report(self):
        """Test report generation"""
        monitor = SelfMonitor()
        
        report = monitor.generate_meta_cognitive_report()
        
        assert isinstance(report, str)
        assert "Meta-Cognitive Status Report" in report
        assert "Self-Model Accuracy" in report
        assert "Value Alignment" in report
        assert "Behavioral Consistency" in report
    
    def test_health_identifies_low_accuracy(self):
        """Test that health report identifies low accuracy issues"""
        monitor = SelfMonitor()
        
        # Add failed predictions
        for _ in range(10):
            monitor.prediction_history.append({
                "category": "action",
                "correct": False,
                "confidence": 0.5
            })
        
        health = monitor.get_meta_cognitive_health()
        
        assert "Self-model accuracy needs improvement" in health["areas_needing_attention"]
    
    def test_health_identifies_inconsistencies(self):
        """Test that health report identifies behavioral inconsistencies"""
        monitor = SelfMonitor()
        
        # Simulate many inconsistencies
        monitor.stats["behavioral_inconsistencies"] = 50
        monitor.stats["total_observations"] = 100
        
        health = monitor.get_meta_cognitive_health()
        
        # Consistency should be low
        assert health["behavioral_consistency"] < 0.8
        assert "Behavioral consistency issues detected" in health["areas_needing_attention"]


class TestEnhancedPercepts:
    """Test enhanced percept types"""
    
    def test_behavioral_inconsistency_percept_structure(self):
        """Test structure of behavioral inconsistency percept"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        # Build consistent history
        for _ in range(15):
            snapshot = WorkspaceSnapshot(
                goals=[],
                percepts={},
                emotions={"valence": 0.8, "arousal": 0.5, "dominance": 0.5},
                memories=[],
                timestamp=datetime.now(),
                cycle_count=0,
                metadata={}
            )
            outcome = {"action_type": "SPEAK", "success": True}
            monitor.update_self_model(snapshot, outcome)
        
        # Create inconsistent state
        deviant_snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        percept = monitor.analyze_behavioral_consistency(deviant_snapshot)
        
        assert percept is not None
        assert percept.modality == "introspection"
        assert percept.raw["type"] == "behavioral_inconsistency"
        assert "description" in percept.raw
        assert "inconsistencies" in percept.raw
        assert "severity" in percept.raw
        assert "self_explanation_attempt" in percept.raw
    
    def test_capability_assessment_percept_structure(self):
        """Test structure of capability assessment percept"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Add limitations
        monitor.self_model["limitations"]["TOOL_CALL"] = [
            {"reason": "Failed", "timestamp": datetime.now().isoformat()}
            for _ in range(5)
        ]
        
        action = Action(type=ActionType.TOOL_CALL)
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": [action]}
        )
        
        percept = monitor.assess_capability_claims(snapshot)
        
        assert percept is not None
        assert percept.modality == "introspection"
        assert percept.raw["type"] == "capability_assessment"
        assert "description" in percept.raw
        assert "issues" in percept.raw
    
    def test_percept_includes_confidence_metadata(self):
        """Test that percepts include confidence/severity metadata"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        # Create scenario that triggers inconsistency
        for _ in range(15):
            snapshot = WorkspaceSnapshot(
                goals=[],
                percepts={},
                emotions={"valence": 0.9, "arousal": 0.5, "dominance": 0.5},
                memories=[],
                timestamp=datetime.now(),
                cycle_count=0,
                metadata={}
            )
            outcome = {"action_type": "SPEAK", "success": True}
            monitor.update_self_model(snapshot, outcome)
        
        deviant_snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.8, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        percept = monitor.analyze_behavioral_consistency(deviant_snapshot)
        
        assert percept is not None
        assert "severity" in percept.metadata
        assert isinstance(percept.metadata["severity"], float)
        assert 0.0 <= percept.metadata["severity"] <= 1.0


class TestConfigurationSupport:
    """Test configuration options"""
    
    def test_custom_update_frequency(self):
        """Test custom self-model update frequency"""
        config = {"self_model_update_frequency": 3}
        monitor = SelfMonitor(config=config)
        
        assert monitor.self_model_update_frequency == 3
    
    def test_custom_confidence_threshold(self):
        """Test custom prediction confidence threshold"""
        config = {"prediction_confidence_threshold": 0.8}
        monitor = SelfMonitor(config=config)
        
        assert monitor.prediction_confidence_threshold == 0.8
    
    def test_custom_inconsistency_threshold(self):
        """Test custom inconsistency severity threshold"""
        config = {"inconsistency_severity_threshold": 0.7}
        monitor = SelfMonitor(config=config)
        
        assert monitor.inconsistency_severity_threshold == 0.7
    
    def test_disable_capability_tracking(self):
        """Test disabling capability tracking"""
        config = {"enable_capability_tracking": False}
        monitor = SelfMonitor(config=config)
        
        assert monitor.enable_capability_tracking is False
        
        # Should return None when disabled
        workspace = GlobalWorkspace()
        monitor.workspace = workspace
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        percept = monitor.assess_capability_claims(snapshot)
        assert percept is None
    
    def test_disable_value_alignment_tracking(self):
        """Test disabling value alignment tracking"""
        config = {"enable_value_alignment_tracking": False}
        monitor = SelfMonitor(config=config)
        
        assert monitor.enable_value_alignment_tracking is False
        
        # Should return empty list when disabled
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        misalignments = monitor.detect_value_action_misalignment(snapshot)
        assert len(misalignments) == 0


class TestStatsTracking:
    """Test statistics tracking for new features"""
    
    def test_self_model_updates_stat(self):
        """Test self_model_updates stat is tracked"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        assert monitor.stats["self_model_updates"] == 0
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        outcome = {"action_type": "SPEAK", "success": True}
        monitor.update_self_model(snapshot, outcome)
        
        assert monitor.stats["self_model_updates"] == 1
    
    def test_predictions_made_stat(self):
        """Test predictions_made stat is tracked"""
        monitor = SelfMonitor()
        
        assert monitor.stats["predictions_made"] == 0
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        monitor.predict_behavior(snapshot)
        
        assert monitor.stats["predictions_made"] == 1
    
    def test_behavioral_inconsistencies_stat(self):
        """Test behavioral_inconsistencies stat is tracked"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"self_model_update_frequency": 1})
        
        assert monitor.stats["behavioral_inconsistencies"] == 0
        
        # Build history and create inconsistency
        for _ in range(15):
            snapshot = WorkspaceSnapshot(
                goals=[],
                percepts={},
                emotions={"valence": 0.9, "arousal": 0.5, "dominance": 0.5},
                memories=[],
                timestamp=datetime.now(),
                cycle_count=0,
                metadata={}
            )
            outcome = {"action_type": "SPEAK", "success": True}
            monitor.update_self_model(snapshot, outcome)
        
        deviant_snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.8, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        percept = monitor.analyze_behavioral_consistency(deviant_snapshot)
        
        if percept:  # Only if severity threshold met
            assert monitor.stats["behavioral_inconsistencies"] >= 1
