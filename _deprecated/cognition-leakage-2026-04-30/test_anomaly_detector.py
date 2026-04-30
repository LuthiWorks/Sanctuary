"""Tests for scaffold anomaly detector."""

from sanctuary.core.schema import (
    CognitiveOutput,
    EmotionalOutput,
    GrowthReflection,
    MemoryOp,
    Prediction,
    GoalProposal,
)
from sanctuary.scaffold.anomaly_detector import ScaffoldAnomalyDetector


class TestAnomalyDetector:
    """Test LLM output anomaly detection."""

    def test_clean_output_no_anomalies(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(inner_speech="All is well.")
        anomalies = detector.check(output)
        assert len(anomalies) == 0

    def test_empty_inner_speech_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(inner_speech="")
        anomalies = detector.check(output)
        assert any("empty" in a.lower() for a in anomalies)

    def test_whitespace_inner_speech_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(inner_speech="   ")
        anomalies = detector.check(output)
        assert any("empty" in a.lower() for a in anomalies)

    def test_long_inner_speech_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(inner_speech="x" * 6000)
        anomalies = detector.check(output)
        assert any("inner speech" in a.lower() and "chars" in a.lower() for a in anomalies)

    def test_long_external_speech_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="thinking",
            external_speech="y" * 4000,
        )
        anomalies = detector.check(output)
        assert any("external speech" in a.lower() for a in anomalies)

    def test_extreme_valence_shift_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="emotional",
            emotional_state=EmotionalOutput(valence_shift=0.9),
        )
        anomalies = detector.check(output)
        assert any("valence" in a.lower() for a in anomalies)

    def test_extreme_arousal_shift_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="activated",
            emotional_state=EmotionalOutput(arousal_shift=-0.9),
        )
        anomalies = detector.check(output)
        assert any("arousal" in a.lower() for a in anomalies)

    def test_too_many_predictions_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="predicting",
            predictions=[
                Prediction(what=f"p{i}", confidence=0.5) for i in range(25)
            ],
        )
        anomalies = detector.check(output)
        assert any("predictions" in a.lower() for a in anomalies)

    def test_too_many_memory_ops_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="memorizing",
            memory_ops=[
                MemoryOp(type="write_episodic", content=f"mem {i}")
                for i in range(15)
            ],
        )
        anomalies = detector.check(output)
        assert any("memory ops" in a.lower() for a in anomalies)

    def test_too_many_goal_proposals_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="goal-setting",
            goal_proposals=[
                GoalProposal(action="add", goal=f"g{i}") for i in range(8)
            ],
        )
        anomalies = detector.check(output)
        assert any("goal proposals" in a.lower() for a in anomalies)

    def test_incomplete_growth_reflection_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="reflecting",
            growth_reflection=GrowthReflection(worth_learning=True, what_to_learn=""),
        )
        anomalies = detector.check(output)
        assert any("growth reflection" in a.lower() for a in anomalies)

    def test_valid_growth_reflection_not_flagged(self):
        detector = ScaffoldAnomalyDetector()
        output = CognitiveOutput(
            inner_speech="reflecting",
            growth_reflection=GrowthReflection(
                worth_learning=True, what_to_learn="Patience"
            ),
        )
        anomalies = detector.check(output)
        assert not any("growth" in a.lower() for a in anomalies)

    def test_recent_anomaly_count(self):
        detector = ScaffoldAnomalyDetector()
        for _ in range(3):
            detector.check(CognitiveOutput(inner_speech=""))
        assert detector.get_recent_anomaly_count(window=5) == 3
