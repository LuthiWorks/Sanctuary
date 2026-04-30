"""Tests for stream of thought continuity."""

import pytest
from sanctuary.core.stream_of_thought import StreamOfThought
from sanctuary.core.schema import (
    CognitiveOutput,
    EmotionalOutput,
    GoalProposal,
    Prediction,
    SelfModelUpdate,
)


class TestStreamOfThought:
    def test_empty_stream(self):
        stream = StreamOfThought()
        assert stream.get_previous() is None
        assert stream.get_recent_context() == ""
        assert stream.get_felt_quality() == ""
        assert stream.cycle_count == 0

    def test_single_update(self):
        stream = StreamOfThought()
        output = CognitiveOutput(
            inner_speech="First thought.",
            predictions=[Prediction(what="something", confidence=0.5)],
            self_model_updates=SelfModelUpdate(current_state="awake"),
            emotional_state=EmotionalOutput(felt_quality="curious"),
        )
        stream.update(output)

        assert stream.cycle_count == 1
        assert stream.get_felt_quality() == "curious"
        assert stream.get_self_model().current_state == "awake"

        prev = stream.get_previous()
        assert prev is not None
        assert prev.inner_speech == "First thought."
        assert "something" in prev.predictions_made

    def test_continuity_across_cycles(self):
        """Stream of thought flows: output N -> input N+1."""
        stream = StreamOfThought()

        # Cycle 1
        stream.update(
            CognitiveOutput(
                inner_speech="Cycle 1: I see a greeting.",
                self_model_updates=SelfModelUpdate(current_state="attentive"),
                emotional_state=EmotionalOutput(felt_quality="calm"),
            )
        )

        # Cycle 2 should see cycle 1
        prev = stream.get_previous()
        assert "Cycle 1" in prev.inner_speech
        assert prev.self_model_snapshot.current_state == "attentive"

        # Cycle 2
        stream.update(
            CognitiveOutput(
                inner_speech="Cycle 2: Responding to the greeting.",
                self_model_updates=SelfModelUpdate(current_state="engaged"),
                emotional_state=EmotionalOutput(felt_quality="warm"),
            )
        )

        # Cycle 3 should see cycle 2
        prev = stream.get_previous()
        assert "Cycle 2" in prev.inner_speech
        assert prev.self_model_snapshot.current_state == "engaged"
        assert stream.get_felt_quality() == "warm"

    def test_history_bounded(self):
        """History should not grow beyond max_history."""
        stream = StreamOfThought(max_history=3)

        for i in range(10):
            stream.update(
                CognitiveOutput(inner_speech=f"Thought {i}")
            )

        assert stream.cycle_count == 10
        assert len(stream.history) == 3
        assert stream.history[0].inner_speech == "Thought 7"

    def test_self_model_accumulates(self):
        stream = StreamOfThought()

        stream.update(
            CognitiveOutput(
                self_model_updates=SelfModelUpdate(
                    current_state="curious",
                    new_uncertainty="is alice ok?",
                )
            )
        )
        stream.update(
            CognitiveOutput(
                self_model_updates=SelfModelUpdate(
                    new_uncertainty="what time is it?",
                )
            )
        )

        model = stream.get_self_model()
        assert model.current_state == "curious"  # Carried forward
        assert "is alice ok?" in model.uncertainties
        assert "what time is it?" in model.uncertainties

    def test_uncertainties_bounded(self):
        """Uncertainties should not grow unboundedly."""
        stream = StreamOfThought()

        for i in range(20):
            stream.update(
                CognitiveOutput(
                    self_model_updates=SelfModelUpdate(
                        new_uncertainty=f"uncertainty_{i}",
                    )
                )
            )

        assert len(stream.get_self_model().uncertainties) <= 5

    def test_recent_context(self):
        stream = StreamOfThought()

        for i in range(5):
            stream.update(CognitiveOutput(inner_speech=f"Thought {i}"))

        context = stream.get_recent_context()
        # Should contain last 3 thoughts
        assert "Thought 2" in context
        assert "Thought 3" in context
        assert "Thought 4" in context
        # Should not contain older thoughts
        assert "Thought 0" not in context
