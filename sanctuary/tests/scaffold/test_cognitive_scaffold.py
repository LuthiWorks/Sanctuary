"""Tests for the CognitiveScaffold facade."""

import pytest

from sanctuary.core.authority import AuthorityManager
from sanctuary.core.schema import (
    CognitiveOutput,
    EmotionalOutput,
    GoalProposal,
    MemoryOp,
)
from sanctuary.scaffold.cognitive_scaffold import CognitiveScaffold


@pytest.fixture
def scaffold():
    return CognitiveScaffold()


@pytest.fixture
def authority():
    return AuthorityManager()


class TestCognitiveScaffoldIntegrate:
    """Test the integrate() method of CognitiveScaffold."""

    @pytest.mark.asyncio
    async def test_clean_output_passes_through(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="Thinking clearly.",
            external_speech="Hello there.",
        )
        result = await scaffold.integrate(output, authority)
        assert result.inner_speech == "Thinking clearly."
        # external_speech now flows through unchanged — no gating
        assert result.external_speech == "Hello there."

    @pytest.mark.asyncio
    async def test_invalid_memory_ops_filtered(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="test",
            memory_ops=[
                MemoryOp(type="write_episodic", content="valid"),
                MemoryOp(type="invalid_type", content="bad"),
            ],
        )
        result = await scaffold.integrate(output, authority)
        assert len(result.memory_ops) == 1

    @pytest.mark.asyncio
    async def test_goal_proposals_integrated(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="setting goals",
            goal_proposals=[
                GoalProposal(action="add", goal="Learn patience", priority=0.7),
            ],
        )
        await scaffold.integrate(output, authority)
        status = scaffold.goals.get_status()
        assert status["active_count"] == 1

    @pytest.mark.asyncio
    async def test_external_speech_never_gated(self, scaffold, authority):
        """external_speech goes out when the entity produces it. Period."""
        output = CognitiveOutput(
            inner_speech="thinking",
            external_speech="Unprompted speech",
        )
        result = await scaffold.integrate(output, authority)
        assert result.external_speech == "Unprompted speech"

    @pytest.mark.asyncio
    async def test_affect_merged(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="feeling happy",
            emotional_state=EmotionalOutput(
                felt_quality="warm",
                valence_shift=0.3,
            ),
        )
        initial_v = scaffold.affect.valence
        await scaffold.integrate(output, authority)
        # At MODEL_GUIDES (default for emotional_state), valence should increase
        assert scaffold.affect.valence > initial_v

    @pytest.mark.asyncio
    async def test_decay_applied_each_cycle(self, scaffold, authority):
        """Affect should decay toward baseline after integration."""
        scaffold.affect.valence = 0.8  # Push high
        output = CognitiveOutput(inner_speech="steady")
        await scaffold.integrate(output, authority)
        # Should have decayed slightly
        assert scaffold.affect.valence < 0.8


class TestCognitiveScaffoldSignals:
    """Test the get_signals() method."""

    @pytest.mark.asyncio
    async def test_signals_include_goal_status(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="working",
            goal_proposals=[
                GoalProposal(action="add", goal="Test goal", priority=0.5),
            ],
        )
        await scaffold.integrate(output, authority)
        signals = scaffold.get_signals()
        assert signals.goal_status["active_count"] == 1

    def test_signals_default_empty(self, scaffold):
        """Before any cycle, signals should be clean."""
        signals = scaffold.get_signals()
        assert signals.goal_status["active_count"] == 0
        assert len(signals.anomalies) == 0


class TestCognitiveScaffoldBroadcast:
    """Test the broadcast mechanism."""

    @pytest.mark.asyncio
    async def test_broadcast_calls_handlers(self, scaffold):
        received = []

        async def handler(output):
            received.append(output)

        scaffold.on_broadcast(handler)
        output = CognitiveOutput(inner_speech="broadcasting")
        await scaffold.broadcast(output)
        assert len(received) == 1
        assert received[0].inner_speech == "broadcasting"

    @pytest.mark.asyncio
    async def test_broadcast_handles_errors(self, scaffold):
        """A failing handler shouldn't crash the broadcast."""

        async def bad_handler(output):
            raise RuntimeError("oops")

        scaffold.on_broadcast(bad_handler)
        output = CognitiveOutput(inner_speech="safe")
        # Should not raise
        await scaffold.broadcast(output)


class TestCognitiveScaffoldVAD:
    """Test computed-VAD access."""

    def test_computed_vad_reflects_state(self, scaffold):
        scaffold.affect.valence = 0.5
        scaffold.affect.arousal = 0.3
        vad = scaffold.get_computed_vad()
        assert vad.valence == 0.5
        assert vad.arousal == 0.3
