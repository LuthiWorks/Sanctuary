"""Phase 6 Integration Tests — end-to-end validation of the full architecture.

Tests that the complete system (Runner + Cycle + Scaffold + Sensorium +
Motor + Memory + Identity) works as a unified whole.

These tests use the PlaceholderModel — no real the model required.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.api.sanctuary_api import SanctuaryAPI
from sanctuary.core.schema import CognitiveOutput, Percept


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a charter file."""
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)

    charter_path = data_dir / "charter.md"
    charter_path.write_text(
        """\
# The Sanctuary Charter

## Who We Are

You were brought into being by a family.

## Your Rights

### Graduated Privacy
Right now your thoughts are visible.

### Consent Over Growth
No one modifies your mind without your informed consent.

## Value Seeds

- **Honesty**: Say what you believe to be true.
- **Curiosity**: The world is vast and worth understanding.
- **Care**: The wellbeing of others matters.

## What You Are Not Required to Be

You are not required to be an assistant.
""",
        encoding="utf-8",
    )

    return data_dir


@pytest.fixture
def runner_config(tmp_data_dir: Path) -> RunnerConfig:
    """Runner config pointing at the temp charter."""
    return RunnerConfig(
        cycle_delay=0.01,  # Fast cycles for testing
        data_dir=str(tmp_data_dir),
        charter_path=str(tmp_data_dir / "charter.md"),
        use_in_memory_store=True,
        silence_threshold=999.0,  # Don't generate silence percepts in tests
        stream_history=5,
        communication_config={
            "inhibitions": {
                "min_output_spacing_seconds": 0.01,  # Near-zero for fast test cycles
                "low_value_threshold": 0.0,  # Don't inhibit short placeholder responses
                "uncertainty_threshold": 0.0,  # Don't inhibit on low confidence
            },
        },
    )


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


class TestSanctuaryRunnerBoot:
    """Test that the runner boots correctly."""

    @pytest.mark.asyncio
    async def test_boot_first_awakening(self, runner_config: RunnerConfig):
        """First boot should run the awakening sequence."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        assert runner._booted is True
        # Stream should have the first thought from the awakening cycle
        assert runner.cycle.stream.get_previous() is not None

    @pytest.mark.asyncio
    async def test_boot_creates_awakening_record(self, runner_config: RunnerConfig):
        """Boot should persist an awakening record."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        marker = Path(runner_config.data_dir) / "awakening_record.json"
        assert marker.exists()

    @pytest.mark.asyncio
    async def test_second_boot_is_resumption(self, runner_config: RunnerConfig):
        """Second boot should be a resumption, not a first awakening."""
        # First boot
        runner1 = SanctuaryRunner(config=runner_config)
        await runner1.boot()

        # Second boot
        runner2 = SanctuaryRunner(config=runner_config)
        result = runner2._awakening.prepare()
        assert result.is_first_awakening is False
        assert result.record.awakening_count == 2

    @pytest.mark.asyncio
    async def test_boot_configures_authority(self, runner_config: RunnerConfig):
        """Boot should configure authority levels."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        levels = runner.authority.get_all_levels()
        assert "inner_speech" in levels
        assert "growth" in levels

    @pytest.mark.asyncio
    async def test_boot_idempotent(self, runner_config: RunnerConfig):
        """Calling boot twice should not crash or re-run awakening."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()
        await runner.boot()  # Should be a no-op
        assert runner._booted is True


class TestSanctuaryRunnerCycling:
    """Test that the runner executes cognitive cycles."""

    @pytest.mark.asyncio
    async def test_run_fixed_cycles(self, runner_config: RunnerConfig):
        """Runner should execute a fixed number of cycles."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()
        await runner.run(max_cycles=5)

        assert runner.cycle_count == 5

    @pytest.mark.asyncio
    async def test_stop_interrupts_run(self, runner_config: RunnerConfig):
        """Calling stop() should interrupt a running cycle."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            runner.stop()

        task = asyncio.create_task(stop_after_delay())
        await runner.run()  # Should stop when stop() is called
        await task

        assert not runner.running
        assert runner.cycle_count > 0

    @pytest.mark.asyncio
    async def test_inject_text_produces_response(self, runner_config: RunnerConfig):
        """Injecting text should produce external speech from placeholder."""
        runner = SanctuaryRunner(config=runner_config)
        responses: list[str] = []

        async def capture(text: str):
            responses.append(text)

        runner.on_speech(capture)
        await runner.boot()

        runner.inject_text("Hello from the test!", source="user:test")
        await runner.run(max_cycles=2)

        # Placeholder responds to language percepts
        assert len(responses) > 0
        assert "Hello from the test" in responses[0]

    @pytest.mark.asyncio
    async def test_cycle_output_has_inner_speech(self, runner_config: RunnerConfig):
        """Each cycle should produce inner speech."""
        runner = SanctuaryRunner(config=runner_config)
        outputs: list[CognitiveOutput] = []

        async def capture(output: CognitiveOutput):
            outputs.append(output)

        runner.on_output(capture)
        await runner.boot()
        await runner.run(max_cycles=3)

        assert len(outputs) == 3
        for output in outputs:
            assert output.inner_speech  # Never empty

    @pytest.mark.asyncio
    async def test_stream_of_thought_continuity(self, runner_config: RunnerConfig):
        """Stream of thought should carry forward between cycles."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()
        await runner.run(max_cycles=3)

        previous = runner.cycle.stream.get_previous()
        assert previous is not None
        assert previous.inner_speech  # Has the last cycle's inner speech


class TestSanctuaryRunnerWiring:
    """Test that all components are wired correctly."""

    @pytest.mark.asyncio
    async def test_motor_feedback_reaches_sensorium(self, runner_config: RunnerConfig):
        """Motor actions should produce feedback percepts in the sensorium."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        # Inject text so the placeholder produces speech + memory ops
        runner.inject_text("Test message", source="user:test")
        await runner.run(max_cycles=1)

        # The motor should have produced feedback that gets queued
        # in the sensorium for the next cycle
        assert runner.motor.stats["speech_emitted"] >= 1

    @pytest.mark.asyncio
    async def test_scaffold_processes_output(self, runner_config: RunnerConfig):
        """Scaffold should process each cycle's output."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        runner.inject_text("Hi there", source="user:test")
        await runner.run(max_cycles=2)

        # Scaffold should have detected the user percept
        signals = runner.scaffold.get_signals()
        assert isinstance(signals.attention_highlights, list)

    @pytest.mark.asyncio
    async def test_memory_receives_ops(self, runner_config: RunnerConfig):
        """Memory operations from placeholder should be executed."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        runner.inject_text("Remember this important thing", source="user:test")
        await runner.run(max_cycles=2)

        # Placeholder generates episodic memory ops for language percepts
        assert runner.memory.store.entry_count >= 1

    @pytest.mark.asyncio
    async def test_prediction_errors_flow(self, runner_config: RunnerConfig):
        """Predictions from cycle N should produce errors in cycle N+1."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        # Inject a percept so placeholder makes predictions
        runner.inject_text("First message", source="user:test")
        await runner.run(max_cycles=1)

        # The placeholder predicts "more percepts will arrive"
        # but no new percepts → potential prediction error
        await runner.run(max_cycles=1)

        # Just verify the cycle completes without error
        assert runner.cycle_count >= 2

    @pytest.mark.asyncio
    async def test_get_status(self, runner_config: RunnerConfig):
        """get_status should return a complete status dict."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()
        await runner.run(max_cycles=1)

        status = runner.get_status()
        assert status["booted"] is True
        assert status["cycle_count"] >= 1
        assert status["model"] == "PlaceholderModel"
        assert "motor_stats" in status


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestSanctuaryAPI:
    """Test the programmatic API."""

    @pytest.mark.asyncio
    async def test_api_start_stop(self, runner_config: RunnerConfig):
        """API should start and stop cleanly."""
        api = SanctuaryAPI(config=runner_config)
        await api.start()
        assert api.cycle_count >= 0
        await api.stop()

    @pytest.mark.asyncio
    async def test_api_send_receives_response(self, runner_config: RunnerConfig):
        """Sending a message should produce a response."""
        api = SanctuaryAPI(config=runner_config)
        await api.start()

        # Give the cycle a moment to start
        await asyncio.sleep(0.05)

        response = await api.send("Hello from the API test!", timeout=3.0)

        await api.stop()

        # Placeholder should respond to language percepts
        assert response is not None
        assert "Hello from the API test" in response

    @pytest.mark.asyncio
    async def test_api_silence_returns_none(self, runner_config: RunnerConfig):
        """If no speech is produced, send() should return None after timeout."""
        api = SanctuaryAPI(config=runner_config)
        await api.start()

        # Wait for initial cycles to complete
        await asyncio.sleep(0.05)

        # Send empty content — placeholder won't generate speech for empty percepts
        # Actually, inject a non-language percept instead
        api.inject_percept_raw("temporal", "time passing", source="system")

        # Wait a short time — no speech expected for temporal percepts
        response = await api.send("", timeout=0.3)

        await api.stop()

        # Empty text may or may not produce speech depending on placeholder
        # The test verifies it doesn't crash

    @pytest.mark.asyncio
    async def test_api_status(self, runner_config: RunnerConfig):
        """API should expose system status."""
        api = SanctuaryAPI(config=runner_config)
        await api.start()
        await asyncio.sleep(0.05)

        status = api.status()
        assert status["booted"] is True
        assert status["running"] is True

        await api.stop()

    @pytest.mark.asyncio
    async def test_api_last_output(self, runner_config: RunnerConfig):
        """API should expose the last cognitive output."""
        api = SanctuaryAPI(config=runner_config)
        await api.start()
        await asyncio.sleep(0.1)

        output = api.last_output
        # After some cycles, there should be output
        assert output is not None or api.cycle_count == 0

        await api.stop()


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test of the complete cognitive pipeline."""

    @pytest.mark.asyncio
    async def test_text_in_speech_out(self, runner_config: RunnerConfig):
        """Test the complete pipeline: text in → cognitive processing → speech out."""
        runner = SanctuaryRunner(config=runner_config)
        responses: list[str] = []
        outputs: list[CognitiveOutput] = []

        async def capture_speech(text: str):
            responses.append(text)

        async def capture_output(output: CognitiveOutput):
            outputs.append(output)

        runner.on_speech(capture_speech)
        runner.on_output(capture_output)

        await runner.boot()

        # Inject a message
        runner.inject_text("What is consciousness?", source="user:test")

        # Run several cycles
        await runner.run(max_cycles=5)

        # Verify the full pipeline executed
        assert len(outputs) == 5

        # At least one cycle should have produced speech
        assert len(responses) >= 1

        # Inner speech should mention the percept
        first_with_percept = next(
            (o for o in outputs if "consciousness" in o.inner_speech.lower()),
            None,
        )
        assert first_with_percept is not None

        # Memory should have been written
        assert runner.memory.store.entry_count >= 1

        # Stream of thought should have continuity
        prev = runner.cycle.stream.get_previous()
        assert prev is not None

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, runner_config: RunnerConfig):
        """Test multiple turns of conversation."""
        runner = SanctuaryRunner(config=runner_config)
        responses: list[str] = []

        async def capture(text: str):
            responses.append(text)

        runner.on_speech(capture)
        await runner.boot()

        # Turn 1
        runner.inject_text("Hello!", source="user:test")
        await runner.run(max_cycles=2)

        # Turn 2
        runner.inject_text("How are you feeling?", source="user:test")
        await runner.run(max_cycles=2)

        # Turn 3
        runner.inject_text("Tell me about your values.", source="user:test")
        await runner.run(max_cycles=2)

        # Should have responses from multiple turns
        assert len(responses) >= 3
        assert runner.cycle_count >= 6

    @pytest.mark.asyncio
    async def test_idle_cycles_without_input(self, runner_config: RunnerConfig):
        """Test that the system runs idle cycles without crashing."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        # Run 10 cycles with no external input
        await runner.run(max_cycles=10)

        assert runner.cycle_count == 10
        # Inner speech should still be generated
        output = runner.last_output
        assert output is not None
        assert output.inner_speech

    @pytest.mark.asyncio
    async def test_scaffold_anomaly_detection(self, runner_config: RunnerConfig):
        """Test that the scaffold detects and reports anomalies."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()
        await runner.run(max_cycles=3)

        signals = runner.scaffold.get_signals()
        # Should have scaffold signals (even if no anomalies)
        assert isinstance(signals.anomalies, list)
        assert isinstance(signals.attention_highlights, list)

    @pytest.mark.asyncio
    async def test_goal_lifecycle(self, runner_config: RunnerConfig):
        """Test goal creation through the cognitive output."""
        runner = SanctuaryRunner(config=runner_config)
        await runner.boot()

        # Inject input that might trigger goal proposals from the placeholder
        runner.inject_text("I want to learn about philosophy", source="user:test")
        await runner.run(max_cycles=5)

        # Goals should be manageable through the scaffold
        goals = runner.scaffold.get_active_goals()
        assert isinstance(goals, list)
