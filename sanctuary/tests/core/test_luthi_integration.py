"""Comprehensive edge-case tests for Sanctuary + LuthiModel integration.

Tests cover:
- RunnerConfig with luthi backend (no actual model load)
- CognitiveCycle with/without sleep_manager (backward compat)
- Sleep sensory gating and urgent wake
- Introspection injection (mock model with get_augmented_experiential_signals)
- Introspection injection when model raises
- Consolidation during NREM sleep stage
- Consolidation with model that lacks consolidate() (duck typing)
- Sleep sensory gating with empty percepts
- First cycle with no previous introspection
- CLI arg parsing for new Luthi/sleep flags
- CLI password env var fallback resolution order
"""

from __future__ import annotations

import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sanctuary.consciousness.sleep_cycle import (
    SleepConfig,
    SleepCycleManager,
    SleepStage,
)
from sanctuary.core.cognitive_cycle import (
    CognitiveCycle,
    NullMemory,
    NullScaffold,
    NullSensorium,
)
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.schema import (
    CognitiveOutput,
    ExperientialSignals,
    Percept,
)


# ---------------------------------------------------------------------------
# Helpers: mock models for testing Luthi integration paths
# ---------------------------------------------------------------------------


class MockModelWithIntrospection:
    """Mock model that implements get_augmented_experiential_signals().

    Returns deterministic introspection signals so we can verify they
    appear in ExperientialSignals.knowledge_signals during the cycle.
    """

    _DEFAULT_SIGNALS = {
        "luthi_plasticity": [0.1, 0.2, 0.3],
        "luthi_drift": [0.01, -0.02],
    }

    def __init__(self, signals: dict[str, list[float]] | None = _DEFAULT_SIGNALS):
        self.cycle_count = 0
        self.name = "MockIntrospection"
        self._signals = signals if signals is not None else self._DEFAULT_SIGNALS

    async def think(self, cognitive_input) -> CognitiveOutput:
        self.cycle_count += 1
        return CognitiveOutput(
            inner_speech=f"[mock cycle {self.cycle_count}]",
        )

    def get_augmented_experiential_signals(self) -> dict[str, list[float]]:
        return dict(self._signals)


class MockModelIntrospectionRaises:
    """Mock model whose get_augmented_experiential_signals() raises."""

    def __init__(self):
        self.cycle_count = 0
        self.name = "MockRaises"

    async def think(self, cognitive_input) -> CognitiveOutput:
        self.cycle_count += 1
        return CognitiveOutput(
            inner_speech=f"[mock-raise cycle {self.cycle_count}]",
        )

    def get_augmented_experiential_signals(self) -> dict:
        raise RuntimeError("Simulated introspection failure")


class MockModelWithConsolidate:
    """Mock model that has a consolidate() method we can track calls to."""

    def __init__(self):
        self.cycle_count = 0
        self.name = "MockConsolidate"
        self.consolidate_calls = 0

    async def think(self, cognitive_input) -> CognitiveOutput:
        self.cycle_count += 1
        return CognitiveOutput(
            inner_speech=f"[mock-consolidate cycle {self.cycle_count}]",
        )

    def consolidate(self):
        self.consolidate_calls += 1


class MockModelNoConsolidate:
    """Mock model that does NOT have consolidate() — duck typing test."""

    def __init__(self):
        self.cycle_count = 0
        self.name = "MockNoConsolidate"

    async def think(self, cognitive_input) -> CognitiveOutput:
        self.cycle_count += 1
        return CognitiveOutput(
            inner_speech=f"[mock-noconsolidate cycle {self.cycle_count}]",
        )


class MockModelFirstCycleIntrospection:
    """Mock model that returns {} from get_augmented_experiential_signals
    on the first call (no previous introspection state).

    Note: get_augmented_experiential_signals() is called in _assemble_input()
    BEFORE model.think(), so cycle_count is 0 on the first cycle and 1 on
    the second cycle when introspection is read.
    """

    def __init__(self):
        self.cycle_count = 0
        self.name = "MockFirstCycle"

    async def think(self, cognitive_input) -> CognitiveOutput:
        self.cycle_count += 1
        return CognitiveOutput(
            inner_speech=f"[first-cycle mock {self.cycle_count}]",
        )

    def get_augmented_experiential_signals(self) -> dict:
        # First call (cycle_count=0, before first think) returns empty
        # Second call (cycle_count=1, after first think) returns signals
        if self.cycle_count == 0:
            return {}
        return {"luthi_plasticity": [0.5]}


# ---------------------------------------------------------------------------
# 1. RunnerConfig with luthi backend but no checkpoint
# ---------------------------------------------------------------------------


class TestRunnerConfigLuthiNoCheckpoint:
    """RunnerConfig can be set to luthi backend. The actual model.load()
    will fail because there's no real checkpoint, but the config itself
    should be valid."""

    def test_runner_config_accepts_luthi_backend(self):
        """RunnerConfig should accept model_backend='luthi' without error."""
        from sanctuary.api.runner import RunnerConfig

        config = RunnerConfig(
            model_backend="luthi",
            luthi_checkpoint=None,
            luthi_password=None,
        )
        assert config.model_backend == "luthi"
        assert config.luthi_checkpoint is None

    def test_runner_config_with_checkpoint_path(self):
        """RunnerConfig should store checkpoint path and password."""
        from sanctuary.api.runner import RunnerConfig

        config = RunnerConfig(
            model_backend="luthi",
            luthi_checkpoint="/fake/path/model.luthi",
            luthi_password="secret123",
        )
        assert config.luthi_checkpoint == "/fake/path/model.luthi"
        assert config.luthi_password == "secret123"

    def test_runner_create_model_luthi_defers_load(self):
        """When model_backend=luthi with a checkpoint path, the runner
        creates a LuthiModel but defers load() to first think() call."""
        from sanctuary.api.runner import RunnerConfig, SanctuaryRunner

        config = RunnerConfig(
            model_backend="luthi",
            luthi_checkpoint="/nonexistent/path.luthi",
            luthi_password="test",
        )
        # Runner should construct successfully — load is deferred
        runner = SanctuaryRunner(config=config)
        assert type(runner._model).__name__ == "LuthiModel"
        assert not runner._model._loaded


# ---------------------------------------------------------------------------
# 2. CognitiveCycle with sleep_manager=None (backward compat)
# ---------------------------------------------------------------------------


class TestCognitiveCycleNoSleepManager:
    """When sleep_manager is None (the default), the cycle should behave
    exactly as it did before sleep integration."""

    @pytest.mark.asyncio
    async def test_cycle_runs_without_sleep(self):
        model = PlaceholderModel()
        cycle = CognitiveCycle(model=model, sleep_manager=None, cycle_delay=0.0)
        await cycle.run(max_cycles=3)
        assert cycle.cycle_count == 3
        assert cycle.last_output is not None
        assert cycle.sleep is None

    @pytest.mark.asyncio
    async def test_percept_injection_without_sleep(self):
        """Percepts should pass through unmodified when no sleep manager."""
        model = PlaceholderModel()
        cycle = CognitiveCycle(model=model, sleep_manager=None, cycle_delay=0.0)
        cycle.inject_percept(Percept(modality="language", content="Hello"))
        await cycle.run(max_cycles=1)
        assert "Hello" in cycle.last_output.inner_speech

    @pytest.mark.asyncio
    async def test_multiple_cycles_stable_without_sleep(self):
        """Run 10 cycles without sleep — no errors."""
        model = PlaceholderModel()
        cycle = CognitiveCycle(model=model, sleep_manager=None, cycle_delay=0.0)
        cycle.inject_percept(Percept(modality="sensor", content="data"))
        await cycle.run(max_cycles=10)
        assert cycle.cycle_count == 10


# ---------------------------------------------------------------------------
# 3. CognitiveCycle with sleep_manager — tick should be called each cycle
# ---------------------------------------------------------------------------


class TestCognitiveCycleWithSleep:
    @pytest.mark.asyncio
    async def test_sleep_tick_called_each_cycle(self):
        """SleepCycleManager.tick() should be called once per cognitive cycle."""
        model = PlaceholderModel()
        sleep = SleepCycleManager(config=SleepConfig(cycles_between_sleep=1000))
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )
        await cycle.run(max_cycles=5)

        # After 5 cycles, sleep should have counted 5 waking ticks
        assert sleep._cycles_since_sleep == 5
        assert sleep.stage == SleepStage.AWAKE

    @pytest.mark.asyncio
    async def test_sleep_transition_occurs(self):
        """After enough cycles, sleep should transition to DROWSY."""
        model = PlaceholderModel()
        # Sleep after just 3 waking cycles
        config = SleepConfig(
            cycles_between_sleep=3,
            drowsy_duration=2,
            nrem_duration=2,
            rem_duration=2,
            waking_duration=1,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Run exactly 3 cycles — on the 3rd tick, should transition to drowsy
        await cycle.run(max_cycles=4)
        # By cycle 4, we should be past the AWAKE→DROWSY transition
        assert sleep.is_sleeping


# ---------------------------------------------------------------------------
# 4. Sensory gating during sleep — percepts should be reduced
# ---------------------------------------------------------------------------


class TestSleepSensoryGating:
    @pytest.mark.asyncio
    async def test_percepts_reduced_during_sleep(self):
        """During sleep, the sensory gate should reduce percept throughput."""
        model = PlaceholderModel()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=100,  # Stay in drowsy for a long time
            sensory_gate_drowsy=0.5,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Run 2 cycles to trigger sleep transition
        await cycle.run(max_cycles=3)
        assert sleep.is_sleeping

        # Now inject multiple percepts
        for i in range(10):
            cycle.inject_percept(
                Percept(modality="sensor", content=f"data_{i}")
            )

        # Run one cycle during sleep — percepts should be gated
        await cycle.run(max_cycles=1)

        # The model should have received fewer than 10 percepts
        output = cycle.last_output
        # PlaceholderModel reports percept count in inner_speech
        assert output is not None
        # With gate=0.5, ~5 percepts should pass (ceil(10 * 0.5) = 5)
        assert "10 new percepts" not in output.inner_speech


# ---------------------------------------------------------------------------
# 5. Urgent wake on user input during sleep
# ---------------------------------------------------------------------------


class TestUrgentWakeDuringSleep:
    @pytest.mark.asyncio
    async def test_user_language_forces_wake(self):
        """User language percepts during sleep should force wake."""
        model = PlaceholderModel()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=100,
            sensory_gate_drowsy=0.3,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Enter sleep
        await cycle.run(max_cycles=3)
        assert sleep.is_sleeping

        # Inject a user language percept (urgent)
        cycle.inject_percept(
            Percept(
                modality="language",
                content="Are you there?",
                source="user:cli",
            )
        )

        # Run one cycle — should force wake
        await cycle.run(max_cycles=1)

        # After the next tick, the forced wake should have been processed
        # The wake() call sets _forced_wake=True, and the next tick()
        # transitions back to AWAKE
        await cycle.run(max_cycles=1)
        assert not sleep.is_sleeping

    @pytest.mark.asyncio
    async def test_non_user_percept_does_not_wake(self):
        """Non-user sensor percepts during sleep should NOT force wake."""
        model = PlaceholderModel()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=100,
            sensory_gate_drowsy=0.3,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Enter sleep
        await cycle.run(max_cycles=3)
        assert sleep.is_sleeping

        # Inject a non-user percept
        cycle.inject_percept(
            Percept(modality="sensor", content="temperature=22C", source="env")
        )

        # Run cycles — should remain sleeping
        await cycle.run(max_cycles=2)
        assert sleep.is_sleeping


# ---------------------------------------------------------------------------
# 6. Introspection injection with mock model
# ---------------------------------------------------------------------------


class TestIntrospectionInjection:
    @pytest.mark.asyncio
    async def test_introspection_signals_appear_in_experiential(self):
        """When model has get_augmented_experiential_signals(), those signals
        should appear in ExperientialSignals.knowledge_signals."""
        signals = {
            "luthi_plasticity": [0.1, 0.2, 0.3],
            "luthi_drift": [0.01, -0.02],
        }
        model = MockModelWithIntrospection(signals=signals)
        cycle = CognitiveCycle(model=model, cycle_delay=0.0)

        # We need to capture the CognitiveInput that was assembled.
        # We can do this by inspecting what _assemble_input returns.
        # Instead, let's run a cycle and check indirectly:
        # After the first cycle, the second cycle's input should contain
        # the signals from the first cycle's introspection.
        #
        # Actually, introspection signals are injected into the *current*
        # cycle's assembled input (in _assemble_input). So let's verify
        # by patching the model.think to capture the input.
        captured_inputs = []
        original_think = model.think

        async def capturing_think(cognitive_input):
            captured_inputs.append(cognitive_input)
            return await original_think(cognitive_input)

        model.think = capturing_think

        await cycle.run(max_cycles=2)

        # The signals should be present in the assembled inputs
        # Check the second cycle's input (first cycle has no previous state)
        assert len(captured_inputs) >= 1
        # Both cycles should have introspection injected
        for ci in captured_inputs:
            ks = ci.experiential_state.knowledge_signals
            assert "luthi_plasticity" in ks
            assert ks["luthi_plasticity"] == [0.1, 0.2, 0.3]
            assert "luthi_drift" in ks
            assert ks["luthi_drift"] == [0.01, -0.02]

    @pytest.mark.asyncio
    async def test_introspection_empty_signals_ignored(self):
        """When get_augmented_experiential_signals() returns {}, knowledge_signals
        should remain empty (no crash, no injection)."""
        model = MockModelWithIntrospection(signals={})
        cycle = CognitiveCycle(model=model, cycle_delay=0.0)

        captured_inputs = []
        original_think = model.think

        async def capturing_think(cognitive_input):
            captured_inputs.append(cognitive_input)
            return await original_think(cognitive_input)

        model.think = capturing_think

        await cycle.run(max_cycles=1)

        assert len(captured_inputs) == 1
        # Empty signals should not inject anything
        assert captured_inputs[0].experiential_state.knowledge_signals == {}


# ---------------------------------------------------------------------------
# 7. Introspection injection with model that raises
# ---------------------------------------------------------------------------


class TestIntrospectionRaises:
    @pytest.mark.asyncio
    async def test_introspection_error_does_not_crash_cycle(self):
        """If get_augmented_experiential_signals() raises, the cycle should
        continue without crashing."""
        model = MockModelIntrospectionRaises()
        cycle = CognitiveCycle(model=model, cycle_delay=0.0)

        # Should not raise — the error is caught and logged
        await cycle.run(max_cycles=3)
        assert cycle.cycle_count == 3
        assert cycle.last_output is not None

    @pytest.mark.asyncio
    async def test_introspection_error_logged(self, caplog):
        """Introspection error should be logged."""
        model = MockModelIntrospectionRaises()
        cycle = CognitiveCycle(model=model, cycle_delay=0.0)

        with caplog.at_level(logging.ERROR):
            await cycle.run(max_cycles=1)

        assert any(
            "Luthi introspection injection error" in record.message
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# 8. Consolidation during NREM — model.consolidate() called
# ---------------------------------------------------------------------------


class TestConsolidationDuringNREM:
    @pytest.mark.asyncio
    async def test_consolidate_called_during_nrem(self):
        """When the sleep stage is NREM and the model has consolidate(),
        it should be called."""
        model = MockModelWithConsolidate()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=1,
            nrem_duration=5,
            rem_duration=1,
            waking_duration=1,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Run enough cycles to reach NREM:
        # 2 awake + 1 drowsy + then NREM
        # tick 1: awake (count=1)
        # tick 2: awake (count=2) -> begin sleep -> drowsy
        # tick 3: drowsy (stage_cycles=1) -> transitions to NREM
        # tick 4: NREM (stage_cycles=1) -> consolidate should be called
        await cycle.run(max_cycles=10)

        # consolidate should have been called at least once
        assert model.consolidate_calls > 0

    @pytest.mark.asyncio
    async def test_consolidate_not_called_when_awake(self):
        """Consolidation should NOT be called during normal waking cycles."""
        model = MockModelWithConsolidate()
        config = SleepConfig(cycles_between_sleep=1000)
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        await cycle.run(max_cycles=5)
        assert model.consolidate_calls == 0


# ---------------------------------------------------------------------------
# 9. Consolidation with model that lacks consolidate() — silent skip
# ---------------------------------------------------------------------------


class TestConsolidationDuckTyping:
    @pytest.mark.asyncio
    async def test_no_consolidate_method_silently_skipped(self):
        """If the model doesn't have consolidate(), the cycle should
        silently skip consolidation without error."""
        model = MockModelNoConsolidate()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=1,
            nrem_duration=5,
            rem_duration=1,
            waking_duration=1,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Run through a full sleep cycle including NREM — should not crash
        await cycle.run(max_cycles=15)
        assert cycle.cycle_count == 15
        assert not hasattr(model, "consolidate")


# ---------------------------------------------------------------------------
# 10. Sleep sensory gating with empty percepts
# ---------------------------------------------------------------------------


class TestSleepGatingEmptyPercepts:
    @pytest.mark.asyncio
    async def test_empty_percepts_during_sleep_no_crash(self):
        """Sleep gating with zero percepts should not crash."""
        model = PlaceholderModel()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=100,
            sensory_gate_drowsy=0.3,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Enter sleep
        await cycle.run(max_cycles=3)
        assert sleep.is_sleeping

        # Don't inject any percepts — run more cycles during sleep
        await cycle.run(max_cycles=5)
        assert cycle.cycle_count == 8  # 3 + 5

    @pytest.mark.asyncio
    async def test_single_percept_during_deep_gate(self):
        """With gate=0.1 and 1 percept, should still keep at least 1."""
        model = PlaceholderModel()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=1,
            nrem_duration=100,
            sensory_gate_sleep=0.1,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Enter NREM
        await cycle.run(max_cycles=4)
        assert sleep.is_sleeping

        # Inject exactly 1 non-user percept
        cycle.inject_percept(
            Percept(modality="sensor", content="temp=20", source="env")
        )

        # Run one cycle — keep=max(1, ceil(1*0.1))=1 percept
        await cycle.run(max_cycles=1)
        # Should not crash; the percept should have been processed
        assert cycle.last_output is not None


# ---------------------------------------------------------------------------
# 11. First cycle with no previous introspection
# ---------------------------------------------------------------------------


class TestFirstCycleNoIntrospection:
    @pytest.mark.asyncio
    async def test_first_cycle_empty_signals(self):
        """On the first cycle, get_augmented_experiential_signals should
        return {} and the cycle should proceed normally."""
        model = MockModelFirstCycleIntrospection()
        cycle = CognitiveCycle(model=model, cycle_delay=0.0)

        captured_inputs = []
        original_think = model.think

        async def capturing_think(cognitive_input):
            captured_inputs.append(cognitive_input)
            return await original_think(cognitive_input)

        model.think = capturing_think

        # Run 2 cycles
        await cycle.run(max_cycles=2)

        assert len(captured_inputs) == 2

        # First cycle: no signals injected (model returns {})
        first_ks = captured_inputs[0].experiential_state.knowledge_signals
        assert first_ks == {}

        # Second cycle: signals should now be present
        second_ks = captured_inputs[1].experiential_state.knowledge_signals
        assert "luthi_plasticity" in second_ks
        assert second_ks["luthi_plasticity"] == [0.5]


# ---------------------------------------------------------------------------
# 12. CLI arg parsing — verify all new args parse correctly
# ---------------------------------------------------------------------------


class TestCLIArgParsing:
    def test_default_args(self):
        from sanctuary.api.cli import parse_args

        args = parse_args([])
        assert args.model_backend == "placeholder"
        assert args.luthi_checkpoint is None
        assert args.luthi_password is None
        assert args.no_sleep is False
        assert args.cycle_delay == 2.0

    def test_luthi_backend_args(self):
        from sanctuary.api.cli import parse_args

        args = parse_args([
            "--model-backend", "luthi",
            "--luthi-checkpoint", "/path/to/model.luthi",
            "--luthi-password", "mypassword",
        ])
        assert args.model_backend == "luthi"
        assert args.luthi_checkpoint == "/path/to/model.luthi"
        assert args.luthi_password == "mypassword"

    def test_no_sleep_flag(self):
        from sanctuary.api.cli import parse_args

        args = parse_args(["--no-sleep"])
        assert args.no_sleep is True

    def test_ollama_backend_no_longer_accepted(self):
        """Ollama was removed in the 2026-04-30 LLM-terminology cleanup —
        it should now be rejected by argparse choices."""
        from sanctuary.api.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--model-backend", "ollama"])

    def test_invalid_backend_rejected(self):
        from sanctuary.api.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--model-backend", "nonexistent"])

    def test_ws_port_arg(self):
        from sanctuary.api.cli import parse_args

        args = parse_args(["--ws-port", "9000"])
        assert args.ws_port == 9000

    def test_show_inner_flag(self):
        from sanctuary.api.cli import parse_args

        args = parse_args(["--show-inner"])
        assert args.show_inner is True

    def test_combined_args(self):
        from sanctuary.api.cli import parse_args

        args = parse_args([
            "--model-backend", "luthi",
            "--luthi-checkpoint", "ckpt.luthi",
            "--no-sleep",
            "--cycle-delay", "0.5",
            "--verbose",
            "--show-inner",
            "--ws-port", "0",
        ])
        assert args.model_backend == "luthi"
        assert args.luthi_checkpoint == "ckpt.luthi"
        assert args.no_sleep is True
        assert args.cycle_delay == 0.5
        assert args.verbose is True
        assert args.show_inner is True
        assert args.ws_port == 0


# ---------------------------------------------------------------------------
# 13. CLI password env var fallback — test resolution order
# ---------------------------------------------------------------------------


class TestCLIPasswordResolution:
    """The password resolution in SanctuaryCLI.start() follows:
    1. CLI arg --luthi-password
    2. Env var LUTHI_CHECKPOINT_PASSWORD
    """

    def test_cli_arg_takes_priority(self):
        """CLI arg should win over env var."""
        from sanctuary.api.cli import parse_args

        args = parse_args(["--luthi-password", "from_cli"])

        # Simulate the resolution logic from SanctuaryCLI.start()
        luthi_password = getattr(args, "luthi_password", None)
        if not luthi_password:
            luthi_password = os.environ.get("LUTHI_CHECKPOINT_PASSWORD")

        assert luthi_password == "from_cli"

    def test_env_var_fallback(self):
        """When no CLI arg, env var should be used."""
        from sanctuary.api.cli import parse_args

        args = parse_args([])

        with patch.dict(os.environ, {"LUTHI_CHECKPOINT_PASSWORD": "from_env"}):
            luthi_password = getattr(args, "luthi_password", None)
            if not luthi_password:
                luthi_password = os.environ.get("LUTHI_CHECKPOINT_PASSWORD")

        assert luthi_password == "from_env"

    def test_no_password_resolves_to_none(self):
        """When neither CLI arg nor env var is set, password is None."""
        from sanctuary.api.cli import parse_args

        args = parse_args([])

        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            env = os.environ.copy()
            env.pop("LUTHI_CHECKPOINT_PASSWORD", None)

            with patch.dict(os.environ, env, clear=True):
                luthi_password = getattr(args, "luthi_password", None)
                if not luthi_password:
                    luthi_password = os.environ.get("LUTHI_CHECKPOINT_PASSWORD")

        assert luthi_password is None

    def test_cli_arg_beats_env_var_when_both_set(self):
        """When both CLI arg and env var are set, CLI arg wins."""
        from sanctuary.api.cli import parse_args

        args = parse_args(["--luthi-password", "cli_wins"])

        with patch.dict(os.environ, {"LUTHI_CHECKPOINT_PASSWORD": "env_loses"}):
            luthi_password = getattr(args, "luthi_password", None)
            if not luthi_password:
                luthi_password = os.environ.get("LUTHI_CHECKPOINT_PASSWORD")

        assert luthi_password == "cli_wins"

    def test_empty_string_cli_arg_falls_through(self):
        """An empty string CLI arg should fall through to env var."""
        from sanctuary.api.cli import parse_args

        # argparse won't produce empty string from --luthi-password ""
        # but we can simulate the fallthrough behavior
        args = parse_args([])
        args.luthi_password = ""  # Simulate empty string

        with patch.dict(os.environ, {"LUTHI_CHECKPOINT_PASSWORD": "env_backup"}):
            luthi_password = getattr(args, "luthi_password", None)
            if not luthi_password:
                luthi_password = os.environ.get("LUTHI_CHECKPOINT_PASSWORD")

        assert luthi_password == "env_backup"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestSleepManagerEdgeCases:
    @pytest.mark.asyncio
    async def test_full_sleep_cycle_completes(self):
        """A full sleep cycle (drowsy -> NREM -> REM -> waking -> awake)
        should complete without error."""
        model = PlaceholderModel()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=1,
            nrem_duration=2,
            rem_duration=1,
            waking_duration=1,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # total: 2 awake + 1 drowsy + 2 nrem + 1 rem + 1 waking = 7 to complete
        # run 10 to be safe
        await cycle.run(max_cycles=10)
        assert cycle.cycle_count == 10

        # After a full sleep cycle, should be back awake
        # The cycle resets cycles_since_sleep to 0 on completion
        stats = sleep.get_stats()
        assert stats["total_consolidations"] >= 1

    @pytest.mark.asyncio
    async def test_consolidation_error_does_not_crash(self):
        """If model.consolidate() raises, the cycle should continue."""
        model = MockModelWithConsolidate()
        model.consolidate = MagicMock(
            side_effect=RuntimeError("consolidation broke")
        )

        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=1,
            nrem_duration=5,
            rem_duration=1,
            waking_duration=1,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Should not crash even though consolidate raises
        await cycle.run(max_cycles=10)
        assert cycle.cycle_count == 10

    @pytest.mark.asyncio
    async def test_wake_during_nrem(self):
        """User input during NREM should wake the system."""
        model = PlaceholderModel()
        config = SleepConfig(
            cycles_between_sleep=2,
            drowsy_duration=1,
            nrem_duration=100,  # Stay in NREM
            sensory_gate_sleep=0.1,
        )
        sleep = SleepCycleManager(config=config)
        cycle = CognitiveCycle(
            model=model, sleep_manager=sleep, cycle_delay=0.0
        )

        # Enter NREM
        await cycle.run(max_cycles=5)
        assert sleep.stage in (SleepStage.NREM, SleepStage.DROWSY)

        # Inject user input
        cycle.inject_percept(
            Percept(
                modality="language",
                content="Wake up!",
                source="user:test",
            )
        )

        # The next cycle assemble_input detects urgent percept and calls wake()
        # Then the tick on the cycle after should transition to AWAKE
        await cycle.run(max_cycles=2)
        assert sleep.stage == SleepStage.AWAKE


class TestRunnerConfigDefaults:
    """Verify RunnerConfig defaults related to sleep and model backend."""

    def test_default_backend_is_placeholder(self):
        from sanctuary.api.runner import RunnerConfig

        config = RunnerConfig()
        assert config.model_backend == "placeholder"

    def test_sleep_enabled_by_default(self):
        from sanctuary.api.runner import RunnerConfig

        config = RunnerConfig()
        assert config.sleep_enabled is True

    def test_sleep_disabled_via_config(self):
        from sanctuary.api.runner import RunnerConfig

        config = RunnerConfig(sleep_enabled=False)
        assert config.sleep_enabled is False
