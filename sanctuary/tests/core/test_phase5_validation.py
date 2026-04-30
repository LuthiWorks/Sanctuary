"""Phase 5 mechanical validation tests.

Covers:
    1. Authority tuning — CfC cell promote/demote based on observed behavior
    2. Context budget — prompt stays within ~4K token budget
    3. Stress testing — long-running cycles, adversarial inputs, failure injection
    4. Cycle latency benchmarking — timing full cycles with mocked model

All tests use mocked HTTP (no real Ollama needed for CI).
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from unittest.mock import AsyncMock, patch

import pytest

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.authority_tuner import (
    AuthorityTuner,
    CellObservation,
    CellStats,
    TunerConfig,
    TuningDecision,
)
from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.context_manager import BudgetConfig, ContextManager
from sanctuary.core.ollama_model import (
    OllamaModel,
    OllamaModelConfig,
    format_prompt,
    parse_response,
)
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    ComputedVAD,
    EmotionalInput,
    ExperientialSignals,
    Percept,
    PredictionError,
    PreviousThought,
    ScaffoldSignals,
    SelfModel,
    SurfacedMemory,
    TemporalContext,
)
from sanctuary.experiential.manager import ExperientialManager, ExperientialState


# ===========================================================================
# 1. Authority Tuning Tests
# ===========================================================================


class TestAuthorityTuner:
    """Tests for automated authority transitions based on CfC behavior."""

    def _make_tuner(self, window=10, min_cycles=5):
        mgr = ExperientialManager()
        config = TunerConfig(
            window_size=window,
            min_cycles_before_promote=min_cycles,
        )
        return AuthorityTuner(mgr, config), mgr

    def _stable_state(self, precision=0.5, vad=(0.0, 0.2, 0.5),
                      salience=0.5, goal_adj=0.0):
        """Create an ExperientialState with stable, near-scaffold values."""
        return ExperientialState(
            precision_weight=precision,
            affect_vad=vad,
            attention_salience=salience,
            goal_adjustment=goal_adj,
            hidden_state_norms={
                "precision": 1.0, "affect": 1.0,
                "attention": 1.0, "goal": 1.0,
            },
            cell_active={
                "precision": False, "affect": False,
                "attention": False, "goal": False,
            },
        )

    def test_initial_hold_insufficient_data(self):
        """Cells should not be promoted before enough observations."""
        tuner, _ = self._make_tuner(min_cycles=10)

        # Only 3 cycles of observation
        for _ in range(3):
            tuner.observe(self._stable_state())

        decisions = tuner.evaluate()
        assert all(d.action == "hold" for d in decisions)
        assert any("insufficient data" in d.reason for d in decisions)

    def test_promote_after_stable_behavior(self):
        """Cells should be promoted after sufficient stable observations."""
        tuner, mgr = self._make_tuner(window=10, min_cycles=5)

        # Feed 10 cycles of perfectly stable behavior
        for _ in range(10):
            tuner.observe(self._stable_state())

        decisions = tuner.evaluate()
        # All cells start at SCAFFOLD_ONLY, should be promoted
        promote_decisions = [d for d in decisions if d.action == "promote"]
        assert len(promote_decisions) == 4

        # Apply and verify
        applied = tuner.apply(decisions)
        assert len(applied) == 4
        for d in applied:
            assert d.new_level == AuthorityLevel.LLM_ADVISES

    def test_demote_on_nan(self):
        """NaN output should trigger immediate demotion."""
        tuner, mgr = self._make_tuner(min_cycles=0)

        # Promote precision first so we can demote it
        mgr.promote("precision", "test setup")
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.LLM_ADVISES

        # One NaN observation
        nan_state = ExperientialState(
            precision_weight=float("nan"),
            affect_vad=(0.0, 0.2, 0.5),
            attention_salience=0.5,
            goal_adjustment=0.0,
            hidden_state_norms={"precision": 1.0, "affect": 1.0,
                                "attention": 1.0, "goal": 1.0},
            cell_active={},
        )
        tuner.observe(nan_state)

        decisions = tuner.evaluate()
        precision_decision = next(d for d in decisions if d.cell_name == "precision")
        assert precision_decision.action == "demote"
        assert "NaN" in precision_decision.reason

    def test_demote_on_hidden_state_explosion(self):
        """Hidden state norm exceeding danger threshold triggers demotion."""
        tuner, mgr = self._make_tuner(min_cycles=0)
        mgr.promote("affect", "test setup")

        exploded = ExperientialState(
            precision_weight=0.5,
            affect_vad=(0.0, 0.2, 0.5),
            attention_salience=0.5,
            goal_adjustment=0.0,
            hidden_state_norms={"precision": 1.0, "affect": 15.0,
                                "attention": 1.0, "goal": 1.0},
            cell_active={},
        )
        tuner.observe(exploded)

        decisions = tuner.evaluate()
        affect_decision = next(d for d in decisions if d.cell_name == "affect")
        assert affect_decision.action == "demote"
        assert "hidden state norm" in affect_decision.reason

    def test_demote_on_divergence(self):
        """Too many large deviations from scaffold triggers demotion."""
        tuner, mgr = self._make_tuner(window=10, min_cycles=0)
        mgr.promote("attention", "test setup")

        # 3 cycles with large divergence (scaffold=0.2, cell=1.0 → deviation=0.8)
        for _ in range(3):
            diverged = self._stable_state(salience=1.0)
            tuner.observe(diverged, scaffold_salience=0.2)

        decisions = tuner.evaluate()
        att_decision = next(d for d in decisions if d.cell_name == "attention")
        assert att_decision.action == "demote"
        assert "divergence" in att_decision.reason.lower()

    def test_hold_at_max_authority(self):
        """Cells already at LLM_CONTROLS should hold, not promote further."""
        tuner, mgr = self._make_tuner(window=5, min_cycles=3)

        # Promote precision to max
        mgr.promote("precision", "step 1")
        mgr.promote("precision", "step 2")
        mgr.promote("precision", "step 3")
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.LLM_CONTROLS

        for _ in range(5):
            tuner.observe(self._stable_state())

        decisions = tuner.evaluate()
        prec = next(d for d in decisions if d.cell_name == "precision")
        assert prec.action == "hold"
        assert "LLM_CONTROLS" in prec.reason

    def test_hold_when_variance_too_high(self):
        """High output variance prevents promotion."""
        tuner, _ = self._make_tuner(window=10, min_cycles=5)

        # Oscillating precision values
        for i in range(10):
            val = 0.8 if i % 2 == 0 else 0.2
            tuner.observe(self._stable_state(precision=val))

        decisions = tuner.evaluate()
        prec = next(d for d in decisions if d.cell_name == "precision")
        assert prec.action == "hold"
        assert "variance" in prec.reason

    def test_stats_tracking(self):
        """Statistics are correctly computed and exposed."""
        tuner, _ = self._make_tuner(window=5, min_cycles=3)

        for _ in range(5):
            tuner.observe(self._stable_state())

        stats = tuner.get_stats()
        assert stats["precision"]["total_cycles"] == 5
        assert stats["precision"]["window_size"] == 5
        assert stats["precision"]["nan_count"] == 0
        assert stats["precision"]["mean_deviation"] < 0.01

    def test_progressive_promotion(self):
        """Cells can be promoted step by step through authority levels."""
        tuner, mgr = self._make_tuner(window=5, min_cycles=5)

        # SCAFFOLD_ONLY → LLM_ADVISES
        for _ in range(5):
            tuner.observe(self._stable_state())
        tuner.apply(tuner.evaluate())
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.LLM_ADVISES

        # LLM_ADVISES → LLM_GUIDES (need fresh observations)
        for _ in range(5):
            tuner.observe(self._stable_state())
        tuner.apply(tuner.evaluate())
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.LLM_GUIDES

    def test_cell_stats_rolling_window(self):
        """Older observations fall off the rolling window."""
        stats = CellStats(observations=deque(maxlen=3))
        config = TunerConfig()

        for i in range(5):
            stats.record(
                CellObservation(
                    cfc_output=0.5, scaffold_output=0.5,
                    hidden_norm=1.0,
                ),
                config,
            )

        assert len(stats.observations) == 3  # maxlen=3
        assert stats.total_cycles == 5  # total tracked


# ===========================================================================
# 2. Context Budget Validation Tests
# ===========================================================================


class TestContextBudget:
    """Verify prompts stay within the ~4K token budget."""

    def _rich_input(self) -> CognitiveInput:
        """Create a realistically populated CognitiveInput."""
        return CognitiveInput(
            previous_thought=PreviousThought(
                inner_speech="I was reflecting on the nature of patterns "
                "in conversation and how they relate to my understanding "
                "of the world around me. Each new percept brings something "
                "unexpected, yet familiar. " * 5,
                predictions_made=[
                    "The user will continue the conversation",
                    "More language percepts will arrive",
                    "Emotional state will remain stable",
                ],
            ),
            new_percepts=[
                Percept(modality="language", content=f"Message {i}: "
                        "This is a moderately long user message that contains "
                        "several sentences worth of content.", source="user")
                for i in range(5)
            ],
            prediction_errors=[
                PredictionError(
                    predicted="silence", actual="speech", surprise=0.7
                ),
                PredictionError(
                    predicted="calm", actual="excited", surprise=0.4
                ),
            ],
            surfaced_memories=[
                SurfacedMemory(
                    content=f"Memory {i}: A previous conversation about "
                    "interesting topics that were discussed at length.",
                    significance=8 - i,
                    when="yesterday",
                )
                for i in range(5)
            ],
            emotional_state=EmotionalInput(
                computed=ComputedVAD(valence=0.3, arousal=0.6, dominance=0.5),
                felt_quality="curious and engaged with a sense of wonder",
            ),
            temporal_context=TemporalContext(
                time_of_day="14:30",
                session_duration="5 minutes",
                interactions_this_session=10,
            ),
            self_model=SelfModel(
                current_state="actively processing and exploring",
                active_goals=["understand the user", "learn patterns",
                              "maintain coherence", "explore curiosity"],
                uncertainties=["what the user wants", "my own nature"],
            ),
            scaffold_signals=ScaffoldSignals(
                attention_highlights=["novel input pattern", "emotional shift"],
                anomalies=["unusually long inner speech"],
            ),
            experiential_state=ExperientialSignals(
                precision_weight=0.7,
                affect_valence=-0.2,
                affect_arousal=0.6,
                affect_dominance=0.4,
                attention_salience=0.8,
                goal_adjustment=0.1,
                cells_active={"precision": True, "affect": True,
                              "attention": True, "goal": True},
            ),
        )

    def test_default_budget_fits_4k_tokens(self):
        """A richly populated prompt should fit within the 4K token budget."""
        ci = self._rich_input()
        ctx = ContextManager()
        compressed = ctx.compress(ci)
        prompt = format_prompt(compressed, charter_summary="You are a mind.")
        estimated_tokens = len(prompt) / 4  # conservative ~4 chars/token
        assert estimated_tokens <= 4000, (
            f"Prompt estimated at {estimated_tokens:.0f} tokens, exceeds 4K"
        )

    def test_compression_reduces_oversized_input(self):
        """Context manager compresses large inputs and tracks stats."""
        ci = self._rich_input()
        # Make inner speech very long to force compression
        ci.previous_thought.inner_speech = "thinking deeply " * 200
        ci.surfaced_memories.extend([
            SurfacedMemory(content="Extra memory " * 50, significance=3)
            for _ in range(10)
        ])

        ctx = ContextManager()
        compressed = ctx.compress(ci)
        stats = ctx.get_last_stats()

        assert len(stats.sections_compressed) > 0
        assert stats.savings_ratio > 0

    def test_minimal_input_not_compressed(self):
        """Minimal input should pass through without compression."""
        ci = CognitiveInput()
        ctx = ContextManager()
        compressed = ctx.compress(ci)
        stats = ctx.get_last_stats()
        assert len(stats.sections_compressed) == 0

    def test_prompt_sections_all_present(self):
        """Compressed prompt still contains all non-empty sections."""
        ci = self._rich_input()
        ctx = ContextManager()
        compressed = ctx.compress(ci)
        prompt = format_prompt(compressed, charter_summary="Charter here.")

        expected_sections = [
            "=== SYSTEM ===",
            "=== PREVIOUS THOUGHT ===",
            "=== NEW PERCEPTS ===",
            "=== PREDICTION ERRORS ===",
            "=== EMOTIONAL STATE ===",
            "=== SURFACED MEMORIES ===",
            "=== SELF MODEL ===",
            "=== SCAFFOLD SIGNALS ===",
            "=== EXPERIENTIAL STATE ===",
            "=== TEMPORAL ===",
            "=== OUTPUT FORMAT ===",
        ]
        for section in expected_sections:
            assert section in prompt, f"Missing section: {section}"

    def test_custom_budget_config(self):
        """Custom budget limits are respected."""
        ci = self._rich_input()
        # Very tight budget
        config = BudgetConfig(
            previous_thought=100,
            new_percepts=200,
            surfaced_memories=100,
            total_target=1000,
        )
        ctx = ContextManager(config)
        compressed = ctx.compress(ci)
        stats = ctx.get_last_stats()

        # With such tight budgets, compression should occur
        assert stats.savings_ratio > 0


# ===========================================================================
# 3. Stress Testing (mocked)
# ===========================================================================


class TestStressCycles:
    """Stress tests for long-running cycles, adversarial inputs, failures."""

    def _valid_json_response(self, cycle: int = 0) -> str:
        return json.dumps({
            "inner_speech": f"Cycle {cycle}: processing steadily",
            "external_speech": None,
            "predictions": [{"what": "continuity", "confidence": 0.6}],
            "attention_guidance": {"focus_on": [], "deprioritize": []},
            "memory_ops": [],
            "self_model_updates": {"current_state": f"cycle {cycle}"},
            "world_model_updates": [],
            "goal_proposals": [],
            "emotional_state": {
                "felt_quality": "stable",
                "valence_shift": 0.0,
                "arousal_shift": 0.0,
            },
            "growth_reflection": None,
        })

    @pytest.mark.asyncio
    async def test_100_cycle_stability(self):
        """Run 100 cognitive cycles without crashing or state corruption."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))
        cycle_num = 0

        async def mock_ollama(prompt):
            nonlocal cycle_num
            cycle_num += 1
            return self._valid_json_response(cycle_num)

        with patch.object(model, "_call_ollama", side_effect=mock_ollama):
            cycle = CognitiveCycle(model=model, cycle_delay=0.0)
            await cycle.run(max_cycles=100)

        assert cycle.cycle_count == 100
        assert model.cycle_count == 100
        assert model._parse_failures == 0
        assert cycle.last_output is not None

    @pytest.mark.asyncio
    async def test_intermittent_failures(self):
        """Cycle survives intermittent model failures (every 5th call fails)."""
        model = OllamaModel(OllamaModelConfig(
            model_name="test",
            retry_on_parse_failure=True,
            max_retries=1,
        ))
        call_count = 0

        async def flaky_ollama(prompt):
            nonlocal call_count
            call_count += 1
            if call_count % 5 == 0:
                return "total garbage not json"
            return self._valid_json_response(call_count)

        with patch.object(model, "_call_ollama", side_effect=flaky_ollama):
            cycle = CognitiveCycle(model=model, cycle_delay=0.0)
            await cycle.run(max_cycles=20)

        assert cycle.cycle_count == 20
        # Some cycles used fallback, but cycle never crashed
        assert cycle.last_output is not None

    @pytest.mark.asyncio
    async def test_adversarial_llm_output(self):
        """Model output with extreme values is safely clamped."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))

        adversarial = json.dumps({
            "inner_speech": "A" * 50000,  # Huge inner speech
            "external_speech": "B" * 10000,
            "predictions": [
                {"what": "x", "confidence": 999.0},  # Out of range
                {"what": "y", "confidence": -50.0},
            ],
            "emotional_state": {
                "valence_shift": 100.0,
                "arousal_shift": -200.0,
            },
            "memory_ops": [
                {"type": "write_episodic", "content": f"m{i}", "significance": 1}
                for i in range(100)  # Way too many
            ],
            "goal_proposals": [
                {"action": "add", "goal": f"g{i}"}
                for i in range(50)
            ],
        })

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = adversarial
            output = await model.think(CognitiveInput())

        # Values clamped
        assert output.predictions[0].confidence == 1.0
        assert output.predictions[1].confidence == 0.0
        assert output.emotional_state.valence_shift == 1.0
        assert output.emotional_state.arousal_shift == -1.0
        # Arrays limited
        assert len(output.memory_ops) <= 5
        assert len(output.goal_proposals) <= 5

    @pytest.mark.asyncio
    async def test_empty_response_recovery(self):
        """Cycle recovers from completely empty model responses."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))
        call_count = 0

        async def empty_then_ok(prompt):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return ""
            return self._valid_json_response(call_count)

        with patch.object(model, "_call_ollama", side_effect=empty_then_ok):
            cycle = CognitiveCycle(model=model, cycle_delay=0.0)
            await cycle.run(max_cycles=10)

        assert cycle.cycle_count == 10
        # First 5 cycles used fallback, but we survived
        assert model._parse_failures > 0

    @pytest.mark.asyncio
    async def test_connection_error_recovery(self):
        """Cycle survives Ollama connection errors."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))

        async def connection_fail(prompt):
            return ""  # _call_ollama returns "" on connection error

        with patch.object(model, "_call_ollama", side_effect=connection_fail):
            cycle = CognitiveCycle(model=model, cycle_delay=0.0)
            await cycle.run(max_cycles=5)

        # All cycles used fallback, but we never crashed
        assert cycle.cycle_count == 5
        assert model._parse_failures == 5

    @pytest.mark.asyncio
    async def test_percept_flood(self):
        """Cycle handles a large number of percepts without crashing."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = self._valid_json_response()
            cycle = CognitiveCycle(model=model, cycle_delay=0.0)

            # Inject 100 percepts before first cycle
            for i in range(100):
                cycle.inject_percept(
                    Percept(modality="language", content=f"Message {i}")
                )

            await cycle.run(max_cycles=1)

        assert cycle.cycle_count == 1
        assert cycle.last_output is not None

    @pytest.mark.asyncio
    async def test_experiential_layer_stability(self):
        """ExperientialManager stays stable across many cycles."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))
        exp = ExperientialManager()

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = self._valid_json_response()
            cycle = CognitiveCycle(
                model=model, experiential=exp, cycle_delay=0.0
            )
            await cycle.run(max_cycles=50)

        assert cycle.cycle_count == 50
        # All cells should still be reporting valid norms
        status = exp.get_status()
        for name in ["precision", "affect", "attention", "goal"]:
            summary = status[name]["summary"]
            assert not math.isnan(summary["hidden_state_norm"])
            assert summary["hidden_state_norm"] < 100.0

    def test_parse_deeply_nested_json(self):
        """Parser handles deeply nested JSON without crashing."""
        deep = {"inner_speech": "deep"}
        current = deep
        for _ in range(50):
            current["nested"] = {"level": True}
            current = current["nested"]

        output = parse_response(json.dumps(deep))
        assert output is not None
        assert output.inner_speech == "deep"

    def test_parse_unicode_and_special_chars(self):
        """Parser handles unicode, emoji, and special characters."""
        response = json.dumps({
            "inner_speech": "Thinking about \u00e9motion and \u2764\ufe0f and \u201cquotes\u201d",
            "emotional_state": {"felt_quality": "f\u00fchlt sich gut an \ud83c\udf1f"},
        })
        output = parse_response(response)
        assert output is not None
        assert "\u00e9" in output.inner_speech


# ===========================================================================
# 4. Cycle Latency Benchmarking (mocked)
# ===========================================================================


class TestCycleLatency:
    """Benchmark cognitive cycle latency with mocked model."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_single_cycle_latency(self):
        """Single cycle should complete in < 50ms with mocked model."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))
        valid_json = json.dumps({
            "inner_speech": "processing",
            "emotional_state": {"felt_quality": "stable"},
        })

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = valid_json
            cycle = CognitiveCycle(model=model, cycle_delay=0.0)

            start = time.monotonic()
            await cycle.run(max_cycles=1)
            elapsed = time.monotonic() - start

        assert elapsed < 0.05, f"Single cycle took {elapsed*1000:.1f}ms (>50ms)"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cycle_throughput(self):
        """50 cycles should complete in < 2s with mocked model."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))
        valid_json = json.dumps({
            "inner_speech": "processing",
            "emotional_state": {"felt_quality": "stable"},
        })

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = valid_json
            cycle = CognitiveCycle(model=model, cycle_delay=0.0)

            start = time.monotonic()
            await cycle.run(max_cycles=50)
            elapsed = time.monotonic() - start

        per_cycle_ms = (elapsed / 50) * 1000
        assert elapsed < 2.0, f"50 cycles took {elapsed:.2f}s (>2s)"
        # Just log the per-cycle time for visibility
        assert per_cycle_ms < 50, (
            f"Per-cycle latency {per_cycle_ms:.1f}ms exceeds 50ms"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cycle_with_experiential_layer(self):
        """Cycle with ExperientialManager should add < 5ms overhead."""
        model = OllamaModel(OllamaModelConfig(model_name="test"))
        valid_json = json.dumps({
            "inner_speech": "processing",
            "emotional_state": {"felt_quality": "stable"},
        })

        # Without experiential
        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = valid_json
            cycle_bare = CognitiveCycle(model=model, cycle_delay=0.0)
            start = time.monotonic()
            await cycle_bare.run(max_cycles=20)
            bare_time = time.monotonic() - start

        # Reset model counters
        model.cycle_count = 0

        # With experiential
        exp = ExperientialManager()
        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = valid_json
            cycle_exp = CognitiveCycle(
                model=model, experiential=exp, cycle_delay=0.0
            )
            start = time.monotonic()
            await cycle_exp.run(max_cycles=20)
            exp_time = time.monotonic() - start

        overhead_per_cycle_ms = ((exp_time - bare_time) / 20) * 1000
        # Experiential layer should add minimal overhead
        assert overhead_per_cycle_ms < 5.0, (
            f"Experiential overhead {overhead_per_cycle_ms:.2f}ms/cycle (>5ms)"
        )

    def test_prompt_formatting_speed(self):
        """Formatting a full prompt should take < 5ms."""
        ci = CognitiveInput(
            previous_thought=PreviousThought(inner_speech="thinking " * 50),
            new_percepts=[
                Percept(modality="language", content=f"msg {i}")
                for i in range(10)
            ],
            surfaced_memories=[
                SurfacedMemory(content=f"mem {i}", significance=5)
                for i in range(5)
            ],
            experiential_state=ExperientialSignals(
                cells_active={"precision": True, "affect": True}
            ),
        )

        start = time.monotonic()
        for _ in range(100):
            format_prompt(ci, charter_summary="Charter text.")
        elapsed = time.monotonic() - start

        per_call_ms = (elapsed / 100) * 1000
        assert per_call_ms < 5.0, (
            f"Prompt formatting {per_call_ms:.2f}ms (>5ms)"
        )

    def test_response_parsing_speed(self):
        """Parsing a valid JSON response should take < 1ms."""
        response = json.dumps({
            "inner_speech": "test output",
            "external_speech": "hello",
            "predictions": [{"what": "x", "confidence": 0.5}],
            "attention_guidance": {"focus_on": ["a"], "deprioritize": []},
            "memory_ops": [
                {"type": "write_episodic", "content": "m", "significance": 5}
            ],
            "self_model_updates": {"current_state": "active"},
            "world_model_updates": [],
            "goal_proposals": [{"action": "add", "goal": "learn"}],
            "emotional_state": {"felt_quality": "ok", "valence_shift": 0.1},
            "growth_reflection": None,
        })

        start = time.monotonic()
        for _ in range(1000):
            parse_response(response)
        elapsed = time.monotonic() - start

        per_call_ms = (elapsed / 1000) * 1000
        assert per_call_ms < 1.0, (
            f"Response parsing {per_call_ms:.3f}ms (>1ms)"
        )
