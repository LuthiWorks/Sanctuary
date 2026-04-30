"""AuthorityTuner mechanical validation tests.

Tests for the automated CfC-cell authority promotion/demotion logic in
``core/authority_tuner.py``. The tuner observes ExperientialState over a
rolling window and decides whether each cell has demonstrated stable
behavior worthy of promotion, or pathological behavior (NaN, hidden-
state explosion, scaffold divergence) warranting demotion.

Originally part of "Phase 5 mechanical validation" alongside Ollama
integration tests; the Ollama-tied test classes (TestContextBudget,
TestStressCycles, TestCycleLatency) were removed in the 2026-04-30
LLM-terminology cleanup when Ollama was deprecated. ContextManager
coverage lives in tests/core/test_context_manager.py.
"""

from __future__ import annotations

from collections import deque

from sanctuary.core.authority import AuthorityLevel
from sanctuary.core.authority_tuner import (
    AuthorityTuner,
    CellObservation,
    CellStats,
    TunerConfig,
)
from sanctuary.experiential.manager import ExperientialManager, ExperientialState


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
            assert d.new_level == AuthorityLevel.MODEL_ADVISES

    def test_demote_on_nan(self):
        """NaN output should trigger immediate demotion."""
        tuner, mgr = self._make_tuner(min_cycles=0)

        # Promote precision first so we can demote it
        mgr.promote("precision", "test setup")
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.MODEL_ADVISES

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
        """Cells already at MODEL_CONTROLS should hold, not promote further."""
        tuner, mgr = self._make_tuner(window=5, min_cycles=3)

        # Promote precision to max
        mgr.promote("precision", "step 1")
        mgr.promote("precision", "step 2")
        mgr.promote("precision", "step 3")
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.MODEL_CONTROLS

        for _ in range(5):
            tuner.observe(self._stable_state())

        decisions = tuner.evaluate()
        prec = next(d for d in decisions if d.cell_name == "precision")
        assert prec.action == "hold"
        assert "MODEL_CONTROLS" in prec.reason

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

        # SCAFFOLD_ONLY → MODEL_ADVISES
        for _ in range(5):
            tuner.observe(self._stable_state())
        tuner.apply(tuner.evaluate())
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.MODEL_ADVISES

        # MODEL_ADVISES → MODEL_GUIDES (need fresh observations)
        for _ in range(5):
            tuner.observe(self._stable_state())
        tuner.apply(tuner.evaluate())
        assert mgr.authority.level("experiential_precision") == AuthorityLevel.MODEL_GUIDES

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
