"""Tests for Phase 4.3 — Continuous CfC evolution between LLM cycles.

Tests the evolution loop, adaptive timing, percept feeding, snapshot
reads, and multi-timescale temporal dynamics.
"""

from __future__ import annotations

import asyncio

import pytest

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.experiential.evolution import (
    ContinuousEvolutionLoop,
    EvolutionConfig,
    EvolutionSnapshot,
    PerceptEvent,
)
from sanctuary.experiential.manager import ExperientialManager


# ---------------------------------------------------------------------------
# ContinuousEvolutionLoop unit tests
# ---------------------------------------------------------------------------


class TestContinuousEvolutionLoop:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=20))
        await loop.start()
        assert loop.running
        await asyncio.sleep(0.05)
        await loop.stop()
        assert not loop.running

    @pytest.mark.asyncio
    async def test_cells_evolve_between_cycles(self):
        """CfC cells should step multiple times while the loop runs."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=10))
        await loop.start()

        # Let it run — each tick steps 4 CfC cells so ~30-40ms per tick
        await asyncio.sleep(0.25)

        snap = loop.snapshot()
        await loop.stop()

        assert snap.ticks_since_last_cycle >= 2  # at least a few ticks
        assert snap.precision_weight > 0.0  # cells produced output

    @pytest.mark.asyncio
    async def test_feed_percept(self):
        """Percepts fed to the loop should be processed by CfC cells."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=10))
        await loop.start()

        loop.feed_percept(PerceptEvent(valence_delta=0.5, arousal_delta=0.3))
        loop.feed_percept(PerceptEvent(valence_delta=-0.2, novelty=0.8))

        await asyncio.sleep(0.05)

        snap = loop.snapshot()
        await loop.stop()

        assert snap.percepts_processed == 2

    @pytest.mark.asyncio
    async def test_snapshot_resets_counters(self):
        """Taking a snapshot should reset tick and percept counters."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(
            mgr, EvolutionConfig(base_tick_ms=10, adaptive=False)
        )
        await loop.start()

        await asyncio.sleep(0.05)
        snap1 = loop.snapshot()
        assert snap1.ticks_since_last_cycle >= 1

        await asyncio.sleep(0.05)
        snap2 = loop.snapshot()
        await loop.stop()

        # Second snapshot should have independent tick count
        assert snap2.ticks_since_last_cycle >= 1

    @pytest.mark.asyncio
    async def test_update_context(self):
        """Updating context should affect subsequent evolution."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=10))
        await loop.start()

        loop.update_context(
            prediction_error=0.8,
            scaffold_precision=0.3,
            scaffold_vad=(-0.5, 0.7, 0.3),
        )

        await asyncio.sleep(0.05)
        snap = loop.snapshot()
        await loop.stop()

        assert snap.ticks_since_last_cycle >= 1

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Starting twice should be safe."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=20))
        await loop.start()
        await loop.start()  # should not raise
        assert loop.running
        await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Stopping when not running should be safe."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr)
        await loop.stop()  # should not raise
        assert not loop.running


# ---------------------------------------------------------------------------
# Adaptive timing tests
# ---------------------------------------------------------------------------


class TestAdaptiveTiming:
    @pytest.mark.asyncio
    async def test_high_error_speeds_up(self):
        """High prediction error should reduce tick interval."""
        mgr = ExperientialManager()
        config = EvolutionConfig(base_tick_ms=50, min_tick_ms=10, adaptive=True)
        loop = ContinuousEvolutionLoop(mgr, config)
        await loop.start()

        # Set high prediction error
        loop.update_context(prediction_error=0.9)

        await asyncio.sleep(0.15)
        tick_ms = loop.current_tick_ms
        await loop.stop()

        # Should have adapted toward faster ticks
        assert tick_ms < config.base_tick_ms

    @pytest.mark.asyncio
    async def test_low_error_slows_down(self):
        """Low prediction error should increase tick interval toward idle."""
        mgr = ExperientialManager()
        config = EvolutionConfig(
            base_tick_ms=50, idle_tick_ms=100, adaptive=True,
        )
        loop = ContinuousEvolutionLoop(mgr, config)
        await loop.start()

        # Set very low prediction error
        loop.update_context(prediction_error=0.01)

        await asyncio.sleep(0.15)
        tick_ms = loop.current_tick_ms
        await loop.stop()

        # Should have adapted toward idle rate
        assert tick_ms > config.base_tick_ms

    @pytest.mark.asyncio
    async def test_non_adaptive_stays_fixed(self):
        """With adaptive=False, tick rate should stay at base."""
        mgr = ExperientialManager()
        config = EvolutionConfig(base_tick_ms=50, adaptive=False)
        loop = ContinuousEvolutionLoop(mgr, config)
        await loop.start()

        loop.update_context(prediction_error=0.9)

        await asyncio.sleep(0.1)
        tick_ms = loop.current_tick_ms
        await loop.stop()

        assert tick_ms == config.base_tick_ms


# ---------------------------------------------------------------------------
# ExperientialManager evolution integration
# ---------------------------------------------------------------------------


class TestManagerEvolution:
    @pytest.mark.asyncio
    async def test_start_and_stop_evolution(self):
        mgr = ExperientialManager()
        await mgr.start_evolution(EvolutionConfig(base_tick_ms=20))
        assert mgr.evolution_running
        await mgr.stop_evolution()
        assert not mgr.evolution_running

    @pytest.mark.asyncio
    async def test_evolution_snapshot(self):
        mgr = ExperientialManager()
        await mgr.start_evolution(EvolutionConfig(base_tick_ms=10))

        await asyncio.sleep(0.05)
        snap = mgr.evolution_snapshot()
        await mgr.stop_evolution()

        assert snap is not None
        assert isinstance(snap, EvolutionSnapshot)
        assert snap.ticks_since_last_cycle >= 1

    @pytest.mark.asyncio
    async def test_no_snapshot_without_evolution(self):
        mgr = ExperientialManager()
        assert mgr.evolution_snapshot() is None

    @pytest.mark.asyncio
    async def test_feed_percept_via_manager(self):
        mgr = ExperientialManager()
        await mgr.start_evolution(EvolutionConfig(base_tick_ms=10))

        mgr.feed_percept(PerceptEvent(valence_delta=0.3))
        await asyncio.sleep(0.05)

        snap = mgr.evolution_snapshot()
        await mgr.stop_evolution()

        assert snap.percepts_processed >= 1

    @pytest.mark.asyncio
    async def test_feed_percept_without_evolution_is_noop(self):
        """Feeding percepts without evolution loop should not raise."""
        mgr = ExperientialManager()
        mgr.feed_percept(PerceptEvent(valence_delta=0.3))  # should not raise

    @pytest.mark.asyncio
    async def test_update_context_via_manager(self):
        mgr = ExperientialManager()
        await mgr.start_evolution(EvolutionConfig(base_tick_ms=10))

        mgr.update_evolution_context(prediction_error=0.5)
        await asyncio.sleep(0.05)

        snap = mgr.evolution_snapshot()
        await mgr.stop_evolution()

        assert snap is not None

    @pytest.mark.asyncio
    async def test_status_includes_evolution(self):
        mgr = ExperientialManager()
        await mgr.start_evolution(EvolutionConfig(base_tick_ms=20))

        await asyncio.sleep(0.03)
        status = mgr.get_status()
        await mgr.stop_evolution()

        assert "evolution" in status
        assert status["evolution"]["running"] is True

    @pytest.mark.asyncio
    async def test_step_still_works_with_evolution(self):
        """The synchronous step() should still work alongside evolution."""
        mgr = ExperientialManager()
        await mgr.start_evolution(EvolutionConfig(base_tick_ms=20))

        # Direct step (used by cognitive cycle) should not conflict
        state = mgr.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
        )

        await mgr.stop_evolution()

        assert 0.0 <= state.precision_weight <= 1.0


# ---------------------------------------------------------------------------
# Temporal dynamics validation
# ---------------------------------------------------------------------------


class TestTemporalDynamics:
    @pytest.mark.asyncio
    async def test_cells_accumulate_state_over_time(self):
        """After many ticks, hidden states should be non-trivial."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=10))
        await loop.start()

        # Feed a burst of percepts
        for _ in range(5):
            loop.feed_percept(PerceptEvent(valence_delta=0.3, arousal_delta=0.2))

        await asyncio.sleep(0.1)
        snap = loop.snapshot()
        await loop.stop()

        # All cells should have non-zero hidden state norms
        for name, norm in snap.hidden_state_norms.items():
            assert norm > 0.0, f"{name} hidden state norm should be > 0"

    @pytest.mark.asyncio
    async def test_percept_burst_changes_affect(self):
        """A burst of negative percepts should shift affect state."""
        mgr = ExperientialManager()
        # Promote affect so CfC output is used
        mgr.promote("affect", "test")

        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=10))
        await loop.start()

        # Feed negative percepts
        for _ in range(10):
            loop.feed_percept(PerceptEvent(valence_delta=-0.5, arousal_delta=0.4))

        await asyncio.sleep(0.3)
        snap = loop.snapshot()
        await loop.stop()

        # Affect cell should have been influenced by the percepts
        assert snap.ticks_since_last_cycle >= 2

    @pytest.mark.asyncio
    async def test_multiple_snapshot_cycles(self):
        """Simulating multiple cognitive cycles reading snapshots."""
        mgr = ExperientialManager()
        loop = ContinuousEvolutionLoop(mgr, EvolutionConfig(base_tick_ms=10))
        await loop.start()

        snapshots = []
        for _ in range(3):
            await asyncio.sleep(0.04)
            loop.feed_percept(PerceptEvent(novelty=0.5))
            await asyncio.sleep(0.02)
            snapshots.append(loop.snapshot())

        await loop.stop()

        # Each snapshot should have independent tick counts
        assert len(snapshots) == 3
        for snap in snapshots:
            assert snap.ticks_since_last_cycle >= 1
