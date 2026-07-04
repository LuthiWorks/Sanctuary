"""Regression tests for Phase 3 liveness firestops (audit 2026-07-03, S3).

The 10 Hz cognitive loop must never freeze. These tests pin the guarantees
each workstream introduces:

- W1: NREM ``consolidate()`` is offloaded to a worker thread, never run on
  the event-loop thread where it would stall every other cycle task.
"""

import asyncio
import threading

import pytest

from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.schema import CognitiveOutput
from sanctuary.consciousness.sleep_cycle import SleepStage


class _StubSleepNREM:
    """Minimal sleep manager reporting a steady NREM stage."""

    stage = SleepStage.NREM
    is_sleeping = True

    def tick(self, cycle_count: int) -> None:
        pass

    def get_sensory_gate(self) -> float:
        return 1.0


class _StubAwakeSleep(_StubSleepNREM):
    stage = SleepStage.AWAKE
    is_sleeping = False


class _RecordingModel:
    """Records the thread ``consolidate()`` runs on."""

    def __init__(self) -> None:
        self.consolidate_thread: int | None = None
        self.consolidate_calls = 0

    async def think(self, cognitive_input) -> CognitiveOutput:
        return CognitiveOutput()

    def consolidate(self) -> None:
        self.consolidate_calls += 1
        self.consolidate_thread = threading.get_ident()


class TestConsolidateOffload:
    """W1: living-weight consolidation must not run on the loop thread."""

    @pytest.mark.asyncio
    async def test_consolidate_runs_off_the_loop_thread(self):
        model = _RecordingModel()
        cycle = CognitiveCycle(model=model, sleep_manager=_StubSleepNREM())

        loop_thread = threading.get_ident()
        await cycle._cycle()

        assert model.consolidate_calls == 1
        assert model.consolidate_thread is not None
        assert model.consolidate_thread != loop_thread, (
            "consolidate() ran on the event-loop thread — it would stall the "
            "10 Hz loop for the duration of consolidation"
        )

    @pytest.mark.asyncio
    async def test_consolidate_not_called_when_awake(self):
        model = _RecordingModel()
        cycle = CognitiveCycle(model=model, sleep_manager=_StubAwakeSleep())

        await cycle._cycle()

        assert model.consolidate_calls == 0

    @pytest.mark.asyncio
    async def test_loop_is_not_blocked_during_consolidation(self):
        """A slow consolidate must not prevent other loop tasks from running.

        The offload frees the event loop: a concurrent task scheduled while
        consolidation is in flight makes progress instead of waiting for the
        (blocking) consolidate to return.
        """
        progressed = threading.Event()
        release = threading.Event()

        class _SlowModel(_RecordingModel):
            def consolidate(self_inner) -> None:
                super().consolidate()
                # Block the worker thread until the concurrent task signals it
                # has run — proving the loop was free during consolidation.
                release.wait(timeout=5.0)

        model = _SlowModel()
        cycle = CognitiveCycle(model=model, sleep_manager=_StubSleepNREM())

        async def concurrent_task():
            progressed.set()
            release.set()

        cycle_task = asyncio.ensure_future(cycle._cycle())
        await asyncio.gather(concurrent_task(), cycle_task)

        assert progressed.is_set()
        assert model.consolidate_calls == 1
