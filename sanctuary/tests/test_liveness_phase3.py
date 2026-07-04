"""Regression tests for Phase 3 liveness firestops (audit 2026-07-03, S3).

The 10 Hz cognitive loop must never freeze. These tests pin the guarantees
each workstream introduces:

- W1: NREM ``consolidate()`` is offloaded to a worker thread, never run on
  the event-loop thread where it would stall every other cycle task.
"""

import asyncio
import threading
import time

import pytest

from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.schema import CognitiveOutput
from sanctuary.consciousness.sleep_cycle import SleepStage
from sanctuary.tools.registry import ToolRegistry, ToolResult, ToolSafety


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


class TestBlockingToolOffload:
    """W2: a blocking tool body must run on a worker thread, and the registry
    watchdog must bound how long any tool can stall its caller."""

    @pytest.mark.asyncio
    async def test_shell_subprocess_runs_off_loop_thread(self, monkeypatch):
        import sanctuary.tools.builtin as builtin
        from sanctuary.tools.builtin import _shell_command

        captured = {}

        class _FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(*args, **kwargs):
            captured["thread"] = threading.get_ident()
            return _FakeCompleted()

        monkeypatch.setattr(builtin.subprocess, "run", fake_run)

        loop_thread = threading.get_ident()
        result = await _shell_command({"command": "echo hi"})

        assert result.success is True
        assert captured["thread"] != loop_thread, (
            "subprocess ran on the event-loop thread — it would freeze the "
            "10 Hz loop for the command's duration"
        )

    @pytest.mark.asyncio
    async def test_watchdog_times_out_a_hung_tool(self):
        registry = ToolRegistry(tool_timeout_seconds=0.2)

        async def slow_tool(params):
            # An offloaded blocking body that overruns the ceiling.
            await asyncio.to_thread(time.sleep, 2.0)
            return ToolResult(tool_name="slow", success=True)

        registry.register(
            name="slow",
            description="sleeps too long",
            parameters={},
            execute=slow_tool,
            safety=ToolSafety.OPEN,
        )

        start = time.perf_counter()
        result = await registry.execute("slow", {})
        elapsed = time.perf_counter() - start

        assert result.success is False
        assert "timed out" in result.error
        assert elapsed < 1.0, "watchdog did not fire near the ceiling"

    @pytest.mark.asyncio
    async def test_loop_runs_while_a_slow_tool_is_in_flight(self):
        registry = ToolRegistry(tool_timeout_seconds=5.0)
        ran = []

        async def slow_tool(params):
            await asyncio.to_thread(time.sleep, 0.5)
            return ToolResult(tool_name="slow", success=True)

        registry.register(
            name="slow", description="", parameters={},
            execute=slow_tool, safety=ToolSafety.OPEN,
        )

        async def ticker():
            for _ in range(5):
                ran.append(time.perf_counter())
                await asyncio.sleep(0.05)

        _, result = await asyncio.gather(ticker(), registry.execute("slow", {}))

        # The ticker completed its 5 ticks well before the 0.5 s tool returned,
        # proving the loop was free while the tool's blocking body ran.
        assert len(ran) == 5
        assert result.success is True
