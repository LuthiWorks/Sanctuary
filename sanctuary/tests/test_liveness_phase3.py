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
import torch

from sanctuary.core.async_learner import AsyncLearner, Transition
from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.schema import CognitiveOutput
from sanctuary.consciousness.sleep_cycle import SleepStage
from sanctuary.tools.registry import ToolRegistry, ToolResult, ToolSafety
from sanctuary.core.supervision import supervise


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


class TestBackgroundTaskSupervision:
    """W3: a fire-and-forget task's unexpected death must be observable."""

    @pytest.mark.asyncio
    async def test_death_invokes_on_death_and_logs(self, caplog):
        deaths = []

        async def boom():
            raise ValueError("substrate on fire")

        supervise(
            asyncio.create_task(boom()),
            name="boomer",
            on_death=lambda exc: deaths.append(exc),
        )
        await asyncio.sleep(0.05)  # let the task run and its callback fire

        assert len(deaths) == 1
        assert isinstance(deaths[0], ValueError)
        assert any("boomer" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_fatal_logs_at_critical(self, caplog):
        import logging as _logging

        async def boom():
            raise RuntimeError("cognitive loop died")

        with caplog.at_level(_logging.CRITICAL):
            supervise(asyncio.create_task(boom()), name="runner", fatal=True)
            await asyncio.sleep(0.05)

        assert any(
            r.levelno == _logging.CRITICAL and "runner" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_cancellation_is_not_treated_as_death(self):
        deaths = []

        async def sleeper():
            await asyncio.sleep(10)

        task = supervise(
            asyncio.create_task(sleeper()),
            name="sleeper",
            on_death=lambda exc: deaths.append(exc),
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.01)

        assert deaths == [], "normal cancellation must not fire the death handler"

    @pytest.mark.asyncio
    async def test_clean_exit_does_not_fire_handler(self):
        deaths = []

        async def quick():
            return 42

        supervise(
            asyncio.create_task(quick()),
            name="quick",
            on_death=lambda exc: deaths.append(exc),
        )
        await asyncio.sleep(0.02)

        assert deaths == []


# ---------------------------------------------------------------------------
# W4: the async learner's dead-letter must survive a process exit and be
# requeued into the next learner (audit item 8). Threaded, not asyncio.
# ---------------------------------------------------------------------------

_D = 8


def _txn(idx: int) -> Transition:
    """A detached transition whose s_t[0] encodes its submit order."""
    s_t = torch.zeros(_D)
    s_t[0] = float(idx)
    return Transition(s_t=s_t, a_t=torch.zeros(_D), s_next=torch.ones(_D))


class _BoomOnAllQueued:
    """Blocks the first observe until everything is queued, then dies."""

    def __init__(self, ready: threading.Event):
        self.ready = ready
        self.calls = 0

    def observe_transition(self, *args, **kwargs):
        self.calls += 1
        self.ready.wait(timeout=5.0)
        raise ValueError("boom in the learner")


class _RecordingSink:
    def __init__(self):
        self.seen = []

    def observe_transition(self, s_t, a_t, s_next, ctx):
        self.seen.append(int(s_t[0].item()))
        return {}


def _kill_with_queued(sink, path, n=6):
    """Start a threaded learner, queue n transitions, kill it mid-first-item."""
    ready = threading.Event()
    learner = AsyncLearner(
        sink, threading.Lock(), mode="threaded", maxsize=16, dead_letter_path=path
    )
    learner.start()
    for i in range(n):
        learner.submit(_txn(i))
    ready.set()
    learner.wait_until_drained()
    with pytest.raises(ValueError, match="boom in the learner"):
        learner.stop()
    return learner


class TestDeadLetterDurability:
    def test_dead_letter_persisted_on_death(self, tmp_path):
        path = tmp_path / "dead_letter.pt"
        ready = threading.Event()
        learner = _kill_with_queued(_BoomOnAllQueued(ready), path)

        # item 0 was consumed (and raised); items 1..5 drained to dead_letter
        assert len(learner.dead_letter) == 5
        assert path.exists(), "dead-letter was not persisted; a process exit loses it"

    def test_recovered_and_requeued_by_next_learner(self, tmp_path):
        path = tmp_path / "dead_letter.pt"
        ready = threading.Event()
        _kill_with_queued(_BoomOnAllQueued(ready), path)
        assert path.exists()

        # A fresh, healthy learner reclaims the persisted transitions on start.
        sink = _RecordingSink()
        survivor = AsyncLearner(
            sink, threading.Lock(), mode="threaded", maxsize=16, dead_letter_path=path
        )
        survivor.start()
        survivor.wait_until_drained()
        survivor.stop()

        assert sorted(sink.seen) == [1, 2, 3, 4, 5]
        assert not path.exists(), "recovered file must be consumed and removed"

    def test_no_path_keeps_dead_letter_in_memory_only(self, tmp_path):
        ready = threading.Event()
        learner = AsyncLearner(
            _BoomOnAllQueued(ready), threading.Lock(), mode="threaded", maxsize=16
        )
        learner.start()
        for i in range(4):
            learner.submit(_txn(i))
        ready.set()
        learner.wait_until_drained()
        with pytest.raises(ValueError):
            learner.stop()

        assert learner.dead_letter_path is None
        assert len(learner.dead_letter) == 3  # in memory, not persisted

    def test_corrupt_dead_letter_file_is_quarantined(self, tmp_path):
        path = tmp_path / "dead_letter.pt"
        path.write_bytes(b"not a torch checkpoint")

        sink = _RecordingSink()
        learner = AsyncLearner(
            sink, threading.Lock(), mode="threaded", maxsize=16, dead_letter_path=path
        )
        learner.start()  # must not raise on a corrupt file
        learner.stop()

        assert sink.seen == []
        # Quarantined, not deleted: the original is gone but the evidence and
        # possibly-recoverable bytes are preserved as .corrupt.
        assert not path.exists(), "corrupt file must not recur on the next boot"
        assert path.with_suffix(path.suffix + ".corrupt").exists()

    def test_poison_pill_is_dropped_and_tail_is_preserved(self, tmp_path):
        """A transition that kills the consumer must error once and be dropped,
        never lose the transitions queued behind it, and never re-poison."""
        path = tmp_path / "dead_letter.pt"
        ready = threading.Event()
        _kill_with_queued(_BoomOnAllQueued(ready), path)  # persists [1..5]

        class _DieOnThree:
            def observe_transition(self, s_t, a_t, s_next, ctx):
                if int(s_t[0].item()) == 3:
                    raise ValueError("poison pill")
                return {}

        survivor = AsyncLearner(
            _DieOnThree(), threading.Lock(), mode="threaded",
            maxsize=16, dead_letter_path=path,
        )
        survivor.start()  # recovers 1..5; consumes 1,2; 3 poisons; 4,5 drained
        survivor.wait_until_drained()
        with pytest.raises(ValueError, match="poison pill"):
            survivor.stop()

        # 3 was consumed-and-errored (dropped, not re-queued); 4,5 must survive.
        assert path.exists()
        sink = _RecordingSink()
        healed = AsyncLearner(
            sink, threading.Lock(), mode="threaded", maxsize=16, dead_letter_path=path
        )
        healed.start()
        healed.wait_until_drained()
        healed.stop()

        assert sorted(sink.seen) == [4, 5], "tail behind the poison pill was lost"
        assert not path.exists()
