"""Async actor/learner boundary for Item #6 lived-experience learning (Plan §4).

This is the Sanctuary-side concurrency boundary that lets the cognitive cycle
(the ACTOR) respond fast while the heavy world-model training (the LEARNER --
lived JEPA re-encode + corpus replay + retention gate + M9-head update) runs on
its own cadence. Luthi itself stays synchronous and lock-free; all the threading
lives here.

Why a single coarse ``model_lock`` (and not the snapshot-under-lock the plan
originally favored)
--------------------------------------------------------------------------------
The learner is not merely a *reader* of living state. Walking the actual code
(``living_layer_pc.py``, ``episode_store.py``, ``m9/runner.py``):

  * the actor's perception forward WRITES ``self.weight`` (``pc_self_modify``) and
    the episode buffers (``store()``), and READS the encoder's backprop params;
  * the learner's lived re-encode READS ``self.weight`` *live* (no clone --
    ``living_layer_pc.py:432``) and recalls the episode buffers;
  * the learner's corpus replay WRITES ``self.weight`` + episode buffers (it
    self-modifies on corpus data) and WRITES theta (optimizer step);
  * both optimizer steps WRITE theta, which the actor's forward reads;
  * the actor's ``select_action(budget=0)`` reads ``habit_net``, which the
    learner's M9-head step writes.

Because actor==sink is typically the *same* ``M9Trainer``, the actor and learner
share nearly all of the model's mutable tensors. A clone of the living buffers
for the lived re-encode read does NOT cover the learner's *writes* (corpus replay
self-mod, optimizer steps) racing the actor's reads/writes -- and there is no
path to feed the frozen re-encode a snapshot without either changing Luthi's
``encode()`` (a weight-override API) or giving the learner a separate model copy
(double-buffer). Both are scale-step work. So for the smoke-first arc we use one
coarse ``model_lock``: the actor holds it across its whole model-touching
critical section, the learner holds it across the whole of ``observe_transition``.
This serializes actor-perception against learner-training -- correct and simple.
The throughput cost is irrelevant at CPU/d=64 smoke (correctness is the
deliverable); the real concurrency (a double-buffer that publishes theta the actor
swaps to at cycle boundaries) is deferred to the scale step. See the §4 design
brief (Window B -> Window A, 2026-06-29) for the full argument.

DEADLOCK INVARIANT (real-thread mode)
-------------------------------------
The actor MUST call :meth:`AsyncLearner.submit` OUTSIDE ``model_lock``. In
threaded mode a blocking ``put()`` on a full queue waits for the learner, which
needs ``model_lock`` to drain -> deadlock if the actor still holds it. In drain
mode ``submit`` acquires ``model_lock`` to process synchronously, which
self-deadlocks if the actor already holds it. Either way: release the lock, then
submit. ``luthi_model._think_sync`` enforces this structurally (enqueue happens
after the ``with`` block).
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Transition payload
# ----------------------------------------------------------------------
@dataclass
class Transition:
    """A fully-detached realized ``(s_t, a_t, s_next)`` crossing the queue.

    Every tensor in here must be detached before it is submitted -- no autograd
    graph may cross the actor->learner boundary (a live graph pins the actor's
    forward buffers and, in threaded mode, invites cross-thread autograd
    corruption). :func:`assert_detached` enforces this at submit time.

    Fields mirror the seam's ``observe_transition`` contract:
      * ``s_t``      -- the transition's starting pooled state (prev cycle's s_t).
      * ``a_t``      -- the realized action taken from ``s_t``.
      * ``s_next``   -- the realized pooled next state (this cycle's s_t); the
                        lived-JEPA target.
      * ``plan_snapshot`` -- F4 self-describing MCTS capture (read-only; its
                        tensors are detached at construction).
      * ``context_obs``   -- raw step-0 inputs that produced ``s_t`` (Item #6
                        lived re-encode channel); ``None`` -> head-only update.
    """

    s_t: torch.Tensor
    a_t: torch.Tensor
    s_next: torch.Tensor
    counterpart_present: bool = False
    time_since_emission: float = 0.0
    plan_snapshot: Any = None  # luthi.seam_types.PlanSnapshot | None
    context_obs: Optional[dict] = None


def _iter_payload_tensors(t: Transition):
    """Yield every tensor reachable from a Transition payload."""
    for x in (t.s_t, t.a_t, t.s_next):
        if isinstance(x, torch.Tensor):
            yield x
    if t.context_obs:
        for v in t.context_obs.values():
            if isinstance(v, torch.Tensor):
                yield v
    snap = t.plan_snapshot
    if snap is not None:
        for name in ("visit_distribution", "candidate_actions"):
            v = getattr(snap, name, None)
            if isinstance(v, torch.Tensor):
                yield v


def assert_detached(t: Transition) -> None:
    """Fail loud if any payload tensor still carries an autograd graph.

    The no-graph-crosses-the-queue guarantee, asserted at the boundary rather
    than trusted. Crash over silence: a graph leaking across the queue is a real
    wiring bug, not something to paper over.
    """
    for x in _iter_payload_tensors(t):
        if x.requires_grad or x.grad_fn is not None:
            raise AssertionError(
                "Transition tensor crossing the learner queue still carries an "
                f"autograd graph (requires_grad={x.requires_grad}, "
                f"grad_fn={x.grad_fn}). Detach at the actor before submit()."
            )


def detach_context_obs(obs: Optional[dict]) -> Optional[dict]:
    """Detach+clone every tensor in a ``context_obs`` dict (aliasing defense).

    The payload outlives the actor's forward and (threaded) crosses a thread, so
    a fresh detached clone severs any aliasing with the actor's live buffers.
    Non-tensor values pass through untouched.
    """
    if obs is None:
        return None
    return {
        k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
        for k, v in obs.items()
    }


# ----------------------------------------------------------------------
# Snapshot helper -- SCALE-STEP INFRASTRUCTURE, NOT ACTIVE AT SMOKE
# ----------------------------------------------------------------------
_LIVING_BUFFER_NAMES = (
    "weight",
    "episode_contexts",
    "episode_outputs",
    "episode_saliences",
    "episode_count",
)


def clone_living_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached snapshot of a model's living-state buffers, keyed by module path.

    SCALE-STEP INFRASTRUCTURE -- **NOT ACTIVE AT SMOKE.** §4 uses the coarse
    ``model_lock``; nothing in the smoke path consumes these clones. This read
    barrier exists only to lock in snapshot semantics for the future
    double-buffer / weight-override concurrency work (see module docstring and
    the §4 design brief). It is deliberately standalone so the
    snapshot-isolation test can prove "mutating the live buffer after the clone
    leaves the clone unchanged" without wiring it into the live mechanism.
    """
    snapshot: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        for buf_name in _LIVING_BUFFER_NAMES:
            buf = getattr(module, buf_name, None)
            if isinstance(buf, torch.Tensor):
                snapshot[f"{name}.{buf_name}"] = buf.detach().clone()
    return snapshot


# ----------------------------------------------------------------------
# Sink protocol + the learner
# ----------------------------------------------------------------------
class _Sink(Protocol):
    def observe_transition(
        self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor,
        ctx: dict,
    ) -> dict: ...


_SHUTDOWN = object()  # sentinel pushed by stop() to end the consumer loop


class AsyncLearner:
    """Coarse-lock async actor/learner boundary (§4, smoke-first).

    Two modes:
      * ``"drain"``    -- :meth:`submit` processes synchronously on the caller's
        thread. Deterministic FIFO, zero loss, no background thread. The safe
        default and the deterministic mode the tests assert against.
      * ``"threaded"`` -- a dedicated background consumer thread drains a bounded
        queue. :meth:`submit` does a blocking ``put`` (no silent loss) and bumps
        a backpressure counter when the queue is full. :meth:`start` /
        :meth:`stop` manage the thread.

    The single ``model_lock`` is shared with the actor (``luthi_model``): the
    actor holds it across its whole model-touching critical section, the learner
    holds it across :meth:`_run_under_lock`. See the module docstring for why
    coarse, and the DEADLOCK INVARIANT for the submit-outside-the-lock rule.
    """

    def __init__(
        self,
        sink: _Sink,
        model_lock: "threading.Lock | threading.RLock",
        *,
        mode: str = "drain",
        maxsize: int = 64,
        resilient: bool = False,
    ):
        if mode not in ("drain", "threaded"):
            raise ValueError(f"mode must be 'drain' or 'threaded', got {mode!r}")
        self.sink = sink
        self.model_lock = model_lock
        self.mode = mode
        self.resilient = resilient

        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[BaseException] = None

        # Counters. Each has a single writer thread (produced/overflow_waits:
        # actor via submit; consumed/errors: learner), so plain int ops are safe
        # under the GIL given reads happen after a join/barrier. last_metrics is
        # a reference swap (atomic).
        self.produced = 0
        self.consumed = 0
        self.overflow_waits = 0
        self.errors = 0
        self.last_metrics: dict = {}

    # -- core: the one critical section that excludes the actor --------
    def _run_under_lock(self, transition: Transition) -> dict:
        ctx: dict[str, Any] = {
            "counterpart_present": transition.counterpart_present,
            "time_since_emission": transition.time_since_emission,
            "plan_snapshot": transition.plan_snapshot,
        }
        if transition.context_obs is not None:
            ctx["context_obs"] = transition.context_obs
        with self.model_lock:
            return self.sink.observe_transition(
                transition.s_t, transition.a_t, transition.s_next, ctx,
            )

    # -- actor entry point ---------------------------------------------
    def submit(self, transition: Transition) -> Optional[dict]:
        """Hand a detached transition to the learner.

        MUST be called OUTSIDE ``model_lock`` (see the DEADLOCK INVARIANT).
        Drain mode returns the learner metrics; threaded mode returns ``None``
        (the learner runs later).
        """
        assert_detached(transition)
        self.produced += 1

        if self.mode == "drain":
            metrics = self._run_under_lock(transition)
            self.consumed += 1
            self.last_metrics = metrics or {}
            return self.last_metrics

        # threaded: surface a dead learner to the actor rather than blocking
        # forever on a queue nobody is draining (crash over silence).
        if self._error is not None and not self.resilient:
            raise self._error
        if self._queue.full():
            self.overflow_waits += 1
            logger.debug(
                "async learner queue full (maxsize=%d); actor blocking on put "
                "-- learner is behind (overflow_waits=%d).",
                self._queue.maxsize, self.overflow_waits,
            )
        self._queue.put(transition)  # blocking: backpressure, never silent loss
        return None

    # -- threaded consumer ---------------------------------------------
    def start(self) -> None:
        if self.mode != "threaded":
            return
        if self._thread is not None:
            raise RuntimeError("AsyncLearner already started")
        self._running = True
        self._error = None
        self._thread = threading.Thread(
            target=self._loop, name="luthi-learner", daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._queue.task_done()
                return
            try:
                metrics = self._run_under_lock(item)
                self.consumed += 1
                self.last_metrics = metrics or {}
            except Exception as exc:  # noqa: BLE001 - captured + surfaced at stop
                self.errors += 1
                self._error = exc
                logger.exception(
                    "async learner: observe_transition raised on the learner "
                    "thread.",
                )
                self._queue.task_done()
                if not self.resilient:
                    # Stop the thread; the error is surfaced on the controlling
                    # thread at stop() and to the actor at the next submit().
                    self._running = False
                    return
                continue
            self._queue.task_done()

    def wait_until_drained(self) -> None:
        """Block until every submitted transition has been consumed (threaded).

        Uses ``Queue.join`` -> waits for a ``task_done`` per ``put``. Call before
        :meth:`stop` in tests so counters are settled before assertions.
        """
        if self.mode == "threaded":
            self._queue.join()

    def stop(self, timeout: float = 30.0) -> None:
        """Signal shutdown (after pending items drain, FIFO) and join the thread.

        Re-raises a non-resilient learner-thread error on the caller, so a
        background failure cannot vanish.
        """
        if self.mode != "threaded" or self._thread is None:
            return
        self._queue.put(_SHUTDOWN)
        self._thread.join(timeout)
        alive = self._thread.is_alive()
        self._thread = None
        self._running = False
        if self._error is not None and not self.resilient:
            err, self._error = self._error, None
            raise err
        if alive:
            raise RuntimeError(
                f"AsyncLearner thread did not stop within {timeout}s."
            )
