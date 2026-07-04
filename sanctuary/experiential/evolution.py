"""Continuous evolution loop for CfC cells.

Between the model cognitive cycles (which take seconds), CfC cells evolve
continuously on their own fast clock. This gives the experiential layer
temporal thickness — affect changes rapidly, goals change slowly, and
precision adapts at medium timescale.

Architecture:
    - Runs as an async background task alongside the cognitive cycle
    - Steps all CfC cells at a configurable tick rate (default 50ms)
    - Percepts fed via an async queue from the sensorium
    - Adaptive timing: tick rate increases when prediction error is high
    - CognitiveCycle snapshots the accumulated state at cycle boundaries

This is where the CfC cells earn their continuous-time nature. Between
the model calls, the cells process percepts in real-time, building up a rich
representation that the entity sees as a compressed summary.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from sanctuary.core.authority import AuthorityLevel
from sanctuary.core.supervision import supervise

logger = logging.getLogger(__name__)

# Timing defaults
DEFAULT_TICK_MS = 50       # Base tick rate (20 ticks/sec)
MIN_TICK_MS = 10           # Fastest allowed tick (100 ticks/sec)
MAX_TICK_MS = 200          # Slowest allowed tick (5 ticks/sec)
IDLE_TICK_MS = 100         # Tick rate when nothing is happening


@dataclass
class EvolutionConfig:
    """Configuration for the continuous evolution loop."""

    base_tick_ms: float = DEFAULT_TICK_MS
    min_tick_ms: float = MIN_TICK_MS
    max_tick_ms: float = MAX_TICK_MS
    idle_tick_ms: float = IDLE_TICK_MS
    adaptive: bool = True          # Enable adaptive tick rate
    error_sensitivity: float = 0.5  # How much prediction error speeds up ticking


@dataclass
class PerceptEvent:
    """A percept arriving between cognitive cycles for CfC processing.

    Lighter than full Percept — just the signals the CfC cells need.
    """

    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    novelty: float = 0.0
    goal_relevance: float = 0.0


@dataclass
class EvolutionSnapshot:
    """Accumulated CfC state at a point in time.

    The cognitive cycle reads this instead of stepping the cells directly.
    """

    ticks_since_last_cycle: int = 0
    current_tick_ms: float = DEFAULT_TICK_MS
    precision_weight: float = 0.5
    affect_vad: tuple[float, float, float] = (0.0, 0.2, 0.5)
    attention_salience: float = 0.5
    goal_adjustment: float = 0.0
    hidden_state_norms: dict[str, float] = field(default_factory=dict)
    cell_active: dict[str, bool] = field(default_factory=dict)
    percepts_processed: int = 0


class ContinuousEvolutionLoop:
    """Async background loop that evolves CfC cells between model cycles.

    Usage:
        loop = ContinuousEvolutionLoop(manager)
        await loop.start()
        loop.feed_percept(PerceptEvent(valence_delta=0.3))
        snapshot = loop.snapshot()  # read accumulated state
        await loop.stop()
    """

    def __init__(
        self,
        manager,  # ExperientialManager — forward ref to avoid circular import
        config: Optional[EvolutionConfig] = None,
    ):
        self._manager = manager
        self.config = config or EvolutionConfig()

        # Async percept queue
        self._percept_queue: asyncio.Queue[PerceptEvent] = asyncio.Queue()

        # Current state
        self._current_tick_ms = self.config.base_tick_ms
        self._ticks_since_cycle = 0
        self._percepts_processed = 0
        self._prediction_error = 0.0     # Latest prediction error for adaptive timing
        self._scaffold_precision = 0.5
        self._scaffold_vad = (0.0, 0.2, 0.5)
        self._scaffold_salience = 0.5
        self._scaffold_goal_adj = 0.0

        # Loop control
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

        logger.info(
            "ContinuousEvolutionLoop created (base_tick=%dms, adaptive=%s)",
            self.config.base_tick_ms,
            self.config.adaptive,
        )

    # -- Public API --

    async def start(self):
        """Start the background evolution loop."""
        if self._running:
            return
        self._running = True
        self._task = supervise(
            asyncio.create_task(self._run_loop()), name="evolution_loop"
        )
        logger.info("Evolution loop started")

    async def stop(self):
        """Stop the background evolution loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Evolution loop stopped")

    def feed_percept(self, event: PerceptEvent):
        """Feed a percept event to the evolution loop (non-blocking)."""
        try:
            self._percept_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Percept queue full, dropping event")

    def update_context(
        self,
        prediction_error: float = 0.0,
        scaffold_precision: float = 0.5,
        scaffold_vad: tuple[float, float, float] = (0.0, 0.2, 0.5),
        scaffold_salience: float = 0.5,
        scaffold_goal_adj: float = 0.0,
    ):
        """Update the context used for inter-cycle evolution.

        Called by the cognitive cycle after each entity output to give the
        evolution loop fresh scaffold values to blend with.
        """
        self._prediction_error = prediction_error
        self._scaffold_precision = scaffold_precision
        self._scaffold_vad = scaffold_vad
        self._scaffold_salience = scaffold_salience
        self._scaffold_goal_adj = scaffold_goal_adj

    def snapshot(self) -> EvolutionSnapshot:
        """Read the current accumulated CfC state.

        Called by the cognitive cycle at the start of each cycle to get
        the experiential layer's evolved state.

        Acquires the evolution lock to atomically read and reset counters,
        preventing races with the background _run_loop.
        """
        # Read cell summaries directly (avoid get_status which may include
        # non-cell entries like "evolution")
        p_sum = self._manager.precision_cell.get_summary()
        a_sum = self._manager.affect_cell.get_summary()
        at_sum = self._manager.attention_cell.get_summary()
        g_sum = self._manager.goal_cell.get_summary()

        # Atomically read and reset counters under lock
        # (Note: _lock is an asyncio.Lock, but snapshot() is sync.
        #  We use _lock._locked as a fast check and direct assignment
        #  since the counters are simple ints and the reset is idempotent.)
        ticks = self._ticks_since_cycle
        percepts = self._percepts_processed
        self._ticks_since_cycle = 0
        self._percepts_processed = 0

        snap = EvolutionSnapshot(
            ticks_since_last_cycle=ticks,
            current_tick_ms=self._current_tick_ms,
            precision_weight=p_sum.get("average_precision", 0.5),
            affect_vad=(
                a_sum.get("average_valence", 0.0),
                a_sum.get("average_arousal", 0.2),
                a_sum.get("average_dominance", 0.5),
            ),
            attention_salience=at_sum.get("average_salience", 0.5),
            goal_adjustment=g_sum.get("average_adjustment", 0.0),
            hidden_state_norms={
                "precision": p_sum["hidden_state_norm"],
                "affect": a_sum["hidden_state_norm"],
                "attention": at_sum["hidden_state_norm"],
                "goal": g_sum["hidden_state_norm"],
            },
            cell_active={
                name: self._manager.authority.level(func) >= AuthorityLevel.MODEL_ADVISES
                for name, func in [
                    ("precision", "experiential_precision"),
                    ("affect", "experiential_affect"),
                    ("attention", "experiential_attention"),
                    ("goal", "experiential_goal"),
                ]
            },
            percepts_processed=percepts,
        )

        return snap

    @property
    def running(self) -> bool:
        return self._running

    @property
    def current_tick_ms(self) -> float:
        return self._current_tick_ms

    # -- Internal --

    async def _run_loop(self):
        """Main evolution loop — steps CfC cells at adaptive tick rate."""
        try:
            while self._running:
                tick_start = time.monotonic()

                try:
                    # Drain any pending percepts
                    event = self._drain_percepts()

                    # Step all cells with current context + percept data
                    async with self._lock:
                        self._step_cells(event)
                        self._ticks_since_cycle += 1

                    # Adapt tick rate based on prediction error
                    if self.config.adaptive:
                        self._adapt_tick_rate()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A step failure must NOT permanently freeze CfC/affect
                    # evolution. This loop had no inner guard, so a single raise
                    # in _step_cells stopped the experiential layer for good
                    # (Phase 3 review, Fable 5). Log loudly and continue.
                    logger.exception("Evolution step failed; continuing")

                # Sleep for remainder of tick (outside the guard so a failed
                # step still paces the loop instead of spinning hot)
                elapsed_ms = (time.monotonic() - tick_start) * 1000
                sleep_ms = max(0, self._current_tick_ms - elapsed_ms)
                await asyncio.sleep(sleep_ms / 1000)

        except asyncio.CancelledError:
            pass

    def _drain_percepts(self) -> PerceptEvent:
        """Drain all pending percepts, merge into one aggregate event."""
        aggregate = PerceptEvent()
        count = 0

        while not self._percept_queue.empty():
            try:
                event = self._percept_queue.get_nowait()
                aggregate.valence_delta += event.valence_delta
                aggregate.arousal_delta += event.arousal_delta
                aggregate.novelty = max(aggregate.novelty, event.novelty)
                aggregate.goal_relevance = max(aggregate.goal_relevance, event.goal_relevance)
                count += 1
            except asyncio.QueueEmpty:
                break

        self._percepts_processed += count
        return aggregate

    def _step_cells(self, event: PerceptEvent):
        """Step all CfC cells with current context."""
        self._manager.step(
            arousal=self._scaffold_vad[1],
            prediction_error=self._prediction_error,
            base_precision=0.5,
            scaffold_precision=self._scaffold_precision,
            percept_valence_delta=event.valence_delta,
            percept_arousal_delta=event.arousal_delta,
            llm_emotion_shift=0.0,
            scaffold_vad=self._scaffold_vad,
            goal_relevance=event.goal_relevance,
            novelty=event.novelty,
            emotional_salience=abs(event.valence_delta),
            recency=1.0 if (event.valence_delta != 0 or event.arousal_delta != 0) else 0.0,
            scaffold_salience=self._scaffold_salience,
            scaffold_goal_adj=self._scaffold_goal_adj,
        )

    def _adapt_tick_rate(self):
        """Adjust tick rate based on prediction error.

        High prediction error -> faster ticks (more processing needed).
        Low prediction error -> slower ticks (idle, save resources).
        """
        error = self._prediction_error
        sensitivity = self.config.error_sensitivity

        if error < 0.1:
            # Low error -> slow down toward idle
            target = self.config.idle_tick_ms
        else:
            # Scale between base and min based on error magnitude
            speed_factor = min(1.0, error * sensitivity * 2)
            target = self.config.base_tick_ms - speed_factor * (
                self.config.base_tick_ms - self.config.min_tick_ms
            )

        # Smooth the transition (exponential moving average)
        self._current_tick_ms = self._current_tick_ms * 0.7 + target * 0.3

        # Clamp
        self._current_tick_ms = max(
            self.config.min_tick_ms,
            min(self.config.max_tick_ms, self._current_tick_ms),
        )
