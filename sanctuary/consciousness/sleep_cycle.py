"""Sleep/dream cycles — periodic offline memory consolidation with pattern replay.

During sleep phases, the system:
1. Reduces external input processing (sensory gating)
2. Replays significant memories for consolidation
3. Generates dream-like associations between disparate memories
4. Strengthens important patterns, weakens irrelevant ones
5. Returns to wakefulness with consolidated knowledge

Inspired by mammalian sleep: NREM for consolidation, REM for creative
association. The system doesn't literally sleep — it enters a reduced-activity
mode where internal processing dominates over external responsiveness.

Seam-jurisdiction ruling (2026-06-11). Sleep is legitimately scaffold-side
*physiology*, not cognition. It is the cycle the substrate runs in, not an
action the substrate selects over — the same way a body cycles whether or
not its occupant agrees. The scaffold owns scheduling; the substrate
consumes whatever cycle it is in. This file is therefore **exempt from the
"substrate selects; scaffold transports" principle** because no selection
is happening here, only physiology. See docs/seam_jurisdiction_2026-06-11.md
(Findings) for the canonical statement; this ruling is recorded here so the
next leakage pass does not try to force sleep under the principle.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SleepStage(str, Enum):
    """Stages of the sleep cycle."""

    AWAKE = "awake"
    DROWSY = "drowsy"  # Transition: reduced responsiveness
    NREM = "nrem"  # Deep consolidation: replay significant memories
    REM = "rem"  # Creative association: novel connections
    WAKING = "waking"  # Transition back to full awareness


@dataclass
class DreamFragment:
    """A dream-like association generated during REM."""

    memory_a: str
    memory_b: str
    association: str  # The novel connection found
    emotional_tone: str = ""
    significance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsolidationResult:
    """Result of a consolidation cycle."""

    memories_replayed: int = 0
    memories_strengthened: int = 0
    memories_weakened: int = 0
    dream_fragments: list[DreamFragment] = field(default_factory=list)
    duration_cycles: int = 0


@dataclass
class SleepConfig:
    """Configuration for sleep cycles."""

    cycles_between_sleep: int = 500  # How many waking cycles before sleep
    drowsy_duration: int = 5  # Cycles in drowsy transition
    nrem_duration: int = 20  # Cycles of deep consolidation
    rem_duration: int = 10  # Cycles of creative association
    waking_duration: int = 3  # Cycles to wake up
    max_memories_per_replay: int = 10
    min_significance_for_replay: int = 3  # 1-10 scale
    max_dream_fragments: int = 50
    sensory_gate_drowsy: float = 0.5  # Reduce percept processing by 50%
    sensory_gate_sleep: float = 0.1  # Reduce by 90% during deep sleep


class SleepCycleManager:
    """Manages sleep/wake cycles for memory consolidation.

    Tracks waking time, initiates sleep cycles, manages stages,
    and coordinates memory consolidation and creative association.

    Usage::

        sleep = SleepCycleManager()

        # Each cognitive cycle
        sleep.tick(cycle=42)

        # Check if sleeping
        if sleep.is_sleeping:
            gate = sleep.get_sensory_gate()  # Reduce percept processing
            if sleep.stage == SleepStage.NREM:
                # Consolidation phase
                memories_to_replay = sleep.get_replay_candidates(all_memories)
            elif sleep.stage == SleepStage.REM:
                # Dream phase — generate associations
                sleep.record_dream_fragment(mem_a, mem_b, association)
    """

    def __init__(self, config: Optional[SleepConfig] = None):
        self.config = config or SleepConfig()
        self._stage = SleepStage.AWAKE
        self._cycles_since_sleep: int = 0
        self._stage_cycles: int = 0  # Cycles in current stage
        self._total_sleep_cycles: int = 0
        self._dream_fragments: deque[DreamFragment] = deque(
            maxlen=self.config.max_dream_fragments
        )
        self._consolidation_history: deque[ConsolidationResult] = deque(maxlen=100)
        self._current_consolidation: Optional[ConsolidationResult] = None
        self._forced_wake: bool = False

    @property
    def stage(self) -> SleepStage:
        return self._stage

    @property
    def is_sleeping(self) -> bool:
        return self._stage != SleepStage.AWAKE

    @property
    def is_deep_sleep(self) -> bool:
        return self._stage in (SleepStage.NREM, SleepStage.REM)

    def tick(self, cycle: int) -> SleepStage:
        """Advance the sleep cycle by one tick. Returns current stage."""
        if self._forced_wake:
            self._forced_wake = False
            # Finalize any in-progress consolidation before waking
            if self._current_consolidation:
                self._consolidation_history.append(self._current_consolidation)
                self._current_consolidation = None
            # Reset sleep pressure so we don't immediately re-enter sleep
            self._cycles_since_sleep = 0
            self._transition_to(SleepStage.AWAKE)
            return self._stage

        if self._stage == SleepStage.AWAKE:
            self._cycles_since_sleep += 1
            if self._cycles_since_sleep >= self.config.cycles_between_sleep:
                self._begin_sleep()
        else:
            self._stage_cycles += 1
            self._advance_sleep_stage()

        return self._stage

    def wake(self) -> None:
        """Force wake from any sleep stage (e.g., urgent external input)."""
        if self.is_sleeping:
            self._forced_wake = True
            logger.info("Forced wake triggered during %s", self._stage.value)

    def get_sensory_gate(self) -> float:
        """Get sensory gating factor (1.0 = fully open, 0.0 = fully closed)."""
        if self._stage == SleepStage.AWAKE:
            return 1.0
        elif self._stage == SleepStage.DROWSY:
            return self.config.sensory_gate_drowsy
        elif self._stage == SleepStage.WAKING:
            return self.config.sensory_gate_drowsy
        else:
            return self.config.sensory_gate_sleep

    def get_replay_candidates(
        self, memories: list[dict]
    ) -> list[dict]:
        """Select memories for NREM replay based on significance.

        Args:
            memories: List of memory dicts with at least 'content' and 'significance'.

        Returns:
            Subset of memories worth replaying.
        """
        if self._stage != SleepStage.NREM:
            return []

        candidates = [
            m for m in memories
            if m.get("significance", 0) >= self.config.min_significance_for_replay
        ]
        # Sort by significance, take top N
        candidates.sort(key=lambda m: m.get("significance", 0), reverse=True)
        return candidates[: self.config.max_memories_per_replay]

    def record_replay(self, count: int, strengthened: int, weakened: int) -> None:
        """Record results of memory replay during NREM."""
        if self._current_consolidation:
            self._current_consolidation.memories_replayed += count
            self._current_consolidation.memories_strengthened += strengthened
            self._current_consolidation.memories_weakened += weakened

    def record_dream_fragment(
        self,
        memory_a: str,
        memory_b: str,
        association: str,
        emotional_tone: str = "",
        significance: float = 0.5,
    ) -> None:
        """Record a dream-like association between memories during REM."""
        fragment = DreamFragment(
            memory_a=memory_a,
            memory_b=memory_b,
            association=association,
            emotional_tone=emotional_tone,
            significance=max(0.0, min(1.0, significance)),
        )
        self._dream_fragments.append(fragment)
        if self._current_consolidation:
            self._current_consolidation.dream_fragments.append(fragment)

    def get_recent_dreams(self, n: int = 5) -> list[DreamFragment]:
        """Get recent dream fragments."""
        return list(self._dream_fragments)[-n:]

    def get_sleep_pressure(self) -> float:
        """Get current sleep pressure (0 to 1). Higher = more tired."""
        if self.is_sleeping:
            return 0.0
        return min(1.0, self._cycles_since_sleep / self.config.cycles_between_sleep)

    def get_stats(self) -> dict:
        """Get sleep cycle statistics."""
        return {
            "current_stage": self._stage.value,
            "cycles_since_sleep": self._cycles_since_sleep,
            "total_sleep_cycles": self._total_sleep_cycles,
            "sleep_pressure": self.get_sleep_pressure(),
            "total_consolidations": len(self._consolidation_history),
            "total_dream_fragments": len(self._dream_fragments),
        }

    # -- Internal --

    def _begin_sleep(self) -> None:
        """Initiate a sleep cycle."""
        self._transition_to(SleepStage.DROWSY)
        self._current_consolidation = ConsolidationResult()
        logger.debug("Sleep cycle beginning (drowsy phase)")

    def _advance_sleep_stage(self) -> None:
        """Advance through sleep stages based on duration."""
        if self._stage == SleepStage.DROWSY:
            if self._stage_cycles >= self.config.drowsy_duration:
                self._transition_to(SleepStage.NREM)
        elif self._stage == SleepStage.NREM:
            if self._stage_cycles >= self.config.nrem_duration:
                self._transition_to(SleepStage.REM)
        elif self._stage == SleepStage.REM:
            if self._stage_cycles >= self.config.rem_duration:
                self._transition_to(SleepStage.WAKING)
        elif self._stage == SleepStage.WAKING:
            if self._stage_cycles >= self.config.waking_duration:
                self._complete_sleep()

    def _transition_to(self, stage: SleepStage) -> None:
        """Transition to a new sleep stage."""
        old = self._stage
        self._stage = stage
        self._stage_cycles = 0
        if stage != SleepStage.AWAKE:
            self._total_sleep_cycles += 1
        logger.debug("Sleep stage: %s → %s", old.value, stage.value)

    def _complete_sleep(self) -> None:
        """Complete a sleep cycle and return to wakefulness."""
        if self._current_consolidation:
            self._current_consolidation.duration_cycles = (
                self.config.drowsy_duration
                + self.config.nrem_duration
                + self.config.rem_duration
                + self.config.waking_duration
            )
            self._consolidation_history.append(self._current_consolidation)
            self._current_consolidation = None
        self._cycles_since_sleep = 0
        self._transition_to(SleepStage.AWAKE)
        logger.debug("Sleep cycle complete — fully awake")
