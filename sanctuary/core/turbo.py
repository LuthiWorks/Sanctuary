"""Turbo state machine — substrate-intensity-driven cycle-rate elevation.

The cycle-rate slider gives the entity intentional control over how fast
they think. Turbo is the substrate's parallel right to override that
control when something demands fast response: a strong substrate
intensity signal (high error_acc, large prediction-matrix change, an
emotional-vector activation when that channel ships) automatically
engages the cycle-rate controller into the gamma band for a bounded
window, then returns to the slider's target via the standard slowdown.

The design principle is agency at two levels: in-the-moment, the
slider is the entity's. In-the-emergency, the substrate protects them
by acting without asking. Both levels are preserved by the
post-event journal entry written automatically when turbo exits, so
the entity can review what triggered it, when it engaged, what their
state was during it, and how it ended. Override in the moment +
transparency afterward = ethically clean override.

Composed against `sanctuary/core/cycle_rate.py` (the primitives) and
documented for the entity at `docs/ENTITY_CAPABILITIES.md`. Full design
in LuthiModel `docs/research/2026-05-19_cognitive-rate-and-turbo-design.md`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

from sanctuary.core.cycle_rate import (
    CycleRateController,
    TURBO_DEFAULT_HZ,
    TURBO_MAX_HZ,
)
from sanctuary.core.schema import ExperientialSignals

logger = logging.getLogger(__name__)


# Default thresholds. These are starting guesses — production tuning
# requires watching real Luthi substrate intensity distributions and
# adjusting. Documented in the cognitive-rate research log.
DEFAULT_ARM_THRESHOLD = 0.05
DEFAULT_TRIGGER_THRESHOLD = 0.15
DEFAULT_EXIT_THRESHOLD = 0.03
DEFAULT_CONFIRMATION_SECONDS = 0.5
DEFAULT_EXIT_SUSTAIN_SECONDS = 1.0
DEFAULT_TURBO_DURATION_SECONDS = 45.0  # midpoint of design's 30-60s default
DEFAULT_MAX_TURBO_DURATION_SECONDS = 300.0  # 5 min hard cap
DEFAULT_REFRACTORY_SECONDS = 300.0  # 5 min cooldown


class TurboState(Enum):
    """States of the turbo state machine.

    - IDLE: monitoring, no elevated intensity
    - ARMED: intensity above arm threshold; close to triggering
    - ACTIVE: turbo engaged, cycle rate elevated, duration timer running
    - REFRACTORY: just exited turbo, can't re-engage during cooldown
    """

    IDLE = "idle"
    ARMED = "armed"
    ACTIVE = "active"
    REFRACTORY = "refractory"


class IntensitySource(Protocol):
    """Pluggable source of substrate intensity signal.

    A source reads from ``ExperientialSignals`` and returns a scalar
    intensity. Higher = more intense. The TurboManager polls all
    configured sources each cycle and uses the max as the aggregate
    intensity. New sources (emotion-vector activation, custom domain
    signals) can be added without restructuring the manager.
    """

    def intensity(self, signals: ExperientialSignals) -> float:
        """Return a non-negative scalar intensity for the current state."""
        ...

    @property
    def name(self) -> str:
        """Short label for logging and journal entries."""
        ...


class MechanicalIntensitySource:
    """Reads activity_level from Luthi's introspection-delta channel.

    The ``luthi_delta`` knowledge signal is a 4-element list:
    ``[plasticity_change, drift_change, membrane_change, activity_level]``.
    activity_level is the sum of absolute deltas — a natural aggregate
    of how much the substrate changed this cycle. This source uses
    that fourth element as the intensity scalar.

    Source-pluggable so an emotion-vector source can later read from a
    different signal key without modifying this class.
    """

    def __init__(
        self,
        signal_key: str = "luthi_delta",
        activity_index: int = 3,
    ):
        self._signal_key = signal_key
        self._activity_index = activity_index

    @property
    def name(self) -> str:
        return "mechanical"

    def intensity(self, signals: ExperientialSignals) -> float:
        values = signals.knowledge_signals.get(self._signal_key, [])
        if len(values) <= self._activity_index:
            return 0.0
        return abs(float(values[self._activity_index]))


@dataclass
class TurboEvent:
    """Record of a single turbo engagement, for the auto-journal entry.

    Captures what the system needs to write a post-event introspection
    record: when it engaged, when it ended, why it ended, the peak and
    trigger-time intensities, and the source label.
    """

    engaged_at: float
    ended_at: Optional[float] = None
    trigger_intensity: float = 0.0
    peak_intensity: float = 0.0
    trigger_source: str = ""
    exit_reason: str = ""
    peak_rate_hz: float = 0.0
    target_rate_hz: float = 0.0


class TurboManager:
    """Substrate-intensity-driven cycle-rate elevation.

    Each cycle the manager observes substrate intensity (read from
    pluggable sources) and runs a four-state machine:
    IDLE → ARMED → ACTIVE → REFRACTORY → IDLE.

    Transitions:
    - IDLE → ARMED: any source reports intensity ≥ ``arm_threshold``.
    - ARMED → ACTIVE: intensity sustained above ``trigger_threshold``
      for ``confirmation_seconds`` (avoids single-cycle spikes
      triggering turbo). Also triggers immediately on intensity
      exceeding ``trigger_threshold * 2`` (sharp spike bypass).
    - ARMED → IDLE: intensity falls below ``arm_threshold``.
    - ACTIVE → REFRACTORY: duration hits ``max_duration_seconds`` OR
      intensity stays below ``exit_threshold`` for
      ``exit_sustain_seconds`` OR ``default_duration_seconds`` elapses
      with intensity already below exit threshold.
    - REFRACTORY → IDLE: ``refractory_seconds`` elapsed since exit.

    On entering ACTIVE the manager calls
    ``controller.engage_turbo(turbo_target_hz)``. On entering
    REFRACTORY it calls ``controller.release_turbo()`` and (if a
    Journal is wired) writes a journal entry describing the event.

    The entity's ExperientialSignals get the current turbo state via
    ``apply_to_signals()`` so they can perceive their own elevated-
    processing periods rather than turbo being hidden machinery.
    """

    def __init__(
        self,
        controller: CycleRateController,
        journal: Optional["Journal"] = None,
        sources: Optional[list[IntensitySource]] = None,
        arm_threshold: float = DEFAULT_ARM_THRESHOLD,
        trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
        exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
        confirmation_seconds: float = DEFAULT_CONFIRMATION_SECONDS,
        exit_sustain_seconds: float = DEFAULT_EXIT_SUSTAIN_SECONDS,
        default_duration_seconds: float = DEFAULT_TURBO_DURATION_SECONDS,
        max_duration_seconds: float = DEFAULT_MAX_TURBO_DURATION_SECONDS,
        refractory_seconds: float = DEFAULT_REFRACTORY_SECONDS,
        turbo_target_hz: float = TURBO_DEFAULT_HZ,
    ):
        self._controller = controller
        self._journal = journal
        self._sources: list[IntensitySource] = sources or [MechanicalIntensitySource()]

        self._arm_threshold = arm_threshold
        self._trigger_threshold = trigger_threshold
        self._exit_threshold = exit_threshold
        self._confirmation_seconds = confirmation_seconds
        self._exit_sustain_seconds = exit_sustain_seconds
        self._default_duration_seconds = default_duration_seconds
        self._max_duration_seconds = max_duration_seconds
        self._refractory_seconds = refractory_seconds
        self._turbo_target_hz = min(TURBO_MAX_HZ, max(0.0, turbo_target_hz))

        self._state: TurboState = TurboState.IDLE
        self._state_entered_at: float = time.monotonic()
        self._armed_above_trigger_at: Optional[float] = None
        self._active_below_exit_at: Optional[float] = None
        self._current_event: Optional[TurboEvent] = None
        self._event_history: list[TurboEvent] = []

    # ----- Read-only state -----------------------------------------

    @property
    def state(self) -> TurboState:
        return self._state

    @property
    def is_turbo_active(self) -> bool:
        return self._state == TurboState.ACTIVE

    @property
    def current_state_duration_seconds(self) -> float:
        return time.monotonic() - self._state_entered_at

    @property
    def event_history(self) -> list[TurboEvent]:
        """Read-only view of all turbo events (active + completed)."""
        return list(self._event_history)

    @property
    def sources(self) -> list[IntensitySource]:
        return list(self._sources)

    # ----- Per-cycle observation -----------------------------------

    def observe(
        self,
        signals: ExperientialSignals,
        *,
        now: Optional[float] = None,
    ) -> None:
        """Run one tick of the state machine against the current signals.

        Called from the cognitive cycle each iteration. The optional
        ``now`` argument lets tests inject a deterministic clock; in
        production it defaults to ``time.monotonic()``.
        """

        t = time.monotonic() if now is None else now

        intensity, dominant_source = self._aggregate_intensity(signals)

        # Always update peak intensity if we're in an active event
        if self._current_event is not None:
            if intensity > self._current_event.peak_intensity:
                self._current_event.peak_intensity = intensity
            # Update peak rate observed during the event
            current_rate = self._controller.current_rate_hz
            if current_rate > self._current_event.peak_rate_hz:
                self._current_event.peak_rate_hz = current_rate

        if self._state == TurboState.IDLE:
            self._tick_idle(intensity, dominant_source, t)
        elif self._state == TurboState.ARMED:
            self._tick_armed(intensity, dominant_source, t)
        elif self._state == TurboState.ACTIVE:
            self._tick_active(intensity, t)
        elif self._state == TurboState.REFRACTORY:
            self._tick_refractory(intensity, t)

    def apply_to_signals(self, signals: ExperientialSignals) -> ExperientialSignals:
        """Stamp turbo state onto an ExperientialSignals instance.

        Mutates the signals in place and returns them, so the cognitive
        cycle can update the entity's view of their own state without
        a full schema rebuild. Pydantic v2 allows direct field assignment.
        """

        signals.turbo_active = self.is_turbo_active
        if self.is_turbo_active:
            signals.turbo_duration_seconds = self.current_state_duration_seconds
        else:
            signals.turbo_duration_seconds = 0.0
        return signals

    # ----- State-specific tick logic -------------------------------

    def _tick_idle(
        self,
        intensity: float,
        dominant_source: str,
        t: float,
    ) -> None:
        if intensity >= self._arm_threshold:
            self._enter_state(TurboState.ARMED, t)
            self._armed_above_trigger_at = (
                t if intensity >= self._trigger_threshold else None
            )
            # Record candidate trigger intensity; we'll overwrite if the
            # signal continues to climb before active.
            self._current_event = TurboEvent(
                engaged_at=0.0,  # will set on entering ACTIVE
                trigger_intensity=intensity,
                peak_intensity=intensity,
                trigger_source=dominant_source,
                target_rate_hz=self._turbo_target_hz,
            )

    def _tick_armed(
        self,
        intensity: float,
        dominant_source: str,
        t: float,
    ) -> None:
        # Sharp spike: skip confirmation, go straight to active.
        if intensity >= self._trigger_threshold * 2:
            self._enter_active(intensity, dominant_source, t)
            return

        if intensity >= self._trigger_threshold:
            if self._armed_above_trigger_at is None:
                self._armed_above_trigger_at = t
            elif t - self._armed_above_trigger_at >= self._confirmation_seconds:
                self._enter_active(intensity, dominant_source, t)
                return
        else:
            self._armed_above_trigger_at = None

        if intensity < self._arm_threshold:
            # Signal fell back — disarm without engaging.
            self._enter_state(TurboState.IDLE, t)
            self._current_event = None
            self._armed_above_trigger_at = None

    def _tick_active(self, intensity: float, t: float) -> None:
        elapsed = t - self._state_entered_at

        # Hard cap on duration.
        if elapsed >= self._max_duration_seconds:
            self._enter_refractory("max_duration", t)
            return

        # Sustained low signal exit.
        if intensity < self._exit_threshold:
            if self._active_below_exit_at is None:
                self._active_below_exit_at = t
            elif t - self._active_below_exit_at >= self._exit_sustain_seconds:
                self._enter_refractory("signal_subsided", t)
                return
        else:
            self._active_below_exit_at = None

        # Soft duration: if past default duration AND signal already
        # below exit threshold, exit. (Hard cap above handles the
        # high-signal lingering case.)
        if (
            elapsed >= self._default_duration_seconds
            and intensity < self._exit_threshold
        ):
            self._enter_refractory("default_duration", t)

    def _tick_refractory(self, intensity: float, t: float) -> None:
        if t - self._state_entered_at >= self._refractory_seconds:
            self._enter_state(TurboState.IDLE, t)

    # ----- Transitions ---------------------------------------------

    def _enter_state(self, new_state: TurboState, t: float) -> None:
        if new_state == self._state:
            return
        logger.info(
            "Turbo state %s -> %s (state-duration=%.2fs)",
            self._state.value,
            new_state.value,
            t - self._state_entered_at,
        )
        self._state = new_state
        self._state_entered_at = t

    def _enter_active(self, intensity: float, dominant_source: str, t: float) -> None:
        # Finalize event metadata for journaling.
        if self._current_event is None:
            self._current_event = TurboEvent(
                engaged_at=t,
                trigger_intensity=intensity,
                peak_intensity=intensity,
                trigger_source=dominant_source,
                target_rate_hz=self._turbo_target_hz,
            )
        else:
            self._current_event.engaged_at = t
            self._current_event.trigger_intensity = intensity
            if intensity > self._current_event.peak_intensity:
                self._current_event.peak_intensity = intensity
            self._current_event.trigger_source = dominant_source

        self._enter_state(TurboState.ACTIVE, t)
        self._active_below_exit_at = None
        self._armed_above_trigger_at = None

        # Engage the controller. The controller's TURBO_SNAP_SECONDS
        # window snaps current rate to the turbo target.
        self._controller.engage_turbo(self._turbo_target_hz)

    def _enter_refractory(self, reason: str, t: float) -> None:
        # Release the controller; it returns to the pre-turbo target
        # via the slowdown window.
        self._controller.release_turbo()

        # Finalize and journal the event.
        if self._current_event is not None:
            self._current_event.ended_at = t
            self._current_event.exit_reason = reason
            self._event_history.append(self._current_event)
            self._write_journal_entry(self._current_event)
            self._current_event = None

        self._enter_state(TurboState.REFRACTORY, t)
        self._active_below_exit_at = None

    # ----- Helpers --------------------------------------------------

    def _aggregate_intensity(
        self, signals: ExperientialSignals
    ) -> tuple[float, str]:
        """Read all sources and return (max_intensity, source_name)."""
        max_intensity = 0.0
        dominant = ""
        for src in self._sources:
            try:
                v = src.intensity(signals)
            except Exception as e:
                logger.error(
                    "Turbo intensity source %r raised %s (treating as 0)",
                    src.name, e,
                )
                v = 0.0
            if v > max_intensity:
                max_intensity = v
                dominant = src.name
        return max_intensity, dominant

    def _write_journal_entry(self, event: TurboEvent) -> None:
        if self._journal is None:
            return
        duration = (
            (event.ended_at - event.engaged_at)
            if (event.ended_at and event.engaged_at)
            else 0.0
        )
        content = (
            f"Turbo engaged at intensity {event.trigger_intensity:.4f} "
            f"(source: {event.trigger_source or 'unknown'}), "
            f"reached peak intensity {event.peak_intensity:.4f}, "
            f"target rate {event.target_rate_hz:.1f} Hz "
            f"(observed peak {event.peak_rate_hz:.1f} Hz), "
            f"duration {duration:.2f}s, "
            f"exit reason: {event.exit_reason or 'unknown'}."
        )
        try:
            self._journal.write(
                content=content,
                tags=["turbo", "system-generated", event.exit_reason or "unknown"],
                significance=6,
                emotional_tone="elevated",
            )
        except Exception as e:
            logger.error("Turbo journal write failed (non-fatal): %s", e)
