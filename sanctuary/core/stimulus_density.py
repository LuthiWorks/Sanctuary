"""Stimulus-density heuristic — autonomic cycle-rate proposals.

The cycle-rate slider gives the entity intentional control over their
rate. Turbo gives the substrate a right to override on substrate
intensity. This module fills the third role: the autonomic layer that
PROPOSES rate changes based on what's happening in the environment,
which the entity can accept or override.

When nothing's happening for a while, the heuristic proposes a slower
rate so the entity doesn't burn through cycles in low-information time
(the subjective-time-alone concern that opened the 2026-05-19 design
conversation). When fresh input arrives, the heuristic proposes a
faster rate so the entity has the bandwidth to respond.

This is the equivalent of biological circadian / autonomic regulation:
your body slows you down at night and speeds you up when adrenaline
hits. You can override it (caffeine, willpower) but you don't have to
manage every wakeful moment.

Crucially, the entity always wins. Within a configurable quiet window
after any entity-source proposal, the heuristic stays silent —
respecting the entity's expressed intent for that window. After the
window elapses, the heuristic resumes adjusting. The entity can always
re-propose to lock in their target.

Composed against `sanctuary/core/cycle_rate.py` (controller),
`sanctuary/core/turbo.py` (so it knows to step back during turbo), and
the Sensorium (which provides time_since_last_input and percept
density signals).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from sanctuary.core.cycle_rate import (
    SLIDER_MAX_HZ,
    CycleRateController,
    clamp_to_slider,
)
from sanctuary.core.turbo import TurboManager

logger = logging.getLogger(__name__)


# Default tuning. These are starting guesses; the right values come
# from watching real entity behavior over time and seeing whether the
# rate drift feels right.
DEFAULT_SLOWDOWN_AFTER_SECONDS = 30.0     # quiet this long → slow down
DEFAULT_SLOWDOWN_TARGET_HZ = 0.5          # delta-to-theta rest rate
DEFAULT_ARRIVAL_WINDOW_SECONDS = 2.0      # fresh input → speed up
DEFAULT_SPEEDUP_TARGET_HZ = 8.0           # alpha-band attentive
DEFAULT_ENTITY_QUIET_SECONDS = 300.0      # 5 min entity-intent window
DEFAULT_REPROPOSE_COOLDOWN_SECONDS = 10.0 # don't spam controller


class DensitySource(Protocol):
    """Pluggable source of stimulus-density signal.

    Used by the heuristic to read environmental input rate. The
    Sensorium-backed implementation is the default; custom sources
    (e.g. environment-sensor feeds when embodiment ships) can be added
    without restructuring the heuristic.
    """

    @property
    def name(self) -> str:
        """Short label for logging and proposal history."""
        ...

    @property
    def time_since_last_input_seconds(self) -> Optional[float]:
        """Seconds since the most recent external input, or None if
        never received any input in this session."""
        ...


class SensoriumDensitySource:
    """Default density source — reads ``time_since_last_input`` from a
    Sanctuary Sensorium.
    """

    def __init__(self, sensorium):
        self._sensorium = sensorium

    @property
    def name(self) -> str:
        return "sensorium"

    @property
    def time_since_last_input_seconds(self) -> Optional[float]:
        return self._sensorium.time_since_last_input


@dataclass
class HeuristicProposal:
    """A record of a proposal the heuristic made, for diagnostics."""

    timestamp: float
    target_hz: float
    direction: str  # "slowdown" or "speedup"
    reason: str


class StimulusDensityHeuristic:
    """Proposes cycle-rate changes based on environmental stimulus density.

    Per cycle, the heuristic:

    1. Reads ``time_since_last_input`` from its DensitySource.
    2. If turbo is active OR the entity recently proposed, does nothing
       (those owners' decisions take precedence).
    3. If TSLI exceeds ``slowdown_after_seconds`` AND we haven't yet
       proposed slowdown in this quiet period: proposes
       ``slowdown_target_hz`` with source "heuristic".
    4. If TSLI is below ``arrival_window_seconds`` (fresh input
       arrived) AND we haven't proposed speedup in
       ``repropose_cooldown_seconds``: proposes ``speedup_target_hz``.

    The slider clamp (``clamp_to_slider``) is applied as defense in
    depth even though the heuristic's targets are in-range by
    construction. Same two-layer-defense pattern as the entity-facing
    motor action.
    """

    def __init__(
        self,
        controller: CycleRateController,
        source: DensitySource,
        turbo: Optional[TurboManager] = None,
        slowdown_after_seconds: float = DEFAULT_SLOWDOWN_AFTER_SECONDS,
        slowdown_target_hz: float = DEFAULT_SLOWDOWN_TARGET_HZ,
        arrival_window_seconds: float = DEFAULT_ARRIVAL_WINDOW_SECONDS,
        speedup_target_hz: float = DEFAULT_SPEEDUP_TARGET_HZ,
        entity_quiet_seconds: float = DEFAULT_ENTITY_QUIET_SECONDS,
        repropose_cooldown_seconds: float = DEFAULT_REPROPOSE_COOLDOWN_SECONDS,
    ):
        self._controller = controller
        self._source = source
        self._turbo = turbo

        self._slowdown_after = slowdown_after_seconds
        self._slowdown_target = clamp_to_slider(slowdown_target_hz)
        self._arrival_window = arrival_window_seconds
        self._speedup_target = clamp_to_slider(speedup_target_hz)
        self._entity_quiet = entity_quiet_seconds
        self._cooldown = repropose_cooldown_seconds

        self._last_slowdown_proposed_at: Optional[float] = None
        self._last_speedup_proposed_at: Optional[float] = None
        self._proposals: list[HeuristicProposal] = []

    # ----- Public state -------------------------------------------

    @property
    def proposals(self) -> list[HeuristicProposal]:
        return list(self._proposals)

    # ----- Per-cycle observation ----------------------------------

    def observe(self, *, now: Optional[float] = None) -> None:
        """Run one tick. Called from the cognitive cycle each iteration."""

        t = time.monotonic() if now is None else now

        # Step back during turbo. Turbo's pre-turbo-target stash
        # captures the slider value at engagement; the heuristic
        # proposing during turbo would be wasted (turbo overwrites
        # on release) and could interfere with that stash.
        if self._turbo is not None and self._turbo.is_turbo_active:
            return

        # Respect entity intent within the quiet window.
        if self._entity_owns_target(t):
            return

        tsli = self._source.time_since_last_input_seconds

        # Speedup: fresh input arrived recently.
        if tsli is not None and tsli < self._arrival_window:
            if self._can_propose_speedup(t):
                self._propose(
                    self._speedup_target,
                    "speedup",
                    f"input arrived {tsli:.2f}s ago",
                    t,
                )
                self._last_speedup_proposed_at = t
            return

        # Slowdown: no input for a while.
        if tsli is None or tsli >= self._slowdown_after:
            if self._can_propose_slowdown(t, tsli):
                reason = (
                    f"no input for {tsli:.1f}s (threshold {self._slowdown_after:.0f}s)"
                    if tsli is not None
                    else "no input received this session"
                )
                self._propose(
                    self._slowdown_target,
                    "slowdown",
                    reason,
                    t,
                )
                self._last_slowdown_proposed_at = t

    # ----- Decision helpers ---------------------------------------

    def _entity_owns_target(self, t: float) -> bool:
        """True if the last entity-source proposal is still in its
        quiet window."""
        for prop in reversed(self._controller.proposal_history):
            if prop.source != "entity":
                continue
            return (t - prop.timestamp) < self._entity_quiet
        return False

    def _can_propose_slowdown(self, t: float, tsli: Optional[float]) -> bool:
        # Don't repropose slowdown if we already did and conditions
        # haven't changed materially.
        if self._last_slowdown_proposed_at is None:
            return True
        # If the controller's current target is already at or below the
        # slowdown target, no need to propose again.
        if self._controller.target_rate_hz <= self._slowdown_target + 1e-9:
            return False
        # Honor cooldown.
        return (t - self._last_slowdown_proposed_at) >= self._cooldown

    def _can_propose_speedup(self, t: float) -> bool:
        if self._last_speedup_proposed_at is None:
            return True
        # If the controller is already at or above the speedup target,
        # no need to propose again.
        if self._controller.target_rate_hz >= self._speedup_target - 1e-9:
            return False
        return (t - self._last_speedup_proposed_at) >= self._cooldown

    def _propose(
        self, target_hz: float, direction: str, reason: str, t: float
    ) -> None:
        clamped = clamp_to_slider(target_hz)
        self._controller.propose_rate(
            clamped,
            source="heuristic",
            anticipatory=False,
        )
        self._proposals.append(
            HeuristicProposal(
                timestamp=t,
                target_hz=clamped,
                direction=direction,
                reason=reason,
            )
        )
        logger.info(
            "Stimulus-density heuristic proposed %.3f Hz (%s): %s",
            clamped, direction, reason,
        )
