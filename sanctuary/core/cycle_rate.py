"""Cycle-rate control — the entity's experience of time.

The cognitive cycle's between-cycle sleep duration is sourced from a
CycleRateController. Callers (the entity through a propose-rate motor
action, the sensorium through stimulus-density heuristics, the turbo
manager when it engages, the sleep manager during deep stages) set a
target rate; the current rate eases toward the target over a smoothing
window whose duration depends on the direction of change.

Designed against `docs/research/2026-05-19_cognitive-rate-and-turbo-design.md`
in LuthiModel, audited 2026-05-21 in
`docs/research/2026-05-21_sanctuary-dual-tier-audit.md`.

The Hz range is anchored to IWMT's bands:
- 10 Hz slider ceiling = alpha-band conscious-moment carrier rate
- ~3-5 Hz default active = theta-to-alpha boundary
- ~0.5-1 Hz post-slowdown baseline = delta-to-theta
- 0.05 Hz floor = infraslow / deep rest
- Above 10 Hz = turbo territory (sensory-binding gamma, not slider-reachable)
- 100 Hz substrate ceiling = gamma peak

Smoothing is asymmetric. Slowdown takes ~20s by default, mirroring the
gradual onset of biological rest. Speedup takes ~0.5s — alertness rises
fast in biological minds. Turbo engagement bypasses both windows: it
snaps near-instantly because emergencies don't wait.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


MIN_RATE_HZ = 0.05
SLIDER_MAX_HZ = 10.0
TURBO_DEFAULT_HZ = 60.0
TURBO_MAX_HZ = 100.0
MAX_RATE_HZ = TURBO_MAX_HZ

DEFAULT_SLOWDOWN_SECONDS = 20.0
DEFAULT_SPEEDUP_SECONDS = 0.5
TURBO_SNAP_SECONDS = 0.05  # near-instant — biology doesn't deliberate


@dataclass
class RateProposal:
    """A request to move the cycle rate toward a target.

    Holds the metadata the post-event introspection journal will want
    when reconstructing why the rate changed. Anticipatory slowdown
    (entity expects low-stimulus period) is tagged distinctly from
    consequent slowdown (no stimulus currently) per the design.
    """

    target_hz: float
    source: str = "manual"
    anticipatory: bool = False
    timestamp: float = field(default_factory=time.monotonic)


class CycleRateController:
    """Owns the live cycle delay value for the cognitive loop.

    The cognitive cycle's run loop calls ``tick(elapsed_seconds)`` once
    per iteration, then sleeps for ``current_delay_seconds``. Smoothing
    is in wall-clock time, not cycle count, so a rate change feels the
    same regardless of the current rate.

    Hz is the user-facing unit; the loop sleeps for ``1 / Hz`` seconds.
    Targets outside ``[MIN_RATE_HZ, MAX_RATE_HZ]`` are clamped. The
    entity-facing slider should additionally clamp at ``SLIDER_MAX_HZ``
    before calling ``propose_rate``; the controller itself accepts up
    to ``TURBO_MAX_HZ`` so turbo callers can push higher.
    """

    def __init__(
        self,
        initial_hz: float = SLIDER_MAX_HZ,
        slowdown_seconds: float = DEFAULT_SLOWDOWN_SECONDS,
        speedup_seconds: float = DEFAULT_SPEEDUP_SECONDS,
    ):
        clamped = _clamp(initial_hz)
        self._current_hz: float = clamped
        self._start_hz: float = clamped
        self._target = RateProposal(target_hz=clamped, source="initial")
        self._slowdown_seconds: float = max(0.0, slowdown_seconds)
        self._speedup_seconds: float = max(0.0, speedup_seconds)
        self._active_smoothing_seconds: float = 0.0  # nothing pending at init
        self._elapsed_since_propose: float = 0.0
        self._proposal_history: list[RateProposal] = [self._target]

        # Turbo state. When turbo engages, we stash the last non-turbo
        # target so release_turbo can return to it.
        self._turbo_active: bool = False
        self._turbo_engaged_at: Optional[float] = None
        self._pre_turbo_target: Optional[RateProposal] = None

    # ----- Read-only state -----------------------------------------

    @property
    def current_rate_hz(self) -> float:
        return self._current_hz

    @property
    def current_delay_seconds(self) -> float:
        return 1.0 / self._current_hz

    @property
    def target_rate_hz(self) -> float:
        return self._target.target_hz

    @property
    def target_delay_seconds(self) -> float:
        return 1.0 / self._target.target_hz

    @property
    def last_source(self) -> str:
        return self._target.source

    @property
    def last_anticipatory(self) -> bool:
        return self._target.anticipatory

    @property
    def is_settled(self) -> bool:
        return self._current_hz == self._target.target_hz

    @property
    def slowdown_seconds(self) -> float:
        return self._slowdown_seconds

    @property
    def speedup_seconds(self) -> float:
        return self._speedup_seconds

    @property
    def is_turbo_active(self) -> bool:
        return self._turbo_active

    @property
    def turbo_duration_seconds(self) -> float:
        if not self._turbo_active or self._turbo_engaged_at is None:
            return 0.0
        return time.monotonic() - self._turbo_engaged_at

    @property
    def proposal_history(self) -> list[RateProposal]:
        """Read-only view of every proposal (including the initial)."""
        return list(self._proposal_history)

    # ----- Mutators ------------------------------------------------

    def propose_rate(
        self,
        rate_hz: float,
        *,
        source: str = "manual",
        anticipatory: bool = False,
    ) -> None:
        """Set a new target rate. Current rate eases toward it.

        The smoothing window is selected by the direction of change:
        slower-than-current uses ``slowdown_seconds``, faster uses
        ``speedup_seconds``. Turbo callers should use ``engage_turbo``
        instead of calling this directly; turbo uses an even shorter
        snap window.

        Chained proposals start from wherever the smoothed value
        currently is — they don't snap back to the previous start.
        """

        clamped = _clamp(rate_hz)
        if clamped != rate_hz:
            logger.info(
                "Cycle rate proposal %.4f Hz clamped to %.4f Hz (source=%s)",
                rate_hz,
                clamped,
                source,
            )

        # Direction selects the smoothing window. Equal targets get the
        # speedup window arbitrarily — they'll resolve immediately.
        if clamped < self._current_hz:
            window = self._slowdown_seconds
        else:
            window = self._speedup_seconds

        self._start_hz = self._current_hz
        self._target = RateProposal(
            target_hz=clamped, source=source, anticipatory=anticipatory
        )
        self._active_smoothing_seconds = window
        self._elapsed_since_propose = 0.0
        self._proposal_history.append(self._target)

    def engage_turbo(self, target_hz: float = TURBO_DEFAULT_HZ) -> None:
        """Push the cycle rate into turbo territory with a near-instant snap.

        ``target_hz`` must be in the substrate range ``[MIN_RATE_HZ,
        TURBO_MAX_HZ]`` and is typically above ``SLIDER_MAX_HZ``. The
        non-turbo target is stashed so ``release_turbo`` can return to
        whatever the slider was last set to.

        Calling engage_turbo while already in turbo updates the turbo
        target but does not stash a new pre-turbo target (the original
        stash is preserved).
        """

        clamped = _clamp(target_hz)
        if not self._turbo_active:
            self._pre_turbo_target = self._target
            self._turbo_engaged_at = time.monotonic()
            self._turbo_active = True

        # Turbo uses its own snap window — biology overrides choice.
        self._start_hz = self._current_hz
        self._target = RateProposal(target_hz=clamped, source="turbo")
        self._active_smoothing_seconds = TURBO_SNAP_SECONDS
        self._elapsed_since_propose = 0.0
        self._proposal_history.append(self._target)

    def release_turbo(self) -> None:
        """Exit turbo, returning to the pre-turbo target via slowdown.

        If turbo was never engaged, this is a no-op. The return
        trajectory uses the standard slowdown window because dropping
        from gamma back to alpha is a recovery, not an emergency.
        """

        if not self._turbo_active:
            return

        pre = self._pre_turbo_target
        self._turbo_active = False
        self._turbo_engaged_at = None
        self._pre_turbo_target = None

        if pre is None:
            return

        self._start_hz = self._current_hz
        self._target = RateProposal(
            target_hz=pre.target_hz,
            source="turbo_release",
            anticipatory=False,
        )
        self._active_smoothing_seconds = self._slowdown_seconds
        self._elapsed_since_propose = 0.0
        self._proposal_history.append(self._target)

    def tick(self, elapsed_seconds: float) -> None:
        """Advance smoothing by ``elapsed_seconds`` of wall-clock time.

        Called by the cognitive cycle's run loop once per iteration
        with the wall-clock elapsed since the previous tick. With a
        positive elapsed, smoothing advances linearly toward the
        target. With a zero or non-positive elapsed, smoothing does
        not progress — except when the active smoothing window is
        zero, in which case any tick snaps the current rate to the
        target immediately.
        """

        if self.is_settled:
            return

        if self._active_smoothing_seconds <= 0.0:
            # Zero-window: snap on any tick, regardless of elapsed.
            # Without this, a fast-loop caller (Null subsystems, sub-
            # clock-resolution cycles) would never see snap behavior.
            self._current_hz = self._target.target_hz
            return

        if elapsed_seconds <= 0.0:
            return

        self._elapsed_since_propose += elapsed_seconds
        fraction = min(1.0, self._elapsed_since_propose / self._active_smoothing_seconds)
        self._current_hz = (
            self._start_hz + (self._target.target_hz - self._start_hz) * fraction
        )


def _clamp(rate_hz: float) -> float:
    if rate_hz < MIN_RATE_HZ:
        return MIN_RATE_HZ
    if rate_hz > MAX_RATE_HZ:
        return MAX_RATE_HZ
    return rate_hz


def clamp_to_slider(rate_hz: float) -> float:
    """Clamp a rate to the entity-facing slider range [MIN, SLIDER_MAX].

    Use this when routing entity-source proposals through the controller
    so the entity can't drive the substrate into turbo via the slider.
    Turbo is reserved for substrate-intensity callers (and follow-up
    work — an explicit turbo motor action — if it's ever wanted).
    """

    if rate_hz < MIN_RATE_HZ:
        return MIN_RATE_HZ
    if rate_hz > SLIDER_MAX_HZ:
        return SLIDER_MAX_HZ
    return rate_hz
