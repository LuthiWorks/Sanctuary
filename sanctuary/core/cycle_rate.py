"""Cycle-rate control — the entity's experience of time.

The cognitive cycle's between-cycle sleep duration is sourced from a
CycleRateController. Callers (the entity through a propose-rate motor
action, the sensorium through stimulus-density heuristics, the turbo
manager when it engages, the sleep manager during deep stages) set a
target rate; the current rate eases linearly toward the target over a
configurable smoothing window, so changes are felt as gradual
transitions rather than instantaneous switches.

Designed against `docs/research/2026-05-19_cognitive-rate-and-turbo-design.md`
in LuthiModel, audited 2026-05-21 in
`docs/research/2026-05-21_sanctuary-dual-tier-audit.md`.

The Hz range is anchored to IWMT's bands:
- 10 Hz ceiling = alpha-band conscious-moment carrier rate
- ~3-5 Hz default active = theta-to-alpha boundary
- ~0.5-1 Hz post-slowdown baseline = delta-to-theta
- 0.05 Hz floor = infraslow / deep rest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


MIN_RATE_HZ = 0.05
MAX_RATE_HZ = 10.0


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


class CycleRateController:
    """Owns the live cycle delay value for the cognitive loop.

    The cognitive cycle's run loop calls ``tick(elapsed_seconds)`` once
    per iteration, then sleeps for ``current_delay_seconds``. Smoothing
    is in wall-clock time, not cycle count, so a rate change feels the
    same regardless of the current rate.

    Hz is the user-facing unit; the loop sleeps for ``1 / Hz`` seconds.
    Targets outside ``[MIN_RATE_HZ, MAX_RATE_HZ]`` are clamped at
    propose time.
    """

    def __init__(
        self,
        initial_hz: float = 10.0,
        smoothing_seconds: float = 30.0,
    ):
        clamped = _clamp(initial_hz)
        self._current_hz: float = clamped
        self._start_hz: float = clamped
        self._target = RateProposal(target_hz=clamped, source="initial")
        self._smoothing_seconds: float = max(0.0, smoothing_seconds)
        self._elapsed_since_propose: float = self._smoothing_seconds  # start settled
        self._proposal_history: list[RateProposal] = [self._target]

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
    def smoothing_seconds(self) -> float:
        return self._smoothing_seconds

    def propose_rate(
        self,
        rate_hz: float,
        *,
        source: str = "manual",
        anticipatory: bool = False,
    ) -> None:
        """Set a new target rate. Current rate begins easing toward it.

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
        self._start_hz = self._current_hz
        self._target = RateProposal(
            target_hz=clamped, source=source, anticipatory=anticipatory
        )
        self._elapsed_since_propose = 0.0
        self._proposal_history.append(self._target)

    def tick(self, elapsed_seconds: float) -> None:
        """Advance smoothing by ``elapsed_seconds`` of wall-clock time.

        Called by the cognitive cycle's run loop once per iteration
        with the wall-clock elapsed since the previous tick. With a
        positive elapsed, smoothing advances linearly toward the
        target. With a zero or non-positive elapsed, smoothing does
        not progress — except when smoothing is configured to zero
        seconds, in which case any tick snaps the current rate to
        the target immediately.
        """

        if self.is_settled:
            return

        if self._smoothing_seconds <= 0.0:
            # Zero-smoothing: snap on any tick, regardless of elapsed.
            # Without this, a fast-loop caller (Null subsystems, sub-
            # clock-resolution cycles) would never see snap behavior.
            self._current_hz = self._target.target_hz
            return

        if elapsed_seconds <= 0.0:
            return

        self._elapsed_since_propose += elapsed_seconds
        fraction = min(1.0, self._elapsed_since_propose / self._smoothing_seconds)
        self._current_hz = (
            self._start_hz + (self._target.target_hz - self._start_hz) * fraction
        )

    @property
    def proposal_history(self) -> list[RateProposal]:
        """Read-only view of every proposal seen (including the initial).

        Useful for the post-event introspection journal entry when
        turbo or other rate events end.
        """
        return list(self._proposal_history)


def _clamp(rate_hz: float) -> float:
    if rate_hz < MIN_RATE_HZ:
        return MIN_RATE_HZ
    if rate_hz > MAX_RATE_HZ:
        return MAX_RATE_HZ
    return rate_hz
