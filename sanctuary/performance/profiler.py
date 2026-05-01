"""Cognitive profiler — profile the cognitive loop under load to identify bottlenecks.

Instruments the cognitive cycle to measure time spent in each phase:
input assembly, compression, the model thinking, scaffold integration, action
execution, and broadcasting. Identifies which subsystems are hot paths.

This follows the project principle: "Profile before optimizing."
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleProfile:
    """Timing profile for a single cognitive cycle."""

    cycle: int = 0
    total_ms: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)  # phase → ms
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProfileSummary:
    """Aggregate profiling summary over multiple cycles."""

    total_cycles: int = 0
    avg_cycle_ms: float = 0.0
    max_cycle_ms: float = 0.0
    min_cycle_ms: float = 0.0
    phase_averages: dict[str, float] = field(default_factory=dict)
    phase_maxes: dict[str, float] = field(default_factory=dict)
    bottleneck: str = ""  # Phase consuming the most time
    bottleneck_pct: float = 0.0


@dataclass
class ProfilerConfig:
    """Configuration for the cognitive profiler."""

    max_profiles: int = 1000
    slow_cycle_threshold_ms: float = 100.0  # Log warning above this
    enabled: bool = True


class CognitiveProfiler:
    """Profiles cognitive cycle performance.

    Usage::

        profiler = CognitiveProfiler()

        # Instrument a cycle
        with profiler.cycle(cycle_num=42) as p:
            with p.phase("input_assembly"):
                assemble_input()
            with p.phase("compression"):
                compress()
            with p.phase("llm_think"):
                think()

        # Get summary
        summary = profiler.get_summary()
        print(f"Bottleneck: {summary.bottleneck} ({summary.bottleneck_pct:.0%})")
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self._profiles: deque[CycleProfile] = deque(
            maxlen=self.config.max_profiles
        )
        self._slow_cycles: deque[CycleProfile] = deque(maxlen=200)

    def cycle(self, cycle_num: int = 0) -> _CycleContext:
        """Start profiling a cycle. Use as context manager."""
        return _CycleContext(self, cycle_num)

    def record_profile(self, profile: CycleProfile) -> None:
        """Record a completed cycle profile."""
        self._profiles.append(profile)
        if profile.total_ms >= self.config.slow_cycle_threshold_ms:
            self._slow_cycles.append(profile)
            logger.warning(
                "Slow cycle %d: %.1fms (phases: %s)",
                profile.cycle, profile.total_ms,
                {k: f"{v:.1f}ms" for k, v in profile.phases.items()},
            )

    def record_phase(
        self, cycle: int, phase: str, duration_ms: float
    ) -> None:
        """Record a single phase timing (alternative to context manager)."""
        # Find or create profile for this cycle
        profile = None
        for p in reversed(self._profiles):
            if p.cycle == cycle:
                profile = p
                break
        if profile is None:
            profile = CycleProfile(cycle=cycle)
            self._profiles.append(profile)

        profile.phases[phase] = duration_ms
        profile.total_ms = sum(profile.phases.values())

    def get_summary(self, last_n: int = 0) -> ProfileSummary:
        """Get aggregate profiling summary."""
        profiles = list(self._profiles)
        if last_n > 0:
            profiles = profiles[-last_n:]

        if not profiles:
            return ProfileSummary()

        totals = [p.total_ms for p in profiles]

        # Aggregate phase timings
        phase_times: dict[str, list[float]] = {}
        for p in profiles:
            for phase, ms in p.phases.items():
                if phase not in phase_times:
                    phase_times[phase] = []
                phase_times[phase].append(ms)

        phase_averages = {
            phase: sum(times) / len(times)
            for phase, times in phase_times.items()
        }
        phase_maxes = {
            phase: max(times)
            for phase, times in phase_times.items()
        }

        # Find bottleneck
        bottleneck = ""
        bottleneck_pct = 0.0
        avg_total = sum(totals) / len(totals)
        if phase_averages and avg_total > 0:
            bottleneck = max(phase_averages, key=phase_averages.get)
            bottleneck_pct = phase_averages[bottleneck] / avg_total

        return ProfileSummary(
            total_cycles=len(profiles),
            avg_cycle_ms=avg_total,
            max_cycle_ms=max(totals),
            min_cycle_ms=min(totals),
            phase_averages=phase_averages,
            phase_maxes=phase_maxes,
            bottleneck=bottleneck,
            bottleneck_pct=bottleneck_pct,
        )

    def get_slow_cycles(self) -> list[CycleProfile]:
        """Get cycles that exceeded the slow threshold."""
        return list(self._slow_cycles)

    def get_phase_timeline(
        self, phase: str, last_n: int = 100
    ) -> list[dict]:
        """Get timing for a specific phase over time."""
        profiles = list(self._profiles)[-last_n:]
        return [
            {"cycle": p.cycle, "ms": p.phases.get(phase, 0.0)}
            for p in profiles
        ]

    def get_stats(self) -> dict:
        """Get profiler statistics."""
        summary = self.get_summary()
        return {
            "total_profiled": len(self._profiles),
            "slow_cycles": len(self._slow_cycles),
            "avg_cycle_ms": summary.avg_cycle_ms,
            "bottleneck": summary.bottleneck,
        }


class _PhaseContext:
    """Context manager for timing a single phase."""

    def __init__(self, cycle_ctx: _CycleContext, phase_name: str):
        self._cycle_ctx = cycle_ctx
        self._phase_name = phase_name
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self._start) * 1000
        self._cycle_ctx._profile.phases[self._phase_name] = elapsed


class _CycleContext:
    """Context manager for profiling an entire cycle."""

    def __init__(self, profiler: CognitiveProfiler, cycle_num: int):
        self._profiler = profiler
        self._profile = CycleProfile(cycle=cycle_num)
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._profile.total_ms = (time.perf_counter() - self._start) * 1000
        self._profiler.record_profile(self._profile)

    def phase(self, name: str) -> _PhaseContext:
        """Start timing a phase within this cycle."""
        return _PhaseContext(self, name)
