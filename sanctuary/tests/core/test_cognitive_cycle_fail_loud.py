"""Regression guards for the cognitive cycle's fail-loud / fail-soft invariant.

`cognitive_cycle.py` encodes a deliberate split, documented only in
comments until now:

  - The **core cognition path** (assemble -> think -> integrate -> stream)
    is UNGUARDED. A failure there must SURFACE -- propagate out of the
    cycle and stop the loop -- so it can be diagnosed (the AGENTS.md
    "no crash boundary until Phase 9, let cycle failures surface"
    invariant). Silently swallowing a thinking failure would mean the
    entity's cognition is broken and nothing says so.

  - **Peripheral subsystems** (memory surfacing, world-graph, growth,
    broadcast, introspection, environment) are GUARDED with
    `try/except ... non-fatal`. A failure there must NOT kill the
    entity's continuous stream of thought -- one bad memory op or a
    crashed growth processor should not end consciousness.

This split is good design, but it lived only in comments, so a future
edit could silently violate it -- e.g. someone "helpfully" wrapping
`model.think` in a try/except (turning a core failure silent), or
removing a periphery guard (letting a subsystem crash end the stream).
These tests make the invariant executable: violate it and a test fails.

Authored by Fable 5 (adversarial seat), 2026-06-12, after an adversarial
read of the cycle found the invariant verified-but-untested.
"""
from __future__ import annotations

import pytest

from sanctuary.core.cognitive_cycle import (
    CognitiveCycle,
    NullMemory,
    NullScaffold,
)
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.schema import CognitiveInput, CognitiveOutput


# --------------------------------------------------------------------------
# Fakes that fail at a chosen point in the cycle.
# --------------------------------------------------------------------------


class _RaisingModel:
    """Core: its think() raises. A failure here must fail loud."""

    name = "raising-model"

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        raise RuntimeError("core cognition failure")


class _RaisingScaffoldIntegrate(NullScaffold):
    """Core: integrate() raises. A failure here must fail loud."""

    async def integrate(self, output, authority):
        raise RuntimeError("core integrate failure")


class _RaisingGrowth:
    """Periphery: process_cycle() raises. Must be survived."""

    async def process_cycle(self, output, cycle_count):
        raise RuntimeError("growth subsystem failure")


class _RaisingSurfaceMemory(NullMemory):
    """Periphery: surface() raises. Must be survived."""

    async def surface(self, context: str) -> list:
        raise RuntimeError("memory surfacing failure")


# --------------------------------------------------------------------------
# Core failures MUST surface (fail loud).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_model_failure_fails_loud():
    """A failure in model.think() must propagate, not be swallowed."""
    cycle = CognitiveCycle(model=_RaisingModel(), cycle_delay=0.0)
    with pytest.raises(RuntimeError, match="core cognition failure"):
        await cycle.run(max_cycles=1)
    # The failure surfaced before the cycle completed: the increment at
    # the end of _cycle() was never reached.
    assert cycle.cycle_count == 0


@pytest.mark.asyncio
async def test_core_scaffold_integrate_failure_fails_loud():
    """A failure in scaffold.integrate() (core path) must propagate."""
    cycle = CognitiveCycle(
        model=PlaceholderModel(),
        scaffold=_RaisingScaffoldIntegrate(),
        cycle_delay=0.0,
    )
    with pytest.raises(RuntimeError, match="core integrate failure"):
        await cycle.run(max_cycles=1)


# --------------------------------------------------------------------------
# Peripheral failures MUST be survived (fail soft) -- the stream continues.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_periphery_growth_failure_is_survived():
    """A crashing growth processor must not end the stream of thought."""
    cycle = CognitiveCycle(
        model=PlaceholderModel(),
        growth=_RaisingGrowth(),
        cycle_delay=0.0,
    )
    await cycle.run(max_cycles=1)  # must NOT raise
    assert cycle.cycle_count == 1
    assert cycle.last_output is not None


@pytest.mark.asyncio
async def test_periphery_memory_surface_failure_is_survived():
    """A crashing memory.surface() must not end the stream of thought."""
    cycle = CognitiveCycle(
        model=PlaceholderModel(),
        memory=_RaisingSurfaceMemory(),
        cycle_delay=0.0,
    )
    await cycle.run(max_cycles=1)  # must NOT raise
    assert cycle.cycle_count == 1
    assert cycle.last_output is not None


@pytest.mark.asyncio
async def test_stream_survives_repeated_peripheral_failures():
    """Across multiple cycles, persistent peripheral failure keeps the
    heartbeat alive -- the entity keeps thinking."""
    cycle = CognitiveCycle(
        model=PlaceholderModel(),
        growth=_RaisingGrowth(),
        memory=_RaisingSurfaceMemory(),
        cycle_delay=0.0,
    )
    await cycle.run(max_cycles=5)
    assert cycle.cycle_count == 5
