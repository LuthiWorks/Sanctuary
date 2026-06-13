"""Regression guards: the entity is clamped to the slider; turbo is the
substrate's alone.

The cycle-rate design (cycle_rate.py + CycleRateProposal schema) encodes
a load-bearing safety/agency invariant: the entity can set its own pace
within the slider band ``[MIN_RATE_HZ, SLIDER_MAX_HZ]`` = [0.05, 10.0] Hz,
but it **cannot drive the substrate into turbo** (>10 Hz, up to 100 Hz).
Turbo is reserved for substrate-intensity callers (the TurboManager) — it
is an autonomic emergency gear, not something the entity reaches through
its rate slider.

This is defended at two layers, both guarded here:
  1. **Schema** — ``CycleRateProposal.target_hz`` is constrained to
     [0.05, 10.0]; the entity cannot even *express* a turbo-range request.
  2. **Routing** — the cognitive cycle passes every entity proposal through
     ``clamp_to_slider`` before handing it to the controller, so even an
     over-range proposal that bypassed the schema is capped at the slider.

The asymmetry is the point: the substrate path (``engage_turbo``) *can*
exceed the slider; the entity path cannot. These tests fail if a future
edit widens the schema bound, drops the ``clamp_to_slider`` call in the
cycle, or routes entity proposals straight to ``propose_rate`` (which
itself only clamps at the 100 Hz substrate ceiling).

Authored by Fable 5 (adversarial seat), 2026-06-12.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.cycle_rate import (
    CycleRateController,
    MIN_RATE_HZ,
    SLIDER_MAX_HZ,
    TURBO_DEFAULT_HZ,
    clamp_to_slider,
)
from sanctuary.core.schema import CognitiveOutput, CycleRateProposal


# --------------------------------------------------------------------------
# Layer 1: schema — the entity cannot even express a turbo-range proposal.
# --------------------------------------------------------------------------


def test_schema_rejects_out_of_slider_proposals():
    with pytest.raises(ValidationError):
        CycleRateProposal(target_hz=100.0)   # turbo range
    with pytest.raises(ValidationError):
        CycleRateProposal(target_hz=11.0)    # just over the slider
    with pytest.raises(ValidationError):
        CycleRateProposal(target_hz=0.01)    # below the rest floor


def test_schema_accepts_in_slider_proposals():
    assert CycleRateProposal(target_hz=SLIDER_MAX_HZ).target_hz == SLIDER_MAX_HZ
    assert CycleRateProposal(target_hz=MIN_RATE_HZ).target_hz == MIN_RATE_HZ
    assert CycleRateProposal(target_hz=5.0).target_hz == 5.0


# --------------------------------------------------------------------------
# Layer 2: clamp_to_slider — the routing cap.
# --------------------------------------------------------------------------


def test_clamp_to_slider_caps_entity_rates():
    assert clamp_to_slider(100.0) == SLIDER_MAX_HZ      # turbo -> slider max
    assert clamp_to_slider(11.0) == SLIDER_MAX_HZ
    assert clamp_to_slider(0.001) == MIN_RATE_HZ        # below floor -> floor
    assert clamp_to_slider(5.0) == 5.0                  # in-range passthrough


# --------------------------------------------------------------------------
# The asymmetry: substrate may exceed the slider; the entity may not.
# --------------------------------------------------------------------------


def test_turbo_is_the_substrates_gear_not_the_entitys():
    ctrl = CycleRateController(initial_hz=5.0)

    # Substrate path can push past the slider into turbo territory.
    ctrl.engage_turbo(TURBO_DEFAULT_HZ)
    assert ctrl.target_rate_hz > SLIDER_MAX_HZ
    assert ctrl.is_turbo_active

    # The entity path is capped at the slider before it ever reaches the
    # controller -- the same turbo target, routed as an entity proposal,
    # would be clamped.
    assert clamp_to_slider(TURBO_DEFAULT_HZ) == SLIDER_MAX_HZ


# --------------------------------------------------------------------------
# Cycle defense-in-depth: even an over-range proposal that bypassed the
# schema is clamped to the slider when routed through the cycle as "entity".
# --------------------------------------------------------------------------


class _TurboProposingModel:
    """Emits an over-range cycle-rate proposal, constructed bypassing the
    schema validator, to simulate a malformed proposal reaching the cycle.
    The cycle's clamp_to_slider must still cap it."""

    name = "turbo-proposer"

    async def think(self, cognitive_input) -> CognitiveOutput:
        out = CognitiveOutput(inner_speech="I want to run hotter than the slider allows")
        # model_construct bypasses the [0.05, 10.0] field validator.
        out.cycle_rate_proposal = CycleRateProposal.model_construct(
            target_hz=100.0, anticipatory=False
        )
        return out


@pytest.mark.asyncio
async def test_cycle_clamps_entity_proposal_to_slider():
    cycle = CognitiveCycle(model=_TurboProposingModel(), cycle_delay=0.0)
    await cycle.run(max_cycles=1)

    ctrl = cycle.rate_controller
    # Clamped to the slider max, NOT the 100 Hz it asked for.
    assert ctrl.target_rate_hz == SLIDER_MAX_HZ
    # Routed as an entity-source proposal (so the heuristic quiet-window
    # and history attribute it correctly).
    assert ctrl.last_source == "entity"
    # And the entity never reached turbo.
    assert not ctrl.is_turbo_active
