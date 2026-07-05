"""Cognitive-proprioception delta: the entity feels its substrate change.

Added 2026-07-05 with the metaplasticity/momentum exposure (rich-
parameters analysis finding 2): update_ema and momentum stats now flow
from LuthiModel's get_introspection into the per-cycle delta, so the
mind can feel how its capacity for change — and the recent magnitude of
its change — moved this cycle. Also pins the pre-existing contract:
activity_level aggregates ONLY the original three delta keys (the new
channels must not silently shift the turbo trigger's scale — that
join is an explicit Phase 4 design call, with a band re-warm).
"""

from __future__ import annotations

from sanctuary.core.luthi_model import LuthiModel


def _bare_model() -> LuthiModel:
    """LuthiModel without running its heavyweight __init__ — the delta
    computation is a pure function of _pre_state/_post_state."""
    m = object.__new__(LuthiModel)
    m._introspection_delta = {}
    return m


def _block(**fields) -> dict:
    return dict(fields)


def test_metaplasticity_and_momentum_deltas_computed():
    m = _bare_model()
    m._pre_state = {"blocks": [
        _block(plasticity_mean=1.0, set_point_drift=0.10,
               update_ema_mean=2e-4, momentum_abs_mean=1e-4),
    ]}
    m._post_state = {"blocks": [
        _block(plasticity_mean=1.1, set_point_drift=0.12,
               update_ema_mean=5e-4, momentum_abs_mean=4e-4),
    ]}
    m._compute_introspection_delta()
    d = m._introspection_delta
    assert abs(d["metaplasticity_change"] - 3e-4) < 1e-12
    assert abs(d["momentum_change"] - 3e-4) < 1e-12


def test_activity_level_excludes_new_channels():
    """The turbo trigger's input keeps its scale: only plasticity/drift/
    membrane deltas aggregate into activity_level."""
    m = _bare_model()
    m._pre_state = {"blocks": [
        _block(plasticity_mean=1.0, set_point_drift=0.10,
               update_ema_mean=0.0, momentum_abs_mean=0.0),
    ]}
    m._post_state = {"blocks": [
        _block(plasticity_mean=1.2, set_point_drift=0.15,
               update_ema_mean=100.0, momentum_abs_mean=100.0),
    ]}
    m._compute_introspection_delta()
    d = m._introspection_delta
    expected = abs(0.2) + abs(0.05)  # plasticity + drift only
    assert abs(d["activity_level"] - expected) < 1e-9, (
        "new channels must not leak into activity_level without the "
        "Phase 4 band re-warm decision"
    )


def test_missing_fields_are_tolerated():
    """Old substrates (no update_ema/momentum exposure) produce no new
    keys and no errors — hasattr-gated end to end."""
    m = _bare_model()
    m._pre_state = {"blocks": [_block(plasticity_mean=1.0)]}
    m._post_state = {"blocks": [_block(plasticity_mean=1.05)]}
    m._compute_introspection_delta()
    d = m._introspection_delta
    assert "metaplasticity_change" not in d
    assert "momentum_change" not in d
    assert "plasticity_change" in d
