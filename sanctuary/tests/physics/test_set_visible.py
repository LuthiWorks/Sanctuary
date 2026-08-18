"""Visibility toggling — the seam operation that keeps a changing population cheap.

Why this exists. A creature roster has births and deaths, so the set of bodies
in the world changes over a run. Doing that with ``add_body``/``remove_body``
is very expensive on a compiling backend: MuJoCo regenerates its XML and
recompiles the model, which also tears down the GL renderer. Measured
2026-08-17 with a 17-body world and one attached camera:

    birth via add_body    : 40.26 ms   (11 ms rebuild + 29 ms renderer)
    birth via set_visible :  1.30 ms
    warm render           :  0.71 ms

40 ms is 40% of a 100 ms cognitive cycle, and it lands *exactly* when something
appeared or vanished — the most curriculum-salient moment there is. Worse, a
rebuild raises outright while a viewer is open, so a creature could not be born
at all while Brian and Sandi were watching.

So population change is expressed as visibility, and these tests pin the three
properties that makes safe:

  - hidden means hidden in the model-facing and renderer views,
  - hidden does NOT mean hidden from instrumentation (ground truth reports what
    is true, not what is seen),
  - and toggling must never dirty a compiled model, or the cost comes straight
    back without anything announcing it.

Authored by Opus 5, 2026-08-17.
"""
from __future__ import annotations

import pytest

from sanctuary.physics import BodySpec
from sanctuary.physics.backends import ReferencePhysicsAuthority


def _mujoco():
    pytest.importorskip("mujoco")
    from sanctuary.physics.backends.mujoco_backend import MuJoCoPhysicsAuthority
    return MuJoCoPhysicsAuthority()


@pytest.fixture(params=["reference", "mujoco"])
def world(request):
    if request.param == "mujoco":
        w = _mujoco()
    else:
        w = ReferencePhysicsAuthority()
    w.reset(seed=0)
    w.add_body(BodySpec(body_id="seen", position=(0.0, 1.0, 0.0)))
    w.add_body(BodySpec(body_id="pooled", position=(2.0, 1.0, 0.0)))
    w.step(0.02)
    return w


def _ids(bodies):
    return {b.body_id for b in bodies}


# ---------------------------------------------------------------------------
# The three views
# ---------------------------------------------------------------------------

def test_hiding_removes_a_body_from_the_model_facing_view(world):
    assert "pooled" in _ids(world.observe().bodies)
    world.set_visible("pooled", False)
    assert "pooled" not in _ids(world.observe().bodies)
    assert "seen" in _ids(world.observe().bodies)


def test_hiding_removes_a_body_from_the_renderer_view(world):
    world.set_visible("pooled", False)
    assert "pooled" not in _ids(world.render_state().bodies)


def test_hiding_does_NOT_hide_from_instrumentation(world):
    """Ground truth reports what is true, not what is seen."""
    world.set_visible("pooled", False)
    gt = {b.body_id: b for b in world.ground_truth().bodies}
    assert "pooled" in gt
    assert gt["pooled"].visible is False


def test_revealing_brings_it_back(world):
    world.set_visible("pooled", False)
    world.set_visible("pooled", True)
    assert "pooled" in _ids(world.observe().bodies)
    assert "pooled" in _ids(world.render_state().bodies)


# ---------------------------------------------------------------------------
# Physics is untouched
# ---------------------------------------------------------------------------

def test_a_hidden_body_still_obeys_physics(world):
    """Hiding is a fact about the views, not about the world.

    A hidden body that stopped falling would mean 'invisible' had quietly
    become 'not simulated' -- two very different claims.
    """
    world.set_visible("pooled", False)
    before = {b.body_id: b.position[1] for b in world.ground_truth().bodies}["pooled"]
    for _ in range(20):
        world.step(0.02)
    after = {b.body_id: b.position[1] for b in world.ground_truth().bodies}["pooled"]
    assert after < before, "a hidden body stopped falling"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_unknown_body_raises(world):
    with pytest.raises(KeyError):
        world.set_visible("nobody", False)


# ---------------------------------------------------------------------------
# The guarantee the whole design rests on
# ---------------------------------------------------------------------------

def test_set_visible_does_not_rebuild_the_model():
    """The load-bearing one, named in set_visible's docstring.

    If a future edit makes `_to_xml()` depend on visibility, `set_visible`
    must start dirtying the model -- and this test is what will say so. It
    checks the mechanism (model object identity) rather than wall-clock, so it
    cannot go flaky on a loaded machine.
    """
    w = _mujoco()
    w.reset(seed=0)
    w.add_body(BodySpec(body_id="a", position=(0.0, 1.0, 0.0)))
    w.add_body(BodySpec(body_id="b", position=(1.0, 1.0, 0.0), visible=False))
    w.step(0.02)
    model_before = w._model
    assert model_before is not None

    w.set_visible("b", True)
    w.step(0.02)

    assert w._model is model_before, (
        "set_visible triggered a model rebuild -- the ~40 ms birth stall is back"
    )
    assert w._dirty is False


def test_add_body_DOES_rebuild_the_model():
    """Guards the test above from passing vacuously.

    If add_body stopped rebuilding, the identity check would prove nothing.
    """
    w = _mujoco()
    w.reset(seed=0)
    w.add_body(BodySpec(body_id="a", position=(0.0, 1.0, 0.0)))
    w.step(0.02)
    model_before = w._model

    w.add_body(BodySpec(body_id="c", position=(1.0, 1.0, 0.0)))
    w.step(0.02)

    assert w._model is not model_before
