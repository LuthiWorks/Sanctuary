"""Tests for the physics-authority seam and its reference backend.

Two things are under test: that lawful action -> consequence crosses the seam
(bodies fall, rest, respond to force, deterministically), and that the seam's
load-bearing invariant holds — the model-facing observation never carries the
hidden ground-truth channel (mass, forces) or invisible bodies.
"""

from __future__ import annotations

import pytest

from sanctuary.physics import BodySpec, BodyState, PhysicsAuthority
from sanctuary.physics.backends import ReferencePhysicsAuthority


def _auth() -> PhysicsAuthority:
    return ReferencePhysicsAuthority()


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_reference_is_a_physics_authority():
    assert isinstance(_auth(), PhysicsAuthority)


def test_add_body_returns_id_and_registers():
    a = _auth()
    bid = a.add_body(BodySpec(body_id="cube", position=(0.0, 1.0, 0.0)))
    assert bid == "cube"
    assert a.body_ids == ("cube",)


def test_reset_clears_world():
    a = _auth()
    a.add_body(BodySpec(body_id="cube", position=(0.0, 1.0, 0.0)))
    a.step(0.1)
    a.reset()
    assert a.body_ids == ()
    assert a.time == 0.0
    assert a.step_count == 0


# ---------------------------------------------------------------------------
# Lawful dynamics (action -> consequence)
# ---------------------------------------------------------------------------


def test_body_falls_under_gravity():
    a = _auth()
    a.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))
    heights = []
    for _ in range(5):
        a.step(0.1)
        heights.append(a.observe().bodies[0].position[1])
    # Monotonic descent — the effect (falling) accrues over time (delayed
    # consequence), it isn't instantaneous.
    assert heights == sorted(heights, reverse=True)
    assert heights[-1] < 5.0


def test_body_rests_on_ground():
    a = _auth()
    a.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))
    for _ in range(200):
        a.step(0.1)
    gt = a.ground_truth().bodies[0]
    assert gt.position[1] == pytest.approx(0.0)
    assert gt.velocity[1] == pytest.approx(0.0, abs=1e-6)
    assert gt.resting is True
    # At rest the normal force cancels gravity: net vertical force ~ 0.
    assert gt.net_force[1] == pytest.approx(0.0)


def test_applied_force_changes_trajectory():
    free = _auth()
    free.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    free.step(0.1)

    pushed = _auth()
    pushed.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    pushed.apply_force("b", (0.0, 100.0, 0.0))  # strong upward push
    pushed.step(0.1)

    assert pushed.observe().bodies[0].position[1] > free.observe().bodies[0].position[1]


def test_static_body_does_not_move():
    a = _auth()
    a.add_body(BodySpec(body_id="ground", position=(0.0, 3.0, 0.0), static=True))
    a.apply_force("ground", (10.0, 10.0, 10.0))
    for _ in range(10):
        a.step(0.1)
    gt = a.ground_truth().bodies[0]
    assert gt.position == (0.0, 3.0, 0.0)
    assert gt.velocity == (0.0, 0.0, 0.0)


def test_applied_force_accumulates_then_clears():
    a = _auth()
    a.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    a.apply_force("b", (0.0, 0.0, 10.0))
    a.apply_force("b", (0.0, 0.0, 5.0))  # accumulates within the step
    a.step(0.1)
    assert a.ground_truth().bodies[0].net_force[2] == pytest.approx(15.0)
    # Force is consumed by the step; the next step sees no applied force.
    a.step(0.1)
    assert a.ground_truth().bodies[0].net_force[2] == pytest.approx(0.0)


def test_determinism():
    def run() -> object:
        a = _auth()
        a.add_body(BodySpec(body_id="b", position=(1.0, 5.0, -2.0), velocity=(0.5, 0.0, 0.0)))
        a.apply_force("b", (2.0, 0.0, 0.0))
        for _ in range(50):
            a.step(0.1)
        return a.ground_truth()

    assert run() == run()  # frozen dataclasses compare by value


# ---------------------------------------------------------------------------
# The seam invariant: observation never carries the hidden channel
# ---------------------------------------------------------------------------


def test_observation_excludes_invisible_bodies():
    a = _auth()
    a.add_body(BodySpec(body_id="seen", position=(0.0, 1.0, 0.0), visible=True))
    a.add_body(BodySpec(body_id="hidden", position=(0.0, 1.0, 0.0), visible=False))

    observed_ids = {b.body_id for b in a.observe().bodies}
    truth_ids = {b.body_id for b in a.ground_truth().bodies}

    assert observed_ids == {"seen"}          # model sees only the visible body
    assert truth_ids == {"seen", "hidden"}   # instrumentation sees both


def test_render_state_excludes_invisible_bodies():
    a = _auth()
    a.add_body(BodySpec(body_id="seen", position=(0.0, 1.0, 0.0), visible=True))
    a.add_body(BodySpec(body_id="hidden", position=(0.0, 1.0, 0.0), visible=False))
    rendered = {b.body_id for b in a.render_state().bodies}
    assert rendered == {"seen"}


def test_observation_cannot_carry_hidden_fields():
    """The model-facing body type has kinematics only — no mass, no forces.

    This is the structural guarantee behind the hidden ground-truth channel:
    the observation type literally cannot represent mass/force, so it can't
    leak into the model even by a coding mistake.
    """
    a = _auth()
    a.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0), mass=7.0))
    a.step(0.1)

    ob = a.observe().bodies[0]
    assert isinstance(ob, BodyState)
    assert hasattr(ob, "position") and hasattr(ob, "velocity")
    assert not hasattr(ob, "mass")
    assert not hasattr(ob, "net_force")

    # The same body's ground truth *does* carry the hidden fields.
    gt = a.ground_truth().bodies[0]
    assert gt.mass == 7.0
    assert hasattr(gt, "net_force")


def test_observation_and_ground_truth_agree_on_kinematics():
    a = _auth()
    a.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    for _ in range(7):
        a.step(0.1)
    ob = a.observe().bodies[0]
    gt = next(b for b in a.ground_truth().bodies if b.body_id == "b")
    assert ob.position == gt.position
    assert ob.velocity == gt.velocity


# ---------------------------------------------------------------------------
# Fail-loud boundaries
# ---------------------------------------------------------------------------


def test_duplicate_body_id_raises():
    a = _auth()
    a.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0)))
    with pytest.raises(ValueError):
        a.add_body(BodySpec(body_id="b", position=(0.0, 2.0, 0.0)))


def test_nonpositive_mass_raises():
    a = _auth()
    with pytest.raises(ValueError):
        a.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0), mass=0.0))


def test_force_on_unknown_body_raises():
    a = _auth()
    with pytest.raises(KeyError):
        a.apply_force("ghost", (1.0, 0.0, 0.0))


def test_remove_unknown_body_raises():
    a = _auth()
    with pytest.raises(KeyError):
        a.remove_body("ghost")


def test_nonpositive_dt_raises():
    a = _auth()
    a.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0)))
    with pytest.raises(ValueError):
        a.step(0.0)
