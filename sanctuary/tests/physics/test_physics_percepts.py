"""Tests for the PhysicsObservation -> Percept adapter.

Covers the three things that can go wrong quietly here: the egocentric frame
silently reverting to world coordinates, the hidden ground-truth channel
finding a route into perception, and the numeric path being wired to an encoder
that does not exist.
"""

from __future__ import annotations

import math

import pytest

from sanctuary.physics import BodySpec, PhysicsObservation
from sanctuary.physics.backends import ReferencePhysicsAuthority
from sanctuary.physics.state import BodyState
from sanctuary.sensorium.physics_percepts import (
    BODY_VECTOR_WIDTH,
    EXTEROCEPTIVE,
    PROPRIOCEPTIVE,
    observation_to_percepts,
    observation_to_vector,
)


def _obs(*bodies: BodyState, step: int = 0, time: float = 0.0) -> PhysicsObservation:
    return PhysicsObservation(time=time, step=step, bodies=tuple(bodies))


ROVER = BodyState(body_id="rover", position=(1.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Egocentric framing
# ---------------------------------------------------------------------------


def test_proprioceptive_percept_leads_and_describes_own_body():
    percepts = observation_to_percepts(_obs(ROVER), self_id="rover")

    assert len(percepts) == 1
    assert percepts[0].modality == PROPRIOCEPTIVE
    assert "[body]" in percepts[0].content
    assert "speed 1.00 m/s" in percepts[0].content


def test_other_bodies_are_reported_relative_to_self():
    ball = BodyState(body_id="ball", position=(4.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    percepts = observation_to_percepts(_obs(ROVER, ball), self_id="rover")
    exteroceptive = [p for p in percepts if p.modality == EXTEROCEPTIVE]

    assert len(exteroceptive) == 1
    content = exteroceptive[0].content
    # Displacement is 3 m, not the ball's world x of 4.
    assert "3.00 m away" in content
    assert "offset (3.00, 0.00, 0.00)" in content
    # The rover moves +x at 1 m/s toward a still ball: the gap is closing.
    assert "closing at 1.00 m/s" in content


def test_closing_versus_separating_sign():
    # Ball ahead, moving away faster than the rover -> separating.
    ball = BodyState(body_id="ball", position=(4.0, 0.0, 0.0), velocity=(3.0, 0.0, 0.0))
    percepts = observation_to_percepts(_obs(ROVER, ball), self_id="rover")
    assert "separating at 2.00 m/s" in percepts[1].content


def test_allocentric_mode_has_no_self_percept_and_uses_world_coordinates():
    ball = BodyState(body_id="ball", position=(4.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    percepts = observation_to_percepts(_obs(ROVER, ball), self_id=None)

    assert all(p.modality == EXTEROCEPTIVE for p in percepts)
    assert len(percepts) == 2
    joined = " ".join(p.content for p in percepts)
    assert "[world]" in joined
    assert "at (4.00, 0.00, 0.00)" in joined


def test_empty_allocentric_world_yields_nothing():
    assert observation_to_percepts(_obs(), self_id=None) == []


# ---------------------------------------------------------------------------
# Fail loud
# ---------------------------------------------------------------------------


def test_missing_self_body_raises_rather_than_falling_back_to_world_frame():
    ball = BodyState(body_id="ball", position=(1.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    with pytest.raises(KeyError, match="rover"):
        observation_to_percepts(_obs(ball), self_id="rover")


def test_missing_self_body_error_names_what_was_visible():
    ball = BodyState(body_id="ball", position=(1.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    with pytest.raises(KeyError) as excinfo:
        observation_to_percepts(_obs(ball), self_id="rover")

    assert "ball" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Acuity limits and ordering
# ---------------------------------------------------------------------------


def test_bodies_are_ordered_nearest_first():
    far = BodyState(body_id="far", position=(11.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    near = BodyState(body_id="near", position=(2.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    percepts = observation_to_percepts(_obs(ROVER, far, near), self_id="rover")

    assert "near" in percepts[1].content
    assert "far" in percepts[2].content


def test_equidistant_bodies_break_ties_on_id_for_determinism():
    a = BodyState(body_id="b_body", position=(3.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    b = BodyState(body_id="a_body", position=(-1.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    # Both are exactly 2.0 m from the rover at x=1.
    percepts = observation_to_percepts(_obs(ROVER, a, b), self_id="rover")

    assert "a_body" in percepts[1].content
    assert "b_body" in percepts[2].content


def test_max_range_excludes_distant_bodies():
    near = BodyState(body_id="near", position=(2.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    far = BodyState(body_id="far", position=(50.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    percepts = observation_to_percepts(
        _obs(ROVER, near, far), self_id="rover", max_range=10.0
    )

    joined = " ".join(p.content for p in percepts)
    assert "near" in joined
    assert "far" not in joined


def test_max_bodies_keeps_the_nearest():
    bodies = [
        BodyState(body_id=f"b{i}", position=(float(i + 2), 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
        for i in range(5)
    ]

    percepts = observation_to_percepts(_obs(ROVER, *bodies), self_id="rover", max_bodies=2)

    exteroceptive = [p for p in percepts if p.modality == EXTEROCEPTIVE]
    assert len(exteroceptive) == 2
    assert "b0" in exteroceptive[0].content
    assert "b1" in exteroceptive[1].content


def test_limits_default_to_unlimited():
    bodies = [
        BodyState(body_id=f"b{i}", position=(float(i + 2), 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
        for i in range(20)
    ]

    percepts = observation_to_percepts(_obs(ROVER, *bodies), self_id="rover")

    assert len([p for p in percepts if p.modality == EXTEROCEPTIVE]) == 20


# ---------------------------------------------------------------------------
# The guard: no tensor route until an encoder exists to receive it
# ---------------------------------------------------------------------------


def test_percepts_leave_tensor_data_unset():
    """Luthi's trunk has vision/audio/text encoders and no proprioceptive one.

    Populating ``tensor_data`` would imply a route into the trunk that does not
    exist -- a mechanism reporting healthy while doing nothing, which is the
    failure mode both repos name as their most expensive recurring defect.

    When a body-state encoder lands in LuthiModel, this test is the thing that
    should be changed *deliberately*, together with the wiring that consumes
    the tensor. Do not delete it to make a change pass.
    """
    ball = BodyState(body_id="ball", position=(4.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    percepts = observation_to_percepts(_obs(ROVER, ball), self_id="rover")

    assert percepts, "expected percepts to assert against"
    assert all(p.tensor_data is None for p in percepts)


# ---------------------------------------------------------------------------
# The numeric path
# ---------------------------------------------------------------------------


def test_vector_width_is_constant_regardless_of_body_count():
    ball = BodyState(body_id="ball", position=(4.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    expected = BODY_VECTOR_WIDTH * 4  # self + 3 slots

    empty = observation_to_vector(_obs(ROVER), self_id="rover", max_bodies=3)
    one = observation_to_vector(_obs(ROVER, ball), self_id="rover", max_bodies=3)

    assert len(empty) == expected
    assert len(one) == expected


def test_vector_self_block_carries_position_velocity_speed():
    vec = observation_to_vector(_obs(ROVER), self_id="rover", max_bodies=0)

    assert vec[:3] == (1.0, 0.0, 0.0)
    assert vec[3:6] == (1.0, 0.0, 0.0)
    assert vec[6] == pytest.approx(1.0)


def test_vector_body_block_is_egocentric():
    ball = BodyState(body_id="ball", position=(4.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))

    vec = observation_to_vector(_obs(ROVER, ball), self_id="rover", max_bodies=1)
    block = vec[BODY_VECTOR_WIDTH:]

    assert block[:3] == (3.0, 0.0, 0.0)          # displacement, not world x=4
    assert block[3:6] == (-1.0, 0.0, 0.0)        # relative velocity
    assert block[6] == pytest.approx(3.0)        # distance


def test_vector_pads_absent_bodies_with_zeros():
    vec = observation_to_vector(_obs(ROVER), self_id="rover", max_bodies=2)

    assert vec[BODY_VECTOR_WIDTH:] == (0.0,) * (BODY_VECTOR_WIDTH * 2)


def test_vector_rejects_negative_max_bodies():
    with pytest.raises(ValueError):
        observation_to_vector(_obs(ROVER), self_id="rover", max_bodies=-1)


# ---------------------------------------------------------------------------
# End to end against the real substrate
# ---------------------------------------------------------------------------


def test_falling_body_produces_changing_perception():
    """The point of the whole adapter: lawful consequence becomes perceivable."""
    world = ReferencePhysicsAuthority()
    world.add_body(BodySpec(body_id="rover", position=(0.0, 0.0, 0.0), static=True))
    world.add_body(BodySpec(body_id="ball", position=(0.0, 10.0, 0.0)))

    first = observation_to_percepts(world.observe(), self_id="rover")

    for _ in range(10):
        world.step(0.05)

    later = observation_to_percepts(world.observe(), self_id="rover")

    def distance_of(percepts) -> float:
        ball = next(p for p in percepts if "ball" in p.content)
        return float(ball.content.split(" m away")[0].split()[-1])

    # Gravity pulled it toward the ground, so it is nearer the rover at origin.
    assert distance_of(later) < distance_of(first)
    assert "closing" in next(p for p in later if "ball" in p.content).content


def test_invisible_bodies_never_reach_perception():
    """Occlusion is the authority's job; verify the adapter inherits it."""
    world = ReferencePhysicsAuthority()
    world.add_body(BodySpec(body_id="rover", position=(0.0, 0.0, 0.0), static=True))
    world.add_body(BodySpec(body_id="secret", position=(1.0, 0.0, 0.0), visible=False))

    percepts = observation_to_percepts(world.observe(), self_id="rover")

    assert all("secret" not in p.content for p in percepts)


def test_adapter_signature_cannot_accept_ground_truth():
    """Structural guard: the hidden channel has no parameter to arrive through.

    ``PhysicsGroundTruth`` carries mass and net force. If someone ever adds a
    ground-truth parameter here, the hidden channel gains a route into the
    model and this test should be the thing that stops them.
    """
    import inspect

    params = set(inspect.signature(observation_to_percepts).parameters)
    assert params == {"obs", "self_id", "max_range", "max_bodies"}

    vec_params = set(inspect.signature(observation_to_vector).parameters)
    assert vec_params == {"obs", "self_id", "max_bodies", "max_range"}
