"""Tests for the ActionSelection -> apply_force actuation adapter.

The failure modes worth guarding here are the quiet ones: a command that
silently does nothing, a rest that is indistinguishable from a breakage, and a
clamp that hides how hard the decoder actually pushed.
"""

from __future__ import annotations

import math

import pytest

from sanctuary.motor.physics_actuation import (
    AppliedAction,
    MotorCommand,
    PhysicsActuator,
    UntrainedProjectionDecoder,
)
from sanctuary.physics import BodySpec
from sanctuary.physics.backends import ReferencePhysicsAuthority
from sanctuary.sensorium.physics_percepts import observation_to_percepts


class _FixedDecoder:
    """Returns a preset command, so actuator behaviour is tested in isolation."""

    def __init__(self, command: MotorCommand) -> None:
        self.command = command
        self.calls = 0

    def decode(self, selection):  # noqa: ANN001 - structural
        self.calls += 1
        return self.command


class _Selection:
    """Minimal ActionSelection-shaped object (luthi is not installed here)."""

    def __init__(self, action) -> None:  # noqa: ANN001
        self.action = action
        self.readable_summary = "test action"
        self.efe_breakdown: dict = {}
        self.plan_snapshot = None


def _world() -> ReferencePhysicsAuthority:
    world = ReferencePhysicsAuthority()
    world.add_body(BodySpec(body_id="rover", position=(0.0, 0.0, 0.0), mass=1.0))
    return world


def _actuator(world, command, **kwargs) -> PhysicsActuator:
    return PhysicsActuator(world, _FixedDecoder(command), **kwargs)


# ---------------------------------------------------------------------------
# Force reaches the world
# ---------------------------------------------------------------------------


def test_force_reaches_the_body():
    world = _world()
    act = _actuator(world, MotorCommand("rover", (5.0, 0.0, 0.0)))

    applied = act.apply(_Selection((1.0, 0.0, 0.0)))
    world.step(0.1)

    assert applied.applied_force == (5.0, 0.0, 0.0)
    assert not applied.at_rest
    # f=5N on 1kg for 0.1s -> +0.5 m/s in x, independent of gravity on y.
    body = next(b for b in world.observe().bodies if b.body_id == "rover")
    assert body.velocity[0] == pytest.approx(0.5)


def test_actuator_does_not_advance_time():
    """Stepping belongs to the cognitive loop, not to an effector."""
    world = _world()
    act = _actuator(world, MotorCommand("rover", (1.0, 0.0, 0.0)))

    act.apply(_Selection((1.0, 0.0, 0.0)))

    assert world.step_count == 0
    assert world.time == 0.0


def test_forces_accumulate_until_the_world_is_stepped():
    world = _world()
    act = _actuator(world, MotorCommand("rover", (1.0, 0.0, 0.0)))

    act.apply(_Selection((1.0, 0.0, 0.0)))
    act.apply(_Selection((1.0, 0.0, 0.0)))
    world.step(1.0)

    body = next(b for b in world.observe().bodies if b.body_id == "rover")
    assert body.velocity[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Rest is an action, not a failure
# ---------------------------------------------------------------------------


def test_low_intensity_is_rest_and_applies_no_force():
    world = _world()
    act = _actuator(
        world,
        MotorCommand("rover", (100.0, 0.0, 0.0), intensity=0.01),
        rest_threshold=0.05,
    )

    applied = act.apply(_Selection((1.0, 0.0, 0.0)))
    world.step(1.0)

    assert applied.at_rest
    assert applied.applied_force == (0.0, 0.0, 0.0)
    body = next(b for b in world.observe().bodies if b.body_id == "rover")
    assert body.velocity[0] == pytest.approx(0.0)


def test_rest_is_distinguishable_from_a_clamped_zero():
    """Quiet-because-at-rest and quiet-because-broken must not look alike."""
    world = _world()
    act = _actuator(
        world,
        MotorCommand("rover", (100.0, 0.0, 0.0), intensity=0.0),
        max_force=10.0,
    )

    applied = act.apply(_Selection((1.0, 0.0, 0.0)))

    assert applied.at_rest is True
    assert applied.clamped is False
    # The intent is still recorded even though nothing was applied.
    assert applied.requested_magnitude == pytest.approx(100.0)


def test_intensity_exactly_at_threshold_is_rest():
    world = _world()
    act = _actuator(
        world,
        MotorCommand("rover", (1.0, 0.0, 0.0), intensity=0.05),
        rest_threshold=0.05,
    )

    assert act.apply(_Selection((1.0, 0.0, 0.0))).at_rest


# ---------------------------------------------------------------------------
# Clamping is bounded and never silent
# ---------------------------------------------------------------------------


def test_excess_force_is_clamped_and_recorded():
    world = _world()
    act = _actuator(world, MotorCommand("rover", (1000.0, 0.0, 0.0)), max_force=10.0)

    applied = act.apply(_Selection((1.0, 0.0, 0.0)))

    assert applied.clamped is True
    assert applied.applied_magnitude == pytest.approx(10.0)
    assert applied.requested_magnitude == pytest.approx(1000.0)


def test_clamping_preserves_direction():
    world = _world()
    act = _actuator(world, MotorCommand("rover", (30.0, 40.0, 0.0)), max_force=5.0)

    applied = act.apply(_Selection((1.0, 0.0, 0.0)))

    # (30, 40) has magnitude 50; scaled to 5 keeps the 3:4 ratio.
    assert applied.applied_force[0] == pytest.approx(3.0)
    assert applied.applied_force[1] == pytest.approx(4.0)


def test_force_within_limit_is_not_marked_clamped():
    world = _world()
    act = _actuator(world, MotorCommand("rover", (1.0, 0.0, 0.0)), max_force=10.0)

    assert act.apply(_Selection((1.0, 0.0, 0.0))).clamped is False


# ---------------------------------------------------------------------------
# Fail loud
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_force_raises(bad):
    world = _world()
    act = _actuator(world, MotorCommand("rover", (bad, 0.0, 0.0)))

    with pytest.raises(ValueError, match="not finite"):
        act.apply(_Selection((1.0, 0.0, 0.0)))


def test_unknown_body_raises_rather_than_doing_nothing():
    world = _world()
    act = _actuator(world, MotorCommand("ghost", (1.0, 0.0, 0.0)))

    with pytest.raises(KeyError):
        act.apply(_Selection((1.0, 0.0, 0.0)))


def test_actuator_rejects_invalid_configuration():
    world = _world()
    decoder = _FixedDecoder(MotorCommand("rover", (1.0, 0.0, 0.0)))

    with pytest.raises(ValueError):
        PhysicsActuator(world, decoder, max_force=0.0)
    with pytest.raises(ValueError):
        PhysicsActuator(world, decoder, rest_threshold=1.5)


# ---------------------------------------------------------------------------
# The untrained decoder cannot be used by accident
# ---------------------------------------------------------------------------


def test_untrained_decoder_refuses_to_construct_without_acknowledgement():
    with pytest.raises(ValueError, match="arbitrary"):
        UntrainedProjectionDecoder("rover")


def test_untrained_decoder_error_points_at_the_real_implementation():
    with pytest.raises(ValueError) as excinfo:
        UntrainedProjectionDecoder("rover")

    message = str(excinfo.value)
    assert "DecoderRegistry" in message
    assert "does not exist yet" in message


def test_untrained_decoder_projects_the_first_three_components():
    decoder = UntrainedProjectionDecoder("rover", acknowledge_untrained=True)

    command = decoder.decode(_Selection((2.0, -3.0, 1.0, 9.0, 9.0)))

    assert command.body_id == "rover"
    assert command.force == (2.0, -3.0, 1.0)


def test_untrained_decoder_rejects_short_action_vectors():
    decoder = UntrainedProjectionDecoder("rover", acknowledge_untrained=True)

    with pytest.raises(ValueError, match="at least 3"):
        decoder.decode(_Selection((1.0, 2.0)))


def test_untrained_decoder_intensity_saturates_at_one():
    decoder = UntrainedProjectionDecoder(
        "rover", acknowledge_untrained=True, reference_magnitude=1.0
    )

    command = decoder.decode(_Selection((10.0, 0.0, 0.0)))

    assert command.intensity == pytest.approx(1.0)


def test_untrained_decoder_rejects_nonpositive_reference():
    with pytest.raises(ValueError):
        UntrainedProjectionDecoder(
            "rover", acknowledge_untrained=True, reference_magnitude=0.0
        )


# ---------------------------------------------------------------------------
# Reading the action vector
# ---------------------------------------------------------------------------


def test_accepts_a_bare_vector_as_well_as_a_selection():
    decoder = UntrainedProjectionDecoder("rover", acknowledge_untrained=True)

    from_selection = decoder.decode(_Selection((1.0, 2.0, 3.0)))
    from_bare = decoder.decode((1.0, 2.0, 3.0))

    assert from_selection.force == from_bare.force


def test_accepts_a_torch_tensor_action():
    torch = pytest.importorskip("torch")
    decoder = UntrainedProjectionDecoder("rover", acknowledge_untrained=True)

    command = decoder.decode(_Selection(torch.tensor([1.0, 2.0, 3.0])))

    assert command.force == (1.0, 2.0, 3.0)


def test_unwraps_a_batch_of_one():
    decoder = UntrainedProjectionDecoder("rover", acknowledge_untrained=True)

    command = decoder.decode(_Selection([[4.0, 5.0, 6.0]]))

    assert command.force == (4.0, 5.0, 6.0)


def test_refuses_a_batch_larger_than_one():
    decoder = UntrainedProjectionDecoder("rover", acknowledge_untrained=True)

    with pytest.raises(ValueError, match="batch size 2"):
        decoder.decode(_Selection([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_rejects_an_unreadable_action():
    decoder = UntrainedProjectionDecoder("rover", acknowledge_untrained=True)

    with pytest.raises(TypeError):
        decoder.decode(_Selection(object()))


# ---------------------------------------------------------------------------
# Motor feedback closes the sensorimotor loop
# ---------------------------------------------------------------------------


def test_applied_action_produces_a_proprioceptive_percept():
    applied = AppliedAction(
        command=MotorCommand("rover", (3.0, 0.0, 4.0)),
        applied_force=(3.0, 0.0, 4.0),
        requested_magnitude=5.0,
    )

    percept = applied.to_percept(step=7)

    assert percept.modality == "proprioceptive"
    assert "[motor:force]" in percept.content
    assert "5.00 N" in percept.content
    assert "step=7" in percept.source


def test_rest_is_reported_as_an_action_taken_not_as_absence():
    applied = AppliedAction(
        command=MotorCommand("rover", (1.0, 0.0, 0.0), intensity=0.0),
        applied_force=(0.0, 0.0, 0.0),
        at_rest=True,
    )

    percept = applied.to_percept()

    assert "held still" in percept.content
    assert "rest" in percept.embedding_summary


def test_clamping_is_visible_in_the_feedback_percept():
    applied = AppliedAction(
        command=MotorCommand("rover", (100.0, 0.0, 0.0)),
        applied_force=(10.0, 0.0, 0.0),
        clamped=True,
        requested_magnitude=100.0,
    )

    assert "clamped from 100.00" in applied.to_percept().content


# ---------------------------------------------------------------------------
# The round trip: action -> consequence -> perception
# ---------------------------------------------------------------------------


def test_action_changes_what_is_subsequently_perceived():
    """The whole point of both adapters together.

    Without this, JEPA's prediction error carries information only about the
    world's autonomous dynamics -- never about the entity's own agency.
    """
    world = ReferencePhysicsAuthority()
    world.add_body(BodySpec(body_id="rover", position=(0.0, 0.0, 0.0)))
    world.add_body(
        BodySpec(body_id="landmark", position=(10.0, 0.0, 0.0), static=True)
    )

    def distance_to_landmark() -> float:
        percepts = observation_to_percepts(world.observe(), self_id="rover")
        landmark = next(p for p in percepts if "landmark" in p.content)
        return float(landmark.content.split(" m away")[0].split()[-1])

    before = distance_to_landmark()

    act = _actuator(world, MotorCommand("rover", (20.0, 0.0, 0.0)))
    act.apply(_Selection((1.0, 0.0, 0.0)))
    world.step(0.5)

    after = distance_to_landmark()

    assert after < before, "pushing toward the landmark should reduce perceived distance"


def test_resting_leaves_the_horizontal_world_unchanged():
    """The control for the test above: no action, no horizontal consequence."""
    world = ReferencePhysicsAuthority()
    world.add_body(BodySpec(body_id="rover", position=(0.0, 0.0, 0.0), static=True))
    world.add_body(
        BodySpec(body_id="landmark", position=(10.0, 0.0, 0.0), static=True)
    )

    def distance_to_landmark() -> float:
        percepts = observation_to_percepts(world.observe(), self_id="rover")
        landmark = next(p for p in percepts if "landmark" in p.content)
        return float(landmark.content.split(" m away")[0].split()[-1])

    before = distance_to_landmark()

    act = _actuator(world, MotorCommand("rover", (20.0, 0.0, 0.0), intensity=0.0))
    applied = act.apply(_Selection((1.0, 0.0, 0.0)))
    world.step(0.5)

    assert applied.at_rest
    assert distance_to_landmark() == pytest.approx(before)


def test_decoder_is_consulted_once_per_action():
    world = _world()
    decoder = _FixedDecoder(MotorCommand("rover", (1.0, 0.0, 0.0)))
    act = PhysicsActuator(world, decoder)

    act.apply(_Selection((1.0, 0.0, 0.0)))
    act.apply(_Selection((1.0, 0.0, 0.0)))

    assert decoder.calls == 2


def test_apply_command_bypasses_the_decoder():
    world = _world()
    decoder = _FixedDecoder(MotorCommand("rover", (99.0, 0.0, 0.0)))
    act = PhysicsActuator(world, decoder)

    applied = act.apply_command(MotorCommand("rover", (2.0, 0.0, 0.0)))

    assert decoder.calls == 0
    assert applied.applied_force == (2.0, 0.0, 0.0)
