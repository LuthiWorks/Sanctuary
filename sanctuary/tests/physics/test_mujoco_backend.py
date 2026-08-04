"""Tests for the MuJoCo backend, and its agreement with the reference.

Two jobs. First, that ``MuJoCoPhysicsAuthority`` satisfies the same
``PhysicsAuthority`` contract ``test_seam.py`` holds the reference backend to --
the seam is only worth having if swapping the engine changes nothing above it.
Second, that the two backends *agree qualitatively* where they should and
*differ only where documented*: MuJoCo bodies have volume and compliant
contacts, so exact resting values differ by construction, while falling,
resting, responding to force, and staying put when static must all still hold.

Skipped wholesale when ``mujoco`` is not installed -- it is an optional extra,
and CI runs the reference backend.
"""

from __future__ import annotations

import pytest

from sanctuary.physics import BodySpec, BodyState, PhysicsAuthority
from sanctuary.physics.backends import ReferencePhysicsAuthority

mujoco = pytest.importorskip("mujoco")

from sanctuary.physics.backends.mujoco_backend import (  # noqa: E402
    MuJoCoPhysicsAuthority,
)


@pytest.fixture
def world() -> MuJoCoPhysicsAuthority:
    return MuJoCoPhysicsAuthority()


def _settled(auth: PhysicsAuthority, body_id: str = "ball", steps: int = 200):
    for _ in range(steps):
        auth.step(0.1)
    return next(b for b in auth.ground_truth().bodies if b.body_id == body_id)


# ---------------------------------------------------------------------------
# Contract parity with the seam
# ---------------------------------------------------------------------------


def test_is_a_physics_authority(world):
    assert isinstance(world, PhysicsAuthority)


def test_add_body_returns_id_and_registers(world):
    assert world.add_body(BodySpec(body_id="cube", position=(0.0, 1.0, 0.0))) == "cube"
    assert world.body_ids == ("cube",)


def test_reset_clears_world(world):
    world.add_body(BodySpec(body_id="cube", position=(0.0, 1.0, 0.0)))
    world.step(0.1)
    world.reset()

    assert world.body_ids == ()
    assert world.time == 0.0
    assert world.step_count == 0


def test_duplicate_body_id_raises(world):
    world.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0)))
    with pytest.raises(ValueError):
        world.add_body(BodySpec(body_id="b", position=(0.0, 2.0, 0.0)))


def test_nonpositive_mass_raises(world):
    with pytest.raises(ValueError):
        world.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0), mass=0.0))


def test_force_on_unknown_body_raises(world):
    with pytest.raises(KeyError):
        world.apply_force("ghost", (1.0, 0.0, 0.0))


def test_remove_unknown_body_raises(world):
    with pytest.raises(KeyError):
        world.remove_body("ghost")


def test_nonpositive_dt_raises(world):
    world.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0)))
    with pytest.raises(ValueError):
        world.step(0.0)


def test_rejects_invalid_construction():
    with pytest.raises(ValueError):
        MuJoCoPhysicsAuthority(body_radius=0.0)
    with pytest.raises(ValueError):
        MuJoCoPhysicsAuthority(max_substep=0.0)


# ---------------------------------------------------------------------------
# Lawful dynamics
# ---------------------------------------------------------------------------


def test_body_falls_under_gravity(world):
    world.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))

    heights = []
    for _ in range(5):
        world.step(0.1)
        heights.append(world.observe().bodies[0].position[1])

    assert heights == sorted(heights, reverse=True)
    assert heights[-1] < 5.0


def test_free_fall_matches_analytic_velocity(world):
    """Substepping must not distort the integration."""
    world.add_body(BodySpec(body_id="ball", position=(0.0, 50.0, 0.0)))

    for _ in range(5):
        world.step(0.1)

    # v = g*t after 0.5 s, well clear of the ground.
    assert world.observe().bodies[0].velocity[1] == pytest.approx(-9.81 * 0.5, rel=1e-3)


def test_body_rests_on_ground_at_its_radius(world):
    """A documented divergence: a point rests at 0, a sphere rests at r."""
    world.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))

    gt = _settled(world)

    assert gt.position[1] == pytest.approx(world.body_radius, abs=1e-3)
    assert gt.velocity[1] == pytest.approx(0.0, abs=1e-3)
    assert gt.resting is True


def test_contact_force_cancels_gravity_at_rest(world):
    """Net vertical force ~0 at rest, arrived at by solving the contact."""
    world.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0), mass=3.0))

    gt = _settled(world)

    assert gt.net_force[1] == pytest.approx(0.0, abs=1e-2)


def test_applied_force_changes_trajectory():
    free = MuJoCoPhysicsAuthority()
    free.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    free.step(0.1)

    pushed = MuJoCoPhysicsAuthority()
    pushed.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    pushed.apply_force("b", (0.0, 100.0, 0.0))
    pushed.step(0.1)

    assert (
        pushed.observe().bodies[0].position[1] > free.observe().bodies[0].position[1]
    )


def test_static_body_does_not_move(world):
    world.add_body(BodySpec(body_id="ground", position=(0.0, 3.0, 0.0), static=True))
    world.apply_force("ground", (10.0, 10.0, 10.0))

    for _ in range(10):
        world.step(0.1)

    gt = world.ground_truth().bodies[0]
    assert gt.position == (0.0, 3.0, 0.0)
    assert gt.velocity == (0.0, 0.0, 0.0)


def test_applied_force_accumulates_then_clears(world):
    world.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    world.apply_force("b", (0.0, 0.0, 10.0))
    world.apply_force("b", (0.0, 0.0, 5.0))
    world.step(0.1)

    assert world.ground_truth().bodies[0].net_force[2] == pytest.approx(15.0)

    world.step(0.1)
    assert world.ground_truth().bodies[0].net_force[2] == pytest.approx(0.0)


def test_determinism():
    def run():
        auth = MuJoCoPhysicsAuthority()
        auth.add_body(
            BodySpec(body_id="b", position=(1.0, 5.0, -2.0), velocity=(0.5, 0.0, 0.0))
        )
        auth.apply_force("b", (2.0, 0.0, 0.0))
        for _ in range(50):
            auth.step(0.1)
        return auth.ground_truth()

    assert run() == run()


def test_clock_advances_by_dt_regardless_of_substepping(world):
    world.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))

    world.step(0.1)   # subdivided internally
    world.step(0.001)  # single substep

    assert world.step_count == 2
    assert world.time == pytest.approx(0.101)


# ---------------------------------------------------------------------------
# The seam invariant
# ---------------------------------------------------------------------------


def test_observation_excludes_invisible_bodies(world):
    world.add_body(BodySpec(body_id="seen", position=(0.0, 1.0, 0.0), visible=True))
    world.add_body(BodySpec(body_id="hidden", position=(0.0, 1.0, 0.0), visible=False))

    assert {b.body_id for b in world.observe().bodies} == {"seen"}
    assert {b.body_id for b in world.ground_truth().bodies} == {"seen", "hidden"}


def test_render_state_excludes_invisible_bodies(world):
    world.add_body(BodySpec(body_id="seen", position=(0.0, 1.0, 0.0), visible=True))
    world.add_body(BodySpec(body_id="hidden", position=(0.0, 1.0, 0.0), visible=False))

    assert {b.body_id for b in world.render_state().bodies} == {"seen"}


def test_observation_cannot_carry_hidden_fields(world):
    world.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0), mass=7.0))
    world.step(0.1)

    ob = world.observe().bodies[0]
    assert isinstance(ob, BodyState)
    assert not hasattr(ob, "mass")
    assert not hasattr(ob, "net_force")

    gt = world.ground_truth().bodies[0]
    assert gt.mass == 7.0


def test_observation_and_ground_truth_agree_on_kinematics(world):
    world.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0)))
    for _ in range(7):
        world.step(0.1)

    ob = world.observe().bodies[0]
    gt = next(b for b in world.ground_truth().bodies if b.body_id == "b")

    assert ob.position == gt.position
    assert ob.velocity == gt.velocity


# ---------------------------------------------------------------------------
# Model rebuild (MuJoCo compiles a fixed model; the seam allows mid-run edits)
# ---------------------------------------------------------------------------


def test_adding_a_body_mid_run_preserves_the_others_state(world):
    world.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))
    for _ in range(5):
        world.step(0.1)
    before = world.observe().bodies[0]

    world.add_body(BodySpec(body_id="late", position=(2.0, 1.0, 0.0)))
    after = next(b for b in world.observe().bodies if b.body_id == "ball")

    assert after.position == before.position
    assert after.velocity == before.velocity


def test_removing_a_body_mid_run_preserves_the_others_state(world):
    world.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))
    world.add_body(BodySpec(body_id="doomed", position=(5.0, 5.0, 0.0)))
    for _ in range(5):
        world.step(0.1)
    before = next(b for b in world.observe().bodies if b.body_id == "ball")

    world.remove_body("doomed")
    after = next(b for b in world.observe().bodies if b.body_id == "ball")

    assert world.body_ids == ("ball",)
    assert after.position == before.position


def test_body_added_mid_run_continues_to_fall(world):
    world.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))
    world.step(0.1)
    world.add_body(BodySpec(body_id="late", position=(2.0, 9.0, 0.0)))

    start = next(b for b in world.observe().bodies if b.body_id == "late").position[1]
    for _ in range(5):
        world.step(0.1)
    end = next(b for b in world.observe().bodies if b.body_id == "late").position[1]

    assert end < start


def test_empty_world_still_has_a_clock(world):
    world.step(0.1)

    assert world.step_count == 1
    assert world.time == pytest.approx(0.1)
    assert world.observe().bodies == ()


# ---------------------------------------------------------------------------
# Agreement with the reference backend
# ---------------------------------------------------------------------------


def _both():
    return [ReferencePhysicsAuthority(), MuJoCoPhysicsAuthority()]


@pytest.mark.parametrize("auth", _both(), ids=["reference", "mujoco"])
def test_both_backends_fall_and_come_to_rest(auth):
    auth.add_body(BodySpec(body_id="ball", position=(0.0, 5.0, 0.0)))
    gt = _settled(auth)

    assert gt.resting is True
    assert gt.velocity[1] == pytest.approx(0.0, abs=1e-3)
    # Both rest on the ground; where exactly depends on whether a body is a
    # point or a sphere, which is the documented divergence.
    assert 0.0 <= gt.position[1] <= 0.1


@pytest.mark.parametrize("auth", _both(), ids=["reference", "mujoco"])
def test_both_backends_respond_to_applied_force(auth):
    auth.add_body(BodySpec(body_id="b", position=(0.0, 5.0, 0.0), mass=1.0))
    auth.apply_force("b", (10.0, 0.0, 0.0))
    auth.step(0.1)

    # f=10N on 1kg for 0.1s -> +1 m/s in x, and x is free of gravity.
    assert auth.observe().bodies[0].velocity[0] == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize("auth", _both(), ids=["reference", "mujoco"])
def test_both_backends_hold_static_bodies_still(auth):
    auth.add_body(BodySpec(body_id="g", position=(0.0, 3.0, 0.0), static=True))
    auth.apply_force("g", (10.0, 10.0, 10.0))
    for _ in range(10):
        auth.step(0.1)

    gt = auth.ground_truth().bodies[0]
    assert gt.position == (0.0, 3.0, 0.0)
    assert gt.velocity == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("auth", _both(), ids=["reference", "mujoco"])
def test_both_backends_agree_on_free_fall_velocity(auth):
    """Away from contact, the two engines should agree closely."""
    auth.add_body(BodySpec(body_id="b", position=(0.0, 50.0, 0.0)))
    for _ in range(5):
        auth.step(0.1)

    assert auth.observe().bodies[0].velocity[1] == pytest.approx(-9.81 * 0.5, rel=1e-2)


@pytest.mark.parametrize("auth", _both(), ids=["reference", "mujoco"])
def test_both_backends_keep_the_hidden_channel_out_of_observation(auth):
    auth.add_body(BodySpec(body_id="seen", position=(0.0, 1.0, 0.0)))
    auth.add_body(BodySpec(body_id="hidden", position=(0.0, 1.0, 0.0), visible=False))

    assert {b.body_id for b in auth.observe().bodies} == {"seen"}
    assert not hasattr(auth.observe().bodies[0], "mass")


# ---------------------------------------------------------------------------
# Integration with the perception and actuation adapters
# ---------------------------------------------------------------------------


def test_the_full_loop_runs_on_mujoco():
    """Perception and actuation are backend-agnostic; prove it on the real engine."""
    from sanctuary.motor.physics_actuation import MotorCommand, PhysicsActuator
    from sanctuary.sensorium.physics_percepts import observation_to_percepts

    world = MuJoCoPhysicsAuthority()
    world.add_body(BodySpec(body_id="rover", position=(0.0, 1.0, 0.0)))
    world.add_body(BodySpec(body_id="landmark", position=(10.0, 1.0, 0.0), static=True))

    class _Decoder:
        def decode(self, selection):  # noqa: ANN001
            return MotorCommand("rover", (50.0, 0.0, 0.0))

    def distance() -> float:
        percepts = observation_to_percepts(world.observe(), self_id="rover")
        landmark = next(p for p in percepts if "landmark" in p.content)
        return float(landmark.content.split(" m away")[0].split()[-1])

    before = distance()

    actuator = PhysicsActuator(world, _Decoder(), max_force=100.0)
    for _ in range(5):
        actuator.apply(None)
        world.step(0.05)

    assert distance() < before
