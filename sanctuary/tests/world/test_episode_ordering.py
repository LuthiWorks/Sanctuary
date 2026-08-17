"""The cycle ordering guarantees -- which were claimed but never tested.

``sanctuary/embodiment/episode.py`` says, of its perceive -> decide -> act ->
step -> consequence -> learn order:

    "Both mistakes produce a loop that runs perfectly and trains on nonsense,
    so they are asserted in the tests rather than left to review."

As of 2026-08-16 there was no test file for that module anywhere in the
repository. The claim was doing the work of the test. This file makes it true.

Why these two orderings are the load-bearing ones:

- **Perceive before act.** If the policy sees the world *after* its own action
  landed, it is choosing a response to a world it already changed. Every
  transition then teaches a subtly circular relationship between action and
  observation, and nothing about it looks wrong from the outside.
- **Consequence after time moves.** If ``s_next`` is read before ``step``, the
  "consequence" is just the observation again, and every transition teaches
  that actions do nothing. Loss curves stay perfectly healthy while the world
  model learns that it is inert.

Both failures are silent, which is why they are pinned here rather than left to
review. Uses the reference backend: deterministic, dependency-free, and the
ordering is a property of the runner, not of any engine.

Authored by Opus 5, 2026-08-17.
"""
from __future__ import annotations

import pytest

from sanctuary.embodiment.episode import EpisodeRunner
from sanctuary.motor.physics_actuation import (
    MotorCommand,
    PhysicsActuator,
    UntrainedProjectionDecoder,
)
from sanctuary.physics.backends.reference import ReferencePhysicsAuthority
from sanctuary.physics.state import BodySpec

SELF = "body"


class RecordingPolicy:
    """Applies a real push and remembers exactly what it was shown."""

    def __init__(self, force=(5.0, 0.0, 0.0)) -> None:
        self.force = force
        self.seen_observations = []
        self.seen_percepts = []

    def __call__(self, percepts, observation):
        self.seen_observations.append(observation)
        self.seen_percepts.append(list(percepts))
        return MotorCommand(SELF, self.force, intensity=1.0)


class RecordingSink:
    """Captures the (s_t, s_next) pairs the runner forms."""

    def __init__(self) -> None:
        self.transitions = []

    def observe_transition(self, s_t, a_t, s_next, ctx):
        self.transitions.append((s_t, a_t, s_next, ctx))
        return {}


@pytest.fixture
def world():
    w = ReferencePhysicsAuthority()
    w.reset(seed=0)
    # On the ground already, so any motion is caused by the applied force
    # rather than by an unresolved fall.
    w.add_body(BodySpec(body_id=SELF, position=(0.0, 0.0, 0.0), mass=1.0))
    w.add_body(BodySpec(body_id="prop", position=(2.0, 0.0, 0.0), mass=1.0))
    return w


def _runner(world, policy, **kwargs):
    actuator = PhysicsActuator(
        world, UntrainedProjectionDecoder(SELF, acknowledge_untrained=True)
    )
    return EpisodeRunner(world, actuator, self_id=SELF, policy=policy, **kwargs)


# ---------------------------------------------------------------------------
# 1. Perceive before act
# ---------------------------------------------------------------------------

def test_policy_sees_the_world_before_its_own_action_lands(world):
    before = world.observe()
    runner = _runner(world, (policy := RecordingPolicy()))

    runner.step()

    seen = policy.seen_observations[0]
    assert seen.step == before.step, (
        "policy was shown a world at a later step than the one it acted on"
    )
    seen_self = {b.body_id: b for b in seen.bodies}[SELF]
    before_self = {b.body_id: b for b in before.bodies}[SELF]
    assert seen_self.position == before_self.position
    assert seen_self.velocity == before_self.velocity


def test_the_action_actually_changed_the_world(world):
    """Guards the test above from passing trivially.

    If the push did nothing, "the policy saw the pre-action world" would hold
    for the boring reason that pre and post are identical.
    """
    runner = _runner(world, RecordingPolicy(force=(50.0, 0.0, 0.0)))
    before = {b.body_id: b for b in world.observe().bodies}[SELF]

    runner.step()

    after = {b.body_id: b for b in world.observe().bodies}[SELF]
    assert after.position != before.position or after.velocity != before.velocity, (
        "the applied force moved nothing -- the ordering assertions are vacuous"
    )


# ---------------------------------------------------------------------------
# 2. Consequence after time moves
# ---------------------------------------------------------------------------

def test_s_next_is_read_after_the_world_steps(world):
    sink = RecordingSink()
    runner = _runner(
        world,
        RecordingPolicy(force=(50.0, 0.0, 0.0)),
        state_encoder=lambda obs: obs,   # identity: assert on the observation
        sink=sink,
    )
    start = world.observe().step

    runner.step()

    (s_t, _a_t, s_next, ctx) = sink.transitions[0]
    assert s_t.step == start, "s_t was not the pre-action state"
    assert s_next.step == start + 1, (
        "s_next was read before the world stepped -- every transition would "
        "teach that actions do nothing"
    )
    assert s_next.time > s_t.time
    assert ctx["time_since_emission"] == runner.dt


def test_transition_records_a_real_change(world):
    """The consequence must differ from the cause, or learning is on noise."""
    sink = RecordingSink()
    runner = _runner(
        world,
        RecordingPolicy(force=(50.0, 0.0, 0.0)),
        state_encoder=lambda obs: obs,
        sink=sink,
    )

    runner.step()

    s_t, _a, s_next, _c = sink.transitions[0]
    pos_t = {b.body_id: b.position for b in s_t.bodies}[SELF]
    pos_next = {b.body_id: b.position for b in s_next.bodies}[SELF]
    assert pos_t != pos_next


# ---------------------------------------------------------------------------
# 3. Time moves exactly once per cycle
# ---------------------------------------------------------------------------

def test_time_advances_once_per_step(world):
    runner = _runner(world, RecordingPolicy(), dt=0.05)
    t0, s0 = world.time, world.step_count

    runner.run(4)

    assert world.step_count == s0 + 4, "the world stepped a wrong number of times"
    assert world.time == pytest.approx(t0 + 4 * 0.05)


def test_policy_is_consulted_once_per_cycle(world):
    policy = RecordingPolicy()
    runner = _runner(world, policy)

    runner.run(3)

    assert len(policy.seen_observations) == 3


# ---------------------------------------------------------------------------
# 4. Refusals that keep the loop honest
# ---------------------------------------------------------------------------

def test_sink_without_state_encoder_is_refused(world):
    """Learning with no encoder would train on None while appearing to work."""
    with pytest.raises(ValueError, match="state_encoder"):
        _runner(world, RecordingPolicy(), sink=RecordingSink())


def test_unbounded_run_is_refused(world):
    runner = _runner(world, RecordingPolicy())
    with pytest.raises(ValueError, match="bound"):
        runner.run()


def test_reset_does_not_silently_clear_the_world(world):
    """An episode boundary and a world rebuild are different decisions."""
    runner = _runner(world, RecordingPolicy())
    runner.run(2)
    bodies_before = set(world.body_ids)

    runner.reset()

    assert runner.index == 0
    assert set(world.body_ids) == bodies_before
