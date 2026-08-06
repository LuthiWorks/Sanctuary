"""The episode runner: one cycle of perceive -> decide -> act -> consequence.

Order of operations, which is the whole point of this module
------------------------------------------------------------

Each :meth:`EpisodeRunner.step` does exactly this, and the order is load-bearing:

1. **Perceive** the world as it is now -> ``percepts``, and ``s_t`` if a state
   encoder is wired.
2. **Decide** -- the policy sees those percepts and returns an action.
3. **Act** -- the action is decoded to a force and applied. No time has passed
   yet; forces accumulate on the body.
4. **Step** the world by ``dt``. This is the only place time moves.
5. **Perceive the consequence** -> ``s_next``.
6. **Learn** -- ``observe_transition(s_t, a_t, s_next, ctx)``, if a sink is
   wired.

Step 1 must precede step 3, or the policy is choosing an action in response to a
world its own action already changed. Step 5 must follow step 4, or the
"consequence" is just the observation again and every transition teaches that
actions do nothing. Both mistakes produce a loop that runs perfectly and trains
on nonsense, so they are asserted in the tests rather than left to review.

What this module deliberately does not do
-----------------------------------------

It does not supply a policy. A policy is either a trained model or a person
driving; inventing one here would be the scaffold authoring the entity's
behaviour. :class:`RestingPolicy` exists as an explicit null -- a body that
chooses stillness -- not as a default that pretends to be a decision.

Both the policy and the transition sink are structurally typed so this runs
without ``luthi`` installed, which is the environment the world is developed in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from sanctuary.core.schema import Percept
from sanctuary.motor.physics_actuation import AppliedAction, MotorCommand, PhysicsActuator
from sanctuary.physics.authority import PhysicsAuthority
from sanctuary.physics.state import PhysicsObservation
from sanctuary.sensorium.physics_percepts import observation_to_percepts

__all__ = [
    "EpisodeRunner",
    "Policy",
    "RestingPolicy",
    "StepRecord",
    "DEFAULT_DT",
]

#: Seconds of simulated time per cycle. 50 Hz -- fast enough that contact looks
#: continuous, slow enough that a 10 Hz cognitive loop can keep up at 5:1.
DEFAULT_DT: float = 0.02


@runtime_checkable
class Policy(Protocol):
    """Chooses an action from what the body currently perceives.

    Structurally typed against anything that returns an
    ``ActionSelection``-shaped object (or a bare action vector) -- a trained M9
    actor, a scripted probe, or a human at a keyboard.
    """

    def __call__(self, percepts: list[Percept], observation: PhysicsObservation) -> Any:
        ...


class RestingPolicy:
    """Chooses stillness, explicitly.

    Not a placeholder for a real policy and not a no-op: M9 treats rest as a
    computed action (``a_rest(s_t)``), so a body that rests is *doing
    something*. This returns a zero-intensity command, which the actuator
    records as ``at_rest=True`` rather than as a failed decode.
    """

    def __init__(self, body_id: str) -> None:
        self.body_id = body_id

    def __call__(self, percepts, observation) -> MotorCommand:  # noqa: ANN001
        return MotorCommand(self.body_id, (0.0, 0.0, 0.0), intensity=0.0)


@dataclass
class StepRecord:
    """What happened in one cycle. The unit an episode is made of."""

    index: int
    time: float
    percepts: list[Percept] = field(default_factory=list)
    applied: AppliedAction | None = None
    learned: dict[str, float] = field(default_factory=dict)

    @property
    def at_rest(self) -> bool:
        return bool(self.applied and self.applied.at_rest)


class EpisodeRunner:
    """Drives a body through a world, one cycle at a time.

    Args:
        world: The physics authority. Any backend.
        actuator: Applies decoded commands. Its decoder converts the policy's
            output; pass a policy that already returns :class:`MotorCommand` and
            the actuator's decoder is bypassed via ``apply_command``.
        self_id: The body being driven -- the one perception is egocentric to.
        policy: Chooses actions. Required; see the module docstring.
        dt: Simulated seconds per cycle.
        max_range / max_bodies: Perceptual acuity, passed to
            ``observation_to_percepts``.
        sensorium: Optional object with ``inject_percept``; percepts are fed to
            it as well as recorded.
        state_encoder: Optional ``PhysicsObservation -> s_t``. Required for
            learning; without it transitions cannot be formed and the runner
            says so rather than passing ``None`` into the sink.
        sink: Optional object with ``observe_transition(s_t, a_t, s_next, ctx)``.
        realtime: Sleep so wall-clock tracks simulated time. Off by default --
            tests and training want to run flat out; a viewer wants this on.
        on_step: Optional callback after each cycle, for viewer sync and logging.
    """

    def __init__(
        self,
        world: PhysicsAuthority,
        actuator: PhysicsActuator,
        *,
        self_id: str,
        policy: Policy,
        dt: float = DEFAULT_DT,
        max_range: float | None = None,
        max_bodies: int | None = None,
        sensorium: Any = None,
        state_encoder: Callable[[PhysicsObservation], Any] | None = None,
        sink: Any = None,
        realtime: bool = False,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if sink is not None and state_encoder is None:
            raise ValueError(
                "A transition sink was given without a state_encoder. Learning "
                "needs s_t and s_next; there is no way to form a transition "
                "without one, and passing None into observe_transition would "
                "train on nothing while appearing to work."
            )

        self.world = world
        self.actuator = actuator
        self.self_id = self_id
        self.policy = policy
        self.dt = dt
        self.max_range = max_range
        self.max_bodies = max_bodies
        self.sensorium = sensorium
        self.state_encoder = state_encoder
        self.sink = sink
        self.realtime = realtime
        self.on_step = on_step

        self._index = 0
        self._stop = False

    # -- lifecycle ----------------------------------------------------------

    @property
    def index(self) -> int:
        """Cycles completed since construction (or the last :meth:`reset`)."""
        return self._index

    def reset(self) -> None:
        """Reset the cycle counter. The *world* is the caller's to rebuild.

        Deliberately does not touch the world: an episode boundary and a world
        rebuild are different decisions, and a runner that silently cleared the
        world would make "start a new episode" destroy a scene the caller had
        set up.
        """
        self._index = 0
        self._stop = False

    def stop(self) -> None:
        """Ask :meth:`run` to finish after the current cycle."""
        self._stop = True

    # -- the loop -----------------------------------------------------------

    def perceive(self) -> tuple[list[Percept], PhysicsObservation]:
        """Percepts for the world as it is right now."""
        observation = self.world.observe()
        percepts = observation_to_percepts(
            observation,
            self_id=self.self_id,
            max_range=self.max_range,
            max_bodies=self.max_bodies,
        )
        if self.sensorium is not None:
            for percept in percepts:
                self.sensorium.inject_percept(percept)
        return percepts, observation

    def step(self) -> StepRecord:
        """One full cycle. See the module docstring for why the order matters."""
        started = time.perf_counter()

        # 1. Perceive the world as it is, before acting on it.
        percepts, observation = self.perceive()
        s_t = self.state_encoder(observation) if self.state_encoder else None

        # 2. Decide.
        chosen = self.policy(percepts, observation)

        # 3. Act. No time has passed; forces accumulate on the body.
        if isinstance(chosen, MotorCommand):
            applied = self.actuator.apply_command(chosen)
        else:
            applied = self.actuator.apply(chosen)

        # 4. Advance the world. The only place time moves.
        self.world.step(self.dt)

        # 5. Perceive the consequence.
        next_observation = self.world.observe()
        s_next = self.state_encoder(next_observation) if self.state_encoder else None

        # 6. Learn from it.
        learned: dict[str, float] = {}
        if self.sink is not None:
            learned = (
                self.sink.observe_transition(
                    s_t,
                    applied.command.force,
                    s_next,
                    {
                        "time_since_emission": self.dt,
                        "at_rest": applied.at_rest,
                    },
                )
                or {}
            )

        record = StepRecord(
            index=self._index,
            time=self.world.time,
            percepts=percepts,
            applied=applied,
            learned=learned,
        )
        self._index += 1

        if self.on_step is not None:
            self.on_step(record)

        if self.realtime:
            self._pace(started)

        return record

    def run(
        self,
        steps: int | None = None,
        *,
        until: Callable[[StepRecord], bool] | None = None,
    ) -> list[StepRecord]:
        """Run until ``steps`` cycles, ``until`` returns True, or :meth:`stop`.

        At least one of ``steps`` or ``until`` must bound the run -- an
        unbounded loop with no exit is a hang, not an episode.
        """
        if steps is None and until is None:
            raise ValueError(
                "run() needs a bound: pass steps=, until=, or call stop() from "
                "on_step. An unbounded run with no exit condition is a hang."
            )
        if steps is not None and steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")

        records: list[StepRecord] = []
        while not self._stop:
            if steps is not None and len(records) >= steps:
                break
            record = self.step()
            records.append(record)
            if until is not None and until(record):
                break
        return records

    def _pace(self, started: float) -> None:
        """Sleep so wall-clock tracks simulated time.

        If a cycle already took longer than ``dt``, do not sleep -- and do not
        try to catch up either. Silently running faster than real time in front
        of someone watching would misrepresent how fast the entity is actually
        living.
        """
        elapsed = time.perf_counter() - started
        remaining = self.dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
