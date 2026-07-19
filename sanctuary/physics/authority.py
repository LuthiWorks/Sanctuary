"""The physics-authority seam: the contract every physics backend implements.

This is the swappable boundary from
``docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md``. A backend owns the
world's true physical state and advances it; the rest of the system only ever
talks to it through the three view methods (:meth:`observe`,
:meth:`ground_truth`, :meth:`render_state`), never by reaching into the engine.

Backends: :class:`sanctuary.physics.backends.reference.ReferencePhysicsAuthority`
now (dependency-free); a MuJoCo backend behind this same interface later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sanctuary.physics.state import (
    BodySpec,
    PhysicsGroundTruth,
    PhysicsObservation,
    RenderFrame,
    Vec3,
)


class PhysicsAuthority(ABC):
    """The single source of physical truth for the developmental world.

    Lifecycle: construct -> :meth:`add_body` (populate) -> repeat
    (:meth:`apply_force`, :meth:`step`, read views) -> :meth:`reset` to start
    over. All spatial quantities are SI (metres, m/s, newtons); gravity acts on
    -Y with the ground plane at Y=0.

    Implementations must fail loud, not silently: adding a duplicate body id, or
    forcing/removing an unknown body id, raises rather than being ignored.
    """

    # -- world construction -------------------------------------------------

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Clear the world back to empty, time 0, step 0.

        ``seed`` seeds any stochastic backend; deterministic backends ignore it
        but must accept it so the contract is uniform across engines.
        """

    @abstractmethod
    def add_body(self, spec: BodySpec) -> str:
        """Add a body and return its id. Raise ``ValueError`` on duplicate id."""

    @abstractmethod
    def remove_body(self, body_id: str) -> None:
        """Remove a body. Raise ``KeyError`` if it does not exist."""

    # -- actuation + time ---------------------------------------------------

    @abstractmethod
    def apply_force(self, body_id: str, force: Vec3) -> None:
        """Accumulate a force (newtons) applied to a body during the next
        :meth:`step`. Cleared after that step. Raise ``KeyError`` if unknown.

        This is the actuation channel — the rover pushing the world, or the
        world pushing the rover. A one-step force is the impulse primitive.
        """

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds. Raise ``ValueError`` if
        ``dt <= 0``."""

    # -- the three views ----------------------------------------------------

    @abstractmethod
    def observe(self) -> PhysicsObservation:
        """The **model-facing** view: visible bodies, kinematics only.

        This is the only world data that may become percepts. It never carries
        hidden state (mass, forces) or invisible bodies — the return type
        cannot represent them.
        """

    @abstractmethod
    def ground_truth(self) -> PhysicsGroundTruth:
        """The **instrumentation-only** view: the full true state, all bodies,
        including hidden fields. **Never route this to the model.**"""

    @abstractmethod
    def render_state(self) -> RenderFrame:
        """The **renderer** view: poses of the visible bodies, for Godot."""

    # -- clocks -------------------------------------------------------------

    @property
    @abstractmethod
    def time(self) -> float:
        """Simulated seconds elapsed since the last :meth:`reset`."""

    @property
    @abstractmethod
    def step_count(self) -> int:
        """Number of :meth:`step` calls since the last :meth:`reset`."""

    @property
    @abstractmethod
    def body_ids(self) -> tuple[str, ...]:
        """Ids of all bodies currently in the world (visible and hidden)."""
