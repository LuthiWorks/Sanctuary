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

    @abstractmethod
    def set_visible(self, body_id: str, visible: bool) -> None:
        """Show or hide a body. Raise ``KeyError`` if it does not exist.

        Visibility is a property of the three *views*, not of the physics: a
        hidden body is absent from :meth:`observe` and :meth:`render_state` and
        still present in :meth:`ground_truth`, because the instrumentation
        channel reports what is true rather than what is seen. Its dynamics are
        unchanged — hiding a body does not make it stop falling or stop
        colliding.

        **Why this exists as a seam operation.** A backend may compile its
        world (MuJoCo does), so ``add_body``/``remove_body`` can force a model
        rebuild. Measured on 2026-08-17, one such rebuild mid-run costs ~11 ms
        and additionally tears down the renderer, whose next frame then costs
        ~29 ms instead of ~0.7 ms — a ~40 ms stall, 40% of a cognitive cycle,
        landing exactly when something appeared or vanished. With a viewer
        open, the rebuild raises instead.

        So a system with a changing population — creatures being born and
        dying — must **pre-allocate its bodies and toggle visibility** rather
        than add and remove them. This method is that operation, and it must
        never dirty a compiled model.
        """

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
