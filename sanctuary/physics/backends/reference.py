"""A dependency-free reference implementation of the physics-authority seam.

Point-mass bodies under constant gravity with a ground plane at Y=0 and
semi-implicit Euler integration. This is **not** the real engine — it exists to:

1. prove the seam carries lawful action -> consequence (bodies fall, rest, and
   respond to applied force) without pulling in MuJoCo,
2. give tests and CI a deterministic backend with no native dependency,
3. be the thing a MuJoCo backend is diffed against for the seam's contract.

Deliberately simple: no rotation, no body-body collision, no friction. Those
belong to the real engine. What matters here is the *boundary* behaving
correctly — especially that the model-facing observation and the
instrumentation-only ground truth stay separated.
"""

from __future__ import annotations

from sanctuary.physics.authority import PhysicsAuthority
from sanctuary.physics.state import (
    BodySpec,
    BodyState,
    GroundTruthBody,
    PhysicsGroundTruth,
    PhysicsObservation,
    RenderBody,
    RenderFrame,
    Vec3,
)

GRAVITY: float = -9.81   # m/s^2 on the Y axis
GROUND_Y: float = 0.0    # ground plane
_REST_EPS: float = 1e-9  # |vertical velocity| below this on the ground = resting


class _Body:
    """Mutable internal body. The public state types are frozen; this is not."""

    __slots__ = (
        "position", "velocity", "mass", "static", "visible",
        "pending_force", "last_net_force", "resting",
    )

    def __init__(self, spec: BodySpec) -> None:
        self.position: list[float] = list(spec.position)
        self.velocity: list[float] = list(spec.velocity)
        self.mass: float = spec.mass
        self.static: bool = spec.static
        self.visible: bool = spec.visible
        self.pending_force: list[float] = [0.0, 0.0, 0.0]
        self.last_net_force: Vec3 = (0.0, 0.0, 0.0)
        self.resting: bool = False


class ReferencePhysicsAuthority(PhysicsAuthority):
    """Deterministic point-mass reference backend. See module docstring."""

    def __init__(self) -> None:
        self._bodies: dict[str, _Body] = {}
        self._time: float = 0.0
        self._step: int = 0

    # -- world construction -------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        # Deterministic backend: seed is part of the contract but unused here.
        self._bodies.clear()
        self._time = 0.0
        self._step = 0

    def add_body(self, spec: BodySpec) -> str:
        if spec.body_id in self._bodies:
            raise ValueError(f"Body id already exists: {spec.body_id!r}")
        if spec.mass <= 0.0:
            raise ValueError(f"Body mass must be positive, got {spec.mass}")
        self._bodies[spec.body_id] = _Body(spec)
        return spec.body_id

    def remove_body(self, body_id: str) -> None:
        # Raises KeyError if absent — fail loud, don't silently ignore.
        del self._bodies[body_id]

    # -- actuation + time ---------------------------------------------------

    def apply_force(self, body_id: str, force: Vec3) -> None:
        body = self._bodies[body_id]  # KeyError if unknown — intended
        body.pending_force[0] += force[0]
        body.pending_force[1] += force[1]
        body.pending_force[2] += force[2]

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        for body in self._bodies.values():
            if body.static:
                body.last_net_force = (0.0, 0.0, 0.0)
                body.resting = True
                body.pending_force = [0.0, 0.0, 0.0]
                continue

            fx, fy, fz = body.pending_force
            # True force acting this step: applied force + gravity.
            net_fx = fx
            net_fy = fy + GRAVITY * body.mass
            net_fz = fz

            # Semi-implicit Euler: update velocity, then position.
            body.velocity[0] += (net_fx / body.mass) * dt
            body.velocity[1] += (net_fy / body.mass) * dt
            body.velocity[2] += (net_fz / body.mass) * dt
            body.position[0] += body.velocity[0] * dt
            body.position[1] += body.velocity[1] * dt
            body.position[2] += body.velocity[2] * dt

            # Ground contact: don't fall through the plane.
            resting = False
            if body.position[1] < GROUND_Y:
                body.position[1] = GROUND_Y
                if body.velocity[1] < 0.0:
                    body.velocity[1] = 0.0
            if body.position[1] <= GROUND_Y and abs(body.velocity[1]) < _REST_EPS:
                resting = True
                # The normal force cancels the vertical net force at rest.
                net_fy = 0.0

            body.resting = resting
            body.last_net_force = (net_fx, net_fy, net_fz)
            body.pending_force = [0.0, 0.0, 0.0]

        self._time += dt
        self._step += 1

    # -- the three views ----------------------------------------------------

    def observe(self) -> PhysicsObservation:
        bodies = tuple(
            BodyState(
                body_id=bid,
                position=tuple(b.position),
                velocity=tuple(b.velocity),
            )
            for bid, b in self._bodies.items()
            if b.visible
        )
        return PhysicsObservation(time=self._time, step=self._step, bodies=bodies)

    def ground_truth(self) -> PhysicsGroundTruth:
        bodies = tuple(
            GroundTruthBody(
                body_id=bid,
                position=tuple(b.position),
                velocity=tuple(b.velocity),
                mass=b.mass,
                net_force=b.last_net_force,
                resting=b.resting,
                static=b.static,
                visible=b.visible,
            )
            for bid, b in self._bodies.items()
        )
        return PhysicsGroundTruth(time=self._time, step=self._step, bodies=bodies)

    def render_state(self) -> RenderFrame:
        bodies = tuple(
            RenderBody(body_id=bid, position=tuple(b.position))
            for bid, b in self._bodies.items()
            if b.visible
        )
        return RenderFrame(time=self._time, step=self._step, bodies=bodies)

    # -- clocks -------------------------------------------------------------

    @property
    def time(self) -> float:
        return self._time

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def body_ids(self) -> tuple[str, ...]:
        return tuple(self._bodies.keys())
