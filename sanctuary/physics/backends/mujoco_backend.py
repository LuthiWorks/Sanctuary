"""A MuJoCo backend behind the physics-authority seam.

The engine the developmental world is meant to run on
(``docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md``): real rigid-body
dynamics with contact, so the world pushes back on a body the way a world does.
It implements exactly the same :class:`~sanctuary.physics.authority.PhysicsAuthority`
contract as :class:`~sanctuary.physics.backends.reference.ReferencePhysicsAuthority`,
which is what makes the seam worth having -- nothing above the seam changes.

``mujoco`` is an **optional** dependency (``uv sync --extra mujoco``). It is
imported lazily inside :meth:`MuJoCoPhysicsAuthority.__init__` so that importing
``sanctuary.physics`` keeps working in an environment without it -- the same
posture ``core/luthi_model.py`` takes toward ``luthi``. CI and the world's own
development run fine on the reference backend.

Three places this backend genuinely differs from the reference, stated plainly
rather than papered over
------------------------------------------------------------------------------

1. **Bodies have volume.** The reference is point-mass: a body rests with its
   centre at ``y = 0``. A MuJoCo body is a sphere of :attr:`body_radius`, so it
   rests with its centre at ``y = body_radius``. This is not an incompatibility
   to fix; it is the difference between a point and a thing, and it is most of
   the reason for moving to MuJoCo at all. Tests that assert an exact resting
   height must ask the backend for its radius rather than assume zero.

2. **Contacts are compliant.** MuJoCo's solver settles a resting body to a small
   penetration rather than to an exact plane, so resting height and velocity
   converge to a *tolerance*, not to a float-exact value. Assertions written
   against the reference's ``approx(0.0)`` default tolerance will not transfer,
   and loosening them for MuJoCo is correct rather than a fudge.

3. **The model is compiled, not mutable.** MuJoCo builds a fixed ``mjModel``;
   bodies cannot be added to a live one. :meth:`add_body` and
   :meth:`remove_body` therefore mark the model dirty and it is rebuilt on next
   use, with the kinematic state of surviving bodies carried across. The
   seam's documented lifecycle (construct -> populate -> step) keeps this off
   the hot path, but mid-run edits are supported because the reference allows
   them and the seam must not quietly mean two different things.

Axis convention
---------------

The seam fixes gravity on **-Y** with the ground plane at ``Y = 0``. MuJoCo's
convention is Z-up. The generated model therefore sets
``<option gravity="0 -9.81 0">`` and orients the floor plane with
``zaxis="0 1 0"`` so its normal points along +Y. Every quantity crossing the
seam is in the seam's frame; no transposition happens above this file.

Integration
-----------

The seam's :meth:`step` takes an arbitrary ``dt``, but MuJoCo integrates
accurately only at small timesteps -- driving it at ``dt=0.1`` directly would
resolve contacts badly enough to be a different world. So a ``step(dt)``
subdivides into ``ceil(dt / max_substep)`` equal internal steps summing to
exactly ``dt``. The seam's clock advances by ``dt`` once per call regardless, so
``time`` and ``step_count`` mean the same thing on both backends.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import quoteattr

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

__all__ = ["MuJoCoPhysicsAuthority", "DEFAULT_BODY_RADIUS", "DEFAULT_MAX_SUBSTEP"]

GRAVITY: float = -9.81
GROUND_Y: float = 0.0

#: Metres. Every body is a sphere of this radius. Per-body geometry belongs on
#: ``BodySpec`` when the world needs varied shapes; that is a seam change and
#: deserves its own decision, so it is deliberately not made here.
DEFAULT_BODY_RADIUS: float = 0.05

#: Seconds. Largest internal integration step. MuJoCo's own default is 0.002;
#: this is the ceiling a seam-level ``dt`` is subdivided to reach.
DEFAULT_MAX_SUBSTEP: float = 0.005

#: |velocity| below which a body in contact counts as resting. Looser than the
#: reference's 1e-9 because compliant contact converges to a tolerance.
_REST_VELOCITY_EPS: float = 1e-3


class MuJoCoPhysicsAuthority(PhysicsAuthority):
    """Rigid-body backend. See the module docstring for the divergences."""

    def __init__(
        self,
        *,
        body_radius: float = DEFAULT_BODY_RADIUS,
        max_substep: float = DEFAULT_MAX_SUBSTEP,
        ground_size: float = 100.0,
    ) -> None:
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised by env, not CI
            raise ImportError(
                "MuJoCoPhysicsAuthority requires the optional 'mujoco' "
                "dependency. Install it with `uv sync --extra mujoco`, or use "
                "ReferencePhysicsAuthority, which implements the same seam with "
                "no native dependency."
            ) from exc

        if body_radius <= 0.0:
            raise ValueError(f"body_radius must be positive, got {body_radius}")
        if max_substep <= 0.0:
            raise ValueError(f"max_substep must be positive, got {max_substep}")

        self._mujoco = mujoco
        self.body_radius = body_radius
        self.max_substep = max_substep
        self.ground_size = ground_size

        self._specs: dict[str, BodySpec] = {}
        self._model = None
        self._data = None
        self._dirty = True

        self._time: float = 0.0
        self._step: int = 0

        self._pending_force: dict[str, list[float]] = {}
        self._net_force: dict[str, Vec3] = {}
        self._resting: dict[str, bool] = {}

        # Kinematic state carried across a model rebuild.
        self._carried: dict[str, tuple[Vec3, Vec3]] = {}

    # -- world construction -------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        # MuJoCo is deterministic here (no stochastic elements are configured),
        # so seed is accepted for contract uniformity and unused -- same as the
        # reference backend.
        self._specs.clear()
        self._pending_force.clear()
        self._net_force.clear()
        self._resting.clear()
        self._carried.clear()
        self._model = None
        self._data = None
        self._dirty = True
        self._time = 0.0
        self._step = 0

    def add_body(self, spec: BodySpec) -> str:
        if spec.body_id in self._specs:
            raise ValueError(f"Body id already exists: {spec.body_id!r}")
        if spec.mass <= 0.0:
            raise ValueError(f"Body mass must be positive, got {spec.mass}")

        self._capture_state()
        self._specs[spec.body_id] = spec
        self._carried[spec.body_id] = (spec.position, spec.velocity)
        self._pending_force[spec.body_id] = [0.0, 0.0, 0.0]
        self._net_force[spec.body_id] = (0.0, 0.0, 0.0)
        self._resting[spec.body_id] = False
        self._dirty = True
        return spec.body_id

    def remove_body(self, body_id: str) -> None:
        if body_id not in self._specs:
            raise KeyError(f"Unknown body id: {body_id!r}")

        self._capture_state()
        del self._specs[body_id]
        self._carried.pop(body_id, None)
        self._pending_force.pop(body_id, None)
        self._net_force.pop(body_id, None)
        self._resting.pop(body_id, None)
        self._dirty = True

    # -- actuation + time ---------------------------------------------------

    def apply_force(self, body_id: str, force: Vec3) -> None:
        if body_id not in self._specs:
            raise KeyError(f"Unknown body id: {body_id!r}")
        pending = self._pending_force[body_id]
        pending[0] += force[0]
        pending[1] += force[1]
        pending[2] += force[2]

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self._ensure_built()
        if self._model is None or not self._specs:
            # An empty world still has a clock.
            self._time += dt
            self._step += 1
            self._clear_pending()
            return

        mujoco = self._mujoco
        model, data = self._model, self._data

        substeps = max(1, math.ceil(dt / self.max_substep))
        model.opt.timestep = dt / substeps

        for body_id, force in self._pending_force.items():
            index = self._body_index(body_id)
            data.xfrc_applied[index, 0:3] = force
            data.xfrc_applied[index, 3:6] = 0.0

        for _ in range(substeps):
            mujoco.mj_step(model, data)

        self._record_forces()

        # The seam's clock is authoritative: one call, one step, exactly dt.
        self._time += dt
        self._step += 1

        self._clear_pending()
        data.xfrc_applied[:] = 0.0

    # -- the three views ----------------------------------------------------

    def observe(self) -> PhysicsObservation:
        self._ensure_built()
        bodies = tuple(
            BodyState(
                body_id=body_id,
                position=self._position(body_id),
                velocity=self._velocity(body_id),
            )
            for body_id, spec in self._specs.items()
            if spec.visible
        )
        return PhysicsObservation(time=self._time, step=self._step, bodies=bodies)

    def ground_truth(self) -> PhysicsGroundTruth:
        self._ensure_built()
        bodies = tuple(
            GroundTruthBody(
                body_id=body_id,
                position=self._position(body_id),
                velocity=self._velocity(body_id),
                mass=spec.mass,
                net_force=self._net_force[body_id],
                resting=self._resting[body_id],
                static=spec.static,
                visible=spec.visible,
            )
            for body_id, spec in self._specs.items()
        )
        return PhysicsGroundTruth(time=self._time, step=self._step, bodies=bodies)

    def render_state(self) -> RenderFrame:
        self._ensure_built()
        bodies = tuple(
            RenderBody(body_id=body_id, position=self._position(body_id))
            for body_id, spec in self._specs.items()
            if spec.visible
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
        return tuple(self._specs)

    # -- model construction -------------------------------------------------

    def _ensure_built(self) -> None:
        if not self._dirty:
            return
        self._build()
        self._dirty = False

    def _build(self) -> None:
        mujoco = self._mujoco

        if not self._specs:
            self._model = None
            self._data = None
            return

        self._model = mujoco.MjModel.from_xml_string(self._to_xml())
        self._data = mujoco.MjData(self._model)

        for body_id, (position, velocity) in self._carried.items():
            if body_id not in self._specs or self._specs[body_id].static:
                continue
            joint_index = self._model.body_jntadr[self._body_index(body_id)]
            qpos_adr = self._model.jnt_qposadr[joint_index]
            qvel_adr = self._model.jnt_dofadr[joint_index]
            self._data.qpos[qpos_adr : qpos_adr + 3] = position
            # Identity orientation: this backend does not yet expose rotation
            # across the seam (RenderBody has no quaternion field), so carrying
            # an orientation would be state the seam cannot report.
            self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            self._data.qvel[qvel_adr : qvel_adr + 3] = velocity
            self._data.qvel[qvel_adr + 3 : qvel_adr + 6] = 0.0

        mujoco.mj_forward(self._model, self._data)

    def _to_xml(self) -> str:
        parts = [
            "<mujoco model='sanctuary'>",
            # Seam frame: gravity on -Y, ground plane normal +Y.
            f"<option gravity='0 {GRAVITY} 0' timestep='{self.max_substep}'/>",
            "<worldbody>",
            f"<geom name='floor' type='plane' size='{self.ground_size} "
            f"{self.ground_size} 0.1' zaxis='0 1 0' pos='0 {GROUND_Y} 0'/>",
        ]

        for body_id, spec in self._specs.items():
            name = quoteattr(body_id)
            px, py, pz = spec.position
            parts.append(f"<body name={name} pos='{px} {py} {pz}'>")
            if not spec.static:
                parts.append("<freejoint/>")
            parts.append(
                f"<geom type='sphere' size='{self.body_radius}' "
                f"mass='{spec.mass}'/>"
            )
            parts.append("</body>")

        parts.append("</worldbody>")
        parts.append("</mujoco>")
        return "".join(parts)

    # -- state access -------------------------------------------------------

    def _body_index(self, body_id: str) -> int:
        return self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_BODY, body_id
        )

    def _position(self, body_id: str) -> Vec3:
        if self._model is None:
            return self._carried[body_id][0]
        pos = self._data.xpos[self._body_index(body_id)]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def _velocity(self, body_id: str) -> Vec3:
        if self._model is None:
            return self._carried[body_id][1]
        spec = self._specs[body_id]
        if spec.static:
            return (0.0, 0.0, 0.0)
        joint_index = self._model.body_jntadr[self._body_index(body_id)]
        adr = self._model.jnt_dofadr[joint_index]
        vel = self._data.qvel[adr : adr + 3]
        return (float(vel[0]), float(vel[1]), float(vel[2]))

    def _capture_state(self) -> None:
        """Snapshot kinematics so a rebuild does not reset the world."""
        if self._model is None or self._dirty:
            return
        for body_id in self._specs:
            self._carried[body_id] = (
                self._position(body_id),
                self._velocity(body_id),
            )

    def _record_forces(self) -> None:
        """Net force and resting state, read from MuJoCo rather than recomputed.

        ``net_force`` is the total force acting on the body over the step just
        taken: the externally applied force, plus gravity, plus the constraint
        (contact) force the solver produced. At rest on the ground the contact
        force cancels gravity, so the vertical component goes to ~0 -- the same
        thing the reference backend arranges by special-casing, arrived at here
        by actually solving the contact.
        """
        for body_id, spec in self._specs.items():
            if spec.static:
                self._net_force[body_id] = (0.0, 0.0, 0.0)
                self._resting[body_id] = True
                continue

            index = self._body_index(body_id)
            applied = self._pending_force[body_id]

            joint_index = self._model.body_jntadr[index]
            adr = self._model.jnt_dofadr[joint_index]
            constraint = self._data.qfrc_constraint[adr : adr + 3]

            self._net_force[body_id] = (
                float(applied[0] + constraint[0]),
                float(applied[1] + GRAVITY * spec.mass + constraint[1]),
                float(applied[2] + constraint[2]),
            )

            velocity = self._velocity(body_id)
            speed = math.sqrt(sum(component * component for component in velocity))
            self._resting[body_id] = (
                self._in_contact(index) and speed < _REST_VELOCITY_EPS
            )

    def _in_contact(self, body_index: int) -> bool:
        data, model = self._data, self._model
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            if body_index in (body1, body2):
                return True
        return False

    def _clear_pending(self) -> None:
        for force in self._pending_force.values():
            force[0] = force[1] = force[2] = 0.0
