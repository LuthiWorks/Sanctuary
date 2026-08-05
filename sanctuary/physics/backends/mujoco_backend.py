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

Cameras
-------

This backend adds a capability the seam does not define: **rendered vision**.
:meth:`MuJoCoPhysicsAuthority.attach_camera` and
:meth:`MuJoCoPhysicsAuthority.render_camera` are deliberately *not* on
:class:`~sanctuary.physics.authority.PhysicsAuthority`, because the reference
backend has no geometry to draw and cannot honestly implement them. Rendering is
a property of an engine, not of the seam.

A camera attached to a body is an **eye** -- the visual counterpart of the
egocentric framing ``sanctuary.sensorium.physics_percepts`` applies to spatial
perception. Rendering is offscreen; no window opens.
``sanctuary.sensorium.physics_vision`` turns the resulting frame into a
:class:`~sanctuary.core.schema.Percept` carrying an encoder-ready tensor.

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

import colorsys
import hashlib
import math
from dataclasses import dataclass
from typing import Any
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

__all__ = [
    "MuJoCoPhysicsAuthority",
    "CameraSpec",
    "CameraFrame",
    "DEFAULT_BODY_RADIUS",
    "DEFAULT_MAX_SUBSTEP",
    "DEFAULT_CAMERA_SIZE",
]

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

#: Pixels. Square, and 224 by default because that is what Luthi's VisionEncoder
#: is built for (``image_size=224``, ``patch_size=16`` -> 196 tokens).
DEFAULT_CAMERA_SIZE: int = 224

#: Metres. Nothing nearer than this to a camera is rendered.
#:
#: MuJoCo expresses the near plane as a *fraction of the model's extent*, and
#: extent is inferred from world size -- so a large ground plane silently pushes
#: the near plane outward. At the default 100 m ground that put it at 0.20 m,
#: which made anything within 20 cm of the eye invisible while every frame still
#: rendered cleanly. A body cannot see what it is about to touch, and nothing
#: raises. The extent is therefore pinned below and the near plane set from it,
#: so this number means metres regardless of how big the world is.
DEFAULT_NEAR_PLANE: float = 0.02

#: Metres. Beyond this, nothing is drawn.
DEFAULT_FAR_PLANE: float = 500.0

#: Pinned so near/far planes are absolute distances rather than a function of
#: world size. See :data:`DEFAULT_NEAR_PLANE`.
_MODEL_EXTENT: float = 10.0


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """A camera in the world.

    Cameras are part of the compiled model, like bodies, so attaching one marks
    the model dirty and it is rebuilt on next use.

    ``body_id`` is what makes a camera an *eye* rather than a security camera:
    when set, the camera rides that body and sees the world from it, which is
    the visual counterpart of the egocentric framing
    ``sanctuary.sensorium.physics_percepts`` applies to spatial perception.
    When ``None`` the camera is fixed in the world -- useful for instrumentation
    and for watching a run, but it is not what the entity sees.

    The camera looks along **+X** with **+Y** up, matching the seam's Y-up
    frame. ``offset`` is relative to the body's origin (or the world origin for
    a fixed camera).
    """

    name: str
    body_id: str | None = None
    offset: Vec3 = (0.0, 0.0, 0.0)
    fovy: float = 60.0


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One rendered view.

    ``rgb`` is an ``[H, W, 3]`` uint8 numpy array -- raw pixels, deliberately
    not normalized. Turning it into an encoder-ready tensor is
    ``sanctuary.sensorium.physics_vision``'s job, so the normalization the model
    expects lives next to the rest of the perception code rather than being
    baked into the renderer.
    """

    camera: str
    time: float
    step: int
    rgb: Any  # numpy.ndarray [H, W, 3] uint8

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


class MuJoCoPhysicsAuthority(PhysicsAuthority):
    """Rigid-body backend. See the module docstring for the divergences."""

    def __init__(
        self,
        *,
        body_radius: float = DEFAULT_BODY_RADIUS,
        max_substep: float = DEFAULT_MAX_SUBSTEP,
        ground_size: float = 100.0,
        near_plane: float = DEFAULT_NEAR_PLANE,
        far_plane: float = DEFAULT_FAR_PLANE,
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
        if near_plane <= 0.0:
            raise ValueError(f"near_plane must be positive, got {near_plane}")
        if far_plane <= near_plane:
            raise ValueError(
                f"far_plane ({far_plane}) must exceed near_plane ({near_plane})"
            )
        if near_plane >= body_radius:
            raise ValueError(
                f"near_plane ({near_plane}) is at or beyond body_radius "
                f"({body_radius}): a body-mounted camera would clip its own "
                f"surroundings before seeing anything adjacent. Lower it."
            )

        self._mujoco = mujoco
        self.body_radius = body_radius
        self.max_substep = max_substep
        self.ground_size = ground_size
        self.near_plane = near_plane
        self.far_plane = far_plane

        self._specs: dict[str, BodySpec] = {}
        self._cameras: dict[str, CameraSpec] = {}
        self._model = None
        self._data = None
        self._renderer = None
        self._renderer_size: tuple[int, int] | None = None
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
        self._cameras.clear()
        self._pending_force.clear()
        self._net_force.clear()
        self._resting.clear()
        self._carried.clear()
        self._model = None
        self._data = None
        self._release_renderer()
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

    # -- cameras ------------------------------------------------------------

    def attach_camera(self, spec: CameraSpec) -> str:
        """Add a camera. Raises on a duplicate name or an unknown body."""
        if spec.name in self._cameras:
            raise ValueError(f"Camera name already exists: {spec.name!r}")
        if spec.body_id is not None and spec.body_id not in self._specs:
            raise KeyError(
                f"Cannot attach camera {spec.name!r} to unknown body "
                f"{spec.body_id!r}. Add the body first."
            )

        # A camera inside its own body's geom renders a valid frame that shows
        # the inside of the sphere -- near-black, every step, forever. Nothing
        # downstream would raise: the shape is right, the dtype is right, and a
        # world model would train happily on darkness. Refuse it here.
        if spec.body_id is not None:
            reach = math.sqrt(sum(component * component for component in spec.offset))
            if reach <= self.body_radius:
                raise ValueError(
                    f"Camera {spec.name!r} sits {reach:.3f} m from the origin of "
                    f"body {spec.body_id!r}, which is inside that body's own geom "
                    f"(radius {self.body_radius}). Every frame would render the "
                    f"inside of the body -- a valid, near-black image that no "
                    f"downstream check would flag. Move the camera clear of the "
                    f"body radius, e.g. offset=({self.body_radius * 1.5:.3f}, 0, 0) "
                    f"to look forward along +X."
                )

        self._capture_state()
        self._cameras[spec.name] = spec
        self._dirty = True
        return spec.name

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self._cameras)

    def render_camera(
        self,
        name: str,
        *,
        width: int = DEFAULT_CAMERA_SIZE,
        height: int = DEFAULT_CAMERA_SIZE,
    ) -> CameraFrame:
        """Render what a camera currently sees.

        Offscreen; no window is opened. Returns raw uint8 pixels -- see
        :class:`CameraFrame` for why normalization is not done here.

        Raises:
            KeyError: if no such camera is attached.
            RuntimeError: if the world has no bodies, and therefore no compiled
                model to render. An empty render would be a black image, which
                is indistinguishable from a broken renderer.
        """
        if name not in self._cameras:
            known = ", ".join(repr(n) for n in self._cameras) or "<none>"
            raise KeyError(f"Unknown camera: {name!r}. Attached: {known}.")

        self._ensure_built()
        if self._model is None:
            raise RuntimeError(
                "Cannot render an empty world: there is no compiled model. Add "
                "at least one body first. (A black frame would look identical "
                "to a broken renderer, so this raises instead.)"
            )

        renderer = self._get_renderer(width, height)
        renderer.update_scene(self._data, camera=name)
        rgb = renderer.render().copy()

        return CameraFrame(camera=name, time=self._time, step=self._step, rgb=rgb)

    def _get_renderer(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Camera size must be positive, got {width}x{height}"
            )
        if self._renderer is None or self._renderer_size != (width, height):
            self._release_renderer()
            self._renderer = self._mujoco.Renderer(
                self._model, height=height, width=width
            )
            self._renderer_size = (width, height)
        return self._renderer

    def _release_renderer(self) -> None:
        """Drop the renderer and its GL resources.

        Called whenever the model is rebuilt: a Renderer is bound to the
        ``mjModel`` it was constructed with, so keeping one across a rebuild
        would render the *old* world while every other view reported the new
        one -- a divergence nothing would raise on.
        """
        if self._renderer is not None:
            close = getattr(self._renderer, "close", None)
            if close is not None:
                close()
        self._renderer = None
        self._renderer_size = None

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

        # The renderer is bound to the model it was built against.
        self._release_renderer()

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
            # Pin the extent so the near/far planes below are absolute metres
            # rather than a fraction of however large the ground happens to be.
            f"<statistic extent='{_MODEL_EXTENT}'/>",
            f"<visual><map znear='{self.near_plane / _MODEL_EXTENT}' "
            f"zfar='{self.far_plane / _MODEL_EXTENT}'/></visual>",
            "<worldbody>",
            # Lighting affects rendering only, never dynamics. Without it every
            # camera frame is near-black, which would look like a broken
            # renderer rather than an unlit scene.
            "<light pos='0 20 0' dir='0 -1 0' diffuse='0.9 0.9 0.9'/>",
            f"<geom name='floor' type='plane' size='{self.ground_size} "
            f"{self.ground_size} 0.1' zaxis='0 1 0' pos='0 {GROUND_Y} 0' "
            f"rgba='0.4 0.4 0.45 1'/>",
        ]

        for spec in self._cameras.values():
            if spec.body_id is None:
                parts.append(self._camera_xml(spec))

        for body_id, spec in self._specs.items():
            name = quoteattr(body_id)
            px, py, pz = spec.position
            parts.append(f"<body name={name} pos='{px} {py} {pz}'>")
            if not spec.static:
                parts.append("<freejoint/>")
            parts.append(
                f"<geom type='sphere' size='{self.body_radius}' "
                f"mass='{spec.mass}' rgba='{_body_rgba(body_id)}'/>"
            )
            for camera in self._cameras.values():
                if camera.body_id == body_id:
                    parts.append(self._camera_xml(camera))
            parts.append("</body>")

        parts.append("</worldbody>")
        parts.append("</mujoco>")
        return "".join(parts)

    @staticmethod
    def _camera_xml(spec: CameraSpec) -> str:
        ox, oy, oz = spec.offset
        # xyaxes gives the camera's own x and y axes in the frame it sits in.
        # MuJoCo cameras look along -Z of their frame, so x=(0,0,1), y=(0,1,0)
        # puts -Z along world +X: looking along +X with +Y up.
        return (
            f"<camera name={quoteattr(spec.name)} pos='{ox} {oy} {oz}' "
            f"xyaxes='0 0 1 0 1 0' fovy='{spec.fovy}'/>"
        )

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


def _body_rgba(body_id: str) -> str:
    """A stable, distinct colour per body id.

    Appearance is a **placeholder**: ``BodySpec`` carries no material, so a
    colour is derived from the id instead. It is deterministic (SHA-256, not
    ``hash()``, which is salted per process) so that two runs of the same world
    render identically -- a world model trained on recoloured objects would be
    learning noise.

    This is the same deferred decision as ``body_radius``: per-body appearance
    and geometry belong on ``BodySpec``, which is a seam change and deserves its
    own ruling. Until then, objects at least have to be visually distinguishable
    or vision has nothing to learn from.
    """
    digest = hashlib.sha256(body_id.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
    return f"{red:.3f} {green:.3f} {blue:.3f} 1"
