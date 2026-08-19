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
from dataclasses import dataclass, replace
from typing import Any
from xml.sax.saxutils import quoteattr

from sanctuary.physics.authority import PhysicsAuthority
from sanctuary.physics.state import (
    BodySpec,
    BodyState,
    GroundTruthBody,
    PhysicsGroundTruth,
    IDENTITY_QUAT,
    PhysicsObservation,
    Quat,
    Shape,
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
        self._viewer = None
        self._dirty = True

        self._time: float = 0.0
        self._step: int = 0

        self._pending_force: dict[str, list[float]] = {}
        self._net_force: dict[str, Vec3] = {}
        self._resting: dict[str, bool] = {}

        # Kinematic state carried across a model rebuild.
        self._carried: dict[str, tuple[Vec3, Vec3]] = {}

        # Orientation carried across a model rebuild, alongside _carried's
        # position/velocity. Before 2026-08-18 a rebuild forced every body back
        # to identity, so adding one prop silently snapped the whole world
        # upright -- invisible with spheres, catastrophic with creatures.
        self._carried_orientation: dict[str, Quat] = {}

        # Body-name -> mjModel body index, built once per compiled model.
        # mj_name2id is a string search through the model; it was being called
        # ~4x per body per step, which is most of what made step() O(n^2)-ish.
        self._index_of: dict[str, int] = {}

        # Applied force as it was at the moment of the last step. ground_truth()
        # derives net force lazily (see _ensure_forces), and step() clears the
        # pending forces, so the value has to be snapshotted rather than re-read.
        self._applied_snapshot: dict[str, Vec3] = {}
        self._forces_stale: bool = True

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
        self._carried_orientation.clear()
        self._index_of.clear()
        self._applied_snapshot.clear()
        self._forces_stale = True
        self._model = None
        self._data = None
        self.close_viewer()
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
        # Seed the orientation too. _build() restores every carried body's qpos
        # from these dicts, so a body present in _carried but absent from
        # _carried_orientation would be written back at identity -- silently
        # discarding the orientation its own spec declared.
        self._carried_orientation[spec.body_id] = spec.orientation
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

    def set_visible(self, body_id: str, visible: bool) -> None:
        """Show or hide a body WITHOUT rebuilding the model.

        `visible` is consumed only by observe() / ground_truth() /
        render_state(); `_to_xml()` never reads it, so the compiled model is
        identical either way and there is nothing to recompile. This is the
        whole point: a population that changes by toggling visibility costs
        nothing, where one that changes by add_body/remove_body pays a ~11 ms
        rebuild plus a ~29 ms renderer reconstruction, and cannot happen at
        all while a viewer is open.

        Deliberately does NOT set `_dirty`. If a future edit makes `_to_xml()`
        depend on visibility, this method must start dirtying the model --
        `test_set_visible_does_not_rebuild_the_model` fails loudly if the two
        ever disagree.
        """
        spec = self._specs.get(body_id)
        if spec is None:
            raise KeyError(f"Unknown body id: {body_id!r}")
        self._specs[body_id] = replace(spec, visible=bool(visible))

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

    # -- interactive viewer -------------------------------------------------

    def launch_viewer(self, *, key_callback=None, show_ui: bool = True):
        """Open an interactive window onto this world and return its handle.

        Passive: MuJoCo does not drive the simulation: you keep stepping, and
        call :meth:`sync_viewer` to push each new state to the window. That is
        what makes it a *view of the running loop* rather than a separate
        replay -- what you watch is the same ``mjData`` the entity is acting on.

        Interaction comes for free from the viewer itself (orbit/pan/zoom, and
        ctrl-drag to shove a body around mid-run), plus ``key_callback`` for
        anything the caller wants to bind.

        Requires a display. Raises if a viewer is already open, or if the world
        has no compiled model yet.
        """
        if self._viewer is not None:
            raise RuntimeError(
                "A viewer is already open for this world. Close it before "
                "launching another."
            )

        self._ensure_built()
        if self._model is None:
            raise RuntimeError(
                "Cannot view an empty world: there is no compiled model. Add "
                "at least one body first."
            )

        from mujoco import viewer as _viewer

        self._viewer = _viewer.launch_passive(
            self._model,
            self._data,
            key_callback=key_callback,
            show_left_ui=show_ui,
            show_right_ui=show_ui,
        )
        return self._viewer

    def sync_viewer(self) -> None:
        """Push the current state to the open viewer. No-op if none is open."""
        if self._viewer is not None:
            self._viewer.sync()

    def close_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @property
    def viewer_is_open(self) -> bool:
        """True while the window is open -- False once the user closes it.

        The run loop should watch this: a closed window means the person
        watching has left, and continuing to spin is not what they asked for.
        """
        if self._viewer is None:
            return False
        running = getattr(self._viewer, "is_running", None)
        return bool(running()) if running is not None else True

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
            self._applied_snapshot = {}
            self._forces_stale = True
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

        # Capture what was applied, then defer the derivation. _clear_pending()
        # below wipes _pending_force, so ground_truth() could not recover it.
        self._applied_snapshot = {
            body_id: (force[0], force[1], force[2])
            for body_id, force in self._pending_force.items()
        }
        self._forces_stale = True

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
        # The only consumer of the force derivation, and therefore the only
        # place that pays for it.
        self._ensure_forces()
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
        """Poses for the renderer.

        Perception-critical as of 2026-08-18: Godot's render is what Luthi
        sees, so anything missing here is missing from the entity's visual
        world. Orientation is real (from ``xquat``), not identity -- a renderer
        that cannot face a creature cannot show approach, avoidance or posture,
        which is the whole legibility curriculum.
        """
        self._ensure_built()
        bodies = tuple(
            RenderBody(
                body_id=body_id,
                position=self._position(body_id),
                orientation=self._orientation(body_id),
                shape=spec.shape,
                size=_resolved_size(spec, self.body_radius),
                kind=spec.kind,
            )
            for body_id, spec in self._specs.items()
            if spec.visible
        )
        return RenderFrame(time=self._time, step=self._step, bodies=bodies)

    def _orientation(self, body_id: str) -> Quat:
        """Body orientation as (w, x, y, z), MuJoCo's own convention."""
        if self._model is None:
            return self._carried_orientation.get(body_id, IDENTITY_QUAT)
        quat = self._data.xquat[self._body_index(body_id)]
        return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

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

        # A viewer holds the mjModel/mjData it was opened against. Rebuilding
        # underneath it would leave the window showing a world that no longer
        # exists while every other view reported the new one -- the same
        # divergence the renderer had, except a person is watching this one and
        # would believe what they saw. Refuse rather than silently diverge.
        if self._viewer is not None:
            raise RuntimeError(
                "Cannot rebuild the model while a viewer is open: the window is "
                "bound to the current model, and would keep showing the old "
                "world. Call close_viewer() before adding or removing bodies "
                "or cameras, then relaunch."
            )

        # The renderer is bound to the model it was built against.
        self._release_renderer()

        if not self._specs:
            self._model = None
            self._data = None
            self._index_of.clear()
            return

        self._model = mujoco.MjModel.from_xml_string(self._to_xml())
        self._data = mujoco.MjData(self._model)

        # Resolve every body name once, here, instead of on every read. Must be
        # populated before the _carried loop below, which calls _body_index.
        self._index_of = {
            body_id: mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            for body_id in self._specs
        }

        for body_id, (position, velocity) in self._carried.items():
            if body_id not in self._specs or self._specs[body_id].static:
                continue
            joint_index = self._model.body_jntadr[self._body_index(body_id)]
            qpos_adr = self._model.jnt_qposadr[joint_index]
            qvel_adr = self._model.jnt_dofadr[joint_index]
            self._data.qpos[qpos_adr : qpos_adr + 3] = position
            # Restore the real orientation (fixed 2026-08-18). This used to
            # force identity, on the reasoning that the seam could not report
            # rotation anyway -- but that made every model rebuild silently
            # snap every body upright. Invisible while all bodies were spheres;
            # with creatures it would mean adding one prop re-orients the whole
            # world mid-run, a physically impossible event with nothing raising.
            self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = (
                self._carried_orientation.get(body_id, IDENTITY_QUAT)
            )
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
            # Emit the declared orientation. Without this, BodySpec.orientation
            # would be a field describing a route that does not exist: accepted,
            # stored, reported back by render_state -- and never applied to the
            # body it claims to orient.
            qw, qx, qy, qz = spec.orientation
            parts.append(
                f"<body name={name} pos='{px} {py} {pz}' "
                f"quat='{qw} {qx} {qy} {qz}'>"
            )
            if not spec.static:
                parts.append("<freejoint/>")
            parts.append(
                f"<geom type='{spec.shape.value}' size='{_geom_size(spec, self.body_radius)}' "
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
        """mjModel body index for a name, from the per-model cache.

        The cache is rebuilt in _build(). Falling back to mj_name2id on a miss
        keeps this correct if a caller reaches for a body between edits, but a
        miss should not happen on the hot path -- and if the cache were ever
        stale rather than merely incomplete, the fallback would return the RIGHT
        answer while the cache returned a wrong one silently, so the cache is
        cleared on every rebuild rather than patched.
        """
        index = self._index_of.get(body_id)
        if index is not None:
            return index
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
            self._carried_orientation[body_id] = self._orientation(body_id)

    def _ensure_forces(self) -> None:
        """Derive net force and resting state, if a step has happened since.

        Split out of :meth:`step` on 2026-08-18. The derivation is
        **instrumentation** -- only :meth:`ground_truth` consumes ``_net_force``
        and ``_resting`` -- but it used to run unconditionally on every step,
        where it was measured at 15.05 ms of a 15.94 ms step at 121 bodies:
        **94% of the step, and ~75% of the per-decision budget**, paid whether
        or not anything was watching.

        Doing only this (laziness) would have been a trap: the moment LUTHISCOPE
        polls every step the cost returns in full, and the benchmark would still
        look fixed. So :meth:`_derive_forces` is also vectorized -- the cost is
        removed when nobody is watching *and* when somebody is.
        """
        if not self._forces_stale:
            return
        self._derive_forces()
        self._forces_stale = False

    def _contacting_bodies(self) -> set[int]:
        """Indices of every body currently in contact, in one vectorized pass.

        Was an O(bodies x ncon) Python scan: for each body, walk every contact
        and materialize ``data.contact[i]`` -- a struct wrapper per access. At
        121 resting bodies that is ~14,600 wrapper constructions per step, and
        it was the dominant term in the old cost.

        MuJoCo exposes the contact array natively, so both geom columns can be
        mapped through ``geom_bodyid`` at once and reduced to a set.
        """
        ncon = self._data.ncon
        if ncon == 0:
            return set()
        geoms = self._data.contact.geom[:ncon]          # (ncon, 2)
        return set(self._model.geom_bodyid[geoms].ravel().tolist())

    def _derive_forces(self) -> None:
        """Net force and resting state, read from MuJoCo rather than recomputed.

        ``net_force`` is the total force acting on the body over the step just
        taken: the externally applied force, plus gravity, plus the constraint
        (contact) force the solver produced. At rest on the ground the contact
        force cancels gravity, so the vertical component goes to ~0 -- the same
        thing the reference backend arranges by special-casing, arrived at here
        by actually solving the contact.

        Reads the applied force from ``_applied_snapshot`` rather than
        ``_pending_force``: step() clears the pending forces before this runs,
        so the value has to have been captured at step time.
        """
        in_contact = self._contacting_bodies()

        for body_id, spec in self._specs.items():
            if spec.static:
                self._net_force[body_id] = (0.0, 0.0, 0.0)
                self._resting[body_id] = True
                continue

            index = self._body_index(body_id)
            applied = self._applied_snapshot.get(body_id, (0.0, 0.0, 0.0))

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
                index in in_contact and speed < _REST_VELOCITY_EPS
            )

    def _clear_pending(self) -> None:
        for force in self._pending_force.values():
            force[0] = force[1] = force[2] = 0.0


def _geom_size(spec: BodySpec, default_radius: float) -> str:
    """MuJoCo's `size` attribute for a body's declared shape.

    MuJoCo reads a different number of size components per geom type, so a
    single Vec3 has to be projected per shape. Getting this wrong produces a
    body that compiles and simulates at the wrong dimensions -- no error, just
    a world quietly built to different measurements.

      sphere  -> radius
      capsule -> radius, half-length
      box     -> three half-extents
    """
    x, y, z = _resolved_size(spec, default_radius)
    if spec.shape is Shape.SPHERE:
        return f"{x}"
    if spec.shape is Shape.CAPSULE:
        return f"{x} {y}"
    if spec.shape is Shape.BOX:
        return f"{x} {y} {z}"
    raise ValueError(f"unhandled shape {spec.shape!r} for body {spec.body_id!r}")


def _resolved_size(spec: BodySpec, default_radius: float) -> Vec3:
    """A concrete size, resolving `size=None` against the backend's default.

    The renderer needs a real number -- it cannot draw "unspecified" -- so the
    resolution happens here rather than being pushed across the seam.
    """
    if spec.size is not None:
        return spec.size
    return (default_radius, default_radius, default_radius)


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
