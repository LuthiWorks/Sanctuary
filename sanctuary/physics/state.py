"""Value types crossing the physics-authority seam.

These are frozen dataclasses, not Pydantic models: they are produced every
physics step on a hot path and carry no external/untrusted input (inputs are
validated once at the :class:`~sanctuary.physics.authority.PhysicsAuthority`
boundary), so per-step validation overhead is not wanted here.

The important design property lives in the *type shapes*, not in any runtime
check: the model-facing :class:`PhysicsObservation` is built from
:class:`BodyState`, which has **only** kinematics (id, position, velocity). It
has no field that could hold mass or force. The hidden ground-truth channel
(:class:`GroundTruthBody`) is a separate type with those fields. So the hidden
channel cannot leak into the model's view by accident — the observation type
literally cannot represent it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A 3-vector in world space. Metres, metres/second, or newtons by context.
Vec3 = tuple[float, float, float]

ORIGIN: Vec3 = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Input: how a body enters the world
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BodySpec:
    """Specification for a body to add to the world (seam input).

    Validated once at ``PhysicsAuthority.add_body``; immutable thereafter.
    """

    body_id: str
    position: Vec3
    velocity: Vec3 = ORIGIN
    mass: float = 1.0
    static: bool = False   # immovable (ground, walls) — unaffected by forces
    visible: bool = True    # if False, excluded from the model's observation


# ---------------------------------------------------------------------------
# Model-facing view (kinematics only, visible bodies only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BodyState:
    """A body as the *model* may perceive it: kinematics only.

    Deliberately has no mass/force field. This is the type-level guarantee
    that the hidden ground-truth channel cannot ride along into perception.
    """

    body_id: str
    position: Vec3
    velocity: Vec3


@dataclass(frozen=True, slots=True)
class PhysicsObservation:
    """The model-facing view of the world at a step.

    Occlusion is applied (invisible bodies are absent), and only kinematics are
    present. This is what an adapter turns into percepts for the sensorium.
    """

    time: float
    step: int
    bodies: tuple[BodyState, ...] = ()


# ---------------------------------------------------------------------------
# Instrumentation-only view (the hidden ground-truth channel)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroundTruthBody:
    """A body's full true state, for instrumentation/LUTHISCOPE only.

    Includes fields the model never sees: mass, the net force applied over the
    last step, and whether the body is resting on the ground.
    """

    body_id: str
    position: Vec3
    velocity: Vec3
    mass: float
    net_force: Vec3
    resting: bool
    static: bool
    visible: bool


@dataclass(frozen=True, slots=True)
class PhysicsGroundTruth:
    """The full true physical state at a step. **Instrumentation only.**

    Never pass this to the model — it exists so learned latent geometry (LID,
    eigenspectrum, covariance) can be checked against real physical structure
    rather than a same-dataset correlate (spec sec. 3, sec. 8).
    """

    time: float
    step: int
    bodies: tuple[GroundTruthBody, ...] = ()


# ---------------------------------------------------------------------------
# Renderer view (poses to draw)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RenderBody:
    """A visible body's pose for the renderer (Godot) to draw."""

    body_id: str
    position: Vec3
    # orientation (quaternion) will join here once the backend tracks rotation.


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """What the renderer draws at a step: poses of the visible bodies."""

    time: float
    step: int
    bodies: tuple[RenderBody, ...] = field(default=())
