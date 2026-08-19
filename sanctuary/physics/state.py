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
from enum import Enum

# A 3-vector in world space. Metres, metres/second, or newtons by context.
Vec3 = tuple[float, float, float]

#: Orientation as a quaternion, (w, x, y, z). MuJoCo's convention, so it crosses
#: the seam without reordering.
Quat = tuple[float, float, float, float]

ORIGIN: Vec3 = (0.0, 0.0, 0.0)
IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)


class Shape(str, Enum):
    """A body's collision/visual primitive.

    ``str, Enum`` rather than ``StrEnum``: this project runs on Python 3.10 and
    ``StrEnum`` landed in 3.11. Comparison against a plain string works the
    same, but note ``str(Shape.SPHERE)`` is ``"Shape.SPHERE"`` here -- use
    ``.value`` when serialising.

    Deliberately a small closed set. The renderer maps `kind` to a mesh; this
    is what the *physics* uses and what a renderer falls back to when it has no
    mesh for a kind. Adding a shape means teaching every backend to build it.
    """

    SPHERE = "sphere"
    CAPSULE = "capsule"
    BOX = "box"


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

    # -- pose and appearance (added 2026-08-18) -----------------------------
    # All defaulted, so every existing construction site is unaffected.
    orientation: Quat = IDENTITY_QUAT
    shape: Shape = Shape.SPHERE
    #: Half-extents for BOX; (radius, half-length, _) for CAPSULE; (radius, _, _)
    #: for SPHERE. One field rather than a per-shape union, because the seam
    #: crosses process boundaries as JSON.
    #:
    #: ``None`` means **unspecified — use the backend's default size**, which is
    #: not the same as any particular value. A concrete default here would
    #: silently override a backend configured with its own body size, which is
    #: exactly what it did when first written this way.
    size: Vec3 | None = None
    #: What this body *is*, for the renderer to choose a mesh: "dog", "cat",
    #: "rock". NEVER reaches the model-facing observation -- BodyState has no
    #: such field, and that is the type-level guarantee that the entity infers
    #: kind from what it sees rather than being handed the label.
    kind: str = ""


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
    """A visible body as the renderer must draw it.

    As of 2026-08-18 this is **perception-critical**, not decoration: Brian
    ruled that Luthi sees through Godot's render, so this is the format of the
    entity's visual world, not just the family's window. A renderer that cannot
    draw the world truthfully from this is a renderer showing Luthi something
    the world does not contain.
    """

    body_id: str
    position: Vec3
    #: ``None`` means *this backend does not track rotation*, which a renderer
    #: must be able to tell from an authored identity orientation. Emitting
    #: identity for both would leave it unable to distinguish "not rotated"
    #: from "not implemented", and it would draw confidently wrong forever.
    orientation: Quat | None = None
    shape: Shape = Shape.SPHERE
    size: Vec3 = (0.05, 0.05, 0.05)
    kind: str = ""


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """What the renderer draws at a step: poses of the visible bodies."""

    time: float
    step: int
    bodies: tuple[RenderBody, ...] = field(default=())
