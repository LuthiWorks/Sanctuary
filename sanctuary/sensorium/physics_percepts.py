"""Perception adapter: :class:`PhysicsObservation` -> :class:`Percept`.

This is the missing link named in ``docs/audits/audit_2026-08-01_wiring.md``.
``sanctuary/physics/`` owns a working, tested physics substrate with a clean
three-view seam, and the cognitive cycle knows how to turn percepts into
cognition -- but until now nothing converted one into the other, so the physics
package was imported by nothing except its own tests.

Why this matters more than a plumbing fix: Luthi is trained as a **JEPA**, so
its objective is to predict the latent of what it has not yet seen. That
objective only builds a world model if the model is *in* a world whose next
state is a lawful consequence of its own actions. Sanctuary is the training
software; this adapter is where the world becomes perceivable.

Design commitments
------------------

**Egocentric by default.** When ``self_id`` names the perceiving body, every
other body is reported *relative to it* -- displacement and relative velocity,
not world coordinates. This is not cosmetic. A world model learned in absolute
coordinates has to re-learn every relationship at every location; one learned
egocentrically generalizes, because "the ball is 2 m ahead and closing" is the
same fact wherever the body is standing. Allocentric output (``self_id=None``)
is still available and honest for a disembodied observer, but it is not the
embodied path.

**The hidden channel cannot arrive here.** This module consumes
:class:`PhysicsObservation` only. That type has no field for mass or force, so
the instrumentation-only :class:`~sanctuary.physics.state.PhysicsGroundTruth`
cannot reach the model through this path even by mistake. Do not add a
ground-truth parameter to these functions.

**Acuity limits are perceptual, not truncation.** ``max_range`` and
``max_bodies`` model a body that cannot see everything at once. They default to
unlimited, so a caller only gets a limited view by asking for one. They are a
statement about the senses, not a silent cap on data.

A note on what is *not* wired, so it cannot be mistaken for finished
-------------------------------------------------------------------

Luthi's trunk currently has **vision, audio, and text** encoders only
(``luthi/multimodal_model.py``: sequence order ``[vision, audio, text]``). There
is no proprioceptive or touch encoder -- the LuthiModel README lists touch as
"eventually". So there is no encoder that can receive a body-state *vector*
today.

Because of that, :func:`observation_to_percepts` deliberately leaves
``Percept.tensor_data`` as ``None``. Setting it would imply a route into the
trunk that does not exist, and a mechanism that reports healthy while doing
nothing is the exact failure mode both repos name as their most expensive
recurring defect. The percepts this module returns carry **text**, which the
prompt context genuinely consumes today.

:func:`observation_to_vector` exists for the numeric path and is honest about
having no consumer yet: it is the shape a proprioceptive encoder would take
when one is built. See ``test_physics_percepts.py`` for the guard that fails if
``tensor_data`` starts being populated without an encoder to receive it.
"""

from __future__ import annotations

import math
from typing import Iterable

from sanctuary.core.schema import Percept
from sanctuary.physics.state import BodyState, PhysicsObservation, Vec3

__all__ = [
    "observation_to_percepts",
    "observation_to_vector",
    "PROPRIOCEPTIVE",
    "EXTEROCEPTIVE",
    "BODY_VECTOR_WIDTH",
]

# Modality tags. "proprioceptive" matches the existing motor-feedback percepts
# in ``Sensorium.perceive_own_action``; exteroception rides the established
# "sensor" tag rather than inventing a new one the prompt builders don't know.
PROPRIOCEPTIVE = "proprioceptive"
EXTEROCEPTIVE = "sensor"

#: Floats contributed per body to :func:`observation_to_vector` --
#: 3 displacement + 3 relative velocity + 1 scalar distance (or, for the self
#: block, 3 position + 3 velocity + 1 speed).
BODY_VECTOR_WIDTH = 7


# ---------------------------------------------------------------------------
# Small vector helpers (kept local: the physics package is dependency-free and
# this adapter should not be the thing that drags numpy into the sense path)
# ---------------------------------------------------------------------------


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _fmt(v: Vec3) -> str:
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


def _find_self(obs: PhysicsObservation, self_id: str) -> BodyState:
    """Return the perceiving body, or raise.

    Fails loud rather than degrading to an allocentric view. A missing
    ``self_id`` means the body is absent or was marked ``visible=False`` in its
    :class:`~sanctuary.physics.state.BodySpec` -- both are real defects, and a
    silent fallback would hand the model world coordinates while the caller
    believed it was sending egocentric ones. That is a corruption the training
    signal would absorb without ever erroring.
    """
    for body in obs.bodies:
        if body.body_id == self_id:
            return body
    visible = ", ".join(repr(b.body_id) for b in obs.bodies) or "<none>"
    raise KeyError(
        f"Perceiving body {self_id!r} is not present in the observation. "
        f"Visible bodies: {visible}. A body the model perceives from must be "
        f"visible=True in its BodySpec; an invisible or absent self body "
        f"cannot produce egocentric perception."
    )


def _ordered(
    bodies: Iterable[BodyState],
    origin: Vec3,
    max_range: float | None,
    max_bodies: int | None,
) -> list[tuple[BodyState, Vec3, float]]:
    """Bodies as ``(body, displacement, distance)``, nearest first.

    Ties break on ``body_id`` so the ordering is total and deterministic --
    percept order must not depend on dict iteration if runs are to be
    reproducible.
    """
    scored = []
    for body in bodies:
        disp = _sub(body.position, origin)
        dist = _norm(disp)
        if max_range is not None and dist > max_range:
            continue
        scored.append((body, disp, dist))

    scored.sort(key=lambda item: (item[2], item[0].body_id))
    if max_bodies is not None:
        scored = scored[:max_bodies]
    return scored


# ---------------------------------------------------------------------------
# The text path (functional today)
# ---------------------------------------------------------------------------


def observation_to_percepts(
    obs: PhysicsObservation,
    *,
    self_id: str | None = None,
    max_range: float | None = None,
    max_bodies: int | None = None,
) -> list[Percept]:
    """Turn one physics observation into percepts the sensorium can accept.

    Args:
        obs: The model-facing view from
            :meth:`~sanctuary.physics.authority.PhysicsAuthority.observe`.
            Occlusion has already been applied by the authority.
        self_id: The perceiving body. When given, output is **egocentric** and
            a proprioceptive percept for the body itself leads the list. When
            ``None``, output is allocentric (world coordinates, no self
            percept) -- an observer's view, not an embodied one.
        max_range: Perceptual reach in metres. Bodies further than this are not
            perceived. ``None`` (default) means unlimited.
        max_bodies: At most this many other bodies, nearest first. ``None``
            (default) means unlimited.

    Returns:
        Percepts in a deterministic order: the proprioceptive percept first
        (when ``self_id`` is given), then exteroceptive percepts nearest-first.
        An empty world yields just the proprioceptive percept; an empty
        allocentric world yields ``[]``.

    Raises:
        KeyError: if ``self_id`` is given but that body is not in ``obs``.
    """
    percepts: list[Percept] = []
    source = f"physics:step={obs.step}"

    if self_id is None:
        origin: Vec3 = (0.0, 0.0, 0.0)
        others: Iterable[BodyState] = obs.bodies
        frame = "world"
    else:
        me = _find_self(obs, self_id)
        origin = me.position
        others = (b for b in obs.bodies if b.body_id != self_id)
        frame = "self"

        speed = _norm(me.velocity)
        percepts.append(
            Percept(
                modality=PROPRIOCEPTIVE,
                content=(
                    f"[body] at {_fmt(me.position)}, "
                    f"velocity {_fmt(me.velocity)}, "
                    f"speed {speed:.2f} m/s, "
                    f"height {me.position[1]:.2f} m"
                ),
                source=f"{source}:self={self_id}",
                embedding_summary=f"own body, speed {speed:.2f} m/s",
            )
        )

    for body, disp, dist in _ordered(others, origin, max_range, max_bodies):
        if self_id is None:
            detail = f"at {_fmt(body.position)}, velocity {_fmt(body.velocity)}"
        else:
            # Relative velocity, and its sign along the line of sight: negative
            # closing rate means the gap is shrinking. This is the quantity an
            # embodied predictor actually needs, and deriving it here keeps
            # every consumer from re-deriving it inconsistently.
            rel_v = _sub(body.velocity, _self_velocity(obs, self_id))
            closing = 0.0 if dist == 0.0 else (
                (disp[0] * rel_v[0] + disp[1] * rel_v[1] + disp[2] * rel_v[2]) / dist
            )
            detail = (
                f"{dist:.2f} m away, offset {_fmt(disp)}, "
                f"relative velocity {_fmt(rel_v)}, "
                f"{'separating' if closing > 0 else 'closing'} at {abs(closing):.2f} m/s"
            )

        percepts.append(
            Percept(
                modality=EXTEROCEPTIVE,
                content=f"[{frame}] {body.body_id}: {detail}",
                source=f"{source}:body={body.body_id}",
                embedding_summary=f"{body.body_id} at {dist:.2f} m",
            )
        )

    return percepts


def _self_velocity(obs: PhysicsObservation, self_id: str) -> Vec3:
    for body in obs.bodies:
        if body.body_id == self_id:
            return body.velocity
    return (0.0, 0.0, 0.0)  # unreachable: _find_self already validated


# ---------------------------------------------------------------------------
# The numeric path (no consumer yet -- see the module docstring)
# ---------------------------------------------------------------------------


def observation_to_vector(
    obs: PhysicsObservation,
    *,
    self_id: str,
    max_bodies: int,
    max_range: float | None = None,
) -> tuple[float, ...]:
    """Fixed-width egocentric encoding of an observation.

    **This has no consumer today.** Luthi's trunk has vision, audio, and text
    encoders and no proprioceptive one, so nothing can currently receive this.
    It exists because the shape is knowable now and the training ground will
    need it the moment a body-state encoder lands -- not because it is wired.
    Do not route it into ``Percept.tensor_data``; see the module docstring.

    Layout, always exactly ``BODY_VECTOR_WIDTH * (1 + max_bodies)`` floats:

    * block 0 -- the self body: ``position(3), velocity(3), speed(1)``
    * blocks 1..N -- nearest other bodies, nearest first:
      ``displacement(3), relative_velocity(3), distance(1)``

    Absent bodies are zero-filled, so the width is constant regardless of how
    many bodies exist. Zero-fill is indistinguishable from a coincident,
    co-moving body at distance zero; that ambiguity is acceptable only because
    a body cannot occupy the observer's exact position in any world we build.
    If that ever stops holding, this needs a presence mask.

    Raises:
        KeyError: if ``self_id`` is not in ``obs``.
        ValueError: if ``max_bodies`` is negative.
    """
    if max_bodies < 0:
        raise ValueError(f"max_bodies must be >= 0, got {max_bodies}")

    me = _find_self(obs, self_id)
    out: list[float] = [
        *me.position,
        *me.velocity,
        _norm(me.velocity),
    ]

    others = (b for b in obs.bodies if b.body_id != self_id)
    selected = _ordered(others, me.position, max_range, max_bodies)

    for body, disp, dist in selected:
        rel_v = _sub(body.velocity, me.velocity)
        out.extend((*disp, *rel_v, dist))

    padding = (max_bodies - len(selected)) * BODY_VECTOR_WIDTH
    out.extend([0.0] * padding)

    return tuple(out)
