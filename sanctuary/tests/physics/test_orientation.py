"""Orientation across the seam, and the rebuild bug it uncovered.

Brian ruled on 2026-08-18 that Luthi sees through Godot's render, which makes
``render_state()`` the format of the entity's visual world rather than a
convenience for the family's window. Orientation stopped being cosmetic at that
moment: approach, avoidance and posture are unreadable without facing, and those
*are* the legibility curriculum for the creature roster.

Two defects were found while wiring it, both of the same family -- a field that
describes a route which does not exist:

1. ``_to_xml()`` never emitted ``quat``, so ``BodySpec.orientation`` was
   accepted, stored, and reported back by ``render_state()`` while never being
   applied to the body it claimed to orient.
2. ``_build()`` wrote identity into every carried body's quaternion, on the
   since-outdated reasoning that the seam could not report rotation anyway. That
   meant **every model rebuild silently snapped every body upright** -- invisible
   while all bodies were spheres, and with creatures it would mean adding one
   prop re-orients the whole world mid-run, a physically impossible event with
   nothing raising.

Authored by Opus 5, 2026-08-18.
"""
from __future__ import annotations

import pytest

from sanctuary.physics.state import BodySpec, IDENTITY_QUAT, Shape

#: 45 degrees about +Z.
TILT = (0.9239, 0.0, 0.0, 0.3827)


def _mj():
    pytest.importorskip("mujoco")
    from sanctuary.physics.backends.mujoco_backend import MuJoCoPhysicsAuthority
    return MuJoCoPhysicsAuthority()


def _orientation_of(world, body_id):
    return {b.body_id: b.orientation for b in world.render_state().bodies}[body_id]


# ---------------------------------------------------------------------------
# A declared orientation is actually applied
# ---------------------------------------------------------------------------

def test_declared_orientation_reaches_the_body():
    """Otherwise BodySpec.orientation is a field describing nothing."""
    world = _mj()
    world.reset(seed=0)
    world.add_body(BodySpec(
        body_id="tilted", position=(0.0, 0.6, 0.0), shape=Shape.CAPSULE,
        size=(0.08, 0.2, 0.08), orientation=TILT,
    ))
    world.step(0.02)

    actual = _orientation_of(world, "tilted")
    for got, want in zip(actual, TILT):
        assert got == pytest.approx(want, abs=1e-3)


def test_default_orientation_is_identity():
    world = _mj()
    world.reset(seed=0)
    world.add_body(BodySpec(body_id="plain", position=(0.0, 0.5, 0.0)))
    world.step(0.02)

    for got, want in zip(_orientation_of(world, "plain"), IDENTITY_QUAT):
        assert got == pytest.approx(want, abs=1e-6)


# ---------------------------------------------------------------------------
# The rebuild bug
# ---------------------------------------------------------------------------

def test_orientation_survives_a_model_rebuild():
    """The load-bearing regression.

    Adding a body recompiles the model. Before the fix that reset every body's
    quaternion to identity -- so a creature would snap upright the moment
    anything else entered the world, and nothing would report it.
    """
    world = _mj()
    world.reset(seed=0)
    world.add_body(BodySpec(
        body_id="tilted", position=(0.0, 0.6, 0.0), shape=Shape.CAPSULE,
        size=(0.08, 0.2, 0.08), orientation=TILT,
    ))
    for _ in range(80):
        world.step(0.02)
    before = _orientation_of(world, "tilted")

    world.add_body(BodySpec(body_id="newcomer", position=(2.0, 0.5, 0.0)))
    world.step(0.02)
    after = _orientation_of(world, "tilted")

    for got, want in zip(after, before):
        assert got == pytest.approx(want, abs=1e-6), (
            "a model rebuild changed an existing body's orientation"
        )
    assert max(abs(a - b) for a, b in zip(after, IDENTITY_QUAT)) > 1e-3, (
        "the body ended at identity -- the test cannot distinguish the fix "
        "from the bug it was written for"
    )


# ---------------------------------------------------------------------------
# Honest reporting per backend
# ---------------------------------------------------------------------------

def test_reference_backend_reports_none_not_identity():
    """It tracks no rotation. Saying identity would be a claim it cannot make."""
    from sanctuary.physics.backends.reference import ReferencePhysicsAuthority

    world = ReferencePhysicsAuthority()
    world.reset(seed=0)
    world.add_body(BodySpec(body_id="a", position=(0.0, 1.0, 0.0),
                            orientation=TILT))
    world.step(0.02)

    assert _orientation_of(world, "a") is None


# ---------------------------------------------------------------------------
# Shape is simulated, not merely declared
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,size", [
    (Shape.SPHERE, (0.1, 0.1, 0.1)),
    (Shape.CAPSULE, (0.05, 0.2, 0.05)),
    (Shape.BOX, (0.1, 0.15, 0.2)),
])
def test_declared_shape_is_built(shape, size):
    """A shape the physics ignores would be a world quietly built to different
    measurements than the one described."""
    world = _mj()
    world.reset(seed=0)
    world.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0),
                            shape=shape, size=size))
    world.step(0.02)

    assert f"type='{shape.value}'" in world._to_xml()
    drawn = {b.body_id: b for b in world.render_state().bodies}["b"]
    assert drawn.shape is shape
    assert drawn.size == size


def test_unspecified_size_uses_the_backend_default():
    """`size=None` means unspecified, which is not the same as any value.

    A concrete default would silently override a backend configured with its
    own body size -- which is exactly what it did when first written that way.
    """
    pytest.importorskip("mujoco")
    from sanctuary.physics.backends.mujoco_backend import MuJoCoPhysicsAuthority

    world = MuJoCoPhysicsAuthority(body_radius=0.5)
    world.reset(seed=0)
    world.add_body(BodySpec(body_id="b", position=(0.0, 2.0, 0.0)))
    world.step(0.02)

    drawn = {b.body_id: b for b in world.render_state().bodies}["b"]
    assert drawn.size == (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# The kind label must not leak into perception
# ---------------------------------------------------------------------------

def test_kind_is_on_the_render_channel_and_nowhere_else():
    """Luthi infers what a thing is from seeing it, never from a label.

    `kind` selects a mesh for the renderer. If it reached the model-facing
    observation, the scaffold would be handing over the answer to the very
    perceptual task the creature roster exists to teach.
    """
    from dataclasses import fields
    from sanctuary.physics.state import BodyState

    world = _mj()
    world.reset(seed=0)
    world.add_body(BodySpec(body_id="dog_0", position=(0.0, 0.5, 0.0),
                            kind="dog"))
    world.step(0.02)

    assert {b.body_id: b for b in world.render_state().bodies}["dog_0"].kind == "dog"
    assert "kind" not in {f.name for f in fields(BodyState)}
    observed = world.observe().bodies[0]
    assert not hasattr(observed, "kind")
