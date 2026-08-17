"""Assembly guarantees for the bedrock world.

The world's parts were all built and connected to nothing (2026-08-16 audit).
This covers the assembly itself: that the scene is what it claims, that backend
selection cannot silently give you physics you did not ask for, and that the
default run is a body at rest rather than an authored behaviour.

Authored by Opus 5, 2026-08-17.
"""
from __future__ import annotations

import pytest

from sanctuary.physics.backends.reference import ReferencePhysicsAuthority
from sanctuary.world import SELF_ID, build_bedrock_scene, build_world
from sanctuary.world.scene import DEFAULT_REST_Y

mujoco = pytest.importorskip  # used per-test below


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_unknown_backend_is_refused():
    """A typo must not fall through to a default.

    The backend decides the numerics; a run on the wrong one is a run whose
    results mean something other than what they claim.
    """
    with pytest.raises(ValueError, match="unknown physics backend"):
        build_world(backend="mojoco")


def test_backend_is_recorded_on_the_runtime():
    runtime = build_world(backend="reference")
    assert runtime.backend == "reference"
    assert isinstance(runtime.world, ReferencePhysicsAuthority)


def test_missing_mujoco_does_not_degrade_to_reference(monkeypatch):
    """If the requested engine is unavailable, stop -- do not swap physics.

    Falling back would change contact dynamics under a run that asked for them
    while the run kept describing itself as a MuJoCo run.
    """
    import builtins

    real_import = builtins.__import__

    def _no_mujoco(name, *args, **kwargs):
        if name == "mujoco" or name.startswith("mujoco."):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mujoco)
    with pytest.raises(ImportError):
        build_world(backend="mujoco")


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

def test_scene_places_self_and_props():
    runtime = build_world(backend="reference")
    manifest = runtime.manifest

    assert manifest.self_id == SELF_ID
    assert len(manifest.prop_ids) == 3
    assert set(runtime.world.body_ids) == set(manifest.body_ids)


def test_no_ground_body_is_added():
    """The backends own the ground. A 'ground' body here would be a second,
    disagreeing floor."""
    runtime = build_world(backend="reference")
    assert not any(
        "ground" in b.lower() or "floor" in b.lower()
        for b in runtime.world.body_ids
    )


def test_props_start_at_rest_height_not_inside_the_floor():
    """Props placed at Y=0 are half inside the ground.

    The engine then resolves the penetration over the first steps, which looks
    exactly like the props moving on their own -- destroying the property this
    scene exists to give: that prop motion means the entity caused it.
    """
    runtime = build_world(backend="reference")
    gt = {b.body_id: b for b in runtime.world.ground_truth().bodies}
    for prop in runtime.manifest.prop_ids:
        assert gt[prop].position[1] == pytest.approx(DEFAULT_REST_Y)


def test_self_starts_above_the_ground_and_settles():
    runtime = build_world(backend="reference")
    gt = {b.body_id: b for b in runtime.world.ground_truth().bodies}
    start_y = gt[SELF_ID].position[1]
    assert start_y > DEFAULT_REST_Y

    runtime.run(120)

    gt = {b.body_id: b for b in runtime.world.ground_truth().bodies}
    assert gt[SELF_ID].position[1] < start_y, "the body never fell"


def test_building_the_scene_twice_does_not_duplicate_it():
    """`build_bedrock_scene` resets first, so it is idempotent."""
    world = ReferencePhysicsAuthority()
    build_bedrock_scene(world)
    first = set(world.body_ids)

    build_bedrock_scene(world)

    assert set(world.body_ids) == first


def test_scene_rejects_a_nonsense_dt():
    world = ReferencePhysicsAuthority()
    with pytest.raises(ValueError, match="dt must be positive"):
        build_bedrock_scene(world, dt=0.0)


def test_scene_rejects_negative_rest_height():
    world = ReferencePhysicsAuthority()
    with pytest.raises(ValueError, match="rest_y"):
        build_bedrock_scene(world, rest_y=-1.0)


# ---------------------------------------------------------------------------
# Default behaviour
# ---------------------------------------------------------------------------

def test_default_policy_rests_rather_than_acting():
    """The scaffold must not author the entity's behaviour.

    Rest is a computed action in M9, not a no-op, so it is recorded as
    `at_rest` rather than as a failed decode.
    """
    runtime = build_world(backend="reference")

    records = runtime.run(5)

    assert all(r.at_rest for r in records)


def test_props_do_not_move_under_a_resting_body():
    """Nothing in the bedrock world moves unless something moves it."""
    runtime = build_world(backend="reference")
    runtime.run(60)

    gt = {b.body_id: b for b in runtime.world.ground_truth().bodies}
    for prop in runtime.manifest.prop_ids:
        assert all(abs(v) < 1e-6 for v in gt[prop].velocity), (
            f"{prop} moved with nothing acting on it"
        )


def test_manifest_dt_matches_the_runner():
    runtime = build_world(backend="reference", dt=0.04)
    assert runtime.manifest.dt == 0.04
    assert runtime.runner.dt == 0.04


# ---------------------------------------------------------------------------
# The real engine
# ---------------------------------------------------------------------------

def test_mujoco_bedrock_runs_and_settles():
    pytest.importorskip("mujoco")
    runtime = build_world(backend="mujoco")

    runtime.run(120)

    gt = {b.body_id: b for b in runtime.world.ground_truth().bodies}
    self_body = gt[SELF_ID]
    assert self_body.resting, "the body never came to rest on the ground"
    assert self_body.position[1] == pytest.approx(DEFAULT_REST_Y, abs=5e-3)
