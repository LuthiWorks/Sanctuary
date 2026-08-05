"""Tests for camera rendering and the frame -> Percept vision adapter.

The failure modes here are all quiet ones. A camera inside its own body renders
a valid black image forever. A normalization mismatch produces a correctly
shaped tensor the encoder accepts and learns miscoloured input from. A renderer
held across a model rebuild shows the old world while every other view reports
the new one. None of those raise on their own.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sanctuary.physics import BodySpec

mujoco = pytest.importorskip("mujoco")

from sanctuary.physics.backends.mujoco_backend import (  # noqa: E402
    CameraFrame,
    CameraSpec,
    MuJoCoPhysicsAuthority,
)
from sanctuary.sensorium.physics_vision import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    VISUAL,
    frame_to_percept,
    frame_to_tensor,
)

# A radius that makes objects legible at a few metres, rather than the 5 cm
# default which subtends almost no angle.
VISIBLE_RADIUS = 0.5


def _seeing_world() -> MuJoCoPhysicsAuthority:
    world = MuJoCoPhysicsAuthority(body_radius=VISIBLE_RADIUS)
    world.add_body(BodySpec(body_id="rover", position=(0.0, 1.0, 0.0)))
    world.add_body(BodySpec(body_id="target", position=(3.0, 1.0, 0.0), static=True))
    world.attach_camera(
        CameraSpec(name="eye", body_id="rover", offset=(VISIBLE_RADIUS + 0.1, 0.0, 0.0))
    )
    return world


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_returns_a_frame_of_the_requested_size():
    frame = _seeing_world().render_camera("eye", width=224, height=224)

    assert isinstance(frame, CameraFrame)
    assert frame.rgb.shape == (224, 224, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.width == 224 and frame.height == 224


def test_the_camera_actually_sees_the_world():
    """Removing what is in front of the eye must change what the eye reports."""
    world = _seeing_world()
    with_target = world.render_camera("eye").rgb

    world.remove_body("target")
    without_target = world.render_camera("eye").rgb

    changed = (np.abs(with_target.astype(int) - without_target.astype(int)).sum(axis=2) > 10)
    assert changed.sum() > 500, "the target should occupy a substantial part of the view"


def test_moving_the_body_changes_the_view():
    world = _seeing_world()
    before = world.render_camera("eye").rgb

    world.apply_force("rover", (200.0, 0.0, 0.0))
    world.step(0.2)
    after = world.render_camera("eye").rgb

    assert not np.array_equal(before, after)


def test_frame_carries_the_world_clock():
    world = _seeing_world()
    world.step(0.1)
    world.step(0.1)

    frame = world.render_camera("eye")

    assert frame.step == 2
    assert frame.time == pytest.approx(0.2)


def test_camera_inside_its_own_body_is_refused():
    """A valid, near-black frame forever -- nothing downstream would flag it."""
    world = MuJoCoPhysicsAuthority(body_radius=0.5)
    world.add_body(BodySpec(body_id="rover", position=(0.0, 1.0, 0.0)))

    with pytest.raises(ValueError, match="inside that body's own geom"):
        world.attach_camera(CameraSpec(name="eye", body_id="rover"))


def test_camera_inside_body_error_suggests_a_working_offset():
    world = MuJoCoPhysicsAuthority(body_radius=0.5)
    world.add_body(BodySpec(body_id="rover", position=(0.0, 1.0, 0.0)))

    with pytest.raises(ValueError) as excinfo:
        world.attach_camera(CameraSpec(name="eye", body_id="rover", offset=(0.1, 0, 0)))

    assert "offset=(0.750, 0, 0)" in str(excinfo.value)


def test_world_fixed_camera_needs_no_offset_guard():
    world = MuJoCoPhysicsAuthority(body_radius=0.5)
    world.add_body(BodySpec(body_id="thing", position=(3.0, 1.0, 0.0)))

    world.attach_camera(CameraSpec(name="observer"))

    assert world.render_camera("observer").rgb.shape == (224, 224, 3)


def test_duplicate_camera_name_raises():
    world = _seeing_world()
    with pytest.raises(ValueError):
        world.attach_camera(CameraSpec(name="eye", body_id="rover", offset=(2.0, 0, 0)))


def test_camera_on_unknown_body_raises():
    world = MuJoCoPhysicsAuthority()
    with pytest.raises(KeyError):
        world.attach_camera(CameraSpec(name="eye", body_id="ghost", offset=(1.0, 0, 0)))


def test_unknown_camera_raises():
    world = _seeing_world()
    with pytest.raises(KeyError, match="nope"):
        world.render_camera("nope")


def test_rendering_an_empty_world_raises_rather_than_returning_black():
    world = MuJoCoPhysicsAuthority()
    world.attach_camera(CameraSpec(name="observer"))

    with pytest.raises(RuntimeError, match="empty world"):
        world.render_camera("observer")


def test_renderer_follows_a_model_rebuild():
    """A stale renderer would draw the old world while every other view moved on."""
    world = _seeing_world()
    world.render_camera("eye")  # build a renderer against the current model

    world.add_body(BodySpec(body_id="newcomer", position=(1.2, 1.0, 0.0), static=True))
    after = world.render_camera("eye").rgb

    world.remove_body("newcomer")
    without = world.render_camera("eye").rgb

    assert not np.array_equal(after, without)


def test_objects_close_to_the_eye_are_visible():
    """Regression: the near plane must not scale with world size.

    MuJoCo expresses the near plane as a fraction of the model extent, which is
    inferred from world size -- so the default 100 m ground silently pushed it
    to 0.20 m and made anything within 20 cm of the eye invisible, while every
    frame still rendered cleanly. A body could not see what it was about to
    touch, and nothing raised. The extent is now pinned so the near plane is an
    absolute distance.
    """
    world = MuJoCoPhysicsAuthority(body_radius=0.5)
    world.add_body(BodySpec(body_id="rover", position=(0.0, 1.0, 0.0)))
    world.attach_camera(CameraSpec(name="eye", body_id="rover", offset=(0.6, 0.0, 0.0)))
    empty_view = world.render_camera("eye").rgb

    # Surface only 0.1 m from the eye -- inside the old 0.20 m near plane.
    world.add_body(BodySpec(body_id="close", position=(1.2, 1.0, 0.0), static=True))
    close_view = world.render_camera("eye").rgb

    assert not np.array_equal(empty_view, close_view)


def test_near_plane_is_absolute_regardless_of_ground_size():
    for ground in (5.0, 100.0, 1000.0):
        world = MuJoCoPhysicsAuthority(body_radius=0.5, ground_size=ground)
        world.add_body(BodySpec(body_id="b", position=(0.0, 1.0, 0.0)))
        world.attach_camera(CameraSpec(name="eye", body_id="b", offset=(0.6, 0, 0)))
        world.render_camera("eye")

        model = world._model
        actual = model.vis.map.znear * model.stat.extent
        assert actual == pytest.approx(world.near_plane, rel=1e-3), (
            f"near plane drifted to {actual} m at ground_size={ground}"
        )


def test_near_plane_beyond_body_radius_is_refused():
    """A camera on a body would clip everything adjacent to it."""
    with pytest.raises(ValueError, match="near_plane"):
        MuJoCoPhysicsAuthority(body_radius=0.05, near_plane=0.1)


def test_camera_survives_reset_as_absent():
    world = _seeing_world()
    world.reset()

    assert world.camera_names == ()


def test_body_colours_are_deterministic_across_runs():
    """Recoloured objects between runs would be noise for a world model."""
    first = _seeing_world().render_camera("eye").rgb
    second = _seeing_world().render_camera("eye").rgb

    assert np.array_equal(first, second)


def test_different_bodies_get_different_colours():
    world = MuJoCoPhysicsAuthority(body_radius=VISIBLE_RADIUS)
    world.add_body(BodySpec(body_id="rover", position=(0.0, 1.0, 0.0)))
    world.add_body(BodySpec(body_id="alpha", position=(3.0, 1.0, 0.0), static=True))
    world.attach_camera(
        CameraSpec(name="eye", body_id="rover", offset=(VISIBLE_RADIUS + 0.1, 0, 0))
    )
    alpha_view = world.render_camera("eye").rgb

    other = MuJoCoPhysicsAuthority(body_radius=VISIBLE_RADIUS)
    other.add_body(BodySpec(body_id="rover", position=(0.0, 1.0, 0.0)))
    other.add_body(BodySpec(body_id="omega", position=(3.0, 1.0, 0.0), static=True))
    other.attach_camera(
        CameraSpec(name="eye", body_id="rover", offset=(VISIBLE_RADIUS + 0.1, 0, 0))
    )
    omega_view = other.render_camera("eye").rgb

    assert not np.array_equal(alpha_view, omega_view)


# ---------------------------------------------------------------------------
# The encoder contract
# ---------------------------------------------------------------------------


def test_tensor_has_the_shape_encode_vision_expects():
    tensor = frame_to_tensor(_seeing_world().render_camera("eye"))

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_normalization_matches_the_training_pipeline_exactly():
    """Reproduce luthi/coco_data.py::_load_image on a known frame."""
    frame = _seeing_world().render_camera("eye")

    expected = torch.from_numpy(frame.rgb).to(torch.float32).div(255.0)
    expected = expected.permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    expected = ((expected - mean) / std).unsqueeze(0)

    assert torch.allclose(frame_to_tensor(frame), expected)


def test_imagenet_constants_match_luthi():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def test_a_mid_grey_frame_normalizes_to_the_expected_values():
    """An independent check that does not restate the implementation."""
    grey = np.full((224, 224, 3), 128, dtype=np.uint8)
    tensor = frame_to_tensor(CameraFrame(camera="c", time=0.0, step=0, rgb=grey))

    for channel in range(3):
        expected = (128 / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert tensor[0, channel].mean().item() == pytest.approx(expected, rel=1e-5)


def test_non_square_frame_is_refused_not_rescaled():
    oblong = np.zeros((100, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="square"):
        frame_to_tensor(CameraFrame(camera="c", time=0.0, step=0, rgb=oblong))


def test_wrong_size_frame_is_refused():
    small = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="expects"):
        frame_to_tensor(CameraFrame(camera="c", time=0.0, step=0, rgb=small))


def test_expected_size_none_allows_another_resolution():
    small = np.zeros((64, 64, 3), dtype=np.uint8)

    tensor = frame_to_tensor(
        CameraFrame(camera="c", time=0.0, step=0, rgb=small), expected_size=None
    )

    assert tensor.shape == (1, 3, 64, 64)


def test_non_rgb_frame_is_refused():
    mono = np.zeros((224, 224), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        frame_to_tensor(CameraFrame(camera="c", time=0.0, step=0, rgb=mono))


# ---------------------------------------------------------------------------
# The percept
# ---------------------------------------------------------------------------


def test_percept_carries_the_image_itself():
    """Unlike proprioception, this channel has an encoder to route to."""
    percept = frame_to_percept(_seeing_world().render_camera("eye"))

    assert percept.modality == VISUAL
    assert percept.tensor_data is not None
    assert percept.tensor_data.shape == (1, 3, 224, 224)


def test_percept_text_describes_seeing_not_the_scene():
    """Naming what is in view would hand the model labels it never earned."""
    percept = frame_to_percept(
        _seeing_world().render_camera("eye"), description="forward eye"
    )

    assert "[vision:eye]" in percept.content
    assert "forward eye" in percept.content
    assert "224x224" in percept.content
    assert "target" not in percept.content


def test_percept_source_identifies_camera_and_step():
    world = _seeing_world()
    world.step(0.1)

    percept = frame_to_percept(world.render_camera("eye"))

    assert "camera=eye" in percept.source
    assert "step=1" in percept.source


def test_tensor_data_is_excluded_from_serialization():
    """Percept declares tensor_data in-memory only; confirm it does not ship."""
    percept = frame_to_percept(_seeing_world().render_camera("eye"))

    assert "tensor_data" not in percept.model_dump()


# ---------------------------------------------------------------------------
# Alongside the other adapters
# ---------------------------------------------------------------------------


def test_vision_and_spatial_perception_describe_the_same_world():
    from sanctuary.sensorium.physics_percepts import observation_to_percepts

    world = _seeing_world()

    spatial = observation_to_percepts(world.observe(), self_id="rover")
    visual = frame_to_percept(world.render_camera("eye"))

    # Both report the same instant, through different channels.
    assert any("target" in p.content for p in spatial)
    assert visual.tensor_data is not None
    assert all(p.tensor_data is None for p in spatial)  # proprioception still has no encoder
