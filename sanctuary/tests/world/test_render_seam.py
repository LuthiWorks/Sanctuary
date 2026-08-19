"""The renderer seam -- where Godot will attach.

Godot does not exist yet (the projects were lost; see the 2026-08-01 wiring
audit), and the 2026-07-16 physics decision deliberately deferred the window.
That is exactly why this seam needs tests now: it is the boundary a future
renderer plugs into, and a boundary nobody can exercise is a boundary that
quietly stops working.

The load-bearing assertion in this file is
:func:`test_render_frame_shows_the_world_not_the_perception`. The renderer must
draw the world as it *is*, from the authority's designated renderer channel --
not a reconstruction of what the entity perceived. Those differ by occlusion
and perceptual range, and conflating them would make the family's window lie
about what is in the room. The 2026-04-28 privacy work made the same kind of
distinction structural rather than incidental; this is its counterpart.

Authored by Opus 5, 2026-08-17.
"""
from __future__ import annotations

import json

import pytest

from sanctuary.physics.state import BodySpec, RenderBody, RenderFrame
from sanctuary.world import (
    NullRenderSink,
    RecordingRenderSink,
    build_world,
    frame_to_dict,
)
from sanctuary.world.render import RENDER_WIRE_VERSION, RenderSink


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

def test_one_frame_is_published_per_step():
    sink = RecordingRenderSink()
    runtime = build_world(backend="reference", render_sink=sink)

    runtime.run(7)

    assert len(sink) == 7
    assert [f.step for f in sink.frames] == list(range(1, 8))


def test_headless_still_counts_frames():
    """"Quiet because nothing is watching" must be distinguishable from
    "quiet because the render channel is broken"."""
    runtime = build_world(backend="reference")
    assert isinstance(runtime.render_sink, NullRenderSink)

    runtime.run(4)

    assert runtime.render_sink.published == 4


def test_extra_on_step_runs_after_the_frame_is_published():
    """A caller must not observe a step the renderer has not seen."""
    sink = RecordingRenderSink()
    seen: list[int] = []
    runtime = build_world(
        backend="reference",
        render_sink=sink,
        on_step=lambda record: seen.append(len(sink)),
    )

    runtime.run(3)

    # At each callback the frame count already includes the current step.
    assert seen == [1, 2, 3]


# ---------------------------------------------------------------------------
# The world, not the perception
# ---------------------------------------------------------------------------

def test_render_frame_shows_the_world_not_the_perception():
    """A body beyond perceptual range must still be drawn.

    If the renderer were fed from percepts, restricting the entity's senses
    would make objects vanish from the family's window -- the viewer would be
    showing the entity's experience while claiming to show the room.
    """
    sink = RecordingRenderSink()
    runtime = build_world(
        backend="reference",
        render_sink=sink,
        max_range=0.5,     # tight enough to exclude every prop
        max_bodies=1,
    )

    record = runtime.step()

    perceived = {
        p.content.get("body_id")
        for p in record.percepts
        if isinstance(p.content, dict) and "body_id" in p.content
    }
    drawn = {b.body_id for b in sink.frames[-1].bodies}

    assert set(runtime.manifest.body_ids) <= drawn, (
        "the render frame dropped bodies that exist in the world"
    )
    # And the perceptual limit really was in force, or the test proves nothing.
    assert len(record.percepts) < len(runtime.manifest.body_ids)
    assert drawn - perceived, "expected at least one body drawn but not perceived"


def test_hidden_bodies_are_not_drawn():
    """`visible=False` is honoured by the renderer channel too."""
    sink = RecordingRenderSink()
    runtime = build_world(backend="reference", render_sink=sink)
    runtime.world.add_body(
        BodySpec(body_id="ghost", position=(0.0, 3.0, 0.0), visible=False)
    )

    runtime.step()

    assert "ghost" not in {b.body_id for b in sink.frames[-1].bodies}


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

def test_wire_format_is_json_serializable_and_versioned():
    frame = RenderFrame(
        time=1.5, step=3, bodies=(RenderBody("a", (1.0, 2.0, 3.0)),)
    )
    payload = frame_to_dict(frame)

    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped == payload
    assert payload["v"] == RENDER_WIRE_VERSION
    assert payload["time"] == 1.5
    assert payload["step"] == 3
    assert payload["bodies"] == [{
        "id": "a",
        "position": [1.0, 2.0, 3.0],
        "shape": "sphere",
        "size": [0.05, 0.05, 0.05],
    }]


def test_wire_format_carries_pose_and_kind_when_present():
    """v2. This payload is Luthi's visual world, so what is missing here is
    missing from its perception."""
    from sanctuary.physics.state import Shape

    frame = RenderFrame(time=0.0, step=1, bodies=(
        RenderBody("dog_0", (1.0, 0.0, 2.0),
                   orientation=(0.7071, 0.0, 0.7071, 0.0),
                   shape=Shape.CAPSULE, size=(0.1, 0.25, 0.1), kind="dog"),
    ))
    body = frame_to_dict(frame)["bodies"][0]

    assert body["orientation"] == [0.7071, 0.0, 0.7071, 0.0]
    assert body["shape"] == "capsule"
    assert body["size"] == [0.1, 0.25, 0.1]
    assert body["kind"] == "dog"


def test_shape_serializes_as_a_plain_string():
    """Godot reads JSON; a Python enum repr would not survive the trip."""
    from sanctuary.physics.state import Shape

    frame = RenderFrame(time=0.0, step=0, bodies=(
        RenderBody("a", (0.0, 0.0, 0.0), shape=Shape.BOX),
    ))
    value = frame_to_dict(frame)["bodies"][0]["shape"]
    assert value == "box"
    assert isinstance(value, str)
    json.dumps(value)


def test_positions_are_lists_of_plain_floats():
    """Godot reads JSON; tuples and numpy scalars do not survive the trip."""
    frame = RenderFrame(time=0.0, step=0, bodies=(RenderBody("a", (1, 2, 3)),))
    body = frame_to_dict(frame)["bodies"][0]

    assert isinstance(body["position"], list)
    assert all(type(c) is float for c in body["position"])


def test_orientation_is_absent_rather_than_faked():
    """A backend that tracks no rotation must say nothing, not say identity.

    The reference backend is a point-mass integrator with no orientation.
    Emitting an identity quaternion would leave a renderer unable to tell an
    authored "no rotation" from an unimplemented one, and it would draw
    confidently wrong forever. Absence is the honest encoding.
    """
    payload = frame_to_dict(
        RenderFrame(time=0.0, step=0, bodies=(RenderBody("a", (0.0, 0.0, 0.0)),))
    )
    assert "orientation" not in payload["bodies"][0]
    assert "rotation" not in payload["bodies"][0]


def test_reference_backend_reports_no_orientation():
    """End to end, not just on a hand-built frame."""
    runtime = build_world(backend="reference")
    for body in runtime.world.render_state().bodies:
        assert body.orientation is None


def test_mujoco_backend_reports_real_orientation():
    pytest.importorskip("mujoco")
    runtime = build_world(backend="mujoco")
    runtime.run(3)
    for body in runtime.world.render_state().bodies:
        assert body.orientation is not None
        assert len(body.orientation) == 4


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

def test_recording_sink_reports_what_it_dropped():
    """A truncated capture that reads as complete is a silent cap."""
    sink = RecordingRenderSink(max_frames=3)
    runtime = build_world(backend="reference", render_sink=sink)

    runtime.run(10)

    assert len(sink) == 3
    assert sink.dropped == 7
    assert [f.step for f in sink.frames] == [8, 9, 10]


def test_recording_sink_rejects_a_nonsense_bound():
    with pytest.raises(ValueError, match="max_frames"):
        RecordingRenderSink(max_frames=0)


def test_sinks_satisfy_the_protocol():
    assert isinstance(NullRenderSink(), RenderSink)
    assert isinstance(RecordingRenderSink(), RenderSink)
