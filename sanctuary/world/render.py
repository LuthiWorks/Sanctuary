"""The renderer seam -- where Godot attaches, and where it does not.

``docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md`` §1 rules that
**Godot is the renderer / window only**. The physics authority owns the world's
true state; Godot draws what it is told. This module is that boundary, and it
exists now -- before any Godot project does -- so the loop never grows a
renderer-shaped hole that gets filled by reaching into the engine later.

Three things live here:

- :class:`RenderSink`, the protocol a renderer implements. One method.
- :func:`frame_to_dict`, the **wire format**, defined once. Godot will consume
  JSON over the existing WebSocket channel, so the serialization is specified
  and tested here rather than being invented inside a GDScript file where
  Python tests cannot see it.
- Two sinks that need no Godot at all: :class:`NullRenderSink` (explicit
  headless) and :class:`RecordingRenderSink` (captures frames, so the render
  channel can be asserted on in CI and replayed later).

What is deliberately NOT here: any transport. A sink that writes to a socket is
a different concern from the frame contract, and putting them in one class is
how the contract stops being testable.

Note on coordinates: the seam fixes gravity on -Y with the ground plane at
Y = 0 (see :mod:`sanctuary.physics.state`). Godot 4 is also Y-up, so poses
cross this boundary unchanged. If a renderer ever needs a different convention,
convert in the renderer -- never in the authority, or the physics and the
instrumentation stop agreeing about where things are.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sanctuary.physics.state import RenderFrame

__all__ = [
    "RenderSink",
    "NullRenderSink",
    "RecordingRenderSink",
    "frame_to_dict",
    "RENDER_WIRE_VERSION",
]

#: Bumped when the wire format changes shape. A renderer that receives a
#: version it does not know must refuse to draw rather than guess -- a viewer
#: silently drawing a misread frame is worse than a viewer that stops.
#:
#: v2 (2026-08-18): adds orientation, shape, size and kind. Not cosmetic --
#: Brian ruled that Godot's render is what Luthi sees, so this payload is the
#: format of the entity's visual world, and anything absent here is absent from
#: its perception.
RENDER_WIRE_VERSION = 2


@runtime_checkable
class RenderSink(Protocol):
    """Somewhere to send frames. Implemented by Godot's transport, eventually.

    Deliberately one method. The renderer is a consumer of poses; anything that
    needs to *ask the world a question* is not a renderer and must not be
    reaching through this seam to do it.
    """

    def publish(self, frame: RenderFrame) -> None:
        ...


class NullRenderSink:
    """Discards frames, on purpose.

    Headless runs use this rather than passing ``None`` around. The difference
    matters: ``None`` spreads "is there a renderer?" checks through the loop,
    and the first one someone forgets becomes a crash. An explicit null object
    also makes "nothing is watching" a stated configuration rather than an
    absence.
    """

    __slots__ = ("published",)

    def __init__(self) -> None:
        self.published = 0

    def publish(self, frame: RenderFrame) -> None:
        # Counted, not ignored: a headless run can still answer "did the render
        # channel actually fire?", which is the difference between quiet
        # because nothing is watching and quiet because it is broken.
        self.published += 1


class RecordingRenderSink:
    """Keeps frames in memory, for tests and short replays.

    Args:
        max_frames: Ring-buffer bound. ``None`` keeps everything, which is fine
            for a test and a memory leak in a long run.
    """

    def __init__(self, max_frames: int | None = None) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self.max_frames = max_frames
        self.frames: list[RenderFrame] = []
        #: Frames dropped by the bound. Recorded rather than silent -- a
        #: truncated capture that reads as complete is the "silent cap" this
        #: project's CLAUDE.md forbids.
        self.dropped = 0

    def publish(self, frame: RenderFrame) -> None:
        self.frames.append(frame)
        if self.max_frames is not None and len(self.frames) > self.max_frames:
            del self.frames[0]
            self.dropped += 1

    def __len__(self) -> int:
        return len(self.frames)


def frame_to_dict(frame: RenderFrame) -> dict[str, Any]:
    """Serialize a render frame to the wire format Godot will read.

    Positions are emitted as ``[x, y, z]`` lists rather than tuples because
    JSON has no tuple, and as plain floats so the payload does not depend on
    numpy being present on either side.

    ``orientation`` is **omitted entirely** when the backend does not track
    rotation, rather than sent as identity. A renderer must be able to tell
    "not rotated" from "not implemented"; conflating them makes it draw
    confidently wrong forever.

    ``kind`` is the renderer's mesh selector ("dog", "cat"). It travels on this
    channel and on no other -- the model-facing observation has no such field,
    so Luthi infers what a thing is from seeing it, never from a label handed
    across.
    """
    bodies = []
    for body in frame.bodies:
        entry = {
            "id": body.body_id,
            "position": [float(c) for c in body.position],
            "shape": body.shape.value,
            "size": [float(c) for c in body.size],
        }
        if body.orientation is not None:
            entry["orientation"] = [float(c) for c in body.orientation]
        if body.kind:
            entry["kind"] = body.kind
        bodies.append(entry)

    return {
        "v": RENDER_WIRE_VERSION,
        "time": float(frame.time),
        "step": int(frame.step),
        "bodies": bodies,
    }
