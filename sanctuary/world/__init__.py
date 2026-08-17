"""The developmental world -- the physical one.

Not to be confused with :mod:`sanctuary.environment`, which is the *text-based*
inner landscape (rooms, concepts, a place to be). This package is the lawful
physical world the entity is embodied in: mass, gravity, contact, and
consequences that follow from what it does.

Layout:

- :mod:`sanctuary.world.scene` -- what is placed in the world.
- :mod:`sanctuary.world.runtime` -- assembling authority + body + loop.
- :mod:`sanctuary.world.render` -- the renderer seam where Godot attaches.

Physics itself lives in :mod:`sanctuary.physics` behind the authority seam;
perception in :mod:`sanctuary.sensorium.physics_percepts`; actuation in
:mod:`sanctuary.motor.physics_actuation`; the loop in
:mod:`sanctuary.embodiment.episode`. This package connects them and adds no
physics of its own.
"""

from __future__ import annotations

from sanctuary.world.render import (
    NullRenderSink,
    RecordingRenderSink,
    RenderSink,
    frame_to_dict,
)
from sanctuary.world.runtime import BACKENDS, WorldRuntime, build_world
from sanctuary.world.scene import SELF_ID, SceneManifest, build_bedrock_scene

__all__ = [
    "build_world",
    "WorldRuntime",
    "BACKENDS",
    "build_bedrock_scene",
    "SceneManifest",
    "SELF_ID",
    "RenderSink",
    "NullRenderSink",
    "RecordingRenderSink",
    "frame_to_dict",
]
