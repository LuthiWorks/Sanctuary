"""The embodied loop — where the training ground actually runs.

`sanctuary/physics/` owns the world, `sensorium/physics_percepts` and
`sensorium/physics_vision` turn it into perception, and
`motor/physics_actuation` turns a chosen action into force. This package is the
thing that puts them in a cycle and presses play.

    perceive -> decide -> act -> step the world -> perceive the consequence
                              \-> observe_transition (learn from it)

Everything crossing this loop is already tested in isolation. What the runner
adds is *sequencing*, which is where the remaining defects live: the order
forces are cleared in, whether a percept describes the world before or after a
step, whether the clock the learner sees matches the clock the physics used.
"""

from __future__ import annotations

from sanctuary.embodiment.episode import (
    EpisodeRunner,
    Policy,
    RestingPolicy,
    StepRecord,
)

__all__ = ["EpisodeRunner", "Policy", "RestingPolicy", "StepRecord"]
