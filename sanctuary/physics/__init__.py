"""The physics-authority seam for the developmental world.

Architecture decision: ``docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md``.

The developmental (rover) world is a hybrid: **Godot renders, an external
authority owns the physics.** This package is that authority's boundary — the
*seam*. The seam is the load-bearing commitment; the engine behind it is
swappable, and both engines now exist:

* :class:`~sanctuary.physics.backends.reference.ReferencePhysicsAuthority` —
  dependency-free point-mass dynamics. Deterministic, instant, and what CI runs.
* :class:`~sanctuary.physics.backends.mujoco_backend.MuJoCoPhysicsAuthority` —
  real rigid-body dynamics with solved contact, for the developmental world.
  Requires the optional ``mujoco`` extra (``uv sync --extra mujoco``) and is
  imported lazily, so this package stays importable without it.

They differ only where physics genuinely differs — a MuJoCo body has volume and
compliant contacts, so it rests at its radius rather than at exactly zero.
``tests/physics/test_mujoco_backend.py`` holds both to the same contract and
diffs their behaviour against each other.

One producer (a :class:`PhysicsAuthority`), three consumers, each with its own
view of the world so the wrong data can never reach the wrong place:

* :meth:`PhysicsAuthority.observe` -> :class:`PhysicsObservation` — the
  **model-facing** view. Partial/occluded (invisible bodies excluded), and
  kinematics only. Safe to turn into percepts.
* :meth:`PhysicsAuthority.ground_truth` -> :class:`PhysicsGroundTruth` — the
  **instrumentation-only** view (LUTHISCOPE). The full true state, including
  hidden fields (mass, forces). This is the spec's "hidden ground-truth
  channel, exposed only to instrumentation, not to the model" (spec sec. 3).
  **It must never be fed to the model.**
* :meth:`PhysicsAuthority.render_state` -> :class:`RenderFrame` — poses for the
  **renderer** (Godot) to draw.

The observation/ground-truth split is structural, not a convention:
``PhysicsObservation`` has no field that could carry mass or force, so the
hidden channel cannot leak into the model by accident. See
``test_seam.py::test_observation_cannot_carry_hidden_fields``.
"""

from __future__ import annotations

from sanctuary.physics.authority import PhysicsAuthority
from sanctuary.physics.state import (
    BodySpec,
    BodyState,
    GroundTruthBody,
    PhysicsGroundTruth,
    PhysicsObservation,
    RenderBody,
    RenderFrame,
)

__all__ = [
    "PhysicsAuthority",
    "BodySpec",
    "BodyState",
    "GroundTruthBody",
    "PhysicsGroundTruth",
    "PhysicsObservation",
    "RenderBody",
    "RenderFrame",
]
