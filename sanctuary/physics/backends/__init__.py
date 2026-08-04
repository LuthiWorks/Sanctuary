"""Physics backends behind the :class:`~sanctuary.physics.authority.PhysicsAuthority` seam.

- :class:`reference.ReferencePhysicsAuthority` — dependency-free, deterministic.
  Point-mass gravity + ground plane. Proves the seam and serves tests/CI.
- :class:`mujoco_backend.MuJoCoPhysicsAuthority` — real rigid-body dynamics with
  contact, for the developmental world. Requires the optional ``mujoco`` extra
  (``uv sync --extra mujoco``).

``MuJoCoPhysicsAuthority`` is exported lazily so that importing this package —
and therefore ``sanctuary.physics`` — does not require ``mujoco`` to be
installed. Referencing the name imports the backend module; only then is the
optional dependency needed, and its absence raises with install instructions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sanctuary.physics.backends.reference import ReferencePhysicsAuthority

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sanctuary.physics.backends.mujoco_backend import MuJoCoPhysicsAuthority

__all__ = ["ReferencePhysicsAuthority", "MuJoCoPhysicsAuthority"]


def __getattr__(name: str) -> Any:  # PEP 562
    if name == "MuJoCoPhysicsAuthority":
        from sanctuary.physics.backends.mujoco_backend import (
            MuJoCoPhysicsAuthority as _MuJoCoPhysicsAuthority,
        )

        return _MuJoCoPhysicsAuthority
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
