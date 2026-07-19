"""Physics backends behind the :class:`~sanctuary.physics.authority.PhysicsAuthority` seam.

- :class:`reference.ReferencePhysicsAuthority` — dependency-free, deterministic.
  Point-mass gravity + ground plane. Proves the seam and serves tests/CI.
- (later) a MuJoCo backend for contact-rich rover/terrain dynamics, behind the
  same interface.
"""

from __future__ import annotations

from sanctuary.physics.backends.reference import ReferencePhysicsAuthority

__all__ = ["ReferencePhysicsAuthority"]
