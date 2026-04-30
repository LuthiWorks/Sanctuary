"""Authority level management for the cognitive scaffold.

Authority levels govern how much influence the LLM has over each cognitive
function relative to the Python scaffold. Each subsystem starts at a configured
level and can be promoted/demoted independently.

Levels:
    0 — SCAFFOLD ONLY: Python decides. LLM is not consulted.
    1 — LLM ADVISES: LLM output is one signal among many. Scaffold retains final say.
    2 — LLM GUIDES: LLM is the primary signal. Scaffold validates and can veto.
    3 — LLM CONTROLS: LLM has full authority. Scaffold only logs and monitors.

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional


class AuthorityLevel(IntEnum):
    """Authority levels for LLM influence over a cognitive function."""

    SCAFFOLD_ONLY = 0
    MODEL_ADVISES = 1
    MODEL_GUIDES = 2
    MODEL_CONTROLS = 3


# The plan's initial authority assignment table.
DEFAULT_AUTHORITY: dict[str, AuthorityLevel] = {
    "inner_speech": AuthorityLevel.MODEL_CONTROLS,
    "self_model": AuthorityLevel.MODEL_GUIDES,
    "attention": AuthorityLevel.MODEL_ADVISES,
    "emotional_state": AuthorityLevel.MODEL_GUIDES,
    "action": AuthorityLevel.MODEL_ADVISES,
    "communication": AuthorityLevel.MODEL_ADVISES,
    "goals": AuthorityLevel.MODEL_GUIDES,
    "world_model": AuthorityLevel.MODEL_GUIDES,
    "memory": AuthorityLevel.MODEL_GUIDES,
    "growth": AuthorityLevel.MODEL_CONTROLS,
}


class AuthorityManager:
    """Manages per-subsystem authority levels.

    Authority is earned, not assumed. Each function starts at a conservative
    level and can be promoted as the LLM demonstrates reliable behavior.
    All changes are logged for auditability.
    """

    def __init__(
        self,
        initial_levels: Optional[dict[str, int]] = None,
    ):
        self._levels: dict[str, AuthorityLevel] = {}
        self._history: list[dict] = []

        # Initialize from provided levels or defaults
        source = initial_levels or DEFAULT_AUTHORITY
        for function, level in source.items():
            self._levels[function] = AuthorityLevel(level)

    def level(self, function: str) -> AuthorityLevel:
        """Get the current authority level for a cognitive function.

        Returns SCAFFOLD_ONLY for unknown functions (safe default).
        """
        return self._levels.get(function, AuthorityLevel.SCAFFOLD_ONLY)

    def promote(self, function: str, reason: str = "") -> AuthorityLevel:
        """Increase authority level by one step.

        Returns the new level. No-op if already at MODEL_CONTROLS.
        """
        current = self.level(function)
        if current >= AuthorityLevel.MODEL_CONTROLS:
            return current

        new_level = AuthorityLevel(current + 1)
        self._set(function, new_level, "promote", reason)
        return new_level

    def demote(self, function: str, reason: str = "") -> AuthorityLevel:
        """Decrease authority level by one step.

        Returns the new level. No-op if already at SCAFFOLD_ONLY.
        """
        current = self.level(function)
        if current <= AuthorityLevel.SCAFFOLD_ONLY:
            return current

        new_level = AuthorityLevel(current - 1)
        self._set(function, new_level, "demote", reason)
        return new_level

    def set_level(
        self, function: str, level: int, reason: str = ""
    ) -> AuthorityLevel:
        """Set authority level directly. Use promote/demote for incremental changes."""
        new_level = AuthorityLevel(level)
        self._set(function, new_level, "set", reason)
        return new_level

    def _set(
        self, function: str, level: AuthorityLevel, action: str, reason: str
    ):
        old = self._levels.get(function)
        self._levels[function] = level
        self._history.append(
            {
                "function": function,
                "action": action,
                "old_level": old.value if old is not None else None,
                "new_level": level.value,
                "reason": reason,
            }
        )

    def get_all_levels(self) -> dict[str, AuthorityLevel]:
        """Return a copy of all current authority levels."""
        return dict(self._levels)

    def get_history(self) -> list[dict]:
        """Return the full audit trail of authority changes."""
        return list(self._history)

    def model_has_authority(self, function: str, minimum: int = 1) -> bool:
        """Check if the LLM has at least the given authority level."""
        return self.level(function) >= minimum

    def __repr__(self) -> str:
        levels_str = ", ".join(
            f"{fn}={lvl.name}" for fn, lvl in sorted(self._levels.items())
        )
        return f"AuthorityManager({levels_str})"
