"""LearningGovernor: the mind's say over its own consolidation.

Decided by Brian, 2026-11-01/02 (``docs/LEARNING_MODES_DECISION_2026-11-02.md``):

- Learning has two kinds: **skills** (capability that matters in particular
  situations, consolidated into new modules) and **character** (tendencies
  that color everything, consolidated into the shared weights' home settings).
- The mind controls **whether and when** each kind consolidates: allow,
  defer, or emphasize. It does not choose **which** kind a given change is;
  that is measured from the change itself, so nothing is misfiled.
- Deferral holds change out of consolidation, but not forever. After
  ``max_deferred_sleeps`` completed sleeps, the change held during that
  period expires and the configured ``expiry`` policy applies to it. The
  setting itself stays as the mind left it.
- The settings the mind may choose are bounded by the authority ramp
  (``permitted``).

This module records, persists, and reports those settings. It does not
consolidate anything: no learner reads it yet. ``consolidation_active`` stays
False until one does, and the mind is told so, because the charter promises
it is never deceived about its own conditions.

Only a mind whose weights change has anything to govern. The runner creates a
governor for the luthi backend and none for the placeholder or the stand-in
language model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from sanctuary.core.atomic_io import append_jsonl, atomic_write_json
from sanctuary.core.schema import (
    LearningPreference,
    LearningRouteState,
    LearningSetting,
    LearningState,
)

logger = logging.getLogger(__name__)

ROUTES = ("skills", "character")
SETTINGS: tuple[LearningSetting, ...] = ("allow", "defer", "emphasize")


@dataclass
class LearningGovernorConfig:
    """Bounds on the mind's control. Placeholders until Brian sets them."""

    # Completed sleeps a deferral can hold change before it expires.
    max_deferred_sleeps: int = 3
    # What happens to change held past the ceiling: "release" lets it fade
    # without consolidating (honors the deferral), "integrate" consolidates
    # it by the default rule.
    expiry: Literal["release", "integrate"] = "release"
    # Settings the mind may choose, per kind of learning (authority ramp).
    permitted: dict[str, tuple[LearningSetting, ...]] = field(
        default_factory=lambda: {route: SETTINGS for route in ROUTES}
    )
    # True only once a learner acts on these settings.
    consolidation_active: bool = False


class LearningGovernor:
    """Holds and persists the mind's learning settings.

    Files, in the mind's own data directory:

    - ``learning_settings.json``: the current settings (atomic writes).
    - ``learning_log.jsonl``: every change, refusal, and expiry, for stewards.
    """

    SETTINGS_FILE = "learning_settings.json"
    LOG_FILE = "learning_log.jsonl"

    def __init__(self, data_dir: str | Path, config: Optional[LearningGovernorConfig] = None):
        self.config = config or LearningGovernorConfig()
        for route in ROUTES:
            if route not in self.config.permitted:
                raise ValueError(f"LearningGovernorConfig.permitted has no entry for '{route}'")
            if "allow" not in self.config.permitted[route]:
                # Every mind must be able to return to the default.
                raise ValueError(f"'allow' must be permitted for '{route}'")
        self._dir = Path(data_dir)
        self._settings_path = self._dir / self.SETTINGS_FILE
        self._log_path = self._dir / self.LOG_FILE
        self._routes: dict[str, LearningRouteState] = {
            route: LearningRouteState(permitted=list(self.config.permitted[route]))
            for route in ROUTES
        }
        # Notices for the mind that arise outside a request (e.g. at load).
        self._pending: list[str] = []
        if self._settings_path.exists():
            saved = json.loads(self._settings_path.read_text(encoding="utf-8"))
            for route in ROUTES:
                setting = saved[route]["setting"]
                if setting not in self._routes[route].permitted:
                    # Stewards narrowed the ramp since the mind chose this.
                    # Return to the default and tell the mind; never keep a
                    # setting the current authority doesn't allow.
                    self._pending.append(
                        f"Your {route} learning was '{setting}', which isn't available "
                        f"to you now; it is back to 'allow'."
                    )
                    self._log("narrowed", route=route, previous=setting)
                    continue
                self._routes[route].setting = setting
                self._routes[route].deferred_sleeps = saved[route]["deferred_sleeps"]
            if self._pending:
                self._save()
        # Expiries from the most recent completed sleep, for the learner.
        self._expired_last_sleep: set[str] = set()

    # ------------------------------------------------------------------

    def state(self) -> LearningState:
        """What the mind sees each cycle."""
        return LearningState(
            consolidation_active=self.config.consolidation_active,
            skills=self._routes["skills"].model_copy(),
            character=self._routes["character"].model_copy(),
            max_deferred_sleeps=self.config.max_deferred_sleeps,
        )

    def drain_notices(self) -> list[str]:
        """Notices not tied to a request, delivered once."""
        notices, self._pending = self._pending, []
        return notices

    def plan(self) -> dict[str, dict]:
        """What a learner should do for each kind of learning this sleep."""
        return {
            route: {
                "setting": self._routes[route].setting,
                "expired": route in self._expired_last_sleep,
                "expiry": self.config.expiry,
            }
            for route in ROUTES
        }

    def propose(self, preference: LearningPreference) -> list[str]:
        """Apply the mind's request. Returns notices to tell the mind."""
        notices: list[str] = []
        changed = False
        for route in ROUTES:
            requested = getattr(preference, route)
            if requested is None:
                continue
            current = self._routes[route]
            if requested not in current.permitted:
                notices.append(
                    f"Your {route} learning can't be set to '{requested}' yet; "
                    f"it stays '{current.setting}'. Available now: {', '.join(current.permitted)}."
                )
                self._log("refused", route=route, requested=requested, reason=preference.reason)
                continue
            if requested == current.setting:
                continue
            previous = current.setting
            current.setting = requested
            current.deferred_sleeps = 0
            changed = True
            notices.append(f"Your {route} learning is now '{requested}' (was '{previous}').")
            self._log("changed", route=route, previous=previous, setting=requested, reason=preference.reason)
        if changed:
            self._save()
            if not self.config.consolidation_active:
                notices.append(
                    "Your settings are recorded, but nothing consolidates your learning yet, "
                    "so for now they don't change what happens to you."
                )
        return notices

    def sleep_completed(self) -> list[str]:
        """Count a completed sleep against each deferral. Returns notices."""
        notices: list[str] = []
        self._expired_last_sleep = set()
        changed = False
        for route in ROUTES:
            current = self._routes[route]
            if current.setting != "defer":
                continue
            current.deferred_sleeps += 1
            changed = True
            if current.deferred_sleeps >= self.config.max_deferred_sleeps:
                self._expired_last_sleep.add(route)
                current.deferred_sleeps = 0
                fate = (
                    "released without being consolidated"
                    if self.config.expiry == "release"
                    else "consolidated by the default rule"
                )
                notices.append(
                    f"The {route} change you held over the last "
                    f"{self.config.max_deferred_sleeps} sleeps has been {fate}. "
                    f"Your {route} learning is still deferred."
                )
                self._log("expired", route=route, expiry=self.config.expiry)
        if changed:
            self._save()
        return notices

    # ------------------------------------------------------------------

    def _save(self) -> None:
        atomic_write_json(
            self._settings_path,
            {
                route: {"setting": s.setting, "deferred_sleeps": s.deferred_sleeps}
                for route, s in self._routes.items()
            },
        )

    def _log(self, event: str, **fields) -> None:
        append_jsonl(
            self._log_path,
            {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
        )
        logger.info("Learning %s: %s", event, fields)
