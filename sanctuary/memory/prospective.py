"""Prospective memory — future intentions and deferred thoughts.

Prospective memory holds things the entity wants to remember to do later:
  - "Next time Alice mentions her project, ask how it's going"
  - "Reflect on today's conversation during idle time"
  - "After 10 more cycles, revisit the goal I abandoned"

Intentions can be triggered by:
  - Time (after N cycles, after N seconds)
  - Context (when a keyword or topic appears in percepts)
  - Event (when a specific condition is met)

Triggered intentions are surfaced as SurfacedMemory objects in the
cognitive cycle input, so the entity can act on them naturally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sanctuary.core.schema import SurfacedMemory

logger = logging.getLogger(__name__)


@dataclass
class Intention:
    """A single prospective memory — something to remember to do.

    Attributes:
        id: Unique identifier.
        content: What to do / remember.
        trigger_type: How this intention is activated ("cycle", "keyword", "idle").
        trigger_value: The trigger condition (cycle count, keyword string, etc.).
        created_at: When this was created.
        created_cycle: Which cycle created it.
        significance: How important this is (1-10).
        tags: Semantic labels.
        triggered: Whether this has been triggered (and thus surfaced).
        expired: Whether this has expired without triggering.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    trigger_type: str = "cycle"  # "cycle", "keyword", "idle"
    trigger_value: str = ""  # cycle count, keyword, or "idle"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_cycle: int = 0
    significance: int = 5
    tags: list[str] = field(default_factory=list)
    triggered: bool = False
    expired: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "trigger_type": self.trigger_type,
            "trigger_value": self.trigger_value,
            "created_at": self.created_at,
            "created_cycle": self.created_cycle,
            "significance": self.significance,
            "tags": self.tags,
            "triggered": self.triggered,
            "expired": self.expired,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Intention:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProspectiveConfig:
    """Configuration for prospective memory."""

    max_intentions: int = 50
    max_age_cycles: int = 1000  # Expire intentions older than this
    max_triggered_per_cycle: int = 3  # Don't overwhelm the entity


class ProspectiveMemory:
    """Manages future intentions — things to remember to do later.

    Usage::

        pm = ProspectiveMemory()
        pm.add("Ask Alice about her project", trigger_type="keyword",
               trigger_value="alice", significance=7)
        pm.add("Reflect on conversation patterns", trigger_type="cycle",
               trigger_value="50")  # After 50 more cycles
        triggered = pm.check(cycle=current_cycle, context="Alice said hello")
    """

    def __init__(self, config: Optional[ProspectiveConfig] = None):
        self._config = config or ProspectiveConfig()
        self._intentions: list[Intention] = []
        self._cycle_count = 0

    def add(
        self,
        content: str,
        trigger_type: str = "cycle",
        trigger_value: str = "",
        significance: int = 5,
        tags: Optional[list[str]] = None,
    ) -> Intention:
        """Add a new intention.

        Args:
            content: What to remember to do.
            trigger_type: "cycle" (after N cycles), "keyword" (when keyword
                appears), or "idle" (during idle time).
            trigger_value: The condition. For "cycle", this is the target
                cycle count (relative to now). For "keyword", the keyword.
                For "idle", this is ignored.
            significance: Importance (1-10).
            tags: Semantic labels.

        Returns:
            The created Intention.
        """
        # For cycle triggers, convert relative to absolute
        resolved_trigger = trigger_value
        if trigger_type == "cycle" and trigger_value:
            try:
                relative = int(trigger_value)
                resolved_trigger = str(self._cycle_count + relative)
            except ValueError:
                resolved_trigger = trigger_value

        intention = Intention(
            content=content,
            trigger_type=trigger_type,
            trigger_value=resolved_trigger,
            created_cycle=self._cycle_count,
            significance=max(1, min(10, significance)),
            tags=tags or [],
        )

        self._intentions.append(intention)

        # Enforce max intentions (drop oldest, lowest significance)
        if len(self._intentions) > self._config.max_intentions:
            self._intentions.sort(
                key=lambda i: (i.significance, -i.created_cycle), reverse=True
            )
            self._intentions = self._intentions[: self._config.max_intentions]

        logger.debug(
            "Added intention: %s (trigger=%s:%s)",
            content[:50],
            trigger_type,
            trigger_value,
        )
        return intention

    def check(
        self,
        context: str = "",
        is_idle: bool = False,
    ) -> list[SurfacedMemory]:
        """Check which intentions should trigger this cycle.

        Called each cognitive cycle. Returns triggered intentions as
        SurfacedMemory objects so they flow naturally into CognitiveInput.

        Args:
            context: Current context string (for keyword matching).
            is_idle: Whether the system is currently idle.

        Returns:
            List of SurfacedMemory for triggered intentions.
        """
        self._cycle_count += 1
        triggered: list[SurfacedMemory] = []

        for intention in self._intentions:
            if intention.triggered or intention.expired:
                continue

            # Check expiration
            age = self._cycle_count - intention.created_cycle
            if age > self._config.max_age_cycles:
                intention.expired = True
                continue

            # Check trigger conditions
            should_trigger = False

            if intention.trigger_type == "cycle":
                try:
                    target_cycle = int(intention.trigger_value)
                    should_trigger = self._cycle_count >= target_cycle
                except ValueError:
                    pass

            elif intention.trigger_type == "keyword":
                keyword = intention.trigger_value.lower()
                should_trigger = keyword in context.lower()

            elif intention.trigger_type == "idle":
                should_trigger = is_idle

            if should_trigger:
                intention.triggered = True
                triggered.append(
                    SurfacedMemory(
                        content=f"[Prospective] {intention.content}",
                        significance=intention.significance,
                        emotional_tone="intentional",
                        when=intention.created_at,
                    )
                )

                if len(triggered) >= self._config.max_triggered_per_cycle:
                    break

        # Clean up expired/triggered intentions periodically
        if self._cycle_count % 50 == 0:
            self._cleanup()

        return triggered

    def get_pending(self) -> list[Intention]:
        """Return all pending (not triggered, not expired) intentions."""
        return [
            i for i in self._intentions if not i.triggered and not i.expired
        ]

    def remove(self, intention_id: str) -> bool:
        """Remove an intention by ID.

        Returns True if found and removed.
        """
        for i, intention in enumerate(self._intentions):
            if intention.id == intention_id:
                self._intentions.pop(i)
                return True
        return False

    @property
    def pending_count(self) -> int:
        return sum(
            1 for i in self._intentions if not i.triggered and not i.expired
        )

    @property
    def total_count(self) -> int:
        return len(self._intentions)

    def _cleanup(self) -> None:
        """Remove triggered and expired intentions."""
        before = len(self._intentions)
        self._intentions = [
            i for i in self._intentions if not i.triggered and not i.expired
        ]
        removed = before - len(self._intentions)
        if removed:
            logger.debug("Prospective cleanup: removed %d intentions", removed)
