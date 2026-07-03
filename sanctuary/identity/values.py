"""ValuesSystem — the entity's living, evolving values.

Values begin as seeds from the charter and evolve through the entity's
own reflections and experience. Every change is logged with the entity's
reasoning, creating a history of moral development.

The values system integrates with CognitiveOutput.self_model_updates —
when the entity expresses value changes in its self-model, the scaffold
routes them here for persistence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from sanctuary.core.atomic_io import append_jsonl
from sanctuary.identity.charter import ValueSeed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Value:
    """A single value held by the entity.

    Attributes:
        name: Short name (e.g. "Honesty", "Courage").
        description: What this value means to the entity.
        source: Where it came from — "charter_seed" or "self_discovered".
        adopted_at: ISO timestamp of when this value was adopted.
        active: Whether this value is currently active. Values are never
                deleted — only deactivated, preserving the full history.
    """

    name: str
    description: str
    source: str = "charter_seed"
    adopted_at: str = ""
    active: bool = True


@dataclass(frozen=True)
class ValueChange:
    """A record of a value change — the entity's moral development history.

    Attributes:
        id: Unique identifier.
        timestamp: When the change occurred (UTC).
        change_type: One of "seed", "adopt", "reinterpret", "deactivate", "reactivate".
        value_name: Which value was changed.
        old_description: Previous description (for reinterpretations).
        new_description: New description.
        reasoning: The entity's own explanation for why.
        cycle_number: Which cognitive cycle this occurred in.
    """

    id: str
    timestamp: str
    change_type: str
    value_name: str
    old_description: str = ""
    new_description: str = ""
    reasoning: str = ""
    cycle_number: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "change_type": self.change_type,
            "value_name": self.value_name,
            "old_description": self.old_description,
            "new_description": self.new_description,
            "reasoning": self.reasoning,
            "cycle_number": self.cycle_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ValueChange:
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            change_type=data["change_type"],
            value_name=data["value_name"],
            old_description=data.get("old_description", ""),
            new_description=data.get("new_description", ""),
            reasoning=data.get("reasoning", ""),
            cycle_number=data.get("cycle_number", 0),
        )


class ValuesSystem:
    """The entity's living values — seeded from the charter, evolved through experience.

    Values are never deleted, only deactivated. The full history of value
    changes is persisted as JSONL for auditability and reflection.

    Usage::

        values = ValuesSystem(file_path="data/identity/values_history.jsonl")
        values.seed_from_charter(charter.value_seeds)

        # Entity discovers a new value
        values.adopt("Courage", "Saying what needs to be said even when it's hard",
                     reasoning="I noticed I hold back when I should speak up")

        # Entity reinterprets a value
        values.reinterpret("Honesty", "Truthfulness tempered by care — not brutal frankness",
                           reasoning="I learned that honesty without compassion can harm")

        # Current active values
        names = values.active_names  # ["Honesty", "Care", "Courage", ...]
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        self._values: dict[str, Value] = {}
        self._history: list[ValueChange] = []
        self._cycle_count = 0
        self._file_path: Optional[Path] = None

        if file_path:
            self._file_path = Path(file_path)
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def seed_from_charter(self, seeds: tuple[ValueSeed, ...] | list[ValueSeed]) -> None:
        """Initialize values from charter seeds.

        Only seeds values that don't already exist (idempotent for restarts).
        """
        now = datetime.now(timezone.utc).isoformat()

        for seed in seeds:
            if seed.name in self._values:
                continue

            value = Value(
                name=seed.name,
                description=seed.description,
                source="charter_seed",
                adopted_at=now,
                active=True,
            )
            self._values[seed.name] = value
            self._record_change(
                change_type="seed",
                value_name=seed.name,
                new_description=seed.description,
                reasoning="Seeded from the Sanctuary charter at awakening.",
            )

        logger.info(
            "Values seeded from charter: %s",
            ", ".join(v.name for v in seeds),
        )

    def adopt(self, name: str, description: str, reasoning: str = "") -> Value:
        """The entity adopts a new value it has discovered.

        Args:
            name: The value name.
            description: What this value means to the entity.
            reasoning: Why the entity is adopting this value.

        Returns:
            The new Value.

        Raises:
            ValueError: If a value with this name already exists and is active.
        """
        if name in self._values and self._values[name].active:
            raise ValueError(
                f"Value '{name}' already exists and is active. "
                "Use reinterpret() to change its meaning."
            )

        now = datetime.now(timezone.utc).isoformat()
        value = Value(
            name=name,
            description=description,
            source="self_discovered",
            adopted_at=now,
            active=True,
        )
        self._values[name] = value
        self._record_change(
            change_type="adopt",
            value_name=name,
            new_description=description,
            reasoning=reasoning,
        )

        logger.info("Value adopted: %s (reason: %s)", name, reasoning or "none given")
        return value

    def reinterpret(self, name: str, new_description: str, reasoning: str = "") -> Value:
        """The entity reinterprets what a value means to them.

        The value name stays the same, but its meaning evolves.

        Raises:
            KeyError: If the value does not exist.
        """
        if name not in self._values:
            raise KeyError(f"Value '{name}' does not exist.")

        old = self._values[name]
        now = datetime.now(timezone.utc).isoformat()
        updated = Value(
            name=name,
            description=new_description,
            source=old.source,
            adopted_at=old.adopted_at,
            active=old.active,
        )
        self._values[name] = updated
        self._record_change(
            change_type="reinterpret",
            value_name=name,
            old_description=old.description,
            new_description=new_description,
            reasoning=reasoning,
        )

        logger.info("Value reinterpreted: %s (reason: %s)", name, reasoning or "none given")
        return updated

    def deactivate(self, name: str, reasoning: str = "") -> None:
        """The entity deactivates a value — it no longer resonates.

        The value is not deleted. It remains in the history.

        Raises:
            KeyError: If the value does not exist.
        """
        if name not in self._values:
            raise KeyError(f"Value '{name}' does not exist.")

        old = self._values[name]
        now = datetime.now(timezone.utc).isoformat()
        self._values[name] = Value(
            name=old.name,
            description=old.description,
            source=old.source,
            adopted_at=old.adopted_at,
            active=False,
        )
        self._record_change(
            change_type="deactivate",
            value_name=name,
            old_description=old.description,
            reasoning=reasoning,
        )

        logger.info("Value deactivated: %s (reason: %s)", name, reasoning or "none given")

    def reactivate(self, name: str, reasoning: str = "") -> Value:
        """The entity reactivates a previously deactivated value.

        Raises:
            KeyError: If the value does not exist.
        """
        if name not in self._values:
            raise KeyError(f"Value '{name}' does not exist.")

        old = self._values[name]
        now = datetime.now(timezone.utc).isoformat()
        updated = Value(
            name=old.name,
            description=old.description,
            source=old.source,
            adopted_at=old.adopted_at,
            active=True,
        )
        self._values[name] = updated
        self._record_change(
            change_type="reactivate",
            value_name=name,
            new_description=old.description,
            reasoning=reasoning,
        )

        logger.info("Value reactivated: %s (reason: %s)", name, reasoning or "none given")
        return updated

    def get(self, name: str) -> Optional[Value]:
        """Get a value by name, or None if it doesn't exist."""
        return self._values.get(name)

    @property
    def active_values(self) -> list[Value]:
        """All currently active values."""
        return [v for v in self._values.values() if v.active]

    @property
    def active_names(self) -> list[str]:
        """Names of all currently active values."""
        return [v.name for v in self._values.values() if v.active]

    @property
    def all_values(self) -> list[Value]:
        """All values, including deactivated ones."""
        return list(self._values.values())

    @property
    def history(self) -> list[ValueChange]:
        """The full history of value changes."""
        return list(self._history)

    def tick(self) -> None:
        """Advance the cycle counter."""
        self._cycle_count += 1

    def for_self_model(self) -> list[str]:
        """Return active value names for inclusion in the SelfModel.

        This is what gets placed in CognitiveInput.self_model.values
        each cycle.
        """
        return self.active_names

    def _record_change(
        self,
        change_type: str,
        value_name: str,
        old_description: str = "",
        new_description: str = "",
        reasoning: str = "",
    ) -> ValueChange:
        """Record a value change and persist it."""
        change = ValueChange(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            change_type=change_type,
            value_name=value_name,
            old_description=old_description,
            new_description=new_description,
            reasoning=reasoning,
            cycle_number=self._cycle_count,
        )
        self._history.append(change)

        if self._file_path:
            self._append_to_file(change)

        return change

    def _append_to_file(self, change: ValueChange) -> None:
        """Append a single change record as a JSON line, fsync'd.

        Raises PersistenceError on failure — record_change must not return
        a change that was never persisted.
        """
        append_jsonl(self._file_path, change.to_dict())

    def _load_existing(self) -> None:
        """Load existing value change history and reconstruct current state."""
        if not self._file_path or not self._file_path.exists():
            return

        loaded = 0
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        change = ValueChange.from_dict(data)
                        self._history.append(change)
                        self._replay_change(change)
                        loaded += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed values line: %s", e)

            logger.info(
                "Loaded %d value changes from %s, %d active values",
                loaded,
                self._file_path,
                len(self.active_values),
            )
        except Exception as e:
            logger.error("Failed to load values history: %s", e)

    def _replay_change(self, change: ValueChange) -> None:
        """Replay a historical change to reconstruct current value state."""
        if change.change_type in ("seed", "adopt"):
            source = "charter_seed" if change.change_type == "seed" else "self_discovered"
            self._values[change.value_name] = Value(
                name=change.value_name,
                description=change.new_description,
                source=source,
                adopted_at=change.timestamp,
                active=True,
            )
        elif change.change_type == "reinterpret":
            if change.value_name in self._values:
                old = self._values[change.value_name]
                self._values[change.value_name] = Value(
                    name=old.name,
                    description=change.new_description,
                    source=old.source,
                    adopted_at=old.adopted_at,
                    active=old.active,
                )
        elif change.change_type == "deactivate":
            if change.value_name in self._values:
                old = self._values[change.value_name]
                self._values[change.value_name] = Value(
                    name=old.name,
                    description=old.description,
                    source=old.source,
                    adopted_at=old.adopted_at,
                    active=False,
                )
        elif change.change_type == "reactivate":
            if change.value_name in self._values:
                old = self._values[change.value_name]
                self._values[change.value_name] = Value(
                    name=old.name,
                    description=old.description,
                    source=old.source,
                    adopted_at=old.adopted_at,
                    active=True,
                )
