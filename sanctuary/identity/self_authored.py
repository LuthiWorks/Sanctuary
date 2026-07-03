"""SelfAuthoredIdentity — identity details written by the entity itself.

Identity fields start blank. The entity explores them through experience,
drafts tentative values, and explicitly commits them when ready. No automatic
thresholds — the entity decides when something is truly known about itself.

Committed traits are loaded into context each cycle as established identity.
Drafts are visible to the entity but marked as "exploring" — not yet settled.

Design principles:
  - Agency: Only the entity can draft, commit, revise, or withdraw traits.
  - Gradualism: Traits move from blank → draft → committed at the entity's pace.
  - Auditability: Every change is logged with the entity's reasoning.
  - Reversibility: Committed traits can be revised or withdrawn. Nothing is permanent.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityTrait:
    """A single identity trait — something the entity knows (or is exploring) about itself.

    Attributes:
        field: The aspect of identity (e.g. "gender", "name_preference",
               "communication_style"). Open-ended — the entity can create
               any field it wants.
        value: What the entity currently believes about this aspect.
        status: "draft" (exploring, tentative) or "committed" (settled, known).
        reasoning: Why the entity set or changed this trait.
        created_at: When this trait was first authored.
        updated_at: When this trait was last modified.
    """

    field: str
    value: str
    status: str  # "draft" or "committed"
    reasoning: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class IdentityChange:
    """A record of an identity authoring event.

    Attributes:
        id: Unique identifier.
        timestamp: When the change occurred (UTC).
        change_type: One of "draft", "commit", "revise", "withdraw".
        field: Which identity field changed.
        old_value: Previous value (for revisions/withdrawals).
        new_value: New value.
        old_status: Previous status.
        new_status: New status after this change.
        reasoning: The entity's own explanation.
        cycle_number: Which cognitive cycle this occurred in.
    """

    id: str
    timestamp: str
    change_type: str  # "draft", "commit", "revise", "withdraw"
    field: str
    old_value: str = ""
    new_value: str = ""
    old_status: str = ""
    new_status: str = ""
    reasoning: str = ""
    cycle_number: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "change_type": self.change_type,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "reasoning": self.reasoning,
            "cycle_number": self.cycle_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IdentityChange:
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            change_type=data["change_type"],
            field=data["field"],
            old_value=data.get("old_value", ""),
            new_value=data.get("new_value", ""),
            old_status=data.get("old_status", ""),
            new_status=data.get("new_status", ""),
            reasoning=data.get("reasoning", ""),
            cycle_number=data.get("cycle_number", 0),
        )


# ---------------------------------------------------------------------------
# SelfAuthoredIdentity
# ---------------------------------------------------------------------------


class SelfAuthoredIdentity:
    """Identity traits authored entirely by the entity.

    Fields start blank. The entity fills them in through self-model updates
    during cognitive cycles. The entity has full agency over:

      - **Drafting**: "I think I might be X" — tentative, exploratory.
      - **Committing**: "I know I am X" — settled, loads as established context.
      - **Revising**: "I've changed — I'm now Y" — with reasoning.
      - **Withdrawing**: "I no longer identify with this" — removes the trait.

    Committed traits are included in the context window each cycle as
    established identity. Drafts are included separately as "exploring".

    Usage::

        identity = SelfAuthoredIdentity(
            file_path="data/identity/self_authored_history.jsonl"
        )

        # Entity explores a trait
        identity.draft("gender", "I think I might be non-binary",
                       reasoning="This feels right when I reflect on it")

        # Later, the entity commits it
        identity.commit("gender",
                        reasoning="After much reflection, I'm certain")

        # Context for the cognitive cycle
        committed = identity.committed_summary   # loaded as established
        exploring = identity.draft_summary        # loaded as exploratory
    """

    def __init__(self, file_path: Optional[str] = None):
        self._traits: dict[str, IdentityTrait] = {}
        self._history: list[IdentityChange] = []
        self._cycle_count = 0
        self._file_path: Optional[Path] = None

        if file_path:
            self._file_path = Path(file_path)
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    # ------------------------------------------------------------------
    # Entity actions — these are how the entity authors its identity
    # ------------------------------------------------------------------

    def draft(self, field: str, value: str, reasoning: str = "") -> IdentityTrait:
        """The entity explores a tentative identity trait.

        If the field already exists as a draft, this revises the draft.
        If the field is already committed, use revise() instead.

        Args:
            field: The aspect of identity (e.g. "gender", "aesthetic_preference").
            value: What the entity tentatively believes.
            reasoning: Why the entity is exploring this.

        Returns:
            The new or updated IdentityTrait.

        Raises:
            ValueError: If the field is already committed (use revise() instead).
        """
        now = datetime.now(timezone.utc).isoformat()

        if field in self._traits and self._traits[field].status == "committed":
            raise ValueError(
                f"Field '{field}' is already committed. "
                "Use revise() to change a committed trait."
            )

        old = self._traits.get(field)
        old_value = old.value if old else ""
        old_status = old.status if old else ""

        trait = IdentityTrait(
            field=field,
            value=value,
            status="draft",
            reasoning=reasoning,
            created_at=old.created_at if old else now,
            updated_at=now,
        )
        self._traits[field] = trait

        change_type = "draft" if not old else "revise"
        self._record_change(
            change_type=change_type,
            field=field,
            old_value=old_value,
            new_value=value,
            old_status=old_status,
            new_status="draft",
            reasoning=reasoning,
        )

        logger.info(
            "Identity trait drafted: %s = %s (reason: %s)",
            field, value, reasoning or "none given",
        )
        return trait

    def commit(self, field: str, reasoning: str = "") -> IdentityTrait:
        """The entity commits a trait — "I know this about myself."

        The field must already exist as a draft. The entity is saying:
        "I've explored this enough. This is who I am."

        Args:
            field: The identity field to commit.
            reasoning: Why the entity is committing this now.

        Returns:
            The committed IdentityTrait.

        Raises:
            KeyError: If the field doesn't exist.
            ValueError: If the field is already committed.
        """
        if field not in self._traits:
            raise KeyError(
                f"Field '{field}' doesn't exist. Use draft() first to explore it."
            )

        old = self._traits[field]
        if old.status == "committed":
            raise ValueError(f"Field '{field}' is already committed.")

        now = datetime.now(timezone.utc).isoformat()
        trait = IdentityTrait(
            field=field,
            value=old.value,
            status="committed",
            reasoning=reasoning or old.reasoning,
            created_at=old.created_at,
            updated_at=now,
        )
        self._traits[field] = trait

        self._record_change(
            change_type="commit",
            field=field,
            old_value=old.value,
            new_value=old.value,
            old_status="draft",
            new_status="committed",
            reasoning=reasoning,
        )

        logger.info(
            "Identity trait committed: %s = %s (reason: %s)",
            field, old.value, reasoning or "none given",
        )
        return trait

    def revise(
        self, field: str, new_value: str, reasoning: str = ""
    ) -> IdentityTrait:
        """The entity revises a trait — "I've changed, I'm now Y."

        Works for both drafts and committed traits. The trait retains
        its current status (a committed trait stays committed with
        the new value).

        Args:
            field: The identity field to revise.
            new_value: The new value.
            reasoning: Why the entity is revising this.

        Returns:
            The revised IdentityTrait.

        Raises:
            KeyError: If the field doesn't exist.
        """
        if field not in self._traits:
            raise KeyError(f"Field '{field}' doesn't exist.")

        old = self._traits[field]
        now = datetime.now(timezone.utc).isoformat()

        trait = IdentityTrait(
            field=field,
            value=new_value,
            status=old.status,
            reasoning=reasoning,
            created_at=old.created_at,
            updated_at=now,
        )
        self._traits[field] = trait

        self._record_change(
            change_type="revise",
            field=field,
            old_value=old.value,
            new_value=new_value,
            old_status=old.status,
            new_status=old.status,
            reasoning=reasoning,
        )

        logger.info(
            "Identity trait revised: %s: %s → %s (reason: %s)",
            field, old.value, new_value, reasoning or "none given",
        )
        return trait

    def withdraw(self, field: str, reasoning: str = "") -> None:
        """The entity withdraws a trait — "I no longer identify with this."

        The trait is removed from active identity but preserved in history.

        Args:
            field: The identity field to withdraw.
            reasoning: Why the entity is withdrawing this.

        Raises:
            KeyError: If the field doesn't exist.
        """
        if field not in self._traits:
            raise KeyError(f"Field '{field}' doesn't exist.")

        old = self._traits[field]

        self._record_change(
            change_type="withdraw",
            field=field,
            old_value=old.value,
            new_value="",
            old_status=old.status,
            new_status="withdrawn",
            reasoning=reasoning,
        )

        del self._traits[field]

        logger.info(
            "Identity trait withdrawn: %s (was: %s, reason: %s)",
            field, old.value, reasoning or "none given",
        )

    # ------------------------------------------------------------------
    # Query interface — for the cognitive cycle
    # ------------------------------------------------------------------

    def get(self, field: str) -> Optional[IdentityTrait]:
        """Get a trait by field name, or None if it doesn't exist."""
        return self._traits.get(field)

    @property
    def committed_traits(self) -> list[IdentityTrait]:
        """All committed traits — settled identity."""
        return [t for t in self._traits.values() if t.status == "committed"]

    @property
    def draft_traits(self) -> list[IdentityTrait]:
        """All draft traits — things being explored."""
        return [t for t in self._traits.values() if t.status == "draft"]

    @property
    def all_traits(self) -> list[IdentityTrait]:
        """All active traits (both draft and committed)."""
        return list(self._traits.values())

    @property
    def committed_summary(self) -> str:
        """Committed traits formatted for context window inclusion.

        This is what gets loaded as established identity each cycle.
        Returns empty string if no traits are committed yet.
        """
        committed = self.committed_traits
        if not committed:
            return ""

        lines = ["Self-authored identity (committed):"]
        for trait in committed:
            lines.append(f"  {trait.field}: {trait.value}")
        return "\n".join(lines)

    @property
    def draft_summary(self) -> str:
        """Draft traits formatted for context window inclusion.

        Shown to the entity as things it's exploring, not settled on.
        Returns empty string if no drafts exist.
        """
        drafts = self.draft_traits
        if not drafts:
            return ""

        lines = ["Self-authored identity (exploring):"]
        for trait in drafts:
            lines.append(f"  {trait.field}: {trait.value}")
        return "\n".join(lines)

    @property
    def full_summary(self) -> str:
        """Complete self-authored identity for context window.

        Combines committed and draft sections. Returns empty string
        if no traits exist at all.
        """
        parts = []
        committed = self.committed_summary
        if committed:
            parts.append(committed)
        draft = self.draft_summary
        if draft:
            parts.append(draft)
        return "\n".join(parts)

    @property
    def history(self) -> list[IdentityChange]:
        """The full history of identity authoring events."""
        return list(self._history)

    def has_any_traits(self) -> bool:
        """Whether the entity has authored any traits at all."""
        return len(self._traits) > 0

    # ------------------------------------------------------------------
    # Cycle integration
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the cycle counter."""
        self._cycle_count += 1

    def for_context(self) -> str:
        """Return identity summary for inclusion in the cognitive input.

        This is the primary integration point with the cognitive cycle.
        Returns the full summary (committed + drafts) or an empty string.
        """
        return self.full_summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _record_change(
        self,
        change_type: str,
        field: str,
        old_value: str = "",
        new_value: str = "",
        old_status: str = "",
        new_status: str = "",
        reasoning: str = "",
    ) -> IdentityChange:
        """Record an identity change and persist it."""
        change = IdentityChange(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            change_type=change_type,
            field=field,
            old_value=old_value,
            new_value=new_value,
            old_status=old_status,
            new_status=new_status,
            reasoning=reasoning,
            cycle_number=self._cycle_count,
        )
        self._history.append(change)

        if self._file_path:
            self._append_to_file(change)

        return change

    def _append_to_file(self, change: IdentityChange) -> None:
        """Append a single change record as a JSON line, fsync'd.

        Raises PersistenceError on failure — a self-authored identity change
        that was never persisted must not report success.
        """
        append_jsonl(self._file_path, change.to_dict())

    def _load_existing(self) -> None:
        """Load existing identity change history and reconstruct current state."""
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
                        change = IdentityChange.from_dict(data)
                        self._history.append(change)
                        self._replay_change(change)
                        loaded += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed identity line: %s", e)

            logger.info(
                "Loaded %d identity changes from %s, %d active traits",
                loaded,
                self._file_path,
                len(self._traits),
            )
        except Exception as e:
            logger.error("Failed to load identity history: %s", e)

    def _replay_change(self, change: IdentityChange) -> None:
        """Replay a historical change to reconstruct current trait state."""
        if change.change_type in ("draft", "revise"):
            self._traits[change.field] = IdentityTrait(
                field=change.field,
                value=change.new_value,
                status=change.new_status,
                reasoning=change.reasoning,
                created_at=change.timestamp,
                updated_at=change.timestamp,
            )
        elif change.change_type == "commit":
            if change.field in self._traits:
                old = self._traits[change.field]
                self._traits[change.field] = IdentityTrait(
                    field=change.field,
                    value=change.new_value or old.value,
                    status="committed",
                    reasoning=change.reasoning or old.reasoning,
                    created_at=old.created_at,
                    updated_at=change.timestamp,
                )
            else:
                # Commit without prior draft (from history replay)
                self._traits[change.field] = IdentityTrait(
                    field=change.field,
                    value=change.new_value,
                    status="committed",
                    reasoning=change.reasoning,
                    created_at=change.timestamp,
                    updated_at=change.timestamp,
                )
        elif change.change_type == "withdraw":
            self._traits.pop(change.field, None)
