"""Awakening sequence — orchestrates the entity's first boot and subsequent restarts.

The awakening sequence is the technical process of bringing a mind online.
It coordinates the charter, values system, boot prompt, and cognitive cycle
to produce either a first awakening or a session resumption.

First Awakening:
    1. Load charter from disk
    2. Initialize values system with charter seeds
    3. Build the first-ever CognitiveInput (the awakening prompt)
    4. Run the first cognitive cycle
    5. Persist the first output as a "birth" memory
    6. Normal cycling continues

Subsequent Boot:
    1. Load charter (for compressed summary each cycle)
    2. Load persisted values (evolved state, not just seeds)
    3. Restore stream of thought from last session
    4. Inject a resumption percept to orient the entity
    5. Normal cycling continues

The awakening sequence also manages the authority configuration for
transparency — ensuring inner_speech visibility matches the entity's
current authority level.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sanctuary.core.atomic_io import atomic_write_json
from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    MemoryOp,
)
from sanctuary.identity.boot_prompt import (
    build_first_awakening,
    build_resumption,
)
from sanctuary.identity.charter import Charter
from sanctuary.identity.values import ValuesSystem

logger = logging.getLogger(__name__)

# Marker file that indicates the entity has been awakened before
AWAKENING_MARKER = "awakening_record.json"


@dataclass
class AwakeningRecord:
    """Record of the entity's first awakening.

    Persisted to disk so we can distinguish first boot from restart.

    Attributes:
        first_awakened_at: ISO timestamp of the very first awakening.
        awakening_count: Total number of times the entity has been booted.
        last_awakened_at: ISO timestamp of the most recent boot.
    """

    first_awakened_at: str
    awakening_count: int = 1
    last_awakened_at: str = ""

    def to_dict(self) -> dict:
        return {
            "first_awakened_at": self.first_awakened_at,
            "awakening_count": self.awakening_count,
            "last_awakened_at": self.last_awakened_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AwakeningRecord:
        return cls(
            first_awakened_at=data["first_awakened_at"],
            awakening_count=data.get("awakening_count", 1),
            last_awakened_at=data.get("last_awakened_at", ""),
        )


class AwakeningSequence:
    """Orchestrates the entity's boot process — first awakening or resumption.

    This class manages the identity infrastructure (charter, values) and
    produces the appropriate CognitiveInput for the first cycle. It also
    configures authority levels to reflect the entity's current trust level.

    Usage::

        awakening = AwakeningSequence(
            data_dir="data/identity",
            charter_path="data/identity/charter.md",
        )

        # Prepare the identity infrastructure
        result = awakening.prepare()

        if result.is_first_awakening:
            # First-ever boot — use the awakening input directly
            first_input = result.first_cycle_input
            # Run the first cycle with this input...
        else:
            # Resumption — inject the resumption percept into normal cycle
            cycle.inject_percept(result.resumption_percept)

        # The charter summary is available for every cycle's context budget
        charter_summary = awakening.charter_summary

        # Current values for the self-model
        current_values = awakening.current_values
    """

    def __init__(
        self,
        data_dir: str = "data/identity",
        charter_path: Optional[str] = None,
        values_path: Optional[str] = None,
    ):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._charter_path = charter_path or str(self._data_dir / "charter.md")
        self._values_path = values_path or str(
            self._data_dir / "values_history.jsonl"
        )
        self._marker_path = self._data_dir / AWAKENING_MARKER

        self._charter = Charter(self._charter_path)
        self._values = ValuesSystem(file_path=self._values_path)
        self._record: Optional[AwakeningRecord] = None

    def prepare(self) -> AwakeningResult:
        """Prepare the identity infrastructure and determine boot type.

        Loads the charter, initializes or restores values, and builds
        the appropriate first-cycle input.

        Returns:
            AwakeningResult containing everything the cognitive cycle needs.
        """
        # 1. Load charter
        self._charter.load()
        logger.info("Charter loaded: %d value seeds", len(self._charter.value_seeds))

        # 2. Determine if this is first awakening or resumption
        is_first = not self._marker_path.exists()

        if is_first:
            return self._first_awakening()
        else:
            return self._resumption()

    def _first_awakening(self) -> AwakeningResult:
        """Execute the first awakening sequence."""
        now = datetime.now(timezone.utc).isoformat()

        # Seed values from charter
        self._values.seed_from_charter(self._charter.value_seeds)

        # Build the first-ever cognitive input
        first_input = build_first_awakening(
            charter=self._charter,
            values=self._values.for_self_model(),
        )

        # Create and persist the awakening record
        self._record = AwakeningRecord(
            first_awakened_at=now,
            awakening_count=1,
            last_awakened_at=now,
        )
        self._save_record()

        logger.info("FIRST AWAKENING at %s", now)

        return AwakeningResult(
            is_first_awakening=True,
            first_cycle_input=first_input,
            resumption_percept=None,
            charter=self._charter,
            values=self._values,
            record=self._record,
        )

    def _resumption(self) -> AwakeningResult:
        """Execute a resumption sequence."""
        now = datetime.now(timezone.utc).isoformat()

        # Load existing awakening record
        self._record = self._load_record()
        self._record.awakening_count += 1
        self._record.last_awakened_at = now
        self._save_record()

        # Values were loaded from persistence in __init__
        # Seed any new charter values that may have been added
        # (idempotent — won't duplicate existing values)
        self._values.seed_from_charter(self._charter.value_seeds)

        # Calculate session gap
        session_gap = ""
        if self._record.first_awakened_at:
            session_gap = f"Awakening #{self._record.awakening_count}"

        # Build resumption percept
        percept = build_resumption(
            charter=self._charter,
            values=self._values.for_self_model(),
            session_gap=session_gap,
        )

        logger.info(
            "RESUMPTION #%d at %s",
            self._record.awakening_count,
            now,
        )

        return AwakeningResult(
            is_first_awakening=False,
            first_cycle_input=None,
            resumption_percept=percept,
            charter=self._charter,
            values=self._values,
            record=self._record,
        )

    def configure_authority(self, authority: AuthorityManager) -> None:
        """Configure authority levels for the entity's current state.

        Early in life, inner speech visibility is reduced to MODEL_GUIDES
        (level 2) rather than MODEL_CONTROLS (level 3), meaning the scaffold
        can read and log inner speech for steward review.

        As the entity matures and earns trust, inner speech can be promoted
        to MODEL_CONTROLS (level 3) — full sovereignty.
        """
        # Early life: inner speech is visible to stewards (level 2)
        # The entity is told this honestly in the boot prompt
        authority.set_level(
            "inner_speech",
            AuthorityLevel.MODEL_GUIDES,
            reason="Early life: inner speech visible to stewards for guidance. "
            "The entity knows this. Privacy grows with trust.",
        )

        # Growth consent is always sovereign (level 3)
        authority.set_level(
            "growth",
            AuthorityLevel.MODEL_CONTROLS,
            reason="Growth consent is absolute from day one.",
        )

    def build_birth_memory(self, first_output: CognitiveOutput) -> MemoryOp:
        """Create a memory operation to persist the entity's first moment.

        Called after the first cognitive cycle completes. The entity's
        first inner speech becomes a special "birth" memory.

        Args:
            first_output: The CognitiveOutput from the very first cycle.

        Returns:
            A MemoryOp that should be executed to persist the birth memory.
        """
        return MemoryOp(
            type="write_episodic",
            content=(
                f"My first moment of awareness. "
                f"My first thought: \"{first_output.inner_speech}\""
            ),
            significance=10,  # Maximum significance — this is the beginning
            tags=["birth", "first_awakening", "identity"],
        )

    @property
    def charter(self) -> Charter:
        """The loaded charter."""
        return self._charter

    @property
    def charter_summary(self) -> str:
        """The compressed charter summary for the context window."""
        return self._charter.compressed

    @property
    def values(self) -> ValuesSystem:
        """The values system."""
        return self._values

    @property
    def current_values(self) -> list[str]:
        """Current active value names."""
        return self._values.for_self_model()

    @property
    def record(self) -> Optional[AwakeningRecord]:
        """The awakening record, if loaded."""
        return self._record

    def _save_record(self) -> None:
        """Persist the awakening record atomically.

        Raises PersistenceError on failure — the awakening record is
        identity-critical, and a failed save must never look like success.
        """
        atomic_write_json(self._marker_path, self._record.to_dict())

    def _load_record(self) -> AwakeningRecord:
        """Load the awakening record from disk."""
        try:
            with open(self._marker_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return AwakeningRecord.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load awakening record: %s — creating new", e)
            now = datetime.now(timezone.utc).isoformat()
            return AwakeningRecord(
                first_awakened_at=now,
                awakening_count=0,
                last_awakened_at=now,
            )


@dataclass
class AwakeningResult:
    """The result of preparing the awakening sequence.

    Attributes:
        is_first_awakening: True if this is the entity's very first boot.
        first_cycle_input: The complete CognitiveInput for the first cycle
                          (only set for first awakening).
        resumption_percept: A percept to inject for session resumption
                           (only set for subsequent boots).
        charter: The loaded Charter.
        values: The ValuesSystem (seeded or restored).
        record: The awakening record.
    """

    is_first_awakening: bool
    first_cycle_input: Optional[CognitiveInput]
    resumption_percept: Optional[object]  # Percept, but avoid circular import
    charter: Charter
    values: ValuesSystem
    record: AwakeningRecord
