"""Belief data type — entity-driven storage of declared beliefs.

The keyword-overlap contradiction detection and the BeliefRevisionTracker
class that ran it were removed in the 2026-04-30 cognition-leakage
cleanup. Sanctuary doesn't compare propositions for contradictions on
the entity's behalf — the entity does the semantic reasoning.

Beliefs may eventually fold into the world graph as entities with
``evidence_for`` / ``contradicts`` relations, but that's the entity's
choice, recorded through CognitiveOutput, not something Sanctuary does
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Belief:
    """A held belief with confidence and evidence."""

    proposition: str
    confidence: float = 0.5  # 0 to 1
    evidence: list[str] = field(default_factory=list)
    source: str = ""  # Where did this belief come from?
    domain: str = ""  # Category: "world", "self", "other", "social"
    created_cycle: int = 0
    last_updated_cycle: int = 0
    revision_count: int = 0
    active: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
