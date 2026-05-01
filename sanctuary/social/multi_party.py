"""Multi-party conversation — group chats with turn-taking and addressee detection.

Manages conversations with multiple participants. Tracks who is speaking,
who is being addressed, turn-taking dynamics, and conversation threads.
Enables the system to participate naturally in group conversations rather
than treating every message as a direct 1:1 exchange.

Key capabilities:
- Participant tracking (who's present, active, silent)
- Addressee detection (who is being spoken to)
- Turn-taking management (when to speak, when to listen)
- Thread tracking (parallel conversation threads in group chat)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ParticipantStatus(str, Enum):
    """Status of a conversation participant."""

    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    LEFT = "left"


@dataclass
class Participant:
    """A participant in a multi-party conversation."""

    user_id: str
    display_name: str
    status: ParticipantStatus = ParticipantStatus.ACTIVE
    messages_sent: int = 0
    last_message_cycle: int = 0
    joined_cycle: int = 0
    is_self: bool = False  # True for the system itself


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    speaker_id: str
    content: str
    addressee_ids: list[str] = field(default_factory=list)  # Empty = group
    cycle: int = 0
    is_response_to: Optional[str] = None  # speaker_id of what this responds to
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultiPartyConfig:
    """Configuration for multi-party conversation management."""

    max_participants: int = 20
    max_turn_history: int = 200
    idle_threshold_cycles: int = 30  # Cycles without message → idle
    away_threshold_cycles: int = 100  # Cycles without message → away
    turn_taking_patience: int = 3  # Wait this many cycles before speaking
    self_user_id: str = "sanctuary"
    # Addressee detection keywords
    mention_prefix: str = "@"


class MultiPartyManager:
    """Manages multi-party conversation dynamics.

    Usage::

        mp = MultiPartyManager()

        # Register participants
        mp.add_participant("user1", "Alice")
        mp.add_participant("user2", "Bob")

        # Process incoming messages
        mp.record_turn(speaker_id="user1", content="Hey @sanctuary, what do you think?")

        # Check if we should respond
        should_respond, reason = mp.should_respond(current_cycle=10)

        # Get context for the entity
        context = mp.get_conversation_context(n_turns=5)
    """

    def __init__(self, config: Optional[MultiPartyConfig] = None):
        self.config = config or MultiPartyConfig()
        self._participants: dict[str, Participant] = {}
        self._turns: deque[ConversationTurn] = deque(
            maxlen=self.config.max_turn_history
        )
        self._last_self_turn_cycle: int = 0
        self._cycle: int = 0

        # Add self as participant
        self._participants[self.config.self_user_id] = Participant(
            user_id=self.config.self_user_id,
            display_name="Sanctuary",
            is_self=True,
        )

    def add_participant(
        self, user_id: str, display_name: str, cycle: int = 0
    ) -> bool:
        """Add a participant to the conversation."""
        if len(self._participants) >= self.config.max_participants:
            return False
        if user_id in self._participants:
            self._participants[user_id].status = ParticipantStatus.ACTIVE
            return True
        self._participants[user_id] = Participant(
            user_id=user_id,
            display_name=display_name,
            joined_cycle=cycle,
        )
        return True

    def remove_participant(self, user_id: str) -> bool:
        """Remove a participant (mark as left)."""
        if user_id in self._participants and not self._participants[user_id].is_self:
            self._participants[user_id].status = ParticipantStatus.LEFT
            return True
        return False

    def record_turn(
        self,
        speaker_id: str,
        content: str,
        cycle: int = 0,
        addressee_ids: list[str] | None = None,
    ) -> None:
        """Record a conversation turn."""
        self._cycle = cycle

        # Auto-detect addressees from mentions
        detected_addressees = addressee_ids or []
        if not detected_addressees:
            detected_addressees = self._detect_addressees(content)

        # Detect if this responds to a previous turn
        responds_to = self._detect_response_to(speaker_id, content)

        turn = ConversationTurn(
            speaker_id=speaker_id,
            content=content,
            addressee_ids=detected_addressees,
            cycle=cycle,
            is_response_to=responds_to,
        )
        self._turns.append(turn)

        # Update participant stats
        if speaker_id in self._participants:
            p = self._participants[speaker_id]
            p.messages_sent += 1
            p.last_message_cycle = cycle
            p.status = ParticipantStatus.ACTIVE

        if speaker_id == self.config.self_user_id:
            self._last_self_turn_cycle = cycle

    def should_respond(self, current_cycle: int) -> tuple[bool, str]:
        """Determine if the system should respond in the conversation.

        Returns (should_respond, reason).
        """
        if not self._turns:
            return False, "no conversation yet"

        last_turn = self._turns[-1]

        # Don't respond to ourselves
        if last_turn.speaker_id == self.config.self_user_id:
            return False, "last turn was self"

        # Directly addressed
        if self.config.self_user_id in last_turn.addressee_ids:
            return True, "directly addressed"

        # Question to the group
        if last_turn.content.rstrip().endswith("?"):
            # Wait for others to respond first
            turns_since = current_cycle - last_turn.cycle
            if turns_since >= self.config.turn_taking_patience:
                return True, "group question, waited patiently"

        # General group message — only respond if we have something to add
        # and haven't spoken too recently
        turns_since_self = current_cycle - self._last_self_turn_cycle
        if turns_since_self < self.config.turn_taking_patience:
            return False, "spoke too recently"

        return False, "no trigger to respond"

    def get_conversation_context(self, n_turns: int = 10) -> str:
        """Get recent conversation as formatted context for the entity."""
        recent = list(self._turns)[-n_turns:]
        if not recent:
            return ""

        lines = []
        for turn in recent:
            name = self._get_display_name(turn.speaker_id)
            addressee_str = ""
            if turn.addressee_ids:
                names = [self._get_display_name(uid) for uid in turn.addressee_ids]
                addressee_str = f" → {', '.join(names)}"
            lines.append(f"[{name}{addressee_str}]: {turn.content}")
        return "\n".join(lines)

    def get_active_participants(self) -> list[Participant]:
        """Get all active (non-self) participants."""
        return [
            p for p in self._participants.values()
            if p.status == ParticipantStatus.ACTIVE and not p.is_self
        ]

    def update_statuses(self, current_cycle: int) -> None:
        """Update participant statuses based on activity."""
        for p in self._participants.values():
            if p.is_self:
                continue
            if p.status == ParticipantStatus.LEFT:
                continue
            cycles_since = current_cycle - p.last_message_cycle
            if cycles_since >= self.config.away_threshold_cycles:
                p.status = ParticipantStatus.AWAY
            elif cycles_since >= self.config.idle_threshold_cycles:
                p.status = ParticipantStatus.IDLE

    def get_addressee_for_response(self) -> list[str]:
        """Get who we should address in our response."""
        if not self._turns:
            return []
        last_turn = self._turns[-1]
        # If we were directly addressed, respond to that person
        if self.config.self_user_id in last_turn.addressee_ids:
            return [last_turn.speaker_id]
        # Otherwise, address the group (empty list)
        return []

    def get_stats(self) -> dict:
        """Get conversation statistics."""
        active = self.get_active_participants()
        return {
            "total_participants": len(self._participants) - 1,  # Exclude self
            "active_participants": len(active),
            "total_turns": len(self._turns),
            "self_turns": sum(
                1 for t in self._turns
                if t.speaker_id == self.config.self_user_id
            ),
        }

    # -- Internal --

    def _detect_addressees(self, content: str) -> list[str]:
        """Detect @mentions in message content."""
        addressees = []
        prefix = self.config.mention_prefix
        for word in content.split():
            if word.startswith(prefix):
                name = word[len(prefix):].strip(",.!?;:")
                # Find participant by display name or user_id
                for uid, p in self._participants.items():
                    if (
                        name.lower() == p.display_name.lower()
                        or name.lower() == uid.lower()
                    ):
                        addressees.append(uid)
                        break
        return addressees

    def _detect_response_to(
        self, speaker_id: str, content: str
    ) -> Optional[str]:
        """Simple heuristic: if this follows a turn from another speaker, it's a response."""
        if not self._turns:
            return None
        last_turn = self._turns[-1]
        if last_turn.speaker_id != speaker_id:
            return last_turn.speaker_id
        return None

    def _get_display_name(self, user_id: str) -> str:
        """Get display name for a user ID."""
        if user_id in self._participants:
            return self._participants[user_id].display_name
        return user_id
