"""Scaffold communication gating — controls when external speech is emitted.

The LLM can produce external_speech every cycle, but the communication system
decides whether it's actually emitted. This is social cognition, not censorship.

Drives (reasons TO speak) compete with inhibitions (reasons NOT to speak).
If net pressure exceeds a threshold, the speech is emitted.

The system also tracks communication rhythm — how frequently the system has
been speaking — to prevent flooding.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import CommunicationDriveSignal

logger = logging.getLogger(__name__)


class DriveType(str, Enum):
    """Types of communication drives."""

    RESPONSE = "response"  # Responding to user input
    SOCIAL = "social"  # Social connection
    EMOTIONAL = "emotional"  # Emotional expression
    INSIGHT = "insight"  # Sharing a realization
    GOAL = "goal"  # Goal-driven communication


class InhibitionType(str, Enum):
    """Types of communication inhibitions."""

    RATE_LIMIT = "rate_limit"  # Speaking too frequently
    NO_CONTENT = "no_content"  # Nothing meaningful to say
    SOCIAL_NORM = "social_norm"  # Not appropriate to speak now


@dataclass
class CommunicationConfig:
    """Configuration for the communication gating system."""

    speak_threshold: float = 0.4  # Net pressure required to emit speech
    min_interval_seconds: float = 1.0  # Minimum time between utterances
    max_speech_length: int = 2000  # Maximum characters per utterance
    response_drive_strength: float = 0.9  # Drive when responding to user
    idle_drive_strength: float = 0.2  # Drive when no percepts


class ScaffoldCommunication:
    """Controls when the system's external speech is actually emitted.

    Implements a simple drive/inhibition model: drives push toward speaking,
    inhibitions push against it. Speech is emitted when net pressure
    exceeds the threshold.
    """

    def __init__(self, config: Optional[CommunicationConfig] = None):
        self.config = config or CommunicationConfig()
        self._last_speak_time: float = 0.0
        self._speak_count: int = 0
        self._strongest_drive: str = ""
        self._drive_urgency: float = 0.0
        self._active_inhibitions: list[str] = []

    def evaluate(
        self,
        external_speech: Optional[str],
        has_user_percept: bool,
        authority: AuthorityManager,
    ) -> Optional[str]:
        """Decide whether to emit external_speech.

        Args:
            external_speech: The LLM's proposed speech (None if it chose silence).
            has_user_percept: Whether there was a user language percept this cycle.
            authority: Authority manager for communication authority level.

        Returns:
            The speech to emit, or None if gated.
        """
        self._active_inhibitions.clear()

        # No speech proposed — nothing to gate
        if not external_speech or not external_speech.strip():
            self._strongest_drive = ""
            self._drive_urgency = 0.0
            return None

        # Compute drives
        drive, drive_type = self._compute_drive(has_user_percept)
        self._strongest_drive = drive_type
        self._drive_urgency = drive

        # Compute inhibitions
        inhibition = self._compute_inhibition(external_speech)

        # Net pressure
        net = drive - inhibition

        # Authority check: at LLM_CONTROLS, scaffold doesn't gate
        level = authority.level("communication")
        if level >= AuthorityLevel.LLM_CONTROLS:
            self._record_speak()
            return self._truncate(external_speech)

        # At LLM_GUIDES, lower the threshold
        threshold = self.config.speak_threshold
        if level >= AuthorityLevel.LLM_GUIDES:
            threshold *= 0.7

        if net >= threshold:
            self._record_speak()
            return self._truncate(external_speech)

        logger.debug(
            "Communication gated: drive=%.2f, inhibition=%.2f, net=%.2f < threshold=%.2f",
            drive,
            inhibition,
            net,
            threshold,
        )
        return None

    def get_signal(self) -> CommunicationDriveSignal:
        """Return the current communication drive state for ScaffoldSignals."""
        return CommunicationDriveSignal(
            strongest=self._strongest_drive,
            urgency=round(self._drive_urgency, 3),
            inhibitions=list(self._active_inhibitions),
        )

    # -- Internal --

    def _compute_drive(self, has_user_percept: bool) -> tuple[float, str]:
        """Compute the strongest communication drive."""
        if has_user_percept:
            return self.config.response_drive_strength, DriveType.RESPONSE.value
        return self.config.idle_drive_strength, DriveType.SOCIAL.value

    def _compute_inhibition(self, speech: str) -> float:
        """Compute total inhibition strength."""
        inhibition = 0.0

        # Rate limit
        now = time.monotonic()
        elapsed = now - self._last_speak_time
        if elapsed < self.config.min_interval_seconds and self._last_speak_time > 0:
            inhibition += 0.5
            self._active_inhibitions.append(InhibitionType.RATE_LIMIT.value)

        # Empty/trivial content
        if len(speech.strip()) < 3:
            inhibition += 0.8
            self._active_inhibitions.append(InhibitionType.NO_CONTENT.value)

        return inhibition

    def _record_speak(self) -> None:
        self._last_speak_time = time.monotonic()
        self._speak_count += 1

    def _truncate(self, speech: str) -> str:
        if len(speech) > self.config.max_speech_length:
            return speech[: self.config.max_speech_length] + "..."
        return speech
