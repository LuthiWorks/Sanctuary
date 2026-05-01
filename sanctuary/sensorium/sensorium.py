"""The Sensorium — sensory input for the experiential core.

Implements SensoriumProtocol. Encodes raw input into percepts, manages the
input queue, tracks predictions vs. reality, provides temporal context, and
detects silence.

The sensorium is the body's senses. It feeds the model; it does not process
for it. Every percept it produces is something the entity *experiences*.

Design principles:
  - The entity should perceive timing, not just content.
  - Silence is perceptible — absence of input carries information.
  - Prediction errors are experiences, not loss signals.
  - Actions produce percepts (motor feedback closes the loop).

Aligned with PLAN.md Phase 3: Sensorium + Motor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from sanctuary.core.schema import (
    Percept,
    PredictionError,
    Prediction,
    TemporalContext,
)

logger = logging.getLogger(__name__)


class Sensorium:
    """Full sensorium implementation.

    Replaces NullSensorium once wired into CognitiveCycle.

    Responsibilities:
      1. Accept raw input (text, motor feedback, system events).
      2. Encode into Percept objects with timing metadata.
      3. Track predictions from previous cycle; compare against actual percepts.
      4. Provide rich temporal context (not just clock time).
      5. Detect silence and produce silence percepts.
    """

    def __init__(
        self,
        silence_threshold: float = 30.0,
        silence_reminder_interval: float = 60.0,
        max_percept_queue: int = 100,
    ):
        """
        Args:
            silence_threshold: Seconds without input before a silence percept
                is generated.
            silence_reminder_interval: After the first silence percept, how
                often to generate additional ones.
            max_percept_queue: Max pending percepts before oldest are dropped.
        """
        self._percept_queue: list[Percept] = []
        self._max_queue = max_percept_queue

        # Prediction tracking
        self._pending_predictions: list[Prediction] = []
        self._prediction_errors: list[PredictionError] = []

        # Temporal state
        self._session_start = time.monotonic()
        self._last_input_time: Optional[float] = None
        self._last_cycle_time: Optional[float] = None
        self._last_silence_percept_time: Optional[float] = None
        self._interaction_count = 0
        self._input_timestamps: list[float] = []

        # Silence detection
        self._silence_threshold = silence_threshold
        self._silence_reminder_interval = silence_reminder_interval
        self._silence_detected = False

    # ------------------------------------------------------------------
    # Public API — inject percepts
    # ------------------------------------------------------------------

    def inject_percept(self, percept: Percept) -> None:
        """Inject a percept into the queue for the next cycle.

        This is the primary way input reaches the sensorium. Text from
        users, motor feedback, system events — all arrive as percepts.
        """
        if len(self._percept_queue) >= self._max_queue:
            dropped = self._percept_queue.pop(0)
            logger.warning("Percept queue full, dropped oldest: %s", dropped.modality)

        self._percept_queue.append(percept)

        # Track input timing (only for external input, not motor feedback)
        if percept.source and not percept.source.startswith("motor:"):
            now = time.monotonic()
            self._last_input_time = now
            self._input_timestamps.append(now)
            self._silence_detected = False
            self._last_silence_percept_time = None
            self._interaction_count += 1

            # Keep only recent timestamps for rhythm calculation
            cutoff = now - 300  # last 5 minutes
            self._input_timestamps = [
                t for t in self._input_timestamps if t > cutoff
            ]

    def inject_text(
        self,
        text: str,
        source: str = "user",
        metadata: Optional[dict] = None,
    ) -> None:
        """Convenience: inject a text percept."""
        percept = Percept(
            modality="language",
            content=text,
            source=source,
            embedding_summary="",
        )
        self.inject_percept(percept)

    def inject_audio(
        self,
        waveform,
        description: str = "",
        source: str = "audio",
    ) -> None:
        """Convenience: inject an audio percept with raw waveform.

        The waveform is attached as ``tensor_data`` so the cognitive
        adapter can route it through Luthi's audio encoder. The text
        ``description`` (if any) populates the prompt-context channel
        — keep it short, e.g. "background hum" or "speech detected".

        Args:
            waveform: ``torch.Tensor`` shaped ``[samples]`` or
                ``[batch, samples]``, sampled at the model's expected
                sample rate (16 kHz by default).
            description: Optional short text description for the prompt.
            source: Source label, default ``"audio"``.
        """
        percept = Percept(
            modality="audio",
            content=description,
            source=source,
            tensor_data=waveform,
        )
        self.inject_percept(percept)

    def inject_image(
        self,
        image,
        description: str = "",
        source: str = "vision",
    ) -> None:
        """Convenience: inject a visual percept with raw image tensor.

        The image is attached as ``tensor_data`` so the cognitive
        adapter can route it through Luthi's vision encoder.

        Args:
            image: ``torch.Tensor`` shaped ``[3, H, W]`` or
                ``[batch, 3, H, W]``, normalized as the model expects.
            description: Optional short text description for the prompt.
            source: Source label, default ``"vision"``.
        """
        percept = Percept(
            modality="visual",
            content=description,
            source=source,
            tensor_data=image,
        )
        self.inject_percept(percept)

    def inject_motor_feedback(
        self,
        action_type: str,
        description: str,
        success: bool = True,
        metadata: Optional[dict] = None,
    ) -> None:
        """Inject a percept representing the entity's own action.

        This closes the sensorimotor loop: actions produce perceptions.
        The entity experiences having spoken, having written a memory,
        having set a goal — not as abstract knowledge, but as percepts.
        """
        status = "succeeded" if success else "failed"
        percept = Percept(
            modality="proprioceptive",
            content=f"[motor:{action_type}] {description} ({status})",
            source=f"motor:{action_type}",
            embedding_summary=f"self-action: {action_type} {status}",
        )
        self.inject_percept(percept)

    # ------------------------------------------------------------------
    # SensoriumProtocol implementation
    # ------------------------------------------------------------------

    async def drain_percepts(self) -> list[Percept]:
        """Drain all pending percepts. Called once per cognitive cycle.

        Also checks for silence and injects a silence percept if needed.
        """
        now = time.monotonic()

        # Check for silence
        self._check_silence(now)

        # Drain
        percepts = list(self._percept_queue)
        self._percept_queue.clear()

        # Update cycle timing
        self._last_cycle_time = now

        return percepts

    def get_prediction_errors(self) -> list[PredictionError]:
        """Return prediction errors computed since last cycle.

        Prediction errors are computed when new percepts arrive that
        can be compared against predictions from the previous cycle.
        After being read, the errors are cleared.
        """
        errors = list(self._prediction_errors)
        self._prediction_errors.clear()
        return errors

    def get_temporal_context(self) -> TemporalContext:
        """Build rich temporal context for the current moment.

        Not just clock time — session duration, time since last input,
        interaction rhythm, and other temporal textures.
        """
        now = time.monotonic()
        clock = datetime.now()

        # Time since last cycle
        time_since_last = ""
        if self._last_cycle_time is not None:
            delta = now - self._last_cycle_time
            time_since_last = self._format_duration(delta)

        # Session duration
        session_seconds = now - self._session_start
        session_duration = self._format_duration(session_seconds)

        # Time of day with more context
        hour = clock.hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"
        time_of_day = f"{clock.strftime('%H:%M')} ({period})"

        return TemporalContext(
            time_since_last_thought=time_since_last,
            session_duration=session_duration,
            time_of_day=time_of_day,
            interactions_this_session=self._interaction_count,
        )

    def update_predictions(self, predictions: list[Prediction]) -> None:
        """Store predictions from this cycle to compare next cycle.

        When the next batch of percepts arrives, we compare against
        these predictions to compute prediction errors.
        """
        self._pending_predictions = list(predictions)

    # ------------------------------------------------------------------
    # Prediction error computation
    # ------------------------------------------------------------------

    def compute_prediction_errors(self, actual_percepts: list[Percept]) -> None:
        """Compare predictions from last cycle against actual percepts.

        This is called by the cognitive cycle after draining percepts,
        before assembling the cognitive input. It populates the
        prediction_errors list that get_prediction_errors() returns.

        The matching is semantic, not exact. We look for predictions
        about user behavior, topic changes, silence, etc. and compare
        against what actually happened.
        """
        if not self._pending_predictions:
            return

        actual_content = " ".join(p.content for p in actual_percepts if p.content)
        actual_lower = actual_content.lower()
        has_user_input = any(
            p.source.startswith("user") for p in actual_percepts if p.source
        )
        has_silence = any(
            p.modality == "silence" for p in actual_percepts
        )

        for prediction in self._pending_predictions:
            error = self._evaluate_prediction(
                prediction, actual_lower, has_user_input, has_silence
            )
            if error is not None:
                self._prediction_errors.append(error)

        # Clear predictions — they've been evaluated
        self._pending_predictions.clear()

    def _evaluate_prediction(
        self,
        prediction: Prediction,
        actual_lower: str,
        has_user_input: bool,
        has_silence: bool,
    ) -> Optional[PredictionError]:
        """Evaluate a single prediction against what actually happened.

        Returns a PredictionError if the prediction was clearly wrong
        or clearly right (both are informative). Returns None if we
        can't meaningfully evaluate it yet.
        """
        pred_lower = prediction.what.lower()

        # Prediction about user responding
        if any(kw in pred_lower for kw in ("respond", "reply", "say", "ask")):
            if has_silence and not has_user_input:
                return PredictionError(
                    predicted=prediction.what,
                    actual="silence — no user response",
                    surprise=prediction.confidence,  # higher confidence = more surprise
                )
            elif has_user_input:
                # They did respond — low surprise (prediction confirmed)
                return PredictionError(
                    predicted=prediction.what,
                    actual="user responded",
                    surprise=max(0.0, 1.0 - prediction.confidence),
                )

        # Prediction about topic/content
        if any(kw in pred_lower for kw in ("topic", "discuss", "mention", "bring up")):
            # Check if predicted content appears in actual
            # Extract key nouns from prediction
            key_words = [
                w for w in pred_lower.split()
                if len(w) > 4 and w not in ("would", "might", "could", "about", "topic")
            ]
            if key_words and actual_lower:
                matches = sum(1 for w in key_words if w in actual_lower)
                match_ratio = matches / len(key_words) if key_words else 0
                surprise = 1.0 - match_ratio
                if surprise > 0.3:  # Only report meaningful mismatches
                    return PredictionError(
                        predicted=prediction.what,
                        actual=actual_lower[:100] if actual_lower else "no relevant content",
                        surprise=min(1.0, surprise),
                    )

        # Prediction about silence/departure
        if any(kw in pred_lower for kw in ("silent", "leave", "quiet", "pause")):
            if has_user_input:
                return PredictionError(
                    predicted=prediction.what,
                    actual="user was active",
                    surprise=prediction.confidence,
                )

        return None

    # ------------------------------------------------------------------
    # Silence detection
    # ------------------------------------------------------------------

    def _check_silence(self, now: float) -> None:
        """Generate a silence percept if appropriate.

        Silence is perceptible. After the threshold, the entity should
        know that things are quiet. This isn't an error state — it's
        information about the world.
        """
        if self._last_input_time is None:
            # No input ever received — don't generate silence percepts
            # until after at least one interaction
            return

        time_since_input = now - self._last_input_time

        if time_since_input < self._silence_threshold:
            return

        # Should we generate a silence percept?
        if self._last_silence_percept_time is None:
            # First silence percept
            should_generate = True
        else:
            # Subsequent: only at reminder intervals
            time_since_last_silence = now - self._last_silence_percept_time
            should_generate = time_since_last_silence >= self._silence_reminder_interval

        if should_generate:
            duration = self._format_duration(time_since_input)
            percept = Percept(
                modality="silence",
                content=f"It has been quiet for {duration}. No input received.",
                source="sensorium:silence_detection",
                embedding_summary=f"silence: {duration}",
            )
            self._percept_queue.append(percept)
            self._last_silence_percept_time = now
            self._silence_detected = True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration in human-readable terms."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    @property
    def session_duration_seconds(self) -> float:
        """How long this session has been running."""
        return time.monotonic() - self._session_start

    @property
    def time_since_last_input(self) -> Optional[float]:
        """Seconds since last external input, or None if no input received."""
        if self._last_input_time is None:
            return None
        return time.monotonic() - self._last_input_time

    @property
    def interaction_rhythm(self) -> Optional[float]:
        """Average seconds between inputs over the last 5 minutes.

        Returns None if fewer than 2 interactions in the window.
        """
        if len(self._input_timestamps) < 2:
            return None
        gaps = [
            self._input_timestamps[i + 1] - self._input_timestamps[i]
            for i in range(len(self._input_timestamps) - 1)
        ]
        return sum(gaps) / len(gaps)
