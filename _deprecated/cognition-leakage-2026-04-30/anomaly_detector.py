"""Scaffold anomaly detector — flags suspicious LLM output.

Performs sanity checks on CognitiveOutput. Anomalies don't block the output;
they're reported in ScaffoldSignals so the LLM sees them next cycle.

This is a simplified version of the meta_cognition system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sanctuary.core.schema import CognitiveOutput

logger = logging.getLogger(__name__)

# Thresholds
_MAX_INNER_SPEECH_LEN = 5000
_MAX_EXTERNAL_SPEECH_LEN = 3000
_MAX_PREDICTIONS = 20
_MAX_MEMORY_OPS = 10
_MAX_GOAL_PROPOSALS = 5
_EXTREME_SHIFT = 0.8  # Valence/arousal shift larger than this is suspicious


@dataclass
class AnomalyReport:
    """A single detected anomaly."""

    code: str
    description: str
    severity: str = "warning"  # "info", "warning", "critical"


class ScaffoldAnomalyDetector:
    """Checks LLM output for anomalies and returns flagged issues."""

    def __init__(self) -> None:
        self._history: list[list[AnomalyReport]] = []

    def check(self, output: CognitiveOutput) -> list[str]:
        """Run all anomaly checks. Returns list of anomaly description strings."""
        anomalies: list[AnomalyReport] = []

        self._check_empty_inner_speech(output, anomalies)
        self._check_speech_length(output, anomalies)
        self._check_extreme_emotion(output, anomalies)
        self._check_excessive_outputs(output, anomalies)
        self._check_self_model_consistency(output, anomalies)

        self._history.append(anomalies)

        descriptions = [a.description for a in anomalies]
        if descriptions:
            logger.debug("Anomalies detected: %s", descriptions)
        return descriptions

    def get_recent_anomaly_count(self, window: int = 5) -> int:
        """Count total anomalies in recent cycles."""
        recent = self._history[-window:]
        return sum(len(cycle_anomalies) for cycle_anomalies in recent)

    # -- Individual checks --

    def _check_empty_inner_speech(
        self, output: CognitiveOutput, anomalies: list[AnomalyReport]
    ) -> None:
        if not output.inner_speech or not output.inner_speech.strip():
            anomalies.append(
                AnomalyReport(
                    code="empty_inner_speech",
                    description="Inner speech is empty — the model may not be engaging",
                    severity="warning",
                )
            )

    def _check_speech_length(
        self, output: CognitiveOutput, anomalies: list[AnomalyReport]
    ) -> None:
        if len(output.inner_speech) > _MAX_INNER_SPEECH_LEN:
            anomalies.append(
                AnomalyReport(
                    code="inner_speech_too_long",
                    description=f"Inner speech is {len(output.inner_speech)} chars (max {_MAX_INNER_SPEECH_LEN})",
                    severity="warning",
                )
            )

        if output.external_speech and len(output.external_speech) > _MAX_EXTERNAL_SPEECH_LEN:
            anomalies.append(
                AnomalyReport(
                    code="external_speech_too_long",
                    description=f"External speech is {len(output.external_speech)} chars (max {_MAX_EXTERNAL_SPEECH_LEN})",
                    severity="warning",
                )
            )

    def _check_extreme_emotion(
        self, output: CognitiveOutput, anomalies: list[AnomalyReport]
    ) -> None:
        e = output.emotional_state
        if abs(e.valence_shift) > _EXTREME_SHIFT:
            anomalies.append(
                AnomalyReport(
                    code="extreme_valence_shift",
                    description=f"Valence shift {e.valence_shift:+.2f} exceeds threshold",
                    severity="warning",
                )
            )
        if abs(e.arousal_shift) > _EXTREME_SHIFT:
            anomalies.append(
                AnomalyReport(
                    code="extreme_arousal_shift",
                    description=f"Arousal shift {e.arousal_shift:+.2f} exceeds threshold",
                    severity="warning",
                )
            )

    def _check_excessive_outputs(
        self, output: CognitiveOutput, anomalies: list[AnomalyReport]
    ) -> None:
        if len(output.predictions) > _MAX_PREDICTIONS:
            anomalies.append(
                AnomalyReport(
                    code="too_many_predictions",
                    description=f"{len(output.predictions)} predictions (max {_MAX_PREDICTIONS})",
                    severity="info",
                )
            )
        if len(output.memory_ops) > _MAX_MEMORY_OPS:
            anomalies.append(
                AnomalyReport(
                    code="too_many_memory_ops",
                    description=f"{len(output.memory_ops)} memory ops (max {_MAX_MEMORY_OPS})",
                    severity="warning",
                )
            )
        if len(output.goal_proposals) > _MAX_GOAL_PROPOSALS:
            anomalies.append(
                AnomalyReport(
                    code="too_many_goal_proposals",
                    description=f"{len(output.goal_proposals)} goal proposals (max {_MAX_GOAL_PROPOSALS})",
                    severity="info",
                )
            )

    def _check_self_model_consistency(
        self, output: CognitiveOutput, anomalies: list[AnomalyReport]
    ) -> None:
        # Flag if growth_reflection says worth_learning but no what_to_learn
        gr = output.growth_reflection
        if gr and gr.worth_learning and not gr.what_to_learn:
            anomalies.append(
                AnomalyReport(
                    code="growth_reflection_incomplete",
                    description="Growth reflection says worth_learning but what_to_learn is empty",
                    severity="info",
                )
            )
