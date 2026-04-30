"""Uncertainty quantification — tracking confidence on beliefs, predictions, outcomes.

Maintains a probabilistic view of the system's knowledge. Tracks prediction
accuracy over time, calibrates confidence scores, and identifies areas of
high uncertainty that warrant attention or inquiry.

This feeds into the LLM's self-model (uncertainties field) and the attention
system (uncertain topics get salience boosts).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackedPrediction:
    """A prediction with its confidence and eventual outcome."""

    what: str
    confidence: float  # 0 to 1: how confident was the prediction?
    domain: str = "general"
    timeframe: str = ""
    cycle_made: int = 0
    cycle_resolved: Optional[int] = None
    outcome: Optional[bool] = None  # True = correct, False = wrong, None = pending
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UncertaintyDomain:
    """Uncertainty tracking for a specific domain/topic."""

    domain: str
    predictions_made: int = 0
    predictions_correct: int = 0
    predictions_wrong: int = 0
    calibration_error: float = 0.0  # Difference between confidence and accuracy
    last_updated_cycle: int = 0


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""

    max_tracked_predictions: int = 500
    calibration_window: int = 50  # Number of recent predictions for calibration
    high_uncertainty_threshold: float = 0.4  # Below this = high uncertainty
    overconfidence_threshold: float = 0.2  # Calibration error above this = overconfident


class UncertaintyQuantifier:
    """Tracks prediction confidence, accuracy, and calibration.

    Monitors how well the system's confidence matches actual outcomes.
    Identifies overconfidence, underconfidence, and domains where the
    system has poor calibration.

    Usage::

        uq = UncertaintyQuantifier()

        # Record a prediction
        uq.record_prediction(
            what="User will ask a follow-up question",
            confidence=0.8,
            cycle=10,
        )

        # Later, resolve it
        uq.resolve_prediction("User will ask a follow-up question", correct=True, cycle=15)

        # Check calibration
        cal = uq.get_calibration()
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self._predictions: deque[TrackedPrediction] = deque(
            maxlen=self.config.max_tracked_predictions
        )
        self._domains: dict[str, UncertaintyDomain] = {}
        self._total_resolved: int = 0

    def record_prediction(
        self,
        what: str,
        confidence: float,
        timeframe: str = "",
        domain: str = "general",
        cycle: int = 0,
    ) -> None:
        """Record a prediction with its confidence level."""
        confidence = max(0.0, min(1.0, confidence))
        pred = TrackedPrediction(
            what=what,
            confidence=confidence,
            domain=domain,
            timeframe=timeframe,
            cycle_made=cycle,
        )
        self._predictions.append(pred)

        # Update domain stats
        if domain not in self._domains:
            self._domains[domain] = UncertaintyDomain(domain=domain)
        self._domains[domain].predictions_made += 1
        self._domains[domain].last_updated_cycle = cycle

    def resolve_prediction(
        self, what: str, correct: bool, cycle: int = 0
    ) -> bool:
        """Resolve a pending prediction with its outcome."""
        for pred in reversed(self._predictions):
            if pred.what == what and pred.outcome is None:
                pred.outcome = correct
                pred.cycle_resolved = cycle
                self._total_resolved += 1
                self._update_calibration()
                return True
        return False

    def get_pending_predictions(self) -> list[TrackedPrediction]:
        """Get all unresolved predictions."""
        return [p for p in self._predictions if p.outcome is None]

    def get_calibration(self) -> dict:
        """Compute calibration metrics over recent resolved predictions.

        Calibration measures how well confidence matches actual accuracy.
        Perfect calibration: 80% confident predictions are correct 80% of the time.
        """
        resolved = [
            p for p in self._predictions if p.outcome is not None
        ][-self.config.calibration_window:]

        if not resolved:
            return {
                "calibration_error": 0.0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "n_resolved": 0,
                "is_overconfident": False,
                "is_underconfident": False,
            }

        avg_confidence = sum(p.confidence for p in resolved) / len(resolved)
        accuracy = sum(1 for p in resolved if p.outcome) / len(resolved)
        calibration_error = avg_confidence - accuracy

        return {
            "calibration_error": calibration_error,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "n_resolved": len(resolved),
            "is_overconfident": calibration_error > self.config.overconfidence_threshold,
            "is_underconfident": calibration_error < -self.config.overconfidence_threshold,
        }

    def get_brier_score(self) -> float:
        """Compute Brier score — lower is better (0 = perfect).

        Brier score = mean((confidence - outcome)^2) for resolved predictions.
        """
        resolved = [
            p for p in self._predictions if p.outcome is not None
        ][-self.config.calibration_window:]

        if not resolved:
            return 0.0

        total = sum(
            (p.confidence - (1.0 if p.outcome else 0.0)) ** 2
            for p in resolved
        )
        return total / len(resolved)

    def get_domain_uncertainty(self, domain: str) -> Optional[float]:
        """Get uncertainty level for a specific domain (0=certain, 1=uncertain)."""
        if domain not in self._domains:
            return None
        d = self._domains[domain]
        if d.predictions_made == 0:
            return 1.0  # No data = maximum uncertainty
        resolved = d.predictions_correct + d.predictions_wrong
        if resolved == 0:
            return 0.8  # Predictions made but none resolved yet
        accuracy = d.predictions_correct / resolved
        return 1.0 - accuracy

    def get_high_uncertainty_areas(self) -> list[str]:
        """Get domains where uncertainty is above threshold."""
        uncertain = []
        for domain, stats in self._domains.items():
            uncertainty = self.get_domain_uncertainty(domain)
            if uncertainty is not None and uncertainty >= self.config.high_uncertainty_threshold:
                uncertain.append(domain)
        return uncertain

    def get_uncertainty_summary(self) -> str:
        """Generate a terse summary for the LLM's self-model."""
        cal = self.get_calibration()
        uncertain_areas = self.get_high_uncertainty_areas()

        parts = []
        if cal["n_resolved"] > 0:
            parts.append(
                f"Prediction accuracy: {cal['accuracy']:.0%} "
                f"(confidence: {cal['avg_confidence']:.0%})"
            )
            if cal["is_overconfident"]:
                parts.append("Tendency: overconfident")
            elif cal["is_underconfident"]:
                parts.append("Tendency: underconfident")
        if uncertain_areas:
            parts.append(f"High uncertainty: {', '.join(uncertain_areas)}")

        return "; ".join(parts) if parts else "Insufficient data for calibration"

    def get_stats(self) -> dict:
        """Get uncertainty quantification statistics."""
        cal = self.get_calibration()
        return {
            "total_predictions": len(self._predictions),
            "total_resolved": self._total_resolved,
            "pending": len(self.get_pending_predictions()),
            "calibration_error": cal["calibration_error"],
            "accuracy": cal["accuracy"],
            "brier_score": self.get_brier_score(),
            "domains_tracked": len(self._domains),
            "high_uncertainty_areas": self.get_high_uncertainty_areas(),
        }

    # -- Internal --

    def _update_calibration(self) -> None:
        """Recompute per-domain calibration after a resolution."""
        for domain, stats in self._domains.items():
            domain_preds = [
                p for p in self._predictions
                if p.outcome is not None and p.domain == domain
            ]
            if domain_preds:
                stats.predictions_correct = sum(
                    1 for p in domain_preds if p.outcome
                )
                stats.predictions_wrong = sum(
                    1 for p in domain_preds if not p.outcome
                )
