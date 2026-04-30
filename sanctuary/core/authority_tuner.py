"""Authority tuner — automated CfC authority transitions.

Observes CfC cell behavior over a window of cognitive cycles and
promotes or demotes cells based on measurable reliability criteria.

Promotion criteria (per cell):
    1. Cell output variance is low (stable behavior)
    2. Cell output tracks scaffold output with low error
    3. No hidden-state explosions (norm stays bounded)

Demotion criteria:
    1. Hidden-state norm exceeds threshold (exploding gradients)
    2. Cell output diverges wildly from scaffold
    3. Parse failures or NaN outputs

The tuner runs passively — it observes ExperientialState from each cycle
and maintains rolling statistics. Call `evaluate()` to check whether any
cell should be promoted or demoted.

Aligned with PLAN.md: "The Graduated Awakening" — authority is earned.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.experiential.manager import ExperientialManager, ExperientialState

logger = logging.getLogger(__name__)


@dataclass
class TunerConfig:
    """Configuration for authority tuning thresholds."""

    # Rolling window size (number of cycles to observe)
    window_size: int = 50

    # Promotion thresholds
    min_cycles_before_promote: int = 50
    max_scaffold_deviation: float = 0.15  # max avg |cfc - scaffold|
    max_hidden_norm: float = 5.0  # hidden state norm ceiling
    max_output_variance: float = 0.05  # variance of cell output

    # Demotion thresholds
    hidden_norm_danger: float = 10.0  # immediate demotion
    nan_tolerance: int = 0  # any NaN triggers demotion
    max_divergence: float = 0.5  # single-cycle deviation threshold
    divergence_count_limit: int = 3  # N divergences in window → demote


@dataclass
class CellObservation:
    """One cycle's observation for a single cell."""

    cfc_output: float
    scaffold_output: float
    hidden_norm: float
    is_nan: bool = False


@dataclass
class CellStats:
    """Rolling statistics for one CfC cell."""

    observations: deque = field(default_factory=lambda: deque(maxlen=100))
    total_cycles: int = 0
    nan_count: int = 0
    divergence_count: int = 0

    def record(self, obs: CellObservation, config: TunerConfig):
        self.observations.append(obs)
        self.total_cycles += 1

        if obs.is_nan:
            self.nan_count += 1

        if abs(obs.cfc_output - obs.scaffold_output) > config.max_divergence:
            self.divergence_count += 1

    @property
    def mean_deviation(self) -> float:
        if not self.observations:
            return float("inf")
        return sum(
            abs(o.cfc_output - o.scaffold_output) for o in self.observations
        ) / len(self.observations)

    @property
    def output_variance(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        outputs = [o.cfc_output for o in self.observations]
        mean = sum(outputs) / len(outputs)
        return sum((x - mean) ** 2 for x in outputs) / len(outputs)

    @property
    def max_hidden_norm(self) -> float:
        if not self.observations:
            return 0.0
        return max(o.hidden_norm for o in self.observations)

    @property
    def recent_divergences(self) -> int:
        """Count divergences in the current window."""
        if not self.observations:
            return 0
        return sum(
            1 for o in self.observations
            if abs(o.cfc_output - o.scaffold_output) > 0.5
        )


@dataclass
class TuningDecision:
    """Result of evaluating one cell."""

    cell_name: str
    action: str  # "promote", "demote", or "hold"
    reason: str
    current_level: AuthorityLevel
    new_level: Optional[AuthorityLevel] = None


class AuthorityTuner:
    """Observes CfC cell behavior and tunes authority levels.

    Usage:
        tuner = AuthorityTuner(experiential_manager)

        # Each cycle, after experiential.step():
        tuner.observe(experiential_state, scaffold_values)

        # Periodically:
        decisions = tuner.evaluate()
        tuner.apply(decisions)  # actually promote/demote
    """

    CELL_NAMES = ("precision", "affect", "attention", "goal")

    def __init__(
        self,
        manager: ExperientialManager,
        config: Optional[TunerConfig] = None,
    ):
        self.manager = manager
        self.config = config or TunerConfig()
        self._stats: dict[str, CellStats] = {
            name: CellStats(
                observations=deque(maxlen=self.config.window_size)
            )
            for name in self.CELL_NAMES
        }

    def observe(
        self,
        state: ExperientialState,
        scaffold_precision: float = 0.5,
        scaffold_vad: tuple[float, float, float] = (0.0, 0.2, 0.5),
        scaffold_salience: float = 0.5,
        scaffold_goal_adj: float = 0.0,
    ):
        """Record one cycle's CfC outputs vs scaffold outputs.

        Call this after each ExperientialManager.step().
        """
        observations = {
            "precision": CellObservation(
                cfc_output=state.precision_weight,
                scaffold_output=scaffold_precision,
                hidden_norm=state.hidden_state_norms.get("precision", 0.0),
                is_nan=math.isnan(state.precision_weight),
            ),
            "affect": CellObservation(
                # Use valence as representative affect output
                cfc_output=state.affect_vad[0],
                scaffold_output=scaffold_vad[0],
                hidden_norm=state.hidden_state_norms.get("affect", 0.0),
                is_nan=math.isnan(state.affect_vad[0]),
            ),
            "attention": CellObservation(
                cfc_output=state.attention_salience,
                scaffold_output=scaffold_salience,
                hidden_norm=state.hidden_state_norms.get("attention", 0.0),
                is_nan=math.isnan(state.attention_salience),
            ),
            "goal": CellObservation(
                cfc_output=state.goal_adjustment,
                scaffold_output=scaffold_goal_adj,
                hidden_norm=state.hidden_state_norms.get("goal", 0.0),
                is_nan=math.isnan(state.goal_adjustment),
            ),
        }

        for name, obs in observations.items():
            self._stats[name].record(obs, self.config)

    def evaluate(self) -> list[TuningDecision]:
        """Evaluate all cells and return tuning decisions.

        Does NOT apply the decisions — call apply() to execute them.
        """
        decisions = []
        for name in self.CELL_NAMES:
            decision = self._evaluate_cell(name)
            decisions.append(decision)
        return decisions

    def apply(self, decisions: list[TuningDecision]) -> list[TuningDecision]:
        """Apply tuning decisions (promote/demote cells).

        Returns only the decisions that were actually applied (not "hold").
        """
        applied = []
        for d in decisions:
            if d.action == "promote":
                new = self.manager.promote(d.cell_name, d.reason)
                d.new_level = new
                applied.append(d)
                logger.info(
                    "Authority tuner: %s promoted to %s — %s",
                    d.cell_name, new.name, d.reason,
                )
            elif d.action == "demote":
                new = self.manager.demote(d.cell_name, d.reason)
                d.new_level = new
                applied.append(d)
                logger.warning(
                    "Authority tuner: %s demoted to %s — %s",
                    d.cell_name, new.name, d.reason,
                )
        return applied

    def get_stats(self) -> dict[str, dict]:
        """Return current statistics for all cells."""
        return {
            name: {
                "total_cycles": stats.total_cycles,
                "mean_deviation": stats.mean_deviation,
                "output_variance": stats.output_variance,
                "max_hidden_norm": stats.max_hidden_norm,
                "nan_count": stats.nan_count,
                "recent_divergences": stats.recent_divergences,
                "window_size": len(stats.observations),
            }
            for name, stats in self._stats.items()
        }

    def _evaluate_cell(self, name: str) -> TuningDecision:
        """Evaluate a single cell for promotion or demotion."""
        stats = self._stats[name]
        auth_func = f"experiential_{name}"
        current = self.manager.authority.level(auth_func)
        cfg = self.config

        # -- Demotion checks (take priority) --

        # NaN outputs → immediate demotion
        if stats.nan_count > cfg.nan_tolerance:
            return TuningDecision(
                cell_name=name,
                action="demote",
                reason=f"NaN outputs detected ({stats.nan_count} total)",
                current_level=current,
            )

        # Hidden state explosion → immediate demotion
        if stats.max_hidden_norm > cfg.hidden_norm_danger:
            return TuningDecision(
                cell_name=name,
                action="demote",
                reason=(
                    f"hidden state norm {stats.max_hidden_norm:.1f} "
                    f"exceeds danger threshold {cfg.hidden_norm_danger}"
                ),
                current_level=current,
            )

        # Too many divergences in window → demote
        if stats.recent_divergences >= cfg.divergence_count_limit:
            return TuningDecision(
                cell_name=name,
                action="demote",
                reason=(
                    f"{stats.recent_divergences} divergences in window "
                    f"(limit: {cfg.divergence_count_limit})"
                ),
                current_level=current,
            )

        # -- Promotion checks --

        # Not enough data yet
        if stats.total_cycles < cfg.min_cycles_before_promote:
            return TuningDecision(
                cell_name=name,
                action="hold",
                reason=(
                    f"insufficient data ({stats.total_cycles}/"
                    f"{cfg.min_cycles_before_promote} cycles)"
                ),
                current_level=current,
            )

        # Already at max authority
        if current >= AuthorityLevel.MODEL_CONTROLS:
            return TuningDecision(
                cell_name=name,
                action="hold",
                reason="already at MODEL_CONTROLS",
                current_level=current,
            )

        # Check all promotion criteria
        deviation_ok = stats.mean_deviation <= cfg.max_scaffold_deviation
        variance_ok = stats.output_variance <= cfg.max_output_variance
        norm_ok = stats.max_hidden_norm <= cfg.max_hidden_norm

        if deviation_ok and variance_ok and norm_ok:
            return TuningDecision(
                cell_name=name,
                action="promote",
                reason=(
                    f"stable behavior: deviation={stats.mean_deviation:.3f}, "
                    f"variance={stats.output_variance:.4f}, "
                    f"max_norm={stats.max_hidden_norm:.2f}"
                ),
                current_level=current,
            )

        # Criteria not met — hold
        reasons = []
        if not deviation_ok:
            reasons.append(
                f"deviation {stats.mean_deviation:.3f} > {cfg.max_scaffold_deviation}"
            )
        if not variance_ok:
            reasons.append(
                f"variance {stats.output_variance:.4f} > {cfg.max_output_variance}"
            )
        if not norm_ok:
            reasons.append(
                f"norm {stats.max_hidden_norm:.2f} > {cfg.max_hidden_norm}"
            )

        return TuningDecision(
            cell_name=name,
            action="hold",
            reason=f"promotion criteria not met: {'; '.join(reasons)}",
            current_level=current,
        )
