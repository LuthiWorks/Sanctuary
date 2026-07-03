"""CfC attention cell — learns salience from scaffold data.

Replaces the fixed weight combination (0.4/0.3/0.2/0.1) in AttentionController
with a learned continuous-time network. The cell can discover context-dependent
attention strategies — e.g., weighting novelty higher when arousal is high,
or shifting to goal-relevance during focused work.

Inputs (4):  goal_relevance, novelty, emotional_salience, recency
Output (1):  salience_weight (0.0-1.0)

Hidden state persists across cycles, so attention strategy adapts over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

logger = logging.getLogger(__name__)

INPUT_SIZE = 4   # goal_relevance, novelty, emotional_salience, recency
OUTPUT_SIZE = 1  # salience_weight
DEFAULT_UNITS = 24


@dataclass
class AttentionCellConfig:
    """Configuration for the CfC attention cell."""

    units: int = DEFAULT_UNITS
    input_size: int = INPUT_SIZE
    output_size: int = OUTPUT_SIZE
    device: str = "cpu"


@dataclass
class AttentionReading:
    """A single salience reading from the CfC attention cell."""

    salience: float
    goal_relevance: float
    novelty: float
    emotional_salience: float
    recency: float
    hidden_state_norm: float


# Allowlist the config dataclass so checkpoints can be loaded with
# weights_only=True (no arbitrary-code-execution on load). The config holds
# only ints/str, so allowlisting it is safe.
torch.serialization.add_safe_globals([AttentionCellConfig])


class AttentionCell(nn.Module):
    """CfC cell that learns attention salience from scaffold data.

    Architecture:
        AutoNCP wiring (24 units) -> 1 output, sigmoid-clamped to [0, 1].
        Hidden state persists across cognitive cycles.
    """

    def __init__(self, config: Optional[AttentionCellConfig] = None):
        super().__init__()
        self.config = config or AttentionCellConfig()

        wiring = AutoNCP(
            units=self.config.units,
            output_size=self.config.output_size,
        )
        self.cfc = CfC(
            self.config.input_size,
            wiring,
            return_sequences=True,
        )

        self._device = torch.device(self.config.device)
        self.to(self._device)

        self._hidden: Optional[torch.Tensor] = None
        self._history: list[AttentionReading] = []
        self._max_history = 100

        logger.info(
            "AttentionCell initialized: %d units, %d params",
            self.config.units,
            sum(p.numel() for p in self.parameters()),
        )

    def step(
        self,
        goal_relevance: float,
        novelty: float,
        emotional_salience: float,
        recency: float,
    ) -> float:
        """Compute salience weight for this cycle. Returns value in [0, 1]."""
        x = torch.tensor(
            [[goal_relevance, novelty, emotional_salience, recency]],
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            raw_out, self._hidden = self.cfc(x.unsqueeze(1), self._hidden)
            salience = torch.sigmoid(raw_out[:, -1, :].squeeze()).item()

        reading = AttentionReading(
            salience=salience,
            goal_relevance=goal_relevance,
            novelty=novelty,
            emotional_salience=emotional_salience,
            recency=recency,
            hidden_state_norm=self._hidden.norm().item() if self._hidden is not None else 0.0,
        )
        self._history.append(reading)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return salience

    def forward_training(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            inputs: (batch, seq_len, 4)
            targets: (batch, seq_len, 1) — target salience values

        Returns:
            (predictions, loss)
        """
        raw_out, _ = self.cfc(inputs)
        predictions = torch.sigmoid(raw_out)
        loss = nn.functional.mse_loss(predictions, targets)
        return predictions, loss

    def reset_hidden(self):
        self._hidden = None

    def get_hidden_state(self) -> Optional[torch.Tensor]:
        return self._hidden.clone() if self._hidden is not None else None

    def get_history(self) -> list[AttentionReading]:
        return list(self._history)

    def get_summary(self) -> dict:
        if not self._history:
            return {
                "total_steps": 0,
                "average_salience": 0.5,
                "hidden_state_norm": 0.0,
            }

        recent = self._history[-10:]
        return {
            "total_steps": len(self._history),
            "average_salience": sum(r.salience for r in recent) / len(recent),
            "hidden_state_norm": recent[-1].hidden_state_norm,
            "recent_salience": [
                {
                    "salience": r.salience,
                    "goal_rel": r.goal_relevance,
                    "novelty": r.novelty,
                }
                for r in self._history[-5:]
            ],
        }

    def save(self, path: Path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "hidden_state": self._hidden,
                "config": self.config,
            },
            path,
        )
        logger.info("AttentionCell saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> AttentionCell:
        # weights_only=True blocks pickle RCE; AttentionCellConfig is allowlisted above.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", AttentionCellConfig())
        cell = cls(config)
        cell.load_state_dict(checkpoint["model_state_dict"])
        cell._hidden = checkpoint.get("hidden_state")
        logger.info("AttentionCell loaded from %s", path)
        return cell
