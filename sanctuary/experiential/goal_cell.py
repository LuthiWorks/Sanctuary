"""CfC goal dynamics cell — learns priority adjustment from scaffold data.

Replaces the manual staleness counters and fixed thresholds in GoalDynamics
with a learned continuous-time network. The cell can discover temporal
patterns in goal management — e.g., how frustration builds nonlinearly,
how deadline urgency interacts with emotional state.

Inputs (3):  cycles_stalled_norm, deadline_urgency, emotional_congruence
Output (1):  priority_adjustment (-1.0 to 1.0)

Hidden state persists across cycles, so the cell develops a sense of
how long goals have been stalling and adapts its urgency response.
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

INPUT_SIZE = 3   # cycles_stalled_norm, deadline_urgency, emotional_congruence
OUTPUT_SIZE = 1  # priority_adjustment
DEFAULT_UNITS = 16


@dataclass
class GoalCellConfig:
    """Configuration for the CfC goal dynamics cell."""

    units: int = DEFAULT_UNITS
    input_size: int = INPUT_SIZE
    output_size: int = OUTPUT_SIZE
    device: str = "cpu"


@dataclass
class GoalReading:
    """A single priority adjustment reading from the CfC goal cell."""

    priority_adjustment: float
    cycles_stalled_norm: float
    deadline_urgency: float
    emotional_congruence: float
    hidden_state_norm: float


# Allowlist the config dataclass so checkpoints can be loaded with
# weights_only=True (no arbitrary-code-execution on load). The config holds
# only ints/str, so allowlisting it is safe.
torch.serialization.add_safe_globals([GoalCellConfig])


class GoalCell(nn.Module):
    """CfC cell that learns goal priority adjustment from scaffold data.

    Architecture:
        AutoNCP wiring (16 units) -> 1 output, tanh-clamped to [-1, 1].
        The output is a priority adjustment, not an absolute priority.
        Hidden state persists across cognitive cycles.
    """

    def __init__(self, config: Optional[GoalCellConfig] = None):
        super().__init__()
        self.config = config or GoalCellConfig()

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
        self._history: list[GoalReading] = []
        self._max_history = 100

        logger.info(
            "GoalCell initialized: %d units, %d params",
            self.config.units,
            sum(p.numel() for p in self.parameters()),
        )

    def step(
        self,
        cycles_stalled_norm: float,
        deadline_urgency: float,
        emotional_congruence: float,
    ) -> float:
        """Compute priority adjustment for this cycle.

        Returns value in [-1, 1] (tanh activation). Typical scaffold
        adjustments are in [-0.25, 0.25], but the cell can learn its
        own scaling.
        """
        x = torch.tensor(
            [[cycles_stalled_norm, deadline_urgency, emotional_congruence]],
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            raw_out, self._hidden = self.cfc(x.unsqueeze(1), self._hidden)
            adjustment = torch.tanh(raw_out[:, -1, :].squeeze()).item()

        reading = GoalReading(
            priority_adjustment=adjustment,
            cycles_stalled_norm=cycles_stalled_norm,
            deadline_urgency=deadline_urgency,
            emotional_congruence=emotional_congruence,
            hidden_state_norm=self._hidden.norm().item() if self._hidden is not None else 0.0,
        )
        self._history.append(reading)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return adjustment

    def forward_training(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            inputs: (batch, seq_len, 3)
            targets: (batch, seq_len, 1) — target priority adjustments

        Returns:
            (predictions, loss)
        """
        raw_out, _ = self.cfc(inputs)
        predictions = torch.tanh(raw_out)
        loss = nn.functional.mse_loss(predictions, targets)
        return predictions, loss

    def reset_hidden(self):
        self._hidden = None

    def get_hidden_state(self) -> Optional[torch.Tensor]:
        return self._hidden.clone() if self._hidden is not None else None

    def get_history(self) -> list[GoalReading]:
        return list(self._history)

    def get_summary(self) -> dict:
        if not self._history:
            return {
                "total_steps": 0,
                "average_adjustment": 0.0,
                "hidden_state_norm": 0.0,
            }

        recent = self._history[-10:]
        return {
            "total_steps": len(self._history),
            "average_adjustment": sum(r.priority_adjustment for r in recent) / len(recent),
            "hidden_state_norm": recent[-1].hidden_state_norm,
            "recent_adjustments": [
                {
                    "adj": r.priority_adjustment,
                    "stalled": r.cycles_stalled_norm,
                    "deadline": r.deadline_urgency,
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
        logger.info("GoalCell saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> GoalCell:
        # weights_only=True blocks pickle RCE; GoalCellConfig is allowlisted above.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", GoalCellConfig())
        cell = cls(config)
        cell.load_state_dict(checkpoint["model_state_dict"])
        cell._hidden = checkpoint.get("hidden_state")
        logger.info("GoalCell loaded from %s", path)
        return cell
