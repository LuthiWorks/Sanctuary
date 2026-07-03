"""CfC precision weighting cell.

Replaces the heuristic precision computation (base + arousal_effect + error_boost)
with a learned continuous-time neural network that can discover nonlinear
relationships and temporal dynamics the heuristic cannot express.

Inputs (3):  arousal, prediction_error, base_precision
Output (1):  precision_weight (0.0-1.0)

The cell maintains hidden state across cycles, so its precision estimate
incorporates temporal context — recent history of arousal and prediction
errors shapes the current output. This is the "temporal thickness" that
the heuristic lacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

logger = logging.getLogger(__name__)

# Cell configuration
INPUT_SIZE = 3       # arousal, prediction_error, base_precision
OUTPUT_SIZE = 1      # precision_weight
DEFAULT_UNITS = 16   # CfC hidden units — small, fast, CPU-trainable


@dataclass
class PrecisionCellConfig:
    """Configuration for the CfC precision cell."""

    units: int = DEFAULT_UNITS
    input_size: int = INPUT_SIZE
    output_size: int = OUTPUT_SIZE
    device: str = "cpu"


@dataclass
class PrecisionReading:
    """A single precision reading from the CfC cell."""

    precision: float
    arousal: float
    prediction_error: float
    base_precision: float
    hidden_state_norm: float  # magnitude of hidden state (introspection)


# Allowlist the config dataclass so checkpoints can be loaded with
# weights_only=True (no arbitrary-code-execution on load). The config holds
# only ints/str, so allowlisting it is safe.
torch.serialization.add_safe_globals([PrecisionCellConfig])


class PrecisionCell(nn.Module):
    """CfC cell that learns precision weighting from scaffold data.

    Architecture:
        AutoNCP wiring (16 units) -> 1 output, sigmoid-clamped to [0, 1].
        Hidden state persists across cognitive cycles.

    Usage:
        cell = PrecisionCell()
        precision = cell.step(arousal=0.7, prediction_error=0.3, base_precision=0.5)
    """

    def __init__(self, config: Optional[PrecisionCellConfig] = None):
        super().__init__()
        self.config = config or PrecisionCellConfig()

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

        # Persistent hidden state — carries across cycles
        self._hidden: Optional[torch.Tensor] = None

        # History for logging / training data collection
        self._history: list[PrecisionReading] = []
        self._max_history = 100

        logger.info(
            "PrecisionCell initialized: %d units, %d params",
            self.config.units,
            sum(p.numel() for p in self.parameters()),
        )

    def step(
        self,
        arousal: float,
        prediction_error: float,
        base_precision: float,
    ) -> float:
        """Compute precision weight for this cycle.

        Evolves the hidden state and returns a precision value in [0, 1].
        """
        x = torch.tensor(
            [[arousal, prediction_error, base_precision]],
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            # Add time dimension: (batch=1, seq=1, features=3)
            raw_out, self._hidden = self.cfc(
                x.unsqueeze(1), self._hidden
            )
            # raw_out is (batch=1, seq=1, output=1); take last timestep
            # Sigmoid to clamp output to [0, 1]
            precision = torch.sigmoid(raw_out[:, -1, :].squeeze()).item()

        reading = PrecisionReading(
            precision=precision,
            arousal=arousal,
            prediction_error=prediction_error,
            base_precision=base_precision,
            hidden_state_norm=self._hidden.norm().item() if self._hidden is not None else 0.0,
        )
        self._history.append(reading)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return precision

    def forward_training(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training (with gradients).

        Args:
            inputs: (batch, seq_len, 3) — arousal, prediction_error, base_precision
            targets: (batch, seq_len, 1) — target precision values

        Returns:
            (predictions, loss) where predictions are sigmoid-clamped.
        """
        raw_out, _ = self.cfc(inputs)  # (batch, seq_len, 1)
        predictions = torch.sigmoid(raw_out)
        loss = nn.functional.mse_loss(predictions, targets)
        return predictions, loss

    def reset_hidden(self):
        """Reset the persistent hidden state (e.g., at session start)."""
        self._hidden = None

    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """Return current hidden state for inspection."""
        return self._hidden.clone() if self._hidden is not None else None

    def get_history(self) -> list[PrecisionReading]:
        """Return recent precision readings."""
        return list(self._history)

    def get_summary(self) -> dict:
        """Summary statistics for monitoring."""
        if not self._history:
            return {
                "total_steps": 0,
                "average_precision": 0.5,
                "hidden_state_norm": 0.0,
            }

        recent = self._history[-10:]
        return {
            "total_steps": len(self._history),
            "average_precision": sum(r.precision for r in recent) / len(recent),
            "hidden_state_norm": recent[-1].hidden_state_norm,
            "recent_precisions": [
                {
                    "precision": r.precision,
                    "arousal": r.arousal,
                    "prediction_error": r.prediction_error,
                }
                for r in self._history[-5:]
            ],
        }

    def save(self, path: Path):
        """Save model weights and hidden state."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "hidden_state": self._hidden,
                "config": self.config,
            },
            path,
        )
        logger.info("PrecisionCell saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> PrecisionCell:
        """Load a trained PrecisionCell from disk."""
        # weights_only=True blocks pickle RCE; PrecisionCellConfig is allowlisted above.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", PrecisionCellConfig())
        cell = cls(config)
        cell.load_state_dict(checkpoint["model_state_dict"])
        cell._hidden = checkpoint.get("hidden_state")
        logger.info("PrecisionCell loaded from %s", path)
        return cell
