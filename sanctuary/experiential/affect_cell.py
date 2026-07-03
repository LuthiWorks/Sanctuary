"""CfC affect cell — learns emotional dynamics from scaffold data.

Replaces the keyword-matching heuristic in ScaffoldAffect with a learned
continuous-time neural network. The cell discovers nonlinear emotional
trajectories and temporal patterns (e.g., sustained stress builds differently
than momentary surprise).

Inputs (3):  percept_valence_delta, percept_arousal_delta, llm_emotion_shift
Output (3):  valence, arousal, dominance

The cell maintains hidden state across cycles, so emotional trajectories
have temporal thickness — a sequence of negative inputs produces a different
response than a single negative input, even at the same magnitude.
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

INPUT_SIZE = 3   # percept_valence_delta, percept_arousal_delta, llm_emotion_shift
OUTPUT_SIZE = 3  # valence, arousal, dominance
DEFAULT_UNITS = 32


@dataclass
class AffectCellConfig:
    """Configuration for the CfC affect cell."""

    units: int = DEFAULT_UNITS
    input_size: int = INPUT_SIZE
    output_size: int = OUTPUT_SIZE
    device: str = "cpu"


@dataclass
class AffectReading:
    """A single VAD reading from the CfC affect cell."""

    valence: float
    arousal: float
    dominance: float
    percept_valence_delta: float
    percept_arousal_delta: float
    llm_emotion_shift: float
    hidden_state_norm: float


# Allowlist the config dataclass so checkpoints can be loaded with
# weights_only=True (no arbitrary-code-execution on load). The config holds
# only ints/str, so allowlisting it is safe.
torch.serialization.add_safe_globals([AffectCellConfig])


class AffectCell(nn.Module):
    """CfC cell that learns emotional dynamics from scaffold data.

    Architecture:
        AutoNCP wiring (32 units) -> 3 outputs.
        Valence clamped to [-1, 1] via tanh.
        Arousal and dominance clamped to [0, 1] via sigmoid.
        Hidden state persists across cognitive cycles.
    """

    def __init__(self, config: Optional[AffectCellConfig] = None):
        super().__init__()
        self.config = config or AffectCellConfig()

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
        self._history: list[AffectReading] = []
        self._max_history = 100

        logger.info(
            "AffectCell initialized: %d units, %d params",
            self.config.units,
            sum(p.numel() for p in self.parameters()),
        )

    def step(
        self,
        percept_valence_delta: float,
        percept_arousal_delta: float,
        llm_emotion_shift: float,
    ) -> tuple[float, float, float]:
        """Compute VAD for this cycle.

        Returns (valence, arousal, dominance) where:
            valence in [-1, 1]
            arousal in [0, 1]
            dominance in [0, 1]
        """
        x = torch.tensor(
            [[percept_valence_delta, percept_arousal_delta, llm_emotion_shift]],
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            raw_out, self._hidden = self.cfc(x.unsqueeze(1), self._hidden)
            out = raw_out[:, -1, :]  # (1, 3)

            valence = torch.tanh(out[:, 0]).item()
            arousal = torch.sigmoid(out[:, 1]).item()
            dominance = torch.sigmoid(out[:, 2]).item()

        reading = AffectReading(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            percept_valence_delta=percept_valence_delta,
            percept_arousal_delta=percept_arousal_delta,
            llm_emotion_shift=llm_emotion_shift,
            hidden_state_norm=self._hidden.norm().item() if self._hidden is not None else 0.0,
        )
        self._history.append(reading)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return valence, arousal, dominance

    def forward_training(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            inputs: (batch, seq_len, 3)
            targets: (batch, seq_len, 3) — target valence, arousal, dominance

        Returns:
            (predictions, loss)
        """
        raw_out, _ = self.cfc(inputs)
        # Apply activations per output channel
        predictions = torch.cat([
            torch.tanh(raw_out[:, :, 0:1]),
            torch.sigmoid(raw_out[:, :, 1:2]),
            torch.sigmoid(raw_out[:, :, 2:3]),
        ], dim=2)
        loss = nn.functional.mse_loss(predictions, targets)
        return predictions, loss

    def reset_hidden(self):
        """Reset persistent hidden state."""
        self._hidden = None

    def get_hidden_state(self) -> Optional[torch.Tensor]:
        return self._hidden.clone() if self._hidden is not None else None

    def get_history(self) -> list[AffectReading]:
        return list(self._history)

    def get_summary(self) -> dict:
        if not self._history:
            return {
                "total_steps": 0,
                "average_valence": 0.0,
                "average_arousal": 0.2,
                "average_dominance": 0.5,
                "hidden_state_norm": 0.0,
            }

        recent = self._history[-10:]
        return {
            "total_steps": len(self._history),
            "average_valence": sum(r.valence for r in recent) / len(recent),
            "average_arousal": sum(r.arousal for r in recent) / len(recent),
            "average_dominance": sum(r.dominance for r in recent) / len(recent),
            "hidden_state_norm": recent[-1].hidden_state_norm,
            "recent_vad": [
                {"v": r.valence, "a": r.arousal, "d": r.dominance}
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
        logger.info("AffectCell saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> AffectCell:
        # weights_only=True blocks pickle RCE; AffectCellConfig is allowlisted above.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", AffectCellConfig())
        cell = cls(config)
        cell.load_state_dict(checkpoint["model_state_dict"])
        cell._hidden = checkpoint.get("hidden_state")
        logger.info("AffectCell loaded from %s", path)
        return cell
