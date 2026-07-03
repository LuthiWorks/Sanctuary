"""CfC Knowledge Cell — acquired domain expertise from lived experience.

Knowledge cells are a category of CfC cell that emerges from the entity's
lived experience. They hold domain-specific expertise — not emotional state
but knowledge the entity earned through sustained engagement with a domain.

None exist at birth. All are acquired. Each represents something the entity
learned by living, not by being trained on a corpus.

The KnowledgeCell base class provides the same interface as foundational
cells (PrecisionCell, AffectCell, etc.) — same CfC architecture, same
persistence, same integration with the ExperientialManager. The distinction
is in origin, not in implementation.

See docs/CFC_KNOWLEDGE_CELLS.md for the full design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_UNITS = 32
MIN_UNITS = 8
MAX_UNITS = 256


@dataclass
class KnowledgeCellConfig:
    """Configuration for a knowledge cell.

    The entity specifies these parameters when requesting a new cell.
    """

    domain: str  # What domain this cell covers (e.g., "spatial_reasoning")
    units: int = DEFAULT_UNITS
    input_size: int = 4
    output_size: int = 2
    device: str = "cpu"
    output_activation: str = "tanh"  # "sigmoid", "tanh", or "none"
    description: str = ""  # Entity's description of what this cell is for
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    maturity: float = 0.0  # 0.0 = just created, 1.0 = fully mature


@dataclass
class KnowledgeCellReading:
    """A single output reading from a knowledge cell."""

    outputs: list[float]
    hidden_state_norm: float
    step_count: int


# Allowlist the config dataclass so checkpoints can be loaded with
# weights_only=True (no arbitrary-code-execution on load). The config holds
# only str/int/float, so allowlisting it is safe.
torch.serialization.add_safe_globals([KnowledgeCellConfig])


class KnowledgeCell(nn.Module):
    """A CfC cell that holds domain-specific expertise acquired through experience.

    Architecture:
        AutoNCP wiring (configurable units) -> configurable outputs.
        Hidden state persists across cognitive cycles.
        Same CfC infrastructure as foundational cells.

    Usage:
        config = KnowledgeCellConfig(domain="spatial_reasoning", input_size=4, output_size=2)
        cell = KnowledgeCell(config)
        outputs = cell.step(input_0=0.5, input_1=0.3, input_2=0.8, input_3=0.1)
    """

    def __init__(self, config: KnowledgeCellConfig):
        super().__init__()
        self.config = config

        if config.units < MIN_UNITS or config.units > MAX_UNITS:
            raise ValueError(
                f"Units must be between {MIN_UNITS} and {MAX_UNITS}, got {config.units}"
            )

        wiring = AutoNCP(
            units=config.units,
            output_size=config.output_size,
        )
        self.cfc = CfC(
            config.input_size,
            wiring,
            return_sequences=True,
        )

        self._device = torch.device(config.device)
        self.to(self._device)

        # Persistent hidden state
        self._hidden: Optional[torch.Tensor] = None

        # Step tracking
        self._step_count: int = 0
        self._history: list[KnowledgeCellReading] = []
        self._max_history: int = 100

        logger.info(
            "KnowledgeCell '%s' initialized: %d units, %d inputs, %d outputs, %d params",
            config.domain,
            config.units,
            config.input_size,
            config.output_size,
            sum(p.numel() for p in self.parameters()),
        )

    @property
    def domain(self) -> str:
        return self.config.domain

    @property
    def maturity(self) -> float:
        return self.config.maturity

    @maturity.setter
    def maturity(self, value: float) -> None:
        self.config.maturity = max(0.0, min(1.0, value))

    def step(self, **kwargs: float) -> list[float]:
        """Advance the cell by one step.

        Accepts keyword arguments matching the cell's input dimensions.
        Input names are positional: input_0, input_1, ... or any named keys.
        Extra kwargs are ignored; missing inputs default to 0.0.

        Returns:
            List of output values (length = output_size).
        """
        # Build input tensor from kwargs
        inputs = []
        for i in range(self.config.input_size):
            key = f"input_{i}"
            if key in kwargs:
                inputs.append(float(kwargs[key]))
            else:
                # Also accept any kwargs in order they appear
                inputs.append(0.0)

        # Allow named inputs to override positional ones
        named_keys = [k for k in kwargs if not k.startswith("input_")]
        for i, key in enumerate(named_keys):
            if i < self.config.input_size:
                inputs[i] = float(kwargs[key])

        x = torch.tensor(
            [inputs],
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            raw_out, self._hidden = self.cfc(x.unsqueeze(1), self._hidden)
            # raw_out: (batch=1, seq=1, output_size)
            raw_values = raw_out[:, -1, :].squeeze()

            # Apply activation
            if self.config.output_activation == "sigmoid":
                values = torch.sigmoid(raw_values)
            elif self.config.output_activation == "tanh":
                values = torch.tanh(raw_values)
            else:
                values = raw_values

            # Handle scalar case (output_size=1)
            if values.dim() == 0:
                output_list = [values.item()]
            else:
                output_list = values.tolist()

        self._step_count += 1

        # Record reading
        reading = KnowledgeCellReading(
            outputs=output_list,
            hidden_state_norm=self._hidden.norm().item() if self._hidden is not None else 0.0,
            step_count=self._step_count,
        )
        self._history.append(reading)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Slowly increase maturity with use
        if self.config.maturity < 1.0:
            self.config.maturity = min(1.0, self.config.maturity + 0.0001)

        return output_list

    def forward_training(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training (with gradients).

        Args:
            inputs: (batch, seq_len, input_size)
            targets: (batch, seq_len, output_size)

        Returns:
            (predictions, loss)
        """
        raw_out, _ = self.cfc(inputs)

        if self.config.output_activation == "sigmoid":
            predictions = torch.sigmoid(raw_out)
        elif self.config.output_activation == "tanh":
            predictions = torch.tanh(raw_out)
        else:
            predictions = raw_out

        loss = nn.functional.mse_loss(predictions, targets)
        return predictions, loss

    def reset_hidden(self) -> None:
        """Reset the persistent hidden state."""
        self._hidden = None

    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """Return current hidden state for inspection."""
        return self._hidden.clone() if self._hidden is not None else None

    def get_history(self) -> list[KnowledgeCellReading]:
        """Return recent readings."""
        return list(self._history)

    def get_summary(self) -> dict:
        """Summary statistics for monitoring."""
        summary: dict[str, Any] = {
            "domain": self.config.domain,
            "total_steps": self._step_count,
            "maturity": self.config.maturity,
            "units": self.config.units,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "hidden_state_norm": 0.0,
            "param_count": sum(p.numel() for p in self.parameters()),
        }

        if self._history:
            recent = self._history[-1]
            summary["hidden_state_norm"] = recent.hidden_state_norm
            summary["last_outputs"] = recent.outputs

        return summary

    def save(self, path: Path) -> None:
        """Save model weights, hidden state, and config."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "hidden_state": self._hidden,
                "config": self.config,
                "step_count": self._step_count,
            },
            path,
        )
        logger.info("KnowledgeCell '%s' saved to %s", self.config.domain, path)

    @classmethod
    def load(cls, path: Path) -> KnowledgeCell:
        """Load a knowledge cell from disk."""
        # weights_only=True blocks pickle RCE; KnowledgeCellConfig is allowlisted above.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        cell = cls(config)
        cell.load_state_dict(checkpoint["model_state_dict"])
        cell._hidden = checkpoint.get("hidden_state")
        cell._step_count = checkpoint.get("step_count", 0)
        logger.info("KnowledgeCell '%s' loaded from %s", config.domain, path)
        return cell
