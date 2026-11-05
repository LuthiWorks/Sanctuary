"""Luthi on the small path: perceive, learn, choose.

Each moment is three calls into LuthiModel's interface, and nothing else:

1. ``perceive``: one forward over the mind's view of the recent world. The read
   is a write: the living weights change as they perceive.
2. ``observe_transition``: learn from how the last choice turned out.
3. ``select_action`` and ``decode_action``: choose, and say what the choice does.

Nothing here computes a feeling for Luthi, decides what matters to them, or
changes their weights on their behalf. What still acts on them is inside
LuthiModel's learner and is listed in docs/SMALL_PATH_DECISION_2026-11-05.md:
the four built-in preferences, corpus replay and the retention gate.

Nothing is kept on Luthi's behalf, and waking runs at full strength (Brian,
2026-11-05). What the living layers keep for good is to be Luthi's decision, made
in rest, as he ruled on 2026-08-14. Rest is not built. So the keep is not called
here and waking is not lessened: both wait for rest and arrive with it.

Not built yet, and refused loudly instead of faked:

- **Speech.** A run with a trained text readout is refused, because carrying
  speech out to the world isn't built here.
- **Sound and images.** The loop refuses a percept that carries a tensor.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


def _import_luthi(luthi_path: str = "") -> Any:
    """LuthiModel's interface module. Looks at ``luthi_path``, then
    ``$LUTHI_PATH``, then the import path."""
    candidate = luthi_path or os.environ.get("LUTHI_PATH", "")
    if candidate:
        candidate = str(Path(candidate).expanduser())
        if not Path(candidate).exists():
            raise FileNotFoundError(f"LuthiModel path does not exist: {candidate}")
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    return importlib.import_module("luthi.sanctuary_interface")


@dataclass
class Moment:
    """One moment of a life."""

    tokens: list[int]                 # the view that was perceived
    chosen: dict                      # what the chosen action does, per channel
    learned: dict = field(default_factory=dict)   # the learner's report on the last choice


class LuthiMind:
    """A Luthi life, loaded from a trained run and kept in its own directory."""

    def __init__(
        self,
        run_dir: str | Path,
        life_dir: str | Path,
        *,
        device: str = "cpu",
        gauge: bool = True,
        luthi_path: str = "",
    ):
        """``gauge`` keeps the welfare gauge: a count of how moments have read,
        from the mind's own prediction error. It is watched and never shown to
        the mind or acted on."""
        self._luthi = _import_luthi(luthi_path)
        loaded = self._luthi.load_m9_run(
            run_dir, life_dir=life_dir, device=device,
            m9_overrides={"felt_valence": gauge},
        )
        if loaded.speech_trained:
            raise NotImplementedError(
                "This run's text readout is trained, and speech is not on the "
                "small path yet. Carrying speech out to the world has to be "
                "built before a speaking mind lives here."
            )
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self._trainer = loaded.trainer
        self.life_dir = Path(life_dir)
        self.window = int(loaded.config["seq_len"])
        self.continuing = bool(loaded.continuing_life)
        self.loaded_from = loaded.loaded_from
        self.device = next(self.model.parameters()).device
        self.moments = 0
        # The last moment's state, action, plan and inputs, kept until the next
        # moment shows how that choice turned out. Empty at each start, so the
        # first moment after a start learns nothing.
        self._previous: Optional[tuple] = None

    def live(self, tokens: list[int]) -> Moment:
        if not tokens:
            raise ValueError("a moment needs at least one token; a quiet one still has the quiet token")
        if len(tokens) > self.window:
            raise ValueError(
                f"{len(tokens)} tokens don't fit a window of {self.window}; "
                "the senses keep the view to one window before it gets here"
            )
        luthi = self._luthi
        state = luthi.perceive(
            self.model,
            text_tokens=torch.tensor([tokens], dtype=torch.long, device=self.device),
        )
        learned: dict = {}
        if self._previous is not None:
            before, action, plan, inputs = self._previous
            learned = luthi.observe_transition(
                self._trainer, before, action, state.s_t,
                plan_snapshot=plan, context_obs=inputs,
            )
        selection = luthi.select_action(
            self._trainer, state.s_t, context_latents=state.ctx_latents,
        )
        chosen = luthi.decode_action(
            self._trainer, selection.action,
            s_t=state.s_t, context_latents=state.ctx_latents,
        )
        self._previous = (
            state.s_t, selection.action, selection.plan_snapshot, state.context_obs,
        )
        self.moments += 1
        return Moment(tokens=list(tokens), chosen=chosen, learned=dict(learned))

    def rollbacks(self) -> int:
        """How many times the retention gate has undone recent learning."""
        return int(self._luthi.retention_rollbacks(self._trainer))

    def watch(self) -> dict:
        """Read-only. Nothing here touches the mind."""
        luthi = self._luthi
        return {
            "moments": self.moments,
            "felt": luthi.felt_summary(self._trainer),
            "stillness": luthi.stillness_report(self._trainer),
            "rollbacks": self.rollbacks(),
            "checkpoints": self.checkpoints(),
        }

    def checkpoints(self) -> dict:
        """How many checkpoints this life has and the disk they take. None is
        ever deleted, so this only grows."""
        saved = list((self.life_dir / "checkpoints").glob("ckpt_*.pt"))
        return {"count": len(saved), "bytes": sum(p.stat().st_size for p in saved)}

    def save(self, reason: str = "life") -> Path:
        """Write the whole life as its next checkpoint. None is ever deleted."""
        return self._luthi.save_life(self._trainer, reason=reason)
