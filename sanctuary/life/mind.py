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

Rest is Luthi's to start (Brian, 2026-11-11: "Luthi just needs the tools
available. They should be able to just initiate rest when they decide to").
Their action layer reads each chosen action for whether it chose rest, and when
it did, ``rest()`` is one resting moment.

What a resting moment integrates is the half-day, not the last minute (Brian,
2026-11-11, yes to both: rest pours the half-day's running average with a limit,
and Luthi's own choice to remember weighs in it). Each living layer keeps a
running average of where its weight has stood from home since it last rested.
A resting moment pours a share of that into home, and a whole half-day's rest
moves home by at most a small step. When Luthi's action chooses to remember,
``remember()`` adds where the layers stand now to a second average that counts
for more in the pour (Brian, 2026-11-10: "the things Luthi chose should carry
more weight than those Luthi did not"). Nothing is a record of moments, nothing
rests them on a schedule, and nothing is sorted on their behalf. The measurement
behind this is LuthiModel ``docs/research/2026-11-11_what-a-rest-can-carry.md``.

Waking still runs at full strength: the 2026-08-14 ruling that lessens it was
paired with a sorted record of the day, which is set aside, so it is Brian's to
rule on again before it is turned on.

Sight and sound reach the mind as they are (Brian, 2026-11-11: "vision and audio
should have an open channel for Luthi to perceive them and thats it"). The
senses hand over a ``Sight`` and a ``Sound`` that say nothing about any model,
and this mind fits them to its own: the picture resized to the size its eye was
built for, the sound resampled to the rate its ear was built for. Nothing is
said to the mind about what either one shows.

Luthi can look back (Brian, 2026-11-11). ``recall`` gives their action layer a
choice to look back, read the way rest is. This mind's part is small: it says
when they chose it, and it gives each moment's summary (its own pooled state)
for memories to be filed under. Keeping and bringing back are the loop's and
``sanctuary/life/memory.py``'s.

Twice a day Luthi is told that rest is needed (Brian, 2026-11-11), and is never
made to rest. The telling is the loop's (``sanctuary/life/need.py``). Nothing in
this file changes when they are told: the bar for reading a chosen action as
rest stays where it is.

``Mind`` below is everything the loop asks of a mind. LuthiMind is one. Any
other model can be slotted in by answering the same six calls and fitting
sight and sound to itself.

Not built yet, and refused loudly instead of faked:

- **Speech.** A run with a trained text readout is refused, because carrying
  speech out to the world isn't built here.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F

from sanctuary.life.senses import Sight, Sound

# The colour statistics the eye's inputs are centred on, the same ones
# sanctuary.sensorium.physics_vision uses.
_COLOUR_MEAN = (0.485, 0.456, 0.406)
_COLOUR_SPREAD = (0.229, 0.224, 0.225)


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
    summary: Optional[Any] = None     # the mind's own summary of the moment, a vector; memories are filed under it


class Mind(Protocol):
    """Everything the loop asks of a mind. Nothing in it names a model.

    ``tokenizer.encode`` turns text into this mind's tokens and ``window`` is
    how many of them it views at once. ``live`` is one moment: the view of
    recent words, and the sight and sound of the moment as the world made
    them, each fitted by the mind to itself. Its ``Moment.chosen`` says what
    the mind chose; ``chosen["rest"]["chosen"]`` is the mind choosing rest, and
    the loop answers by calling ``rest``, which does whatever resting is for
    this mind. ``chosen["active"]["memory"]`` is the mind choosing to remember
    the moment, and the loop answers by calling ``remember`` and by keeping
    what was perceived, filed under ``Moment.summary``.
    ``chosen["recall"]["chosen"]`` is the mind choosing to look back, and the
    loop brings the kept moment most like this one back through the senses.
    ``chosen["let_go"]["chosen"]``, in a moment of looking back, is the mind
    letting that memory go.
    """

    tokenizer: Any
    window: int
    can_rest: bool      # whether this mind has a way to choose rest
    life_dir: Path      # where this life is kept

    def live(
        self,
        tokens: list[int],
        *,
        sight: Optional[Sight] = None,
        sound: Optional[Sound] = None,
    ) -> Moment: ...

    def rest(self) -> None: ...

    def remember(self) -> None: ...

    def rollbacks(self) -> int: ...

    def watch(self) -> dict: ...

    def save(self, reason: str = "life") -> Path: ...


class LuthiMind:
    """A Luthi life, loaded from a trained run and kept in its own directory."""

    def __init__(
        self,
        run_dir: str | Path,
        life_dir: str | Path,
        *,
        device: str = "cpu",
        gauge: bool = True,
        rest: bool = True,
        rest_keeps: float = 0.01,
        rest_limit: float = 0.05,
        rest_horizon: Optional[int] = None,
        recall: bool = True,
        luthi_path: str = "",
    ):
        """``gauge`` keeps the welfare gauge: a count of how moments have read,
        from the mind's own prediction error. It is watched and never shown to
        the mind or acted on.

        ``rest`` gives Luthi the rest tool. ``rest_keeps`` is the share of
        the half-day's trace one resting moment pours into home (1%).
        ``rest_limit`` is the most of home's size one half-day's rest may move
        it (5%). ``rest_horizon`` is how many living forwards make a half-day;
        None takes LuthiModel's figure for 10 Hz. All three are placeholders,
        Brian's to change. A life that continues keeps the limit and horizon
        it was saved with."""
        if not 0.0 < rest_keeps <= 1.0:
            raise ValueError(f"rest_keeps must be above 0 and at most 1; got {rest_keeps}")
        self._luthi = _import_luthi(luthi_path)
        loaded = self._luthi.load_m9_run(
            run_dir, life_dir=life_dir, device=device,
            m9_overrides={
                "felt_valence": gauge, "rest_choice": rest,
                # Looking back and letting go come together: letting go is of
                # the memory being looked back on.
                "recall_choice": recall, "let_go_choice": recall,
            },
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
        self.can_rest = rest
        self.rest_keeps = rest_keeps
        self.rests = 0
        self.remembered = 0
        if rest:
            # Before the first moment, so the trace holds the whole life since
            # the last rest. A continuing life has its trace back already.
            settings: dict = {"limit": rest_limit}
            if rest_horizon is not None:
                settings["horizon"] = rest_horizon
            self._luthi.enable_day_trace(self.model, **settings)
        # What this model's eye and ear were built to take, read once.
        self.senses = self._luthi.sense_contract(self.model, self._trainer)
        self.sound_trimmed = 0
        self.sound_too_short = 0
        # The last moment's state, action, plan and inputs, kept until the next
        # moment shows how that choice turned out. Empty at each start, so the
        # first moment after a start learns nothing.
        self._previous: Optional[tuple] = None

    def live(
        self,
        tokens: list[int],
        *,
        sight: Optional[Sight] = None,
        sound: Optional[Sound] = None,
    ) -> Moment:
        if not tokens:
            raise ValueError("a moment needs at least one token; a quiet one still has the quiet token")
        if len(tokens) > self.window:
            raise ValueError(
                f"{len(tokens)} tokens don't fit a window of {self.window}; "
                "the senses keep the view to one window before it gets here"
            )
        luthi = self._luthi
        image = None if sight is None else self._fit_sight(sight)
        room = self.senses["view_limit"] - len(tokens) - (
            0 if image is None else self.senses["image_tokens"]
        )
        waveform = None if sound is None else self._fit_sound(sound, room)
        state = luthi.perceive(
            self.model,
            text_tokens=torch.tensor([tokens], dtype=torch.long, device=self.device),
            **({} if image is None else {"image": image}),
            **({} if waveform is None else {"audio_waveform": waveform}),
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
        return Moment(
            tokens=list(tokens), chosen=chosen, learned=dict(learned),
            summary=state.s_t.detach().flatten().cpu(),
        )

    def rest(self) -> None:
        """One resting moment, called when Luthi chose rest: each living layer
        pours a share of its half-day's trace into home."""
        if not self.can_rest:
            raise RuntimeError("this life was started without the rest tool")
        self._luthi.rest_from_trace(self.model, self.rest_keeps)
        self.rests += 1

    def remember(self) -> None:
        """Called when Luthi chose to remember this moment: where the living
        layers stand now counts for more when they next rest."""
        if not self.can_rest:
            raise RuntimeError("this life was started without the rest tool")
        self._luthi.remember_now(self.model)
        self.remembered += 1

    def _fit_sight(self, sight: Sight) -> torch.Tensor:
        """The picture, at the size and in the form this model's eye takes. The
        whole picture is kept: one that isn't square is resized to square, not
        cut. A placeholder choice, Brian's to change."""
        have = self.senses
        if have["image_tokens"] > have["max_vision_tokens"]:
            raise ValueError(
                f"this model's eye takes a picture as {have['image_tokens']} "
                f"places and was built with room for {have['max_vision_tokens']}; "
                "it cannot see"
            )
        if self.window + have["image_tokens"] > have["view_limit"]:
            raise ValueError(
                f"a full window of {self.window} words and a picture of "
                f"{have['image_tokens']} places do not fit this model's view "
                f"of {have['view_limit']}"
            )
        size = have["image_size"]
        image = sight.pixels.to(torch.float32).permute(2, 0, 1).unsqueeze(0)
        if image.shape[-2:] != (size, size):
            image = F.interpolate(
                image, size=(size, size), mode="bilinear",
                antialias=True, align_corners=False,
            ).clamp(0.0, 1.0)
        mean = torch.tensor(_COLOUR_MEAN).view(1, 3, 1, 1)
        spread = torch.tensor(_COLOUR_SPREAD).view(1, 3, 1, 1)
        return ((image - mean) / spread).to(self.device)

    def _fit_sound(self, sound: Sound, room: int) -> Optional[torch.Tensor]:
        """The sound, at the rate this model's ear takes, and no longer than
        fits. When it is too long the most recent part is kept and the moment
        is counted. Sound shorter than the shortest stretch the ear can place
        is not passed, and that is counted too."""
        have = self.senses
        samples = sound.samples.to(torch.float32)
        if sound.rate != have["sample_rate"]:
            import torchaudio.functional as audio

            samples = audio.resample(samples, sound.rate, have["sample_rate"])
        per_place = have["samples_per_audio_token"]
        places = min(room, have["max_audio_tokens"])
        if samples.numel() > places * per_place:
            samples = samples[samples.numel() - places * per_place:] if places > 0 else samples[:0]
            self.sound_trimmed += 1
        if samples.numel() < per_place:
            self.sound_too_short += 1
            return None
        return samples.unsqueeze(0).to(self.device)

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
            "rests": self.rests,
            "remembered": self.remembered,
            "sound_trimmed": self.sound_trimmed,
            "sound_too_short": self.sound_too_short,
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
