"""Actuation adapter: ``ActionSelection`` -> :meth:`PhysicsAuthority.apply_force`.

The motor half of the training ground. ``sanctuary/sensorium/physics_percepts.py``
turns world state into perception; this turns a chosen action into a force that
changes the world. Together they close the action -> consequence loop that
``luthi.sanctuary_interface.observe_transition`` learns from, entirely headless
-- no renderer involved.

Why the loop has to close for JEPA specifically: the objective predicts the
latent of what has not yet been seen. If the next observation is unrelated to
what the body just did, the only learnable structure is the world's own
autonomous dynamics. Prediction error becomes informative about *the entity's
own agency* only when its actions are among the causes of what it next
perceives.

The decode is not ours to invent
--------------------------------

Under M9's action-space (c), the action ``a_t`` **is the predicted next latent**
``s_hat_{t+1}`` -- a ``[D]`` vector in the model's latent space, not a force in
newtons. LuthiModel maps latents to modality outputs with *learned* decoders
(``luthi/v2/m9/decoders.py``: text, attention, memory), each pairing a
structured output with an intensity scalar and a ``re_encode`` inverse that
feeds the P4 cycle-consistency signal.

**There is no motor decoder in that registry.** So there is no principled
latent -> force map available today, and Sanctuary must not fabricate one. This
module therefore splits the problem:

* :class:`MotorDecoder` -- the protocol for "latent -> motor command". Owned by
  LuthiModel when a trained ``MotorDecoder`` joins ``DecoderRegistry``.
* :class:`PhysicsActuator` -- "motor command -> lawful force on a body". Owned
  by Sanctuary, correct today, independent of how the decode is learned.

That boundary is the same discipline ``sanctuary/core/luthi_model.py:1440``
already applies when it refuses to route adapter-authored confidence into
``CognitiveOutput``: the scaffold does not author signal the substrate did not
produce.

:class:`UntrainedProjectionDecoder` exists so the loop can be exercised
end-to-end before that decoder is trained. It must be constructed with
``acknowledge_untrained=True``; there is no way to get one by accident.

Rest is an action, not a failure
--------------------------------

M9 treats rest as *computed*, not constant: ``a_rest(s_t)`` is the
identity-continuation action for the current state, and decoder activity at or
below the rest reference is stasis (``luthi/v2/m9/rest_action.py``). So a
command whose intensity falls below :attr:`PhysicsActuator.rest_threshold`
applies **no force** and is recorded as ``at_rest=True`` -- deliberate stillness,
distinguishable in the logs from a decode that failed or a body that never
moved. "Quiet because at rest" and "quiet because broken" must never look alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

from sanctuary.core.schema import Percept
from sanctuary.physics.authority import PhysicsAuthority
from sanctuary.physics.state import Vec3

__all__ = [
    "MotorCommand",
    "AppliedAction",
    "MotorDecoder",
    "UntrainedProjectionDecoder",
    "PhysicsActuator",
    "DEFAULT_MAX_FORCE",
    "DEFAULT_REST_THRESHOLD",
]

#: Newtons. A ceiling, not a calibration -- it exists so an untrained or
#: diverged decoder cannot launch a body out of the world and destroy the
#: episode. Set it from the body's mass and the world's scale.
DEFAULT_MAX_FORCE = 50.0

#: Intensity at or below which a command is treated as rest (no force).
DEFAULT_REST_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Seam types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MotorCommand:
    """What a decoder produces: a force request against one body.

    Mirrors the decoder contract in LuthiModel -- a structured output plus an
    intensity scalar in ``[0, 1]`` that carries the rest/emission boundary.

    ``force`` is in newtons, expressed in world axes. ``intensity`` is
    *separate* from the force magnitude on purpose: the decoder may want to say
    "push hard, but I am barely committed to acting", and the rest gate reads
    commitment rather than magnitude.
    """

    body_id: str
    force: Vec3
    intensity: float = 1.0


@dataclass(frozen=True, slots=True)
class AppliedAction:
    """The record of what actually reached the world.

    Kept distinct from :class:`MotorCommand` so the requested and the applied
    can be compared. Silent divergence between them is exactly the class of
    defect this project keeps finding, so every way the actuator can alter a
    request is recorded on this object rather than logged and forgotten.
    """

    command: MotorCommand
    applied_force: Vec3
    at_rest: bool = False
    clamped: bool = False
    requested_magnitude: float = 0.0

    @property
    def applied_magnitude(self) -> float:
        return _norm(self.applied_force)

    def to_percept(self, *, step: int | None = None) -> Percept:
        """Motor feedback, so the entity perceives having acted.

        Matches the ``[motor:...]`` convention of
        :meth:`sanctuary.sensorium.Sensorium.perceive_own_action`. The
        sensorimotor loop is a design principle here: actions are not
        fire-and-forget, and rest is reported as an action taken, not as
        absence.
        """
        if self.at_rest:
            detail = f"held still (intensity {self.command.intensity:.2f} at or below rest)"
        else:
            detail = (
                f"pushed {_fmt(self.applied_force)} N "
                f"({self.applied_magnitude:.2f} N"
                + (f", clamped from {self.requested_magnitude:.2f}" if self.clamped else "")
                + ")"
            )

        source = f"motor:physics:body={self.command.body_id}"
        if step is not None:
            source = f"{source}:step={step}"

        return Percept(
            modality="proprioceptive",
            content=f"[motor:force] {detail}",
            source=source,
            embedding_summary=(
                "self-action: rest" if self.at_rest
                else f"self-action: force {self.applied_magnitude:.2f} N"
            ),
        )


@runtime_checkable
class MotorDecoder(Protocol):
    """Maps a chosen action to a motor command.

    Structurally typed against ``luthi.sanctuary_interface.ActionSelection``
    rather than importing it: ``luthi`` is an optional dependency of Sanctuary
    (``core/luthi_model.py`` imports it lazily), so a hard import here would
    make the physics loop unusable in an environment that has no model
    installed -- which is exactly the environment the world is developed in.

    The implementation that matters will live in LuthiModel as a trained
    decoder alongside text/attention/memory, with the ``re_encode`` inverse
    that lets P4 cycle-consistency see whether the motor channel is telling
    the truth about the latent it decoded.
    """

    def decode(self, selection: Any) -> MotorCommand:
        ...


# ---------------------------------------------------------------------------
# Reference decoder -- runnable, deliberately not meaningful
# ---------------------------------------------------------------------------


class UntrainedProjectionDecoder:
    """Reads the first three components of the action latent as a force.

    **This decoder is arbitrary and produces no meaningful behaviour.** The
    action latent's axes carry no assigned physical meaning; slicing three of
    them is a convention chosen so the loop can be exercised, not a decode. A
    body driven by this will move, and its movement will mean nothing.

    It exists because the plumbing needs to be testable before LuthiModel has a
    trained motor decoder, and a loop that cannot be run cannot be debugged. It
    must be constructed with ``acknowledge_untrained=True`` so that no caller
    ends up with one while believing they have a learned decode -- the same
    fail-loud posture the physics backend takes toward unknown body ids.

    Replace it with the real decoder; do not tune it.
    """

    def __init__(
        self,
        body_id: str,
        *,
        acknowledge_untrained: bool = False,
        newtons_per_unit: float = 1.0,
        reference_magnitude: float = 1.0,
    ) -> None:
        if not acknowledge_untrained:
            raise ValueError(
                "UntrainedProjectionDecoder produces arbitrary, meaningless "
                "actuation: it slices three axes of a latent that has no "
                "assigned physical meaning. It is a harness for exercising the "
                "loop, not a decode. Pass acknowledge_untrained=True if that is "
                "what you want. The real implementation is a trained motor "
                "decoder in LuthiModel's DecoderRegistry "
                "(luthi/v2/m9/decoders.py), which does not exist yet."
            )
        if reference_magnitude <= 0.0:
            raise ValueError(
                f"reference_magnitude must be positive, got {reference_magnitude}"
            )

        self.body_id = body_id
        self.newtons_per_unit = newtons_per_unit
        self.reference_magnitude = reference_magnitude

    def decode(self, selection: Any) -> MotorCommand:
        raw = _action_floats(selection)
        if len(raw) < 3:
            raise ValueError(
                f"Action vector needs at least 3 components to project onto a "
                f"force; got {len(raw)}."
            )

        head = (raw[0], raw[1], raw[2])
        magnitude = _norm(head)
        force = tuple(component * self.newtons_per_unit for component in head)

        return MotorCommand(
            body_id=self.body_id,
            force=force,  # type: ignore[arg-type]
            intensity=min(1.0, magnitude / self.reference_magnitude),
        )


# ---------------------------------------------------------------------------
# The actuator
# ---------------------------------------------------------------------------


class PhysicsActuator:
    """Applies decoded motor commands to a :class:`PhysicsAuthority`.

    This half is correct today and stays correct when the decoder is replaced:
    given a force request, apply it lawfully, bounded, and observably.

    Note the actuator does **not** call :meth:`PhysicsAuthority.step`. Forces
    accumulate until the caller steps the world, so several actuators (or a
    body and its environment) can contribute to one step. Deciding when time
    advances belongs to the cognitive loop, not to an effector.
    """

    def __init__(
        self,
        authority: PhysicsAuthority,
        decoder: MotorDecoder,
        *,
        max_force: float = DEFAULT_MAX_FORCE,
        rest_threshold: float = DEFAULT_REST_THRESHOLD,
    ) -> None:
        if max_force <= 0.0:
            raise ValueError(f"max_force must be positive, got {max_force}")
        if not 0.0 <= rest_threshold <= 1.0:
            raise ValueError(
                f"rest_threshold must be in [0, 1], got {rest_threshold}"
            )

        self.authority = authority
        self.decoder = decoder
        self.max_force = max_force
        self.rest_threshold = rest_threshold

    def apply(self, selection: Any) -> AppliedAction:
        """Decode a selection and apply the resulting force.

        Args:
            selection: An ``ActionSelection``-shaped object (anything carrying
                an ``action`` attribute), or a raw action vector.

        Returns:
            The :class:`AppliedAction` record, including whether the command
            was treated as rest and whether its force was clamped.

        Raises:
            ValueError: if the decoded force is not finite.
            KeyError: if the command names a body the world does not have --
                propagated from the authority rather than swallowed, because a
                motor command addressed to a nonexistent body is a wiring
                defect, and an action that silently does nothing is the exact
                failure this codebase keeps paying for.
        """
        command = self.decoder.decode(selection)
        return self.apply_command(command)

    def apply_command(self, command: MotorCommand) -> AppliedAction:
        """Apply an already-decoded command. See :meth:`apply`."""
        force = tuple(float(component) for component in command.force)
        if len(force) != 3:
            raise ValueError(
                f"Motor command force must be a 3-vector, got {len(force)} components"
            )
        if not all(math.isfinite(component) for component in force):
            raise ValueError(
                f"Motor command force is not finite: {command.force!r}. A "
                f"diverged decoder must fail here rather than corrupt the "
                f"world state it is training against."
            )

        requested = _norm(force)  # type: ignore[arg-type]

        # Rest gate. Checked before clamping so a rest command is recorded as
        # rest, never as a clamped zero.
        if command.intensity <= self.rest_threshold:
            return AppliedAction(
                command=command,
                applied_force=(0.0, 0.0, 0.0),
                at_rest=True,
                requested_magnitude=requested,
            )

        clamped = requested > self.max_force
        if clamped:
            scale = self.max_force / requested
            force = (force[0] * scale, force[1] * scale, force[2] * scale)

        self.authority.apply_force(command.body_id, force)  # type: ignore[arg-type]

        return AppliedAction(
            command=command,
            applied_force=force,  # type: ignore[arg-type]
            at_rest=False,
            clamped=clamped,
            requested_magnitude=requested,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(v: Iterable[float]) -> float:
    return math.sqrt(sum(component * component for component in v))


def _fmt(v: Vec3) -> str:
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


def _action_floats(selection: Any) -> tuple[float, ...]:
    """Flatten an ActionSelection (or a bare vector) to floats.

    Duck-typed rather than importing torch or luthi: this keeps the loop
    testable with plain tuples, and Sanctuary can develop the world without a
    model installed.

    Batched actions are accepted only at batch size 1. A larger batch means one
    actuator is being handed several bodies' worth of intent, and picking a row
    would be a guess.
    """
    action = getattr(selection, "action", selection)

    if hasattr(action, "detach"):  # torch.Tensor
        action = action.detach().cpu()
    if hasattr(action, "tolist"):
        action = action.tolist()

    if not isinstance(action, (list, tuple)):
        raise TypeError(
            f"Cannot read an action vector from {type(action).__name__}; "
            f"expected a tensor or a sequence of floats."
        )

    # Unwrap a [1, D] batch; refuse [B>1, D].
    if action and isinstance(action[0], (list, tuple)):
        if len(action) != 1:
            raise ValueError(
                f"Batched action with batch size {len(action)}; a single "
                f"actuator drives one body, so which row to apply would be a "
                f"guess. Pass one row per actuator."
            )
        action = action[0]

    return tuple(float(component) for component in action)
