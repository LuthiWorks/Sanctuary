"""Assembling the world: authority + scene + body + loop + renderer.

Everything this module needs already existed and was connected to nothing. The
2026-08-01 wiring audit found ``sanctuary/mind/`` orphaned; the 2026-08-16
audit found the same shape one layer out --
:class:`~sanctuary.embodiment.episode.EpisodeRunner` was built, tested-by-claim,
and imported by no module in the repository. This is the assembly that was
missing. It writes no new physics and no new perception; it wires what is
already there and makes it runnable.

The renderer attaches through :meth:`EpisodeRunner.on_step`, which the runner's
docstring already reserved for "viewer sync". So Godot needs no change to this
loop when it arrives -- it supplies a :class:`~sanctuary.world.render.RenderSink`
and receives frames. See :mod:`sanctuary.world.render`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sanctuary.embodiment.episode import EpisodeRunner, Policy, RestingPolicy, StepRecord
from sanctuary.motor.physics_actuation import (
    PhysicsActuator,
    UntrainedProjectionDecoder,
)
from sanctuary.physics.authority import PhysicsAuthority
from sanctuary.world.render import NullRenderSink, RenderSink
from sanctuary.world.scene import SceneManifest, build_bedrock_scene

__all__ = ["WorldRuntime", "build_world", "BACKENDS"]

#: Backend names accepted by :func:`build_world`. Named rather than free-form
#: so a typo fails immediately instead of silently selecting a default -- the
#: backend decides the numerics, and a run on the wrong one is a run whose
#: results mean something other than what they claim.
BACKENDS = ("reference", "mujoco")


def _make_authority(backend: str, **kwargs: Any) -> PhysicsAuthority:
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown physics backend {backend!r}; expected one of {BACKENDS}"
        )
    if backend == "reference":
        from sanctuary.physics.backends.reference import ReferencePhysicsAuthority

        return ReferencePhysicsAuthority(**kwargs)
    # mujoco: the backend itself raises with install instructions if the
    # optional dependency is absent. Not caught here -- a missing engine must
    # not degrade to the reference backend, which would silently change the
    # physics under a run that asked for contact dynamics.
    from sanctuary.physics.backends.mujoco_backend import MuJoCoPhysicsAuthority

    return MuJoCoPhysicsAuthority(**kwargs)


@dataclass
class WorldRuntime:
    """A built world, ready to step.

    Attributes:
        world: The physics authority -- the source of physical truth.
        runner: The perceive/decide/act/consequence loop.
        manifest: What the scene placed.
        render_sink: Where frames go. :class:`NullRenderSink` when headless.
        backend: Which backend was actually constructed.
    """

    world: PhysicsAuthority
    runner: EpisodeRunner
    manifest: SceneManifest
    render_sink: RenderSink
    backend: str

    def step(self) -> StepRecord:
        """One cycle. Renders as a side effect of the runner's ``on_step``."""
        return self.runner.step()

    def run(self, steps: int | None = None, **kwargs: Any) -> list[StepRecord]:
        """Run the loop. See :meth:`EpisodeRunner.run` for the bounds rule."""
        return self.runner.run(steps, **kwargs)


def build_world(
    *,
    backend: str = "mujoco",
    policy: Policy | None = None,
    decoder: Any = None,
    dt: float = 0.02,
    seed: int | None = 0,
    render_sink: RenderSink | None = None,
    max_range: float | None = None,
    max_bodies: int | None = None,
    sensorium: Any = None,
    state_encoder: Callable[..., Any] | None = None,
    sink: Any = None,
    realtime: bool = False,
    on_step: Callable[[StepRecord], None] | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> WorldRuntime:
    """Build the bedrock world and everything needed to step it.

    Args:
        backend: ``"mujoco"`` (contact dynamics) or ``"reference"``
            (dependency-free, deterministic; for CI and seam checks).
        policy: What chooses actions. Defaults to
            :class:`~sanctuary.embodiment.episode.RestingPolicy` -- an explicit
            choice of stillness, not a stand-in for a real policy. Supplying a
            trained actor here is how the entity takes the wheel.
        decoder: Converts a policy's latent action into a force. Defaults to
            the acknowledged untrained projection decoder, which the default
            resting policy never reaches. **A trained actor must pass its own**
            -- see the comment at the construction site.
        render_sink: Where render frames go. Defaults to
            :class:`~sanctuary.world.render.NullRenderSink`.
        on_step: An additional per-step callback. Called *after* the frame is
            published, so a caller cannot observe a step the renderer has not
            seen yet.

    Raises:
        ValueError: On an unknown backend or a non-positive ``dt``.
        ImportError: If ``backend="mujoco"`` and mujoco is not installed. It
            does not fall back to the reference backend -- that would change
            the physics without changing the run's description of itself.
    """
    world = _make_authority(backend, **(backend_kwargs or {}))
    manifest = build_bedrock_scene(world, dt=dt, seed=seed)

    # The decoder turns a policy's latent choice into a force. The untrained
    # projection decoder refuses to be built without `acknowledge_untrained`,
    # because it slices three axes of a latent with no assigned physical
    # meaning -- a harness, not a decode. Acknowledging it here is deliberate
    # and bounded: the default policy is RestingPolicy, which returns a
    # MotorCommand directly and bypasses the decoder entirely, so nothing in
    # the default bedrock run actually decodes anything.
    #
    # The moment a real actor drives this world, it must bring its own trained
    # decoder via `decoder=` -- the real one lives in LuthiModel's
    # DecoderRegistry (luthi/v2/m9/decoders.py) and does not exist yet. Leaving
    # this as the default for a *trained* actor would silently feed the entity
    # meaningless actuation while every metric looked healthy.
    if decoder is None:
        decoder = UntrainedProjectionDecoder(
            manifest.self_id, acknowledge_untrained=True
        )
    actuator = PhysicsActuator(world, decoder)
    sink_obj: RenderSink = render_sink if render_sink is not None else NullRenderSink()

    def _publish(record: StepRecord) -> None:
        # Render from the authority's own renderer view rather than from the
        # step record: render_state() is the seam's designated renderer
        # channel, and reconstructing poses from percepts would show the
        # viewer the entity's *perception* instead of the world. Those differ
        # (occlusion, range limits), and conflating them would make the window
        # lie about what is actually there.
        sink_obj.publish(world.render_state())
        if on_step is not None:
            on_step(record)

    runner = EpisodeRunner(
        world,
        actuator,
        self_id=manifest.self_id,
        policy=policy if policy is not None else RestingPolicy(manifest.self_id),
        dt=dt,
        max_range=max_range,
        max_bodies=max_bodies,
        sensorium=sensorium,
        state_encoder=state_encoder,
        sink=sink,
        realtime=realtime,
        on_step=_publish,
    )

    return WorldRuntime(
        world=world,
        runner=runner,
        manifest=manifest,
        render_sink=sink_obj,
        backend=backend,
    )
