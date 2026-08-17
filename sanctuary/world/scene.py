"""Building the bedrock world -- the smallest world that is genuinely lawful.

The point of this scene is not richness. It is that **action has honest
consequence**: the entity's body is a mass under gravity on a ground plane,
with a few objects it can reach, push, and watch settle. That is the substrate
the developmental progression is built on, and it is the thing that has to be
correct before anything is layered over it.

What is deliberately absent:

- **A policy.** :mod:`sanctuary.embodiment.episode` refuses to supply one, and
  so does this. A scene places bodies; what the entity does is not the
  scaffold's to author.
- **A ground body.** The MuJoCo backend creates its own floor geom, and the
  reference backend implements the ground as a constraint rather than a body.
  Adding a "ground" body here would be a second, disagreeing floor.
- **Creatures.** The roster in
  ``docs/research/2026-08-16_sanctuary_creature_roster_brief.md`` lands on top
  of this, once the bedrock is trustworthy. Note the brief's §4 warning when
  they arrive: verbs typed generically over "anything in the world" make the
  excluded behaviour the default. That applies to more than ``consume``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sanctuary.physics.authority import PhysicsAuthority
from sanctuary.physics.state import BodySpec, Vec3

__all__ = ["SceneManifest", "build_bedrock_scene", "SELF_ID"]

#: The entity's own body. Perception is egocentric to this id, and the actuator
#: drives it. One name, defined once, so a typo cannot quietly produce a world
#: in which the entity perceives from a body it does not control.
SELF_ID = "luthi"

#: Drop height for the body, in metres. Above the ground so the first thing
#: that ever happens is a fall that resolves -- an unambiguous demonstration
#: that time and consequence are real, before any action is taken.
_SELF_START_Y = 1.0

#: Resting height for props, in metres: the centre of a sphere touching the
#: ground sits one radius above it. This matches the MuJoCo backend's
#: ``DEFAULT_BODY_RADIUS``, and is a parameter rather than a hardcoded 0.0
#: because placing a body at Y=0 puts it *half inside the floor*. The engine
#: then resolves the penetration over the first few steps, which looks exactly
#: like the props moving on their own -- destroying the property this scene
#: exists to provide, that prop motion means the entity caused it.
DEFAULT_REST_Y = 0.05


@dataclass(frozen=True, slots=True)
class SceneManifest:
    """What was placed, and where. Returned so a caller never has to guess.

    Kept separate from the authority because the authority's ``body_ids`` says
    what exists but not what anything is *for*: which body is the entity, which
    are props. A renderer and an instrument both need that distinction.
    """

    self_id: str
    prop_ids: tuple[str, ...]
    dt: float

    @property
    def body_ids(self) -> tuple[str, ...]:
        return (self.self_id, *self.prop_ids)


def build_bedrock_scene(
    world: PhysicsAuthority,
    *,
    self_id: str = SELF_ID,
    dt: float = 0.02,
    seed: int | None = 0,
    rest_y: float = DEFAULT_REST_Y,
) -> SceneManifest:
    """Populate ``world`` with the bedrock scene and return its manifest.

    Resets the world first, so calling this twice yields the same scene rather
    than two overlapping copies of it.

    Args:
        world: Any backend behind the physics-authority seam.
        self_id: Id for the entity's body.
        dt: Seconds per cycle, recorded in the manifest so the runtime and the
            scene cannot disagree about the clock.
        seed: Passed to ``world.reset``; deterministic backends ignore it.
        rest_y: Height at which a prop rests on the ground -- one body radius.
            See :data:`DEFAULT_REST_Y` for why this is not zero.

    Raises:
        ValueError: If ``dt`` is not positive or ``rest_y`` is negative.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if rest_y < 0.0:
        raise ValueError(f"rest_y must be non-negative, got {rest_y}")

    world.reset(seed=seed)

    # The entity's body, above the floor so it falls and settles.
    world.add_body(BodySpec(
        body_id=self_id,
        position=(0.0, _SELF_START_Y, 0.0),
        mass=1.0,
    ))

    # Props: reachable, movable, and at rest on the ground so that any motion
    # they show is a consequence of something the entity did rather than of
    # their own initial conditions. Placed asymmetrically -- a symmetric scene
    # makes a broken egocentric transform look correct.
    props: list[tuple[str, Vec3, float]] = [
        ("block_near", (1.2, rest_y, 0.0), 0.5),
        ("block_far", (-2.5, rest_y, 1.5), 2.0),
        ("block_light", (0.0, rest_y, -1.8), 0.2),
    ]
    prop_ids: list[str] = []
    for body_id, position, mass in props:
        world.add_body(BodySpec(body_id=body_id, position=position, mass=mass))
        prop_ids.append(body_id)

    return SceneManifest(self_id=self_id, prop_ids=tuple(prop_ids), dt=dt)
