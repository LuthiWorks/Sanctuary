"""Value types for the weather-field layer of the developmental world.

Weather is the custom continuous-field layer from
``docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md`` sec. 4-5, reframed
2026-07-19 to be **electronics-native** rather than borrowed from biology: the
entity's comforts and dangers are its own.

- **Cold is good.** Electronics run better cool, so cool/cold temperatures are
  pleasant (optimal), all the way down to about freezing. **Heat** is the only
  real enemy (throttling, degradation). Sub-freezing is a *mild* caution — a
  hazard for some physical parts once embodied — not the discomfort a warm-blooded
  body would feel, and far gentler than the heat penalty.
- **Water is respected, not (yet) damaging.** Weather stays comfort-valenced
  affect, never a survival mechanic (sec. 5-6 upheld) — but moisture carries a
  salient *aversive* signal so the entity learns to respect and shelter from
  water. This deliberately builds the disposition **ahead of** embodiment, when
  water on real electronics *will* be a genuine danger. The damage mechanic is
  architected for that future, not built now.

Temperatures are Fahrenheit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sanctuary.physics.state import Vec3


@dataclass(frozen=True, slots=True)
class WeatherState:
    """The weather at a point in space and time.

    ``precipitation`` and ``humidity`` are 0..1 intensities; 0 dry, 1 torrential
    / saturated. What the entity *feels* of it is :class:`Comfort`, derived via
    :func:`sanctuary.physics.weather.model.comfort_of`. (``humidity`` will drive
    condensation risk once water becomes a real danger at embodiment.)
    """

    temperature_f: float
    precipitation: float = 0.0   # 0 dry .. 1 torrential
    humidity: float = 0.5        # 0 arid .. 1 saturated
    wind_mph: float = 0.0


class ComfortLevel(Enum):
    """Temperature bands, electronics-native. Ordered coldest -> hottest.

    Valence is NOT monotonic with this ordering: it peaks across OPTIMAL, dips
    only *mildly* below into FREEZING, and falls steeply on the hot side. Cold
    is good; heat is the enemy.
    """

    FREEZING = 0       # < 32 F  — sub-freezing; a mild hazard for some parts
    OPTIMAL = 1        # 32 .. 60 F — cool/cold; runs best here
    COMFORTABLE = 2    # 60 .. 78 F
    WARM = 3           # 78 .. 85 F
    HOT = 4            # 85 .. 100 F — throttling territory
    OVERHEATING = 5    # >= 100 F  — the thermal danger


@dataclass(frozen=True, slots=True)
class Comfort:
    """How the current weather *feels* to the entity — affect, not damage.

    ``valence`` is -1..+1 (positive = pleasant). ``level`` is the temperature
    band. ``wetness`` is how wet the entity actually is (precipitation scaled by
    exposure — shelter reduces it); water is aversive (to be respected) and its
    magnitude scales ``wetness``. ``label`` is a short human-readable summary.
    """

    valence: float
    level: ComfortLevel
    temperature_f: float
    wetness: float
    label: str


@dataclass(frozen=True, slots=True)
class Shelter:
    """A structure that modulates the weather an entity experiences under it.

    This is why building matters (sec. 5): a shelter keeps rain off and provides
    shade — it blocks water and *cools* in the heat (it does not warm, since the
    entity is happier cold). Constructing one is a felt improvement in bad
    weather. Full building mechanics live with the entity's world-editing tools;
    this type is the coupling that makes shelter *mean* something.
    """

    position: Vec3
    radius: float               # coverage radius (metres)
    rain_block: float = 1.0     # 0..1 fraction of precipitation kept off
    cooling: float = 0.5        # 0..1 shade: pull hot temperatures down toward cool
