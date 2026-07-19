"""Value types for the weather-field layer of the developmental world.

Weather is the custom continuous-field layer from
``docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md`` sec. 4-5. It is not
rigid-body physics (that's the :mod:`sanctuary.physics` seam); it is temperature
and moisture the entity **feels** and predicts, coupling into physics only where
it should (moisture -> traction).

Key design commitment (sec. 5): weather is **comfort-valenced affect, never a
survival/lethal mechanic.** Warm is pleasant, hot is not; cool is tolerable,
cold is not; light moisture is refreshing, torrential rain is annoying. These
types carry that affect (:class:`Comfort`), not damage or hunger.

Temperatures are in Fahrenheit (Brian's thresholds: 70 comfortable, 85 possibly
too warm, 32 too cold).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sanctuary.physics.state import Vec3


@dataclass(frozen=True, slots=True)
class WeatherState:
    """The weather at a point in space and time.

    ``precipitation`` and ``humidity`` are 0..1 intensities; 0 dry, 1 torrential
    / saturated. This is ground-truth environmental state — what the entity
    *feels* of it is :class:`Comfort`, derived via
    :func:`sanctuary.physics.weather.model.comfort_of`.
    """

    temperature_f: float
    precipitation: float = 0.0   # 0 dry .. 1 torrential
    humidity: float = 0.5        # 0 arid .. 1 saturated
    wind_mph: float = 0.0


class ComfortLevel(Enum):
    """Ordered comfort bands keyed to Brian's named thresholds.

    The integer values are ordered cold -> hot so tests and UI can compare/sort.
    """

    TOO_COLD = 0      # <= 32 F
    COLD = 1          # 32 .. 45 F
    COOL = 2          # 45 .. 60 F
    COMFORTABLE = 3   # 60 .. 78 F  (peak ~70)
    WARM = 4          # 78 .. 85 F
    TOO_WARM = 5      # >= 85 F


@dataclass(frozen=True, slots=True)
class Comfort:
    """How the current weather *feels* to the entity — affect, not damage.

    ``valence`` is -1..+1 (positive = pleasant). ``level`` is the temperature
    band. ``wetness`` is how wet the entity actually is (precipitation scaled by
    exposure — shelter reduces it). ``label`` is a short human-readable summary.
    """

    valence: float
    level: ComfortLevel
    temperature_f: float
    wetness: float
    label: str


@dataclass(frozen=True, slots=True)
class Shelter:
    """A structure that modulates the weather an entity experiences under it.

    This is why building matters (sec. 5 discussion): a shelter blocks rain and
    moderates temperature toward comfortable, so constructing one is a real,
    felt improvement in bad weather. Full building mechanics live with the
    entity's world-editing tools; this type is the coupling that makes shelter
    *mean* something.
    """

    position: Vec3
    radius: float               # coverage radius (metres)
    rain_block: float = 1.0     # 0..1 fraction of precipitation kept off
    temp_moderation: float = 0.5  # 0..1 pull of local temperature toward ideal
