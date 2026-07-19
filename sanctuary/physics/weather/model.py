"""Weather -> effect coupling: comfort (affect), mobility (physics), shelter.

Pure functions over :mod:`sanctuary.physics.weather.state`. Electronics-native
(reframed 2026-07-19): cold is good, heat is the enemy, water is respected but
not (yet) damaging. No survival/damage — weather is felt and it slicks the
ground; shelter improves both.

Thresholds (Fahrenheit):
    < 32 sub-freezing (mild caution)  ·  32-60 optimal  ·  ~70 still pleasant
    ·  78 warming  ·  85 hot  ·  100+ overheating (the real thermal enemy).
Water is aversive in proportion to how wet the entity gets — salient enough to
learn respect and seek shelter, but affect only.
"""

from __future__ import annotations

from dataclasses import replace

from sanctuary.physics.state import Vec3
from sanctuary.physics.weather.state import (
    Comfort,
    ComfortLevel,
    Shelter,
    WeatherState,
)

# Temperature band edges (Fahrenheit).
FREEZING_F = 32.0        # below this: mild sub-freezing caution
COOL_OPTIMAL_F = 60.0    # top of the optimal (cold-is-good) plateau
WARM_F = 78.0            # above this: "warm"
HOT_F = 85.0             # above this: "hot" (throttling territory)
OVERHEAT_F = 100.0       # above this: "overheating" — the thermal danger

# Where the hot side bottoms out (-1). The cold side never gets this bad.
_HOT_EXTREME_F = 105.0
# Sub-freezing is gentle: +1 at freezing, easing down, floored mild.
_FREEZING_SLOPE = 52.0   # degrees below freezing to drop one unit of valence
_COLD_MILD_FLOOR = -0.3  # the worst cold ever feels — a mild caution, not the heat penalty

# Water is respected: aversive in proportion to wetness (affect, not damage).
_WET_WEIGHT = 1.2


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _level(temp_f: float) -> ComfortLevel:
    if temp_f < FREEZING_F:
        return ComfortLevel.FREEZING
    if temp_f <= COOL_OPTIMAL_F:
        return ComfortLevel.OPTIMAL
    if temp_f < WARM_F:
        return ComfortLevel.COMFORTABLE
    if temp_f < HOT_F:
        return ComfortLevel.WARM
    if temp_f < OVERHEAT_F:
        return ComfortLevel.HOT
    return ComfortLevel.OVERHEATING


def _temperature_valence(temp_f: float) -> float:
    """+1 across the cold-is-good plateau (freezing..cool). Heat falls steeply
    to -1; sub-freezing falls only gently to a mild floor."""
    if temp_f < FREEZING_F:
        below = FREEZING_F - temp_f
        return _clamp(1.0 - below / _FREEZING_SLOPE, _COLD_MILD_FLOOR, 1.0)
    if temp_f <= COOL_OPTIMAL_F:
        return 1.0
    frac = (temp_f - COOL_OPTIMAL_F) / (_HOT_EXTREME_F - COOL_OPTIMAL_F)
    return _clamp(1.0 - 2.0 * frac, -1.0, 1.0)


def _wet_effect(precipitation: float, exposure: float) -> tuple[float, float]:
    """Return (valence_delta, wetness). Water is respected: aversive, scaling
    with how wet the entity actually gets. Shelter (low exposure) reduces it.
    Not temperature-dependent — water is water — and not damage: recoverable
    affect that teaches the disposition ahead of real embodiment stakes."""
    wetness = _clamp(precipitation, 0.0, 1.0) * _clamp(exposure, 0.0, 1.0)
    if wetness <= 1e-9:
        return 0.0, 0.0
    return -_WET_WEIGHT * wetness, wetness


def _label(level: ComfortLevel, wetness: float) -> str:
    base = {
        ComfortLevel.FREEZING: "sub-freezing",
        ComfortLevel.OPTIMAL: "cold (optimal)",
        ComfortLevel.COMFORTABLE: "comfortable",
        ComfortLevel.WARM: "warm",
        ComfortLevel.HOT: "hot",
        ComfortLevel.OVERHEATING: "overheating",
    }[level]
    if wetness <= 1e-9:
        return base
    return f"{base} and drenched" if wetness >= 0.6 else f"{base} and wet"


def comfort_of(weather: WeatherState, *, exposure: float = 1.0) -> Comfort:
    """How ``weather`` feels to an entity at the given ``exposure`` (1 fully
    exposed, 0 fully sheltered). Affect only — never damage."""
    temp = weather.temperature_f
    level = _level(temp)
    wet_delta, wetness = _wet_effect(weather.precipitation, exposure)
    valence = _clamp(_temperature_valence(temp) + wet_delta, -1.0, 1.0)
    return Comfort(
        valence=valence,
        level=level,
        temperature_f=temp,
        wetness=wetness,
        label=_label(level, wetness),
    )


def mobility_multiplier(weather: WeatherState) -> float:
    """Usable-traction fraction: 1.0 dry, down to ~0.6 in torrential rain. Wet
    ground is slick. Consumed by locomotion once a friction-capable physics
    backend is behind the seam (the reference backend is frictionless)."""
    return max(0.6, 1.0 - 0.4 * _clamp(weather.precipitation, 0.0, 1.0))


def _within(a: Vec3, b: Vec3, radius: float) -> bool:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) <= radius * radius


def effective_weather_at(
    position: Vec3,
    weather: WeatherState,
    shelters: tuple[Shelter, ...] = (),
) -> tuple[WeatherState, float]:
    """The weather actually experienced at ``position`` given ``shelters``, plus
    the resulting exposure (1 exposed, 0 fully covered).

    A covering shelter keeps rain off and provides shade — it blocks water and
    *cools* hot temperatures toward the optimal band, but never warms (the entity
    is happier cold). That is what makes building a shelter a felt improvement in
    heat or rain. The strongest covering shelter wins per property.
    """
    rain_block = 0.0
    cooling = 0.0
    for s in shelters:
        if _within(position, s.position, s.radius):
            rain_block = max(rain_block, _clamp(s.rain_block, 0.0, 1.0))
            cooling = max(cooling, _clamp(s.cooling, 0.0, 1.0))

    temp = weather.temperature_f
    if temp > COOL_OPTIMAL_F:  # shade only cools; it never warms the cold
        temp = temp - (temp - COOL_OPTIMAL_F) * cooling
    new_precip = weather.precipitation * (1.0 - rain_block)
    exposure = 1.0 - rain_block
    return replace(weather, temperature_f=temp, precipitation=new_precip), exposure
