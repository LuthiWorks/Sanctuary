"""Weather -> effect coupling: comfort (affect), mobility (physics), shelter.

Pure functions over :mod:`sanctuary.physics.weather.state` value types. No
survival/damage — weather is felt (comfort valence) and it makes the ground
slick (mobility), and shelter can improve both.

Thresholds are Brian's (2026-07-16), in Fahrenheit:
    70 comfortable  ·  85 possibly too warm  ·  32 too cold  ·  rain slick.
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
IDEAL_F = 70.0
COOL_F = 60.0        # below this, "cool"
WARM_F = 78.0        # above this, "warm"
TOO_COLD_F = 32.0    # at/below, "too cold"
COLD_F = 45.0        # below this, "cold"
TOO_WARM_F = 85.0    # at/above, "too warm"

# Where valence bottoms out (-1). Asymmetric so the named thresholds land where
# Brian described: 85 reads only mildly bad ("possibly too warm"), 32 clearly bad.
_HOT_EXTREME_F = 95.0
_COLD_EXTREME_F = 20.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _level(temp_f: float) -> ComfortLevel:
    if temp_f <= TOO_COLD_F:
        return ComfortLevel.TOO_COLD
    if temp_f < COLD_F:
        return ComfortLevel.COLD
    if temp_f < COOL_F:
        return ComfortLevel.COOL
    if temp_f < WARM_F:
        return ComfortLevel.COMFORTABLE
    if temp_f < TOO_WARM_F:
        return ComfortLevel.WARM
    return ComfortLevel.TOO_WARM


def _temperature_valence(temp_f: float) -> float:
    """+1 at the ideal, falling to -1 at the (asymmetric) extremes."""
    if temp_f >= IDEAL_F:
        frac = (temp_f - IDEAL_F) / (_HOT_EXTREME_F - IDEAL_F)
    else:
        frac = (IDEAL_F - temp_f) / (IDEAL_F - _COLD_EXTREME_F)
    return _clamp(1.0 - 2.0 * frac, -1.0, 1.0)


def _wet_effect(temp_f: float, precipitation: float, exposure: float) -> tuple[float, float]:
    """Return (valence_delta, wetness). Light warm moisture is refreshing (+);
    torrential or cold rain is unpleasant (-). Shelter (low exposure) reduces
    how wet the entity actually gets."""
    wetness = _clamp(precipitation, 0.0, 1.0) * _clamp(exposure, 0.0, 1.0)
    if wetness <= 1e-9:
        return 0.0, 0.0
    # Refreshing: light rain when it isn't cold. (Brian: "moisture is refreshing.")
    if precipitation <= 0.3 and temp_f >= COOL_F:
        return 0.15 * min(wetness / 0.3, 1.0), wetness
    # Otherwise unpleasant, and worse when cold.
    penalty = 0.5 * wetness
    if temp_f < COOL_F:
        penalty += 0.3 * wetness
    return -penalty, wetness


def _label(level: ComfortLevel, wetness: float, wet_delta: float) -> str:
    base = {
        ComfortLevel.TOO_COLD: "too cold",
        ComfortLevel.COLD: "cold",
        ComfortLevel.COOL: "cool",
        ComfortLevel.COMFORTABLE: "comfortable",
        ComfortLevel.WARM: "warm",
        ComfortLevel.TOO_WARM: "too warm",
    }[level]
    if wetness <= 1e-9:
        return base
    if wet_delta > 0.0:
        return f"{base}, refreshingly damp"
    return f"{base} and drenched" if wetness >= 0.6 else f"{base} and wet"


def comfort_of(weather: WeatherState, *, exposure: float = 1.0) -> Comfort:
    """How ``weather`` feels to an entity at the given ``exposure`` (1 fully
    exposed, 0 fully sheltered). Affect only — never damage."""
    temp = weather.temperature_f
    level = _level(temp)
    wet_delta, wetness = _wet_effect(temp, weather.precipitation, exposure)
    valence = _clamp(_temperature_valence(temp) + wet_delta, -1.0, 1.0)
    return Comfort(
        valence=valence,
        level=level,
        temperature_f=temp,
        wetness=wetness,
        label=_label(level, wetness, wet_delta),
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

    A covering shelter blocks rain and moderates temperature toward the ideal —
    which is what makes building a shelter a felt improvement in bad weather. The
    strongest covering shelter wins per property.
    """
    rain_block = 0.0
    temp_mod = 0.0
    for s in shelters:
        if _within(position, s.position, s.radius):
            rain_block = max(rain_block, _clamp(s.rain_block, 0.0, 1.0))
            temp_mod = max(temp_mod, _clamp(s.temp_moderation, 0.0, 1.0))

    new_temp = weather.temperature_f + (IDEAL_F - weather.temperature_f) * temp_mod
    new_precip = weather.precipitation * (1.0 - rain_block)
    exposure = 1.0 - rain_block
    return replace(weather, temperature_f=new_temp, precipitation=new_precip), exposure
