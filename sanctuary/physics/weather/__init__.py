"""The weather-field layer of the developmental world.

Custom continuous fields (temperature, moisture) that the entity **feels** and
predicts — comfort-valenced affect, never a survival mechanic — plus the couplings
that make weather matter: rain slicks the ground (mobility), and shelter improves
comfort (which is why building matters).

See ``docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md`` sec. 4-5.

    from sanctuary.physics.weather import SyntheticWeatherSource, comfort_of
    weather = SyntheticWeatherSource().at(sim_time)
    feeling = comfort_of(weather)          # -> Comfort(valence, level, label, ...)
"""

from __future__ import annotations

from sanctuary.physics.weather.model import (
    COOL_OPTIMAL_F,
    FREEZING_F,
    HOT_F,
    OVERHEAT_F,
    WARM_F,
    comfort_of,
    effective_weather_at,
    mobility_multiplier,
)
from sanctuary.physics.weather.source import (
    RainWindow,
    SyntheticWeatherSource,
    WeatherSource,
)
from sanctuary.physics.weather.state import (
    Comfort,
    ComfortLevel,
    Shelter,
    WeatherState,
)

__all__ = [
    # state
    "WeatherState",
    "Comfort",
    "ComfortLevel",
    "Shelter",
    # model / coupling
    "comfort_of",
    "mobility_multiplier",
    "effective_weather_at",
    "FREEZING_F",
    "COOL_OPTIMAL_F",
    "WARM_F",
    "HOT_F",
    "OVERHEAT_F",
    # sources
    "WeatherSource",
    "SyntheticWeatherSource",
    "RainWindow",
]
