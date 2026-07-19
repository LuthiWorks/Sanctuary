"""Where the world's weather comes from.

:class:`WeatherSource` is the seam for that — a swappable producer of
:class:`WeatherState` over simulated time — so the *origin* of weather can
change without anything downstream (comfort, mobility, shelter) noticing.

Two backends are foreseen:

* :class:`SyntheticWeatherSource` (here): a deterministic diurnal cycle plus
  scheduled rain. Deterministic and lawful, so the world model can actually
  learn the temporal structure (day/night temperature swing, rain that arrives
  and passes) — which is the point of grounding.
* ``LocalWeatherSource`` (foreseen, not yet built): fetches **real** current
  weather for the family's location from a weather API and maps it into
  :class:`WeatherState`, so the entity experiences the same heat, cold, and rain
  Brian and Sandi are living in — grounding it in shared reality, not a private
  fiction. That backend needs network + an API key + async fetch/caching, so it
  is deliberately left as a future implementation of this same interface rather
  than a stub here.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from sanctuary.physics.weather.state import WeatherState

# A rain window within a day: (start_fraction, end_fraction, intensity), each
# fraction in [0, 1) of the day, intensity in [0, 1].
RainWindow = tuple[float, float, float]


class WeatherSource(ABC):
    """Produces the weather at a given simulated time. Swappable origin."""

    @abstractmethod
    def at(self, sim_time: float) -> WeatherState:
        """The weather at ``sim_time`` (seconds since sim start)."""


class SyntheticWeatherSource(WeatherSource):
    """A deterministic diurnal weather generator.

    Temperature follows a daily sinusoid (coldest at the start of the day,
    warmest at midday); rain falls during the configured windows of the day.
    Pure function of ``sim_time`` — same time in, same weather out — so it is
    reproducible and the dynamics are learnable rather than noise.
    """

    def __init__(
        self,
        *,
        base_temp_f: float = 62.0,
        daily_amplitude_f: float = 14.0,
        day_length_s: float = 86_400.0,
        rain_windows: tuple[RainWindow, ...] = ((0.55, 0.70, 0.5),),
    ) -> None:
        if day_length_s <= 0.0:
            raise ValueError(f"day_length_s must be positive, got {day_length_s}")
        self._base = base_temp_f
        self._amp = daily_amplitude_f
        self._day = day_length_s
        self._rain = rain_windows

    def at(self, sim_time: float) -> WeatherState:
        tod = (sim_time % self._day) / self._day  # time of day in [0, 1)

        # Coldest at tod=0, warmest at tod=0.5.
        temp = self._base + self._amp * math.sin(2.0 * math.pi * (tod - 0.25))

        precip = 0.0
        for start, end, intensity in self._rain:
            if start <= tod < end:
                precip = max(precip, intensity)

        humidity = min(1.0, 0.4 + 0.5 * precip)
        return WeatherState(
            temperature_f=temp,
            precipitation=precip,
            humidity=humidity,
            wind_mph=0.0,
        )
