"""Tests for the weather-field layer: comfort, mobility, shelter, source.

The comfort assertions pin Brian's thresholds (2026-07-16): 70 F comfortable,
85 F possibly too warm, 32 F too cold; light warm rain refreshing, torrential
rain unpleasant; and — crucially — everything is affect, never damage.
"""

from __future__ import annotations

import pytest

from sanctuary.physics.weather import (
    ComfortLevel,
    Shelter,
    SyntheticWeatherSource,
    WeatherState,
    comfort_of,
    effective_weather_at,
    mobility_multiplier,
)


# ---------------------------------------------------------------------------
# Comfort at the named thresholds
# ---------------------------------------------------------------------------


def test_ideal_is_comfortable_and_pleasant():
    c = comfort_of(WeatherState(temperature_f=70.0))
    assert c.level is ComfortLevel.COMFORTABLE
    assert c.valence == pytest.approx(1.0)
    assert c.label == "comfortable"


def test_eightyfive_is_too_warm_and_mildly_unpleasant():
    c = comfort_of(WeatherState(temperature_f=85.0))
    assert c.level is ComfortLevel.TOO_WARM
    assert -0.35 < c.valence < 0.0     # unpleasant, but only mildly ("possibly")


def test_thirtytwo_is_too_cold_and_clearly_unpleasant():
    c = comfort_of(WeatherState(temperature_f=32.0))
    assert c.level is ComfortLevel.TOO_COLD
    assert c.valence < -0.35           # clearly worse than "possibly too warm"


def test_cold_reads_worse_than_the_warm_threshold():
    # 32 (too cold) should feel worse than 85 (possibly too warm).
    assert comfort_of(WeatherState(32.0)).valence < comfort_of(WeatherState(85.0)).valence


def test_valence_is_monotonic_away_from_ideal():
    hot = [comfort_of(WeatherState(t)).valence for t in (70, 78, 85, 95)]
    cold = [comfort_of(WeatherState(t)).valence for t in (70, 60, 45, 32)]
    assert hot == sorted(hot, reverse=True)
    assert cold == sorted(cold, reverse=True)


def test_levels_track_the_bands():
    cases = {
        20.0: ComfortLevel.TOO_COLD,
        40.0: ComfortLevel.COLD,
        50.0: ComfortLevel.COOL,
        70.0: ComfortLevel.COMFORTABLE,
        80.0: ComfortLevel.WARM,
        90.0: ComfortLevel.TOO_WARM,
    }
    for temp, level in cases.items():
        assert comfort_of(WeatherState(temp)).level is level


# ---------------------------------------------------------------------------
# Moisture: refreshing vs annoying
# ---------------------------------------------------------------------------


def test_light_warm_rain_is_refreshing():
    dry = comfort_of(WeatherState(75.0, precipitation=0.0))
    light = comfort_of(WeatherState(75.0, precipitation=0.2))
    assert light.valence > dry.valence
    assert "refresh" in light.label


def test_torrential_rain_is_unpleasant():
    dry = comfort_of(WeatherState(75.0, precipitation=0.0))
    heavy = comfort_of(WeatherState(75.0, precipitation=1.0))
    assert heavy.valence < dry.valence
    assert "drenched" in heavy.label


def test_cold_and_wet_is_worse_than_cold_and_dry():
    dry = comfort_of(WeatherState(40.0, precipitation=0.0))
    wet = comfort_of(WeatherState(40.0, precipitation=0.8))
    assert wet.valence < dry.valence


def test_shelter_from_exposure_means_no_wetness():
    # Fully sheltered from rain (exposure 0) -> not wet, even in a downpour.
    c = comfort_of(WeatherState(70.0, precipitation=1.0), exposure=0.0)
    assert c.wetness == 0.0


# ---------------------------------------------------------------------------
# Mobility coupling (rain -> slick ground)
# ---------------------------------------------------------------------------


def test_mobility_decreases_with_rain():
    assert mobility_multiplier(WeatherState(70.0, precipitation=0.0)) == pytest.approx(1.0)
    assert mobility_multiplier(WeatherState(70.0, precipitation=0.5)) == pytest.approx(0.8)
    assert mobility_multiplier(WeatherState(70.0, precipitation=1.0)) == pytest.approx(0.6)


def test_mobility_is_monotonic_and_bounded():
    vals = [mobility_multiplier(WeatherState(70.0, precipitation=p)) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals, reverse=True)
    assert all(0.6 <= v <= 1.0 for v in vals)


# ---------------------------------------------------------------------------
# Shelter: why building matters
# ---------------------------------------------------------------------------


def test_shelter_blocks_rain_and_moderates_temperature():
    weather = WeatherState(temperature_f=40.0, precipitation=0.8)
    shelter = Shelter(position=(0.0, 0.0, 0.0), radius=3.0, rain_block=1.0, temp_moderation=0.5)
    eff, exposure = effective_weather_at((1.0, 0.0, 0.0), weather, (shelter,))
    assert eff.precipitation == pytest.approx(0.0)       # rain kept off
    assert 40.0 < eff.temperature_f <= 70.0              # pulled toward ideal
    assert exposure == pytest.approx(0.0)


def test_outside_shelter_radius_is_unchanged():
    weather = WeatherState(temperature_f=40.0, precipitation=0.8)
    shelter = Shelter(position=(0.0, 0.0, 0.0), radius=3.0)
    eff, exposure = effective_weather_at((100.0, 0.0, 0.0), weather, (shelter,))
    assert eff == weather
    assert exposure == pytest.approx(1.0)


def test_shelter_improves_comfort_in_cold_rain():
    weather = WeatherState(temperature_f=40.0, precipitation=0.8)
    shelter = Shelter(position=(0.0, 0.0, 0.0), radius=3.0, rain_block=1.0, temp_moderation=0.5)

    exposed = comfort_of(weather, exposure=1.0)
    eff, exposure = effective_weather_at((0.5, 0.0, 0.0), weather, (shelter,))
    sheltered = comfort_of(eff, exposure=exposure)

    assert sheltered.valence > exposed.valence


# ---------------------------------------------------------------------------
# Synthetic weather source
# ---------------------------------------------------------------------------


def test_synthetic_source_is_deterministic():
    src = SyntheticWeatherSource()
    assert src.at(1234.5) == src.at(1234.5)
    # Independent instances with the same config agree too.
    assert SyntheticWeatherSource().at(9999.0) == SyntheticWeatherSource().at(9999.0)


def test_synthetic_source_has_a_diurnal_cycle():
    src = SyntheticWeatherSource(day_length_s=86_400.0)
    cold = src.at(0.0).temperature_f            # start of day = coldest
    warm = src.at(43_200.0).temperature_f       # midday = warmest
    assert warm > cold


def test_synthetic_source_rains_in_its_window_only():
    src = SyntheticWeatherSource(day_length_s=86_400.0, rain_windows=((0.55, 0.70, 0.5),))
    assert src.at(86_400.0 * 0.60).precipitation == pytest.approx(0.5)   # inside window
    assert src.at(86_400.0 * 0.10).precipitation == pytest.approx(0.0)   # dry morning


def test_synthetic_source_rejects_nonpositive_day_length():
    with pytest.raises(ValueError):
        SyntheticWeatherSource(day_length_s=0.0)
