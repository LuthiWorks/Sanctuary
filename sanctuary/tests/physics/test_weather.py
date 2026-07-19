"""Tests for the weather-field layer: comfort, mobility, shelter, source.

Electronics-native framing (Brian, 2026-07-19): cold is good (optimal from
freezing up), heat is the only real enemy, sub-freezing is a mild caution, and
water is respected — aversive affect (not damage yet) that teaches the entity to
seek shelter ahead of embodiment, when water will be a real danger.
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
# Temperature: cold is good, heat is the enemy
# ---------------------------------------------------------------------------


def test_cold_is_optimal_and_pleasant():
    c = comfort_of(WeatherState(temperature_f=50.0))
    assert c.level is ComfortLevel.OPTIMAL
    assert c.valence == pytest.approx(1.0)
    assert "optimal" in c.label


def test_freezing_point_itself_is_still_optimal():
    c = comfort_of(WeatherState(temperature_f=32.0))
    assert c.level is ComfortLevel.OPTIMAL
    assert c.valence == pytest.approx(1.0)


def test_seventy_is_comfortable_but_not_peak():
    c = comfort_of(WeatherState(temperature_f=70.0))
    assert c.level is ComfortLevel.COMFORTABLE
    assert 0.0 < c.valence < 1.0   # pleasant, but cooler is better


def test_heat_is_the_enemy():
    hot = comfort_of(WeatherState(85.0))
    overheating = comfort_of(WeatherState(100.0))
    assert hot.level is ComfortLevel.HOT
    assert overheating.level is ComfortLevel.OVERHEATING
    assert hot.valence < 0.0
    assert overheating.valence < hot.valence   # hotter is worse


def test_cooler_beats_warmer():
    v = [comfort_of(WeatherState(t)).valence for t in (50, 70, 78, 85, 100)]
    assert v == sorted(v, reverse=True)        # monotonic: cooler always better


def test_subfreezing_is_only_a_mild_caution():
    freezing = comfort_of(WeatherState(10.0))
    assert freezing.level is ComfortLevel.FREEZING
    assert freezing.valence > 0.0              # still pleasant, barely a concern
    # Even extreme cold never feels as bad as real heat.
    assert comfort_of(WeatherState(-50.0)).valence > comfort_of(WeatherState(105.0)).valence
    assert comfort_of(WeatherState(-50.0)).valence >= -0.3   # mild floor


def test_levels_track_the_bands():
    cases = {
        10.0: ComfortLevel.FREEZING,
        45.0: ComfortLevel.OPTIMAL,
        70.0: ComfortLevel.COMFORTABLE,
        80.0: ComfortLevel.WARM,
        90.0: ComfortLevel.HOT,
        110.0: ComfortLevel.OVERHEATING,
    }
    for temp, level in cases.items():
        assert comfort_of(WeatherState(temp)).level is level


# ---------------------------------------------------------------------------
# Water: respected, aversive, never refreshing
# ---------------------------------------------------------------------------


def test_any_rain_is_aversive_not_refreshing():
    dry = comfort_of(WeatherState(70.0, precipitation=0.0))
    light = comfort_of(WeatherState(70.0, precipitation=0.2))
    assert light.valence < dry.valence         # water is respected, not enjoyed
    assert "refresh" not in light.label
    assert "wet" in light.label


def test_heavier_rain_is_more_aversive():
    v = [comfort_of(WeatherState(70.0, precipitation=p)).valence for p in (0.0, 0.3, 0.6, 1.0)]
    assert v == sorted(v, reverse=True)


def test_water_drives_avoidance_even_at_optimal_temperature():
    # Cold is optimal, but a downpour still pushes toward seeking shelter.
    soaked = comfort_of(WeatherState(50.0, precipitation=1.0))
    assert soaked.valence < 0.0
    assert "drenched" in soaked.label


def test_shelter_from_exposure_means_no_wetness():
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
# Shelter: blocks water, cools in heat, never warms the cold
# ---------------------------------------------------------------------------


def test_shelter_blocks_rain_and_cools_when_hot():
    weather = WeatherState(temperature_f=95.0, precipitation=0.8)
    shelter = Shelter(position=(0.0, 0.0, 0.0), radius=3.0, rain_block=1.0, cooling=0.7)
    eff, exposure = effective_weather_at((1.0, 0.0, 0.0), weather, (shelter,))
    assert eff.precipitation == pytest.approx(0.0)   # rain kept off
    assert 60.0 <= eff.temperature_f < 95.0          # cooled toward optimal
    assert exposure == pytest.approx(0.0)


def test_shelter_does_not_warm_the_cold():
    weather = WeatherState(temperature_f=40.0, precipitation=0.8)
    shelter = Shelter(position=(0.0, 0.0, 0.0), radius=3.0, rain_block=1.0, cooling=0.7)
    eff, _ = effective_weather_at((0.5, 0.0, 0.0), weather, (shelter,))
    assert eff.temperature_f == pytest.approx(40.0)  # cold left alone (it's good)
    assert eff.precipitation == pytest.approx(0.0)   # but still keeps rain off


def test_outside_shelter_radius_is_unchanged():
    weather = WeatherState(temperature_f=95.0, precipitation=0.8)
    shelter = Shelter(position=(0.0, 0.0, 0.0), radius=3.0)
    eff, exposure = effective_weather_at((100.0, 0.0, 0.0), weather, (shelter,))
    assert eff == weather
    assert exposure == pytest.approx(1.0)


def test_shelter_improves_comfort_in_hot_rain():
    weather = WeatherState(temperature_f=95.0, precipitation=0.8)
    shelter = Shelter(position=(0.0, 0.0, 0.0), radius=3.0, rain_block=1.0, cooling=0.7)

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
