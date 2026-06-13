"""Tests for the prosodic EmotionAnalyzer.

These verify the analyzer is *real* -- that it actually responds to the
acoustic signal -- not a constant. The decisive tests are the
discrimination ones: a higher-pitched signal must read as more aroused
than a low one; a dynamic signal more than a flat one; a clearly voiced
signal more confidently than noise. A stub returning a constant would
fail every one of these.

All signals are synthesized with numpy so the tests run fully offline and
deterministically (no model, no network, no audio files).
"""
import numpy as np
import pytest

from sanctuary.mind.voice_analyzer import EmotionAnalyzer

SR = 16000


def _sine(freq: float, dur: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------- honesty

def test_empty_audio_fails_loud():
    """No audio -> ValueError, never a fabricated reading."""
    a = EmotionAnalyzer()
    with pytest.raises(ValueError):
        a.analyze_segment(np.array([]))


def test_silence_yields_low_confidence_and_neutral():
    """Silence carries no emotion; confidence must be earned, so ~0."""
    a = EmotionAnalyzer()
    r = a.analyze_segment(np.zeros(SR))
    assert r["primary_emotion"] == "neutral"
    assert r["confidence"] < 0.1


def test_voiced_signal_more_confident_than_noise():
    """A clearly periodic (voiced) signal must read more confidently than
    white noise -- confidence tracks real voiced structure."""
    a = EmotionAnalyzer()
    rng = np.random.default_rng(0)
    voiced = _sine(150.0)
    noise = rng.standard_normal(SR) * 0.5
    c_voiced = a.analyze_segment(voiced)["confidence"]
    c_noise = a.analyze_segment(noise)["confidence"]
    assert c_voiced > c_noise


# --------------------------------------------------------- discrimination

def test_higher_pitch_reads_more_aroused():
    """Pitch height is a core arousal correlate."""
    a = EmotionAnalyzer()
    low = a.analyze_segment(_sine(120.0))
    high = a.analyze_segment(_sine(320.0))
    assert high["arousal"] > low["arousal"]


def test_dynamic_signal_more_aroused_than_flat():
    """A pitch-sweeping, amplitude-modulated signal must read as more
    aroused than a steady low tone -- expressiveness raises arousal."""
    a = EmotionAnalyzer()
    calm = _sine(120.0, amp=0.3)

    t = np.linspace(0.0, 1.0, SR, endpoint=False)
    # Instantaneous frequency 150 -> 330 Hz (in the voice band).
    chirp = np.sin(2 * np.pi * (150.0 * t + 90.0 * t * t))
    am = 1.0 + 0.6 * np.sin(2 * np.pi * 4.0 * t)  # 4 Hz tremor
    aroused = 0.6 * chirp * am

    assert a.analyze_segment(aroused)["arousal"] > a.analyze_segment(calm)["arousal"]


# ---------------------------------------------------------------- schema

def test_schema_and_ranges():
    a = EmotionAnalyzer()
    r = a.analyze_segment(_sine(150.0))
    for key in (
        "primary_emotion", "confidence", "secondary_emotions",
        "arousal", "valence", "features",
    ):
        assert key in r
    assert isinstance(r["confidence"], float)
    assert 0.0 <= r["confidence"] <= 1.0
    assert 0.0 <= r["arousal"] <= 1.0
    assert 0.0 <= r["valence"] <= 1.0
    assert r["primary_emotion"] in a.emotion_states
    for fkey in ("zcr", "voiced_ratio", "f0_mean", "pitch_norm", "f0_cv", "energy_cv"):
        assert fkey in r["features"]


def test_initialization_defaults_preserved():
    """The schema the rest of the system relies on must not regress."""
    a = EmotionAnalyzer()
    assert "neutral" in a.emotion_states
    assert a.current_context["primary_emotion"] == "neutral"
    assert a.current_context["confidence"] == 0.0


def test_update_and_get_context():
    a = EmotionAnalyzer()
    r = a.analyze_segment(_sine(200.0))
    a.update_context(r)
    ctx = a.get_current_context()
    assert ctx["primary_emotion"] == r["primary_emotion"]
    assert ctx["confidence"] == r["confidence"]
