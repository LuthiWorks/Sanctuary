"""Voice analysis module for emotional context detection in speech.

This is real prosodic analysis, not a placeholder. Emotional state in a
voice is carried by *prosody* -- the melody and energy of speech, largely
independent of the words -- and classical signal processing recovers the
load-bearing prosodic features without any trained model:

- **Pitch (fundamental frequency, F0)** via short-time autocorrelation.
  Pitch height and how much it moves are among the strongest acoustic
  correlates of emotional arousal: aroused speech (excited, angry,
  fearful) sits higher and varies more; subdued speech (sad, calm) sits
  lower and flatter.
- **Pitch variability** (coefficient of variation across voiced frames):
  expressiveness / emotional intensity.
- **Energy dynamics** (relative variation of frame RMS): monotone vs.
  dynamic delivery. Deliberately *relative* (CV of the envelope), so the
  measure is invariant to recording volume.
- **Zero-crossing rate**: roughness / tension in the signal.
- **Voiced ratio**: the fraction of frames with a clear periodic
  structure. This is what makes the analyzer *honest* -- confidence is
  earned from how much real voiced signal was present, not asserted. A
  near-silent or unvoiced segment yields low confidence, not a fabricated
  one.

Continuous **arousal** and **valence** are derived from these features
(grounded in the affective-circumplex tradition: arousal from energy and
pitch activity; valence weakly from pitch and voicing), and a coarse
discrete emotion label is read off the arousal/valence quadrant. Where the
signal is weak the analyzer reports ``neutral`` with low confidence rather
than guessing -- the entity's hearing should say "I can't tell" instead of
inventing certainty.

All features are scale-invariant where possible (pitch, ratios, coefficients
of variation), so the result does not depend on the caller's recording gain.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


class EmotionAnalyzer:
    """Analyzes emotional content in voice signals via prosodic features."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.emotion_states = [
            "neutral", "happy", "sad", "angry",
            "fearful", "disgust", "surprised"
        ]
        self.current_context = {
            "primary_emotion": "neutral",
            "confidence": 0.0,
            "secondary_markers": []
        }

        # Human voice pitch range (Hz). F0 estimates outside this band are
        # rejected as octave errors / non-pitch artifacts.
        self._f0_min = 80.0
        self._f0_max = 400.0
        # Normalized-autocorrelation peak above which a frame counts as
        # voiced (periodic). Pilot value; conservative.
        self._voiced_peak_threshold = 0.30
        # Frame geometry for short-time analysis.
        self._frame_seconds = 0.040   # 40 ms analysis window
        self._hop_seconds = 0.020     # 20 ms hop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_segment(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> Dict[str, any]:
        """Analyze emotional content in an audio segment.

        Args:
            audio_data: Raw audio waveform (mono). Any amplitude scale --
                the analysis is gain-invariant.
            sample_rate: Sample rate of ``audio_data``; falls back to the
                analyzer's configured rate.

        Returns:
            Dict with ``primary_emotion`` (one of ``emotion_states``),
            ``confidence`` (float in [0, 1], *earned* from voiced signal),
            ``secondary_emotions`` (list[str]), ``arousal`` and ``valence``
            (floats in [0, 1]), plus the raw prosodic features under
            ``features`` for downstream use and instrumentation.

        Raises:
            ValueError: if ``audio_data`` is empty -- there is no fallback
                to a fabricated reading; no audio means no analysis.
        """
        sr = int(sample_rate or self.sample_rate)
        x = np.asarray(audio_data, dtype=np.float64).reshape(-1)
        if x.size == 0:
            raise ValueError("analyze_segment received empty audio_data")

        features = self._extract_features(x, sr)
        arousal, valence = self._features_to_affect(features)
        confidence = self._confidence(features)
        primary, secondary = self._affect_to_emotion(
            arousal, valence, confidence
        )

        return {
            "primary_emotion": primary,
            "confidence": confidence,
            "secondary_emotions": secondary,
            "arousal": arousal,
            "valence": valence,
            "features": features,
        }

    def update_context(self, analysis_result: Dict[str, any]) -> None:
        """Update the ongoing emotional context tracking."""
        self.current_context = {
            "primary_emotion": analysis_result["primary_emotion"],
            "confidence": analysis_result["confidence"],
            "secondary_markers": analysis_result.get("secondary_emotions", []),
        }

    def get_current_context(self) -> Dict[str, any]:
        """Get the current emotional context state."""
        return self.current_context.copy()

    # ------------------------------------------------------------------
    # Feature extraction (the real DSP)
    # ------------------------------------------------------------------
    def _extract_features(self, x: np.ndarray, sr: int) -> Dict[str, float]:
        """Short-time prosodic features over the segment."""
        frame_len = max(256, int(self._frame_seconds * sr))
        hop = max(1, int(self._hop_seconds * sr))

        # Zero-crossing rate over the whole segment (sign changes / sample).
        # Scale-invariant; high for noisy/unvoiced/tense speech.
        if x.size > 1:
            zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
        else:
            zcr = 0.0

        # Frame the signal (pad the tail so short inputs still yield 1 frame).
        if x.size < frame_len:
            frames = [np.pad(x, (0, frame_len - x.size))]
        else:
            starts = range(0, x.size - frame_len + 1, hop)
            frames = [x[s:s + frame_len] for s in starts]

        window = np.hanning(frame_len)
        f0_values: List[float] = []
        frame_rms: List[float] = []
        voiced_flags: List[bool] = []

        for frame in frames:
            fw = frame * window
            rms = float(np.sqrt(np.mean(fw ** 2)))
            frame_rms.append(rms)
            f0, voiced = self._frame_pitch(fw, sr)
            voiced_flags.append(voiced)
            if voiced:
                f0_values.append(f0)

        voiced_ratio = (
            float(np.mean(voiced_flags)) if voiced_flags else 0.0
        )

        # Pitch statistics over voiced frames only.
        if f0_values:
            f0_arr = np.asarray(f0_values)
            f0_mean = float(f0_arr.mean())
            f0_std = float(f0_arr.std())
            # Pitch height normalized into the human voice band -> [0, 1].
            pitch_norm = float(
                np.clip(
                    (f0_mean - self._f0_min) / (self._f0_max - self._f0_min),
                    0.0, 1.0,
                )
            )
            # Coefficient of variation (scale-free expressiveness).
            f0_cv = float(f0_std / (f0_mean + 1e-8))
        else:
            f0_mean = 0.0
            pitch_norm = 0.0
            f0_cv = 0.0

        # Energy dynamics: relative variation of the RMS envelope.
        rms_arr = np.asarray(frame_rms)
        mean_rms = float(rms_arr.mean())
        energy_cv = (
            float(rms_arr.std() / (mean_rms + 1e-8)) if mean_rms > 0 else 0.0
        )

        return {
            "zcr": zcr,
            "voiced_ratio": voiced_ratio,
            "f0_mean": f0_mean,
            "pitch_norm": pitch_norm,
            "f0_cv": f0_cv,
            "energy_cv": energy_cv,
            "n_frames": float(len(frames)),
        }

    def _frame_pitch(self, frame: np.ndarray, sr: int) -> tuple[float, bool]:
        """Estimate F0 of one frame via normalized autocorrelation.

        Returns (f0_hz, voiced). ``voiced`` is True when a clear periodic
        peak exists in the human-pitch lag band.
        """
        energy = float(np.dot(frame, frame))
        if energy <= 1e-12:
            return 0.0, False  # silent frame -> unvoiced, honestly

        # Full autocorrelation via FFT, take the non-negative lags.
        n = frame.size
        fft = np.fft.rfft(frame, n=2 * n)
        acf = np.fft.irfft(fft * np.conjugate(fft))[:n]
        if acf[0] <= 0:
            return 0.0, False
        acf = acf / acf[0]  # normalize so acf[0] == 1

        min_lag = max(1, int(sr / self._f0_max))
        max_lag = min(n - 1, int(sr / self._f0_min))
        if max_lag <= min_lag:
            return 0.0, False

        search = acf[min_lag:max_lag + 1]
        peak_idx = int(np.argmax(search))
        peak_val = float(search[peak_idx])
        lag = min_lag + peak_idx

        if peak_val < self._voiced_peak_threshold or lag <= 0:
            return 0.0, False
        f0 = sr / lag
        return float(f0), True

    # ------------------------------------------------------------------
    # Feature -> affect mapping
    # ------------------------------------------------------------------
    def _features_to_affect(self, f: Dict[str, float]) -> tuple[float, float]:
        """Map prosodic features to continuous (arousal, valence) in [0, 1].

        Arousal: pitch height, pitch movement, energy dynamics, and signal
        roughness all rise with activation. Valence is genuinely weaker from
        prosody alone; we estimate it conservatively from pitch and voicing
        (clear, higher-pitched voiced speech skews positive; rough,
        high-ZCR delivery skews negative) and keep it near neutral when the
        evidence is thin.
        """
        arousal = (
            0.40 * f["pitch_norm"]
            + 0.25 * np.tanh(3.0 * f["f0_cv"])
            + 0.20 * np.tanh(2.0 * f["energy_cv"])
            + 0.15 * min(f["zcr"] * 4.0, 1.0)
        )
        arousal = float(np.clip(arousal, 0.0, 1.0))

        # Valence centered at 0.5, nudged by pitch (up) and roughness (down),
        # scaled by how voiced the segment is (no voicing -> stay neutral).
        valence_raw = (
            0.5
            + 0.25 * (f["pitch_norm"] - 0.5)
            - 0.25 * (min(f["zcr"] * 4.0, 1.0) - 0.5)
        )
        valence = float(np.clip(0.5 + (valence_raw - 0.5) * f["voiced_ratio"], 0.0, 1.0))
        return arousal, valence

    def _confidence(self, f: Dict[str, float]) -> float:
        """Earned confidence: high only when there is genuine voiced signal.

        A near-silent or unvoiced segment carries little emotional
        information, so confidence stays low rather than being asserted.
        """
        # Voiced ratio is the dominant term; a tiny floor of energy dynamics
        # keeps a fully-monotone-but-voiced segment from reading as zero.
        conf = 0.85 * f["voiced_ratio"] + 0.15 * min(f["energy_cv"], 1.0)
        return float(np.clip(conf, 0.0, 1.0))

    def _affect_to_emotion(
        self,
        arousal: float,
        valence: float,
        confidence: float,
    ) -> tuple[str, List[str]]:
        """Read a coarse emotion off the arousal/valence quadrant.

        Affective-circumplex quadrants. When arousal is near baseline or
        confidence is low, report ``neutral`` -- the analyzer does not guess
        a strong emotion from a weak signal.
        """
        secondary: List[str] = []
        # Low confidence or low arousal -> neutral (honest abstention).
        if confidence < 0.35 or arousal < 0.35:
            return "neutral", secondary

        high_arousal = arousal >= 0.6
        positive = valence >= 0.5

        if high_arousal and positive:
            primary = "happy"
            secondary = ["surprised"]
        elif high_arousal and not positive:
            # High arousal + negative: anger vs fear. Roughness already fed
            # arousal; we separate weakly by valence depth.
            primary = "angry" if valence < 0.35 else "fearful"
            secondary = ["surprised"]
        elif not high_arousal and not positive:
            primary = "sad"
        else:
            # Moderate arousal, positive -> mild positive affect.
            primary = "happy"
        return primary, secondary
