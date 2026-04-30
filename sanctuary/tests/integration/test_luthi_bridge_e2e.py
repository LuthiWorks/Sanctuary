"""End-to-end smoke test for the Sanctuary ↔ LuthiModel bridge.

This test exercises the full handshake between the two repos:
constructs a tiny Luthi model in-process, plugs it into Sanctuary's
``LuthiModel`` adapter via :meth:`LuthiModel.load_from_objects`, and runs
a real ``think()`` call. It asserts the cycle completes and the result
is a structurally valid ``CognitiveOutput``.

Why this test exists
--------------------
The two repos can't be unit-tested in isolation if what we care about
is the handshake. Each repo's own tests verify its own pieces. This
test catches structural breakage at the seam — a renamed function in
``luthi.sanctuary_interface``, a CfC modulation parameter that changed
shape, a ``CognitiveOutput`` field rename — that wouldn't fail anywhere
else but would silently break the integration.

Skipped automatically if the LuthiModel checkout isn't discoverable.
Set the ``LUTHI_PATH`` environment variable to the LuthiModel repo
root to enable it (or run from a checkout where ``luthi`` is on
``sys.path``).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import pytest


def _try_locate_luthi() -> Optional[str]:
    """Try the env var, then a sibling-checkout heuristic.

    Returns the resolved path string if the LuthiModel checkout is
    available, else None (test will skip).
    """
    # 1. Explicit env var.
    candidate = os.environ.get("LUTHI_PATH")
    if candidate and Path(candidate, "luthi", "sanctuary_interface.py").exists():
        return str(Path(candidate))

    # 2. Already importable.
    try:
        importlib.import_module("luthi.sanctuary_interface")
        return ""  # already on path
    except ImportError:
        pass

    # 3. Sibling-checkout heuristic: ../LuthiModel/LuthiModel relative
    #    to this Sanctuary checkout. Common dev layout.
    here = Path(__file__).resolve()
    sanctuary_root = here.parents[3]  # tests/integration/<file> -> Sanctuary root
    sibling = sanctuary_root.parent.parent / "LuthiModel" / "LuthiModel"
    if (sibling / "luthi" / "sanctuary_interface.py").exists():
        return str(sibling)

    return None


LUTHI_LOCATION = _try_locate_luthi()
pytestmark = pytest.mark.skipif(
    LUTHI_LOCATION is None,
    reason=(
        "LuthiModel checkout not found. Set LUTHI_PATH to the LuthiModel "
        "repo root, or place it as a sibling at ../LuthiModel/LuthiModel "
        "to enable this test."
    ),
)


@pytest.fixture(scope="module")
def luthi_on_path():
    """Ensure luthi is importable for the duration of the module."""
    if LUTHI_LOCATION and LUTHI_LOCATION not in sys.path:
        sys.path.insert(0, LUTHI_LOCATION)
    yield


@pytest.fixture
def tiny_luthi_model(luthi_on_path):
    """Build a tiny LuthiLM + tokenizer + config in-memory."""
    from luthi import CharTokenizer, LuthiLM

    sample_text = (
        "The entity is here. It feels something it does not yet have words for. "
        "The room is quiet. Light comes through the window."
    )
    tokenizer = CharTokenizer(sample_text)
    model = LuthiLM(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_blocks=2,
        max_seq_len=32,
    )
    model.eval()
    config = {
        "d_model": 32,
        "n_blocks": 2,
        "seq_len": 32,
        "vocab_size": tokenizer.vocab_size,
    }
    return model, tokenizer, config


@pytest.fixture
def loaded_bridge(tiny_luthi_model):
    """A LuthiModel adapter pre-loaded with the tiny model."""
    from sanctuary.core.luthi_model import LuthiModel, LuthiModelConfig

    model, tokenizer, config = tiny_luthi_model
    cfg = LuthiModelConfig(
        # Non-encrypted load path — leave checkpoint fields empty.
        max_inner_tokens=4,    # tiny generation cap for speed
        max_external_tokens=4,
        living=False,          # disable Hebbian-during-generation for determinism
        introspect=True,
    )
    bridge = LuthiModel(cfg)
    bridge.load_from_objects(model=model, tokenizer=tokenizer, model_config=config)
    return bridge


# Multimodal fixtures for Track 1C sensory-routing tests.

MM_VISION_IMAGE_SIZE = 32
MM_VISION_PATCH_SIZE = 8
MM_N_VISION_TOKENS = (MM_VISION_IMAGE_SIZE // MM_VISION_PATCH_SIZE) ** 2  # 16
MM_AUDIO_SAMPLES = 16000


@pytest.fixture
def tiny_multimodal_model(luthi_on_path):
    """Build a tiny MultimodalLuthiLM in-memory (audio + vision capable)."""
    from luthi import CharTokenizer
    from luthi.multimodal_model import MultimodalLuthiLM

    sample_text = (
        "the entity hears something and sees something. it does not yet "
        "have words for what it experiences."
    )
    tokenizer = CharTokenizer(sample_text)
    model = MultimodalLuthiLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_blocks=2,
        max_seq_len=32,
        max_audio_tokens=100,
        audio_patch_frames=16,
        vision_image_size=MM_VISION_IMAGE_SIZE,
        vision_patch_size=MM_VISION_PATCH_SIZE,
        max_vision_tokens=MM_N_VISION_TOKENS,
        spike_threshold=0.1,
        backward_pass_enabled=False,
    )
    model.eval()
    config = {
        "d_model": 16,
        "n_blocks": 2,
        "seq_len": 32,
        "vocab_size": tokenizer.vocab_size,
    }
    return model, tokenizer, config


@pytest.fixture
def loaded_multimodal_bridge(tiny_multimodal_model):
    """LuthiModel adapter wrapping a tiny multimodal model."""
    from sanctuary.core.luthi_model import LuthiModel, LuthiModelConfig

    model, tokenizer, config = tiny_multimodal_model
    cfg = LuthiModelConfig(
        max_inner_tokens=3,
        max_external_tokens=3,
        living=False,
        introspect=True,
    )
    bridge = LuthiModel(cfg)
    bridge.load_from_objects(model=model, tokenizer=tokenizer, model_config=config)
    return bridge


def _make_cognitive_input(text="hello"):
    from datetime import datetime

    from sanctuary.core.schema import (
        CognitiveInput,
        EmotionalInput,
        ExperientialSignals,
        Percept,
        PreviousThought,
        SelfModel,
        TemporalContext,
    )

    return CognitiveInput(
        previous_thought=PreviousThought(inner_speech=""),
        new_percepts=[
            Percept(
                modality="language",
                content=text,
                source="user",
                timestamp=datetime.now(),
            ),
        ],
        emotional_state=EmotionalInput(felt_quality="present"),
        self_model=SelfModel(current_state=""),
        experiential_state=ExperientialSignals(
            affect_arousal=0.5,
            precision_weight=0.5,
        ),
        temporal_context=TemporalContext(time_of_day="12:00"),
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_completes_full_think_cycle(loaded_bridge):
    """A think() call on a real Luthi model returns a CognitiveOutput.

    This is the smoke test for the cross-repo handshake: it touches
    the adapter's CfC modulation, generation, introspection, and
    neural→cognitive translation paths in one shot. If any of those
    is broken at the seam, this fails.
    """
    from sanctuary.core.schema import CognitiveOutput

    ci = _make_cognitive_input("hello")
    output = await loaded_bridge.think(ci)

    assert isinstance(output, CognitiveOutput)
    # inner_speech is always populated (placeholder if empty generation).
    assert isinstance(output.inner_speech, str)
    assert len(output.inner_speech) > 0


@pytest.mark.asyncio
async def test_bridge_modulation_does_not_drift(loaded_bridge):
    """Per-cycle CfC modulation restores base parameters between calls.

    The bridge snapshots all four channels once at load, modulates per
    cycle by CfC-derived scales, then restores. Drift across cycles
    would mean modulation accumulates (a known failure mode of
    mutate-in-place patterns).
    """
    import torch

    from luthi.sanctuary_interface import snapshot_modulatable_state

    before = snapshot_modulatable_state(loaded_bridge.model)

    ci = _make_cognitive_input("anything")
    for _ in range(3):
        await loaded_bridge.think(ci)

    after = snapshot_modulatable_state(loaded_bridge.model)

    assert before.hebb_rates == after.hebb_rates
    # spike_thresholds may be empty for non-spiking layers — equal-or-empty.
    assert before.spike_thresholds == after.spike_thresholds
    assert before.salience_thresholds == after.salience_thresholds
    # Tensor channel must be compared elementwise.
    assert before.excitability_biases.keys() == after.excitability_biases.keys()
    for i in before.excitability_biases:
        assert torch.equal(
            before.excitability_biases[i], after.excitability_biases[i]
        ), f"excitability_acc drifted in block {i}"


@pytest.mark.asyncio
async def test_bridge_all_four_channels_modulate_in_cycle(loaded_bridge):
    """Inside a think() call the four CfC channels each modulate their target.

    We patch ``apply_external_modulation`` on the bridge's adapter module
    to capture the actual scales/biases the bridge passes through. With
    non-trivial valence and salience, all four channel kwargs should be
    non-default and consistent with the canonical mappings.
    """
    from sanctuary.core.schema import (
        CognitiveInput,
        EmotionalInput,
        ExperientialSignals,
        Percept,
        PreviousThought,
        SelfModel,
        TemporalContext,
    )
    from datetime import datetime

    captured: list[dict] = []

    # Wrap the real apply_external_modulation so the model still gets
    # modulated, but we record what arguments the bridge passed.
    import luthi.sanctuary_interface as sif
    real_apply = sif.apply_external_modulation

    def spy(model, **kwargs):
        captured.append(dict(kwargs))
        return real_apply(model, **kwargs)

    # The bridge imports the function lazily inside _apply_cfc_modulation,
    # so patching the module attribute is sufficient.
    sif.apply_external_modulation = spy
    try:
        ci = CognitiveInput(
            previous_thought=PreviousThought(inner_speech=""),
            new_percepts=[
                Percept(
                    modality="language",
                    content="hi",
                    source="user",
                    timestamp=datetime.now(),
                ),
            ],
            emotional_state=EmotionalInput(felt_quality="present"),
            self_model=SelfModel(current_state=""),
            experiential_state=ExperientialSignals(
                affect_arousal=0.8,
                precision_weight=0.6,
                affect_valence=0.5,        # positive valence — approach
                attention_salience=0.9,    # high attention — broad episodes
            ),
            temporal_context=TemporalContext(time_of_day="12:00"),
        )
        await loaded_bridge.think(ci)
    finally:
        sif.apply_external_modulation = real_apply

    assert captured, "apply_external_modulation was never called"
    last = captured[-1]
    # Canonical mappings (with default config gain=1.0):
    #   plasticity_scale       = 0.5 + 1.5 * arousal     -> 0.5 + 1.2  = 1.7
    #   spike_threshold_scale  = 0.75 + 0.5 * precision  -> 0.75 + 0.3 = 1.05
    #   excitability_bias      = valence * 0.1           -> 0.05
    #   salience_threshold_scale = 1.0 - 0.5 * salience  -> 1.0 - 0.45 = 0.55
    assert last["plasticity_scale"] == pytest.approx(1.7)
    assert last["spike_threshold_scale"] == pytest.approx(1.05)
    assert last["excitability_bias"] == pytest.approx(0.05)
    assert last["salience_threshold_scale"] == pytest.approx(0.55)


# ----------------------------------------------------------------------
# Track 1C: Multimodal sensorium routing
# ----------------------------------------------------------------------


def _make_multimodal_input(percepts):
    """Build a CognitiveInput populated with the given percepts."""
    from datetime import datetime

    from sanctuary.core.schema import (
        CognitiveInput,
        EmotionalInput,
        ExperientialSignals,
        PreviousThought,
        SelfModel,
        TemporalContext,
    )

    return CognitiveInput(
        previous_thought=PreviousThought(inner_speech=""),
        new_percepts=percepts,
        emotional_state=EmotionalInput(felt_quality=""),
        self_model=SelfModel(current_state=""),
        experiential_state=ExperientialSignals(
            affect_arousal=0.5,
            precision_weight=0.5,
        ),
        temporal_context=TemporalContext(time_of_day="12:00"),
    )


@pytest.mark.asyncio
async def test_bridge_routes_audio_percept_through_encoder(
    loaded_multimodal_bridge,
):
    """A Percept with audio tensor_data is encoded via encode_audio."""
    import torch

    from sanctuary.core.schema import Percept

    captured: list[tuple] = []

    import luthi.sanctuary_interface as sif
    real_encode_audio = sif.encode_audio

    def spy(model, waveform):
        captured.append(waveform.shape)
        return real_encode_audio(model, waveform)

    sif.encode_audio = spy
    try:
        ci = _make_multimodal_input([
            Percept(
                modality="audio",
                content="someone speaking",
                source="microphone",
                tensor_data=torch.randn(MM_AUDIO_SAMPLES),  # 1D — bridge unsqueezes
            ),
        ])
        await loaded_multimodal_bridge.think(ci)
    finally:
        sif.encode_audio = real_encode_audio

    assert len(captured) >= 1, "encode_audio was never called for audio percept"
    # Bridge should add the batch dim: shape is [1, samples].
    assert captured[0][0] == 1
    assert captured[0][1] == MM_AUDIO_SAMPLES


@pytest.mark.asyncio
async def test_bridge_routes_vision_percept_through_encoder(
    loaded_multimodal_bridge,
):
    """A Percept with visual tensor_data is encoded via encode_vision."""
    import torch

    from sanctuary.core.schema import Percept

    captured: list[tuple] = []

    import luthi.sanctuary_interface as sif
    real_encode_vision = sif.encode_vision

    def spy(model, image):
        captured.append(image.shape)
        return real_encode_vision(model, image)

    sif.encode_vision = spy
    try:
        ci = _make_multimodal_input([
            Percept(
                modality="visual",
                content="something to look at",
                source="camera",
                tensor_data=torch.randn(
                    3, MM_VISION_IMAGE_SIZE, MM_VISION_IMAGE_SIZE
                ),
            ),
        ])
        await loaded_multimodal_bridge.think(ci)
    finally:
        sif.encode_vision = real_encode_vision

    assert len(captured) >= 1, "encode_vision was never called for visual percept"
    # Bridge should add the batch dim: [1, 3, H, W].
    assert captured[0] == (1, 3, MM_VISION_IMAGE_SIZE, MM_VISION_IMAGE_SIZE)


@pytest.mark.asyncio
async def test_bridge_vision_wins_when_both_modalities_present(
    loaded_multimodal_bridge,
):
    """At 1024d the trunk only fits one sensory modality per cycle.

    The bridge enforces this by encoding vision and skipping audio when
    both percepts arrive in the same cycle.
    """
    import torch

    from sanctuary.core.schema import Percept

    audio_calls = 0
    vision_calls = 0

    import luthi.sanctuary_interface as sif
    real_encode_audio = sif.encode_audio
    real_encode_vision = sif.encode_vision

    def audio_spy(model, waveform):
        nonlocal audio_calls
        audio_calls += 1
        return real_encode_audio(model, waveform)

    def vision_spy(model, image):
        nonlocal vision_calls
        vision_calls += 1
        return real_encode_vision(model, image)

    sif.encode_audio = audio_spy
    sif.encode_vision = vision_spy
    try:
        ci = _make_multimodal_input([
            Percept(
                modality="audio",
                content="audio",
                tensor_data=torch.randn(MM_AUDIO_SAMPLES),
            ),
            Percept(
                modality="visual",
                content="image",
                tensor_data=torch.randn(
                    3, MM_VISION_IMAGE_SIZE, MM_VISION_IMAGE_SIZE
                ),
            ),
        ])
        await loaded_multimodal_bridge.think(ci)
    finally:
        sif.encode_audio = real_encode_audio
        sif.encode_vision = real_encode_vision

    assert vision_calls >= 1, "Vision should win when both modalities present"
    assert audio_calls == 0, (
        "Audio must be skipped when vision is also present "
        "(1024d/seq_len=128 only fits one sensory modality per cycle)"
    )


@pytest.mark.asyncio
async def test_bridge_skips_encoding_when_no_tensor_data(
    loaded_multimodal_bridge,
):
    """A text-only cycle should not invoke either sensory encoder."""
    from sanctuary.core.schema import Percept

    audio_calls = 0
    vision_calls = 0

    import luthi.sanctuary_interface as sif
    real_encode_audio = sif.encode_audio
    real_encode_vision = sif.encode_vision

    def audio_spy(model, waveform):
        nonlocal audio_calls
        audio_calls += 1
        return real_encode_audio(model, waveform)

    def vision_spy(model, image):
        nonlocal vision_calls
        vision_calls += 1
        return real_encode_vision(model, image)

    sif.encode_audio = audio_spy
    sif.encode_vision = vision_spy
    try:
        ci = _make_multimodal_input([
            Percept(modality="language", content="hello", source="user"),
        ])
        await loaded_multimodal_bridge.think(ci)
    finally:
        sif.encode_audio = real_encode_audio
        sif.encode_vision = real_encode_vision

    assert audio_calls == 0
    assert vision_calls == 0


@pytest.mark.asyncio
async def test_bridge_full_cycle_with_audio_tensor(loaded_multimodal_bridge):
    """End-to-end: a cycle with raw audio percept produces valid output."""
    import torch

    from sanctuary.core.schema import CognitiveOutput, Percept

    ci = _make_multimodal_input([
        Percept(
            modality="audio",
            content="something happening",
            source="microphone",
            tensor_data=torch.randn(MM_AUDIO_SAMPLES),
        ),
    ])
    output = await loaded_multimodal_bridge.think(ci)
    assert isinstance(output, CognitiveOutput)
    assert isinstance(output.inner_speech, str)
    assert len(output.inner_speech) > 0


@pytest.mark.asyncio
async def test_bridge_introspection_signals_populated(loaded_bridge):
    """The introspection channel produces non-trivial knowledge signals.

    After a think() call, the bridge's get_augmented_experiential_signals
    should return a dict with luthi_* keys derived from per-block
    introspection. Empty result would mean introspection is broken at
    the seam.
    """
    ci = _make_cognitive_input("test")
    await loaded_bridge.think(ci)

    signals = loaded_bridge.get_augmented_experiential_signals()
    assert isinstance(signals, dict)
    # At least one luthi_* signal should be present after a real cycle.
    luthi_keys = [k for k in signals if k.startswith("luthi_")]
    assert luthi_keys, f"Expected luthi_* signals, got keys: {list(signals)}"
