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

    The bridge snapshots base hebb_rate / spike_threshold once at load,
    multiplies them per cycle by CfC-derived scales, then restores.
    Drift across cycles would mean modulation accumulates (a known
    failure mode of mutate-in-place patterns).
    """
    from luthi.sanctuary_interface import snapshot_modulatable_state

    before = snapshot_modulatable_state(loaded_bridge.model)

    ci = _make_cognitive_input("anything")
    for _ in range(3):
        await loaded_bridge.think(ci)

    after = snapshot_modulatable_state(loaded_bridge.model)
    assert before.hebb_rates == after.hebb_rates
    # spike_thresholds may be empty for non-spiking layers — equal-or-empty.
    assert before.spike_thresholds == after.spike_thresholds


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
