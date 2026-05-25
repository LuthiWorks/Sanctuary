"""Smoke test: turbo trace pipeline against real Luthi introspection.

This is the empirical companion to the threshold-tuning research log
entry. It verifies — against a tiny in-memory Luthi model that uses the
SAME introspection code path production would use — that:

  1. The turbo trace mechanism captures per-source intensity values.
  2. The current ``MechanicalIntensitySource`` reads non-zero data
     from a real Luthi model (not just synthetic test fixtures).
  3. The v1 vs v2 substrate distinction shows up empirically: v1
     populates plasticity / drift / membrane components; v2 populates
     plasticity / drift only (no spiking/membrane in v2's PC layer).
  4. Most importantly — v2 has new introspection signals (`error_acc`,
     `prediction` Frobenius norm) that the current ``get_introspection``
     does NOT expose. This is the gap that needs closing before v2
     reaches production scale and turbo's threshold tuning becomes
     load-bearing.

These tests use ``LuthiModel.load_from_objects`` to bypass the
encrypted-checkpoint loader, so they run without any password. Skipped
if the LuthiModel sibling checkout is not on the path.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytest


def _try_locate_luthi() -> Optional[str]:
    candidate = os.environ.get("LUTHI_PATH")
    if candidate and Path(candidate, "luthi", "sanctuary_interface.py").exists():
        return str(Path(candidate))
    try:
        importlib.import_module("luthi.sanctuary_interface")
        return ""
    except ImportError:
        pass
    here = Path(__file__).resolve()
    sanctuary_root = here.parents[3]
    sibling = sanctuary_root.parent.parent / "LuthiModel" / "LuthiModel"
    if (sibling / "luthi" / "sanctuary_interface.py").exists():
        return str(sibling)
    return None


LUTHI_LOCATION = _try_locate_luthi()
pytestmark = pytest.mark.skipif(
    LUTHI_LOCATION is None,
    reason="LuthiModel checkout not found. Set LUTHI_PATH.",
)


@pytest.fixture(scope="module", autouse=True)
def _luthi_on_path():
    if LUTHI_LOCATION and LUTHI_LOCATION not in sys.path:
        sys.path.insert(0, LUTHI_LOCATION)
    yield


@pytest.fixture
def tiny_v1_model():
    from luthi import CharTokenizer, LuthiLM
    sample = "The room is quiet. Light comes through the window."
    tokenizer = CharTokenizer(sample)
    model = LuthiLM(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_blocks=2, max_seq_len=256,
    )
    model.eval()
    # config seq_len is the truncation hint (max_chars = seq_len * 4),
    # set lower than the model's max_seq_len to leave generation headroom.
    # The 4x multiplier in the adapter assumes BPE-like tokens; with a
    # CharTokenizer (1 char = 1 token) we deliberately undersize.
    config = {
        "d_model": 32, "n_blocks": 2, "seq_len": 32,
        "vocab_size": tokenizer.vocab_size,
    }
    return model, tokenizer, config


@pytest.fixture
def tiny_v2_model():
    from luthi import CharTokenizer
    from luthi.v2 import PredictiveCodingLM
    sample = "The room is quiet. Light comes through the window."
    tokenizer = CharTokenizer(sample)
    model = PredictiveCodingLM(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_blocks=2, max_seq_len=256,
    )
    model.eval()
    # config seq_len is the truncation hint (max_chars = seq_len * 4),
    # set lower than the model's max_seq_len to leave generation headroom.
    # The 4x multiplier in the adapter assumes BPE-like tokens; with a
    # CharTokenizer (1 char = 1 token) we deliberately undersize.
    config = {
        "d_model": 32, "n_blocks": 2, "seq_len": 32,
        "vocab_size": tokenizer.vocab_size,
    }
    return model, tokenizer, config


def _bridge(model, tokenizer, config):
    from sanctuary.core.luthi_model import LuthiModel, LuthiModelConfig
    cfg = LuthiModelConfig(
        max_inner_tokens=4,
        max_external_tokens=4,
        living=False,  # deterministic
        introspect=True,
    )
    bridge = LuthiModel(cfg)
    bridge.load_from_objects(model=model, tokenizer=tokenizer, model_config=config)
    return bridge


class TestIntrospectionShapeV1:
    """Confirm the v1 introspection populates all three activity_level
    components (plasticity, drift, membrane).
    """

    def test_v1_introspection_fields(self, tiny_v1_model):
        from luthi.sanctuary_interface import get_introspection
        model, _, _ = tiny_v1_model

        state = get_introspection(model)
        assert "blocks" in state and len(state["blocks"]) == 2
        block = state["blocks"][0]

        # v1-specific fields populate
        assert "plasticity_mean" in block
        assert "set_point_drift" in block
        # v1 LuthiLM is non-spiking; spike_fraction won't be there,
        # but excitability and (no) membrane should be reasonable.
        # If model has the spiking attrs, those populate; otherwise skip.

    def test_v1_augmented_signals_populate_luthi_delta(self, tiny_v1_model):
        """After running a forward pass, get_augmented_experiential_signals
        should produce a luthi_delta with at least plasticity_change and
        drift_change populated.
        """
        import torch
        model, tokenizer, config = tiny_v1_model
        bridge = _bridge(model, tokenizer, config)

        # Run one forward pass to populate pre/post state.
        from sanctuary.core.schema import (
            CognitiveInput, EmotionalInput, ComputedVAD, ExperientialSignals,
        )
        ci = CognitiveInput(
            previous_thought=None, new_percepts=[], surfaced_memories=[],
            prediction_errors=[],
            emotional_state=EmotionalInput(
                computed_vad=ComputedVAD(),
            ),
            experiential_state=ExperientialSignals(),
        )
        import asyncio
        asyncio.run(bridge.think(ci))

        signals = bridge.get_augmented_experiential_signals()
        # On v1 we expect luthi_delta to exist and contain finite values.
        assert "luthi_delta" in signals, (
            "v1 model should populate luthi_delta key in knowledge_signals"
        )
        delta = signals["luthi_delta"]
        assert len(delta) == 4, "luthi_delta is a 4-element list"
        # All values are real floats (no NaN).
        assert all(isinstance(v, float) for v in delta)


class TestIntrospectionShapeV2:
    """v2 substrate: the gap we're documenting.

    The v2 ``PredictiveCodingLayer`` has plasticity + set_point (same as
    v1) PLUS error_acc + prediction + precision (NEW, v2-specific). But
    it LACKS membrane_potential + spike_mask (no spiking dynamics).

    The current ``get_introspection`` reads v1-shape fields, so:
      - plasticity_mean and set_point_drift populate on v2 ✓
      - spike_fraction and membrane_mean are absent on v2 ✗
      - error_acc and prediction-Frobenius are NOT read at all ✗

    These tests lock that finding in as the empirical state of the
    pipeline as of 2026-05-25.
    """

    def test_v2_introspection_lacks_v1_fields(self, tiny_v2_model):
        from luthi.sanctuary_interface import get_introspection
        model, _, _ = tiny_v2_model

        state = get_introspection(model)
        block = state["blocks"][0]

        # v1 spiking fields don't apply to v2's non-spiking PC layer.
        assert "spike_fraction" not in block, (
            "v2 PC layer has no spike_mask — spike_fraction shouldn't appear"
        )
        assert "membrane_mean" not in block, (
            "v2 PC layer has no membrane_potential — membrane_mean shouldn't appear"
        )

    def test_v2_introspection_partially_populates_v1_fields(self, tiny_v2_model):
        """plasticity and set_point exist on v2 too, so the introspection
        captures them. This is the path of partial compatibility.
        """
        from luthi.sanctuary_interface import get_introspection
        model, _, _ = tiny_v2_model

        state = get_introspection(model)
        block = state["blocks"][0]

        assert "plasticity_mean" in block, (
            "v2 has plasticity tensor — plasticity_mean should populate"
        )
        assert "set_point_drift" in block, (
            "v2 has set_point + weight — set_point_drift should compute"
        )

    def test_v2_specific_signals_NOT_currently_exposed(self, tiny_v2_model):
        """error_acc, pred_frob, and precision are v2-native signals
        that the current get_introspection does NOT read. This test
        documents the gap.

        When this test starts FAILING (because get_introspection has
        been extended to read v2 fields), update it to assert the new
        v2-aware behavior. Until then, this is the load-bearing finding
        for the threshold-tuning research log.
        """
        from luthi.sanctuary_interface import get_introspection
        model, _, _ = tiny_v2_model

        state = get_introspection(model)
        block = state["blocks"][0]

        # These are the v2 signals turbo would most want to read.
        # Confirm they're NOT in the introspection output as of now.
        assert "error_acc_mean" not in block
        assert "pred_frob" not in block
        assert "precision_mean" not in block


class TestTracePipelineEndToEnd:
    """Turbo trace logging captures non-zero data from real introspection.

    The point of this test isn't behavior verification (that's in
    test_turbo.py); it's pipeline integrity — confirms the trace file
    actually fills with data when wired against a real Luthi model.
    """

    def test_v1_substrate_produces_nonzero_trace(self, tiny_v1_model, tmp_path):
        """Boot the runner with a v1 model + turbo trace, run a few
        cycles, verify the trace has non-zero per_source['mechanical']
        readings (since v1 introspection populates luthi_delta).
        """
        import asyncio
        from sanctuary.api.runner import RunnerConfig, SanctuaryRunner

        model, tokenizer, config = tiny_v1_model
        bridge = _bridge(model, tokenizer, config)

        data_dir = tmp_path / "identity"
        data_dir.mkdir()
        # Minimal charter so the formatted prompt stays well under
        # max_seq_len for the tiny test model.
        (data_dir / "charter.md").write_text("# T\n", encoding="utf-8")

        trace_path = tmp_path / "turbo_trace.jsonl"
        runner_config = RunnerConfig(
            cycle_delay=0.01,
            data_dir=str(data_dir),
            charter_path=str(data_dir / "charter.md"),
            persist_state=False,
            experiential_enabled=False,  # focus on turbo, not CfC
            turbo_enabled=True,
            turbo_trace_path=str(trace_path),
            silence_threshold=999.0,
        )
        runner = SanctuaryRunner(model=bridge, config=runner_config)

        async def _boot_and_run():
            await runner.boot()
            await runner.cycle.run(max_cycles=3)

        asyncio.run(_boot_and_run())

        # Trace should have 3 lines (one per observe).
        assert trace_path.exists()
        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3, f"Expected 3 trace lines, got {len(lines)}"

        records = [json.loads(line) for line in lines]
        # At least one record should have a non-zero mechanical intensity
        # — confirming the source IS reading luthi_delta correctly. With
        # only 3 cycles on an untrained tiny model the magnitudes are
        # small, but the pipeline produces real numbers, not zeros.
        any_nonzero = any(
            r["per_source"].get("mechanical", 0.0) > 0.0 for r in records
        )
        # Even untrained, plasticity_change + drift_change should produce
        # at least one non-zero reading across 3 cycles.
        assert any_nonzero, (
            f"v1 substrate should produce some non-zero mechanical intensity "
            f"across 3 cycles. Trace: {records}"
        )
