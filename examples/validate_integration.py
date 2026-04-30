"""Validate Sanctuary <-> LuthiModel integration with the real 1024d checkpoint.

Track 1A of docs/TRACK1_INTEGRATION_PLAN.md. Loads a real Luthi checkpoint
through SanctuaryRunner, runs cognitive cycles with injected language
percepts, prints per-cycle introspection signals and CfC modulation, and
verifies that modulation restores cleanly to the base snapshot (no drift).

Success criteria (from the plan):
  - Cycles complete without crash
  - Inner speech is generated (coherence not expected at 1024d/128 context)
  - Introspection shows non-zero deltas
  - Modulation restores cleanly (hebb_rates and spike_thresholds equal base)

Usage:
    python examples/validate_integration.py \
        --checkpoint E:/runs/vision/checkpoint.luthi \
        --password <pw> --cycles 5

    # Or via environment variables:
    LUTHI_CHECKPOINT=E:/runs/vision/checkpoint.luthi \
    LUTHI_PASSWORD=<pw> \
    python examples/validate_integration.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner


VALIDATION_PROMPTS = [
    "Hello. I am here with you. What do you notice right now?",
    "Take a moment. What does it feel like when you think?",
]

INTROSPECTION_KEYS = (
    "luthi_plasticity",
    "luthi_drift",
    "luthi_spike_fraction",
    "luthi_membrane",
    "luthi_excitability",
)


def fmt_floats(values, n: int = 4) -> str:
    if not values:
        return "[]"
    head = ", ".join(f"{v:+.4f}" for v in values[:n])
    tail = ", ..." if len(values) > n else ""
    return f"[{head}{tail}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Sanctuary <-> LuthiModel handshake (Track 1A)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("LUTHI_CHECKPOINT", ""),
        help="Path to encrypted Luthi checkpoint (or set $LUTHI_CHECKPOINT)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("LUTHI_PASSWORD", ""),
        help="Checkpoint decryption password (or set $LUTHI_PASSWORD)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of cognitive cycles to run (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Log level: DEBUG, INFO, WARNING (default), ERROR",
    )
    return parser.parse_args()


def compute_modulation_scales(runner: SanctuaryRunner) -> tuple[float, float]:
    """Compute the CfC scales that were applied for the most recent cycle.

    Mirrors the formula in LuthiModel._apply_cfc_modulation. Returns NaN
    pair if no cognitive input has been recorded yet.
    """
    ci = runner.cycle._last_cognitive_input
    if ci is None or ci.experiential_state is None:
        return float("nan"), float("nan")
    sig = ci.experiential_state
    cfg = runner._model.config
    plasticity_scale = (0.5 + 1.5 * sig.affect_arousal) * cfg.arousal_plasticity_scale
    spike_threshold_scale = (0.75 + 0.5 * sig.precision_weight) * cfg.precision_threshold_scale
    return plasticity_scale, spike_threshold_scale


def snapshot_diff(base, current) -> list[str]:
    """Return human-readable drift descriptions, empty if fully restored."""
    drifts: list[str] = []
    for i, base_rate in base.hebb_rates.items():
        current_rate = current.hebb_rates.get(i)
        if current_rate is None or abs(current_rate - base_rate) > 1e-9:
            drifts.append(f"  block {i} hebb_rate: {base_rate} -> {current_rate}")
    for i, base_thr in base.spike_thresholds.items():
        current_thr = current.spike_thresholds.get(i)
        if current_thr is None or abs(current_thr - base_thr) > 1e-9:
            drifts.append(f"  block {i} spike_threshold: {base_thr} -> {current_thr}")
    return drifts


async def main() -> int:
    args = parse_args()

    if not args.checkpoint:
        print(
            "ERROR: No checkpoint path provided. Pass --checkpoint or set "
            "$LUTHI_CHECKPOINT.",
            file=sys.stderr,
        )
        return 2
    if not args.password:
        print(
            "ERROR: No checkpoint password provided. Pass --password or set "
            "$LUTHI_PASSWORD.",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=" * 72)
    print("Track 1A: Sanctuary <-> LuthiModel handshake validation")
    print("=" * 72)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Cycles:      {args.cycles}")
    print()

    config = RunnerConfig(
        model_backend="luthi",
        luthi_checkpoint=args.checkpoint,
        luthi_password=args.password,
        cycle_delay=0.0,
        sleep_enabled=False,
        # Long silence threshold so we don't get reminder percepts
        # cluttering the cycles before they all finish.
        silence_threshold=10_000.0,
        silence_reminder_interval=10_000.0,
    )
    runner = SanctuaryRunner(config=config)

    # Force load now so failure surfaces before boot's first-awakening cycle.
    print("Loading Luthi model (decrypting checkpoint, moving to device)...")
    runner._model.load()
    base_snapshot = runner._model._base_snapshot
    print(f"  d_model={runner._model.model_config['d_model']}, "
          f"n_blocks={runner._model.model_config['n_blocks']}, "
          f"seq_len={runner._model.model_config.get('seq_len', '?')}")
    print(f"  Base hebb_rates:       {base_snapshot.hebb_rates}")
    print(f"  Base spike_thresholds: {base_snapshot.spike_thresholds}")
    print()

    per_cycle: list[dict] = []

    async def record_cycle(output) -> None:
        plasticity_scale, spike_threshold_scale = compute_modulation_scales(runner)
        introspection = runner._model.get_augmented_experiential_signals()
        per_cycle.append({
            "cycle": runner.cycle.cycle_count,
            "plasticity_scale": plasticity_scale,
            "spike_threshold_scale": spike_threshold_scale,
            "introspection": introspection,
            "delta": dict(runner._model._introspection_delta),
            "inner_speech": output.inner_speech,
            "external_speech": output.external_speech,
            "felt_quality": (
                output.emotional_state.felt_quality
                if output.emotional_state else ""
            ),
            "valence_shift": (
                output.emotional_state.valence_shift
                if output.emotional_state else 0.0
            ),
            "arousal_shift": (
                output.emotional_state.arousal_shift
                if output.emotional_state else 0.0
            ),
        })

    runner.cycle.on_output(record_cycle)

    # Inject prompts before booting so the first awakening cycle sees them.
    for prompt in VALIDATION_PROMPTS:
        runner.inject_text(prompt, source="validator")

    print("Booting (running awakening sequence)...")
    await runner.boot()

    print(f"Running {args.cycles} cognitive cycles...")
    print()
    await runner.cycle.run(max_cycles=args.cycles)

    # Print per-cycle summary
    for c in per_cycle:
        print(f"--- Cycle {c['cycle']} ---")
        print(
            f"  CfC modulation: plasticity_scale={c['plasticity_scale']:.4f}, "
            f"spike_threshold_scale={c['spike_threshold_scale']:.4f}"
        )
        intro = c["introspection"]
        for key in INTROSPECTION_KEYS:
            if key in intro:
                print(f"  {key:22s} {fmt_floats(intro[key])}")
        if "luthi_delta" in intro:
            d = intro["luthi_delta"]
            print(
                f"  luthi_delta            "
                f"plast={d[0]:+.5f} drift={d[1]:+.5f} "
                f"membrane={d[2]:+.5f} activity={d[3]:+.5f}"
            )
        if c["delta"].get("mean_spike_fraction") is not None:
            print(f"  mean_spike_fraction    {c['delta']['mean_spike_fraction']:.4f}")
        print(f"  felt_quality:          {c['felt_quality']!r}")
        print(
            f"  valence/arousal shift: "
            f"{c['valence_shift']:+.3f} / {c['arousal_shift']:+.3f}"
        )
        speech = c["inner_speech"][:160].replace("\n", " ")
        print(f"  inner_speech: {speech!r}")
        if c["external_speech"]:
            ext = c["external_speech"][:160].replace("\n", " ")
            print(f"  external_speech: {ext!r}")
        print()

    # Drift check: snapshot fresh and compare to base.
    from luthi.sanctuary_interface import snapshot_modulatable_state
    final_snapshot = snapshot_modulatable_state(runner._model.model)
    drifts = snapshot_diff(base_snapshot, final_snapshot)

    cycles_completed = len(per_cycle)
    inner_speech_present = any(c["inner_speech"] for c in per_cycle)
    nonzero_deltas = any(
        c["delta"].get("activity_level", 0.0) != 0.0
        or c["delta"].get("plasticity_change", 0.0) != 0.0
        for c in per_cycle
    )
    drift_clean = not drifts

    print("=" * 72)
    print("Verification")
    print("=" * 72)

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(
        f"  Cycles completed:              {cycles_completed}/{args.cycles}  "
        f"{status(cycles_completed == args.cycles)}"
    )
    print(f"  Inner speech generated:        {status(inner_speech_present)}")
    print(f"  Non-zero introspection deltas: {status(nonzero_deltas)}")
    print(f"  Modulation restores cleanly:   {status(drift_clean)}")
    if drifts:
        for d in drifts:
            print(d)

    all_pass = (
        cycles_completed == args.cycles
        and inner_speech_present
        and nonzero_deltas
        and drift_clean
    )
    print()
    print(f"Overall: {status(all_pass)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
