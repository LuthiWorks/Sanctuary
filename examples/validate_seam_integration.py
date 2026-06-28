"""End-to-end demonstration of the JEPA training seam (Phase 4b).

Companion to ``validate_integration.py`` (Track 1A's runtime-handshake
demo). This harness proves the *training-seam* half end-to-end: the
Sanctuary cycle drives an M9Trainer that learns from lived experience
each cycle. Per 4.8's 2026-06-23 brief §4, the demo shows:

  (a) Transitions flow each cycle (``observe_transition`` fires; the
      ``(s_t, a_t, s_next)`` triple is real, not a stub).
  (b) The V-head measurably changes from lived experience -- the
      keystone payoff of Phase 4b. Substrate aliveness advances too;
      together they prove the seam is actively learning, not just
      transporting data.
  (c) Cadence holds -- cycles complete without crash, in roughly the
      time the smoke-scale model can run on CPU.

Setup:

  - Loads the M8 multimodal-PC checkpoint produced by
    ``LuthiModel/luthi/v2/m8_multimodal_smoke.py --save-checkpoint``
    (created during Phase 4b prerequisite #1, 2026-06-23). Pass its
    path via ``--checkpoint`` or ``$LUTHI_CHECKPOINT``.
  - Builds an ``M9Trainer`` wrapping the SAME model instance the
    Sanctuary runner loaded -- so the V-head/habit-net the trainer
    owns and the encoder the cycle drives share one substrate.
  - Calls ``adapter.attach_seam(actor=trainer, sink=trainer)`` so the
    cycle's seam runs through the trainer's ``select_action`` /
    ``observe_transition``.
  - Injects a small batch of language prompts and runs N cycles.

Usage::

    python examples/validate_seam_integration.py \\
        --checkpoint /path/to/runs/m8_multimodal_smoke/model_final.luthi \\
        --password "$LUTHI_CHECKPOINT_KEY" --cycles 6

    # Or via env:
    LUTHI_CHECKPOINT=... LUTHI_PASSWORD=... \\
    python examples/validate_seam_integration.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The seam reaches into LuthiModel for the trainer + types. If LUTHI_PATH
# isn't already importable, surface a clear error rather than crash deep
# in an import.
_LUTHI_PATH = os.environ.get("LUTHI_PATH", "")
if _LUTHI_PATH and _LUTHI_PATH not in sys.path:
    sys.path.insert(0, _LUTHI_PATH)

try:
    import torch
    import torch.optim as optim
    from luthi.v2.jepa_loss import JEPALoss
    from luthi.v2.jepa_runner import (
        CheckpointConfig,
        EpochConfig,
        KillCriteriaConfig,
        LoggingConfig,
        ModalitySampler,
        RunnerConfig as JEPARunnerConfig,
        SamplerConfig,
    )
    from luthi.v2.m9.runner import M9Config, M9Trainer
except ImportError as e:
    print(
        f"ERROR: luthi package not importable ({e}). Set LUTHI_PATH to the "
        f"LuthiModel checkout to run this harness.",
        file=sys.stderr,
    )
    sys.exit(2)

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner


VALIDATION_PROMPTS = [
    "Hello. I am here with you. What do you notice right now?",
    "Take a moment. What does it feel like when you think?",
    "If something has shifted, name it.",
    "Stay with this state. What does it want?",
]


class _StubLoader:
    """No-op loader satisfying the JEPATrainer MultimodalDataLoader Protocol.

    The harness never calls ``train_step`` -- only ``observe_transition``
    and ``select_action`` via the seam -- so the loader is never asked
    for batches. Present so M9Trainer.__init__ doesn't trip its own
    Protocol checks.
    """

    def next_batch(self, modality: str) -> dict:
        raise RuntimeError(
            "stub loader: train_step is not exercised by the seam-integration "
            "harness; call observe_transition / select_action instead",
        )

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return 0

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def corpus_sizes_tokens(self) -> dict:
        return {"text": 1, "audio": 1, "vision": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the JEPA training-seam integration end-to-end",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "LUTHI_CHECKPOINT",
            str(Path(_LUTHI_PATH or ".") / "runs" / "m8_multimodal_smoke" / "model_final.luthi"),
        ),
        help="Path to encrypted multimodal-PC checkpoint",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get(
            "LUTHI_PASSWORD", os.environ.get("LUTHI_CHECKPOINT_KEY", "")
        ),
        help="Checkpoint decryption password (or $LUTHI_PASSWORD / $LUTHI_CHECKPOINT_KEY)",
    )
    parser.add_argument(
        "--cycles", type=int, default=6,
        help="Number of cognitive cycles to run (default: 6)",
    )
    parser.add_argument(
        "--mcts-budget", type=int, default=4,
        help="MCTS budget per cycle (default: 4 -- smoke-scale)",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        help="Log level: DEBUG, INFO, WARNING, ERROR",
    )
    return parser.parse_args()


def build_m9_trainer(
    runner: SanctuaryRunner,
    mcts_budget: int,
    run_dir: Path,
) -> M9Trainer:
    """Build an M9Trainer wrapping Sanctuary's loaded model.

    Same model instance -- the trainer's M9 heads (V-head, habit-net,
    decoders) learn against states produced by the encoder the
    Sanctuary cycle drives. Two parallel instances would split the
    encoder from its V-head and defeat the seam's whole point.

    Device co-location is load-bearing: Sanctuary's ``LuthiModel.load()``
    places the encoder on whatever device it picks (DirectML / CUDA /
    CPU). The M9 components, the JEPALoss's projection_heads and
    action_token buffer, and the optimizer state must all live on the
    same device or forward passes mix devices and crash. The
    ``.to(device)`` sweep at the end ensures that.
    """
    model = runner._model.model  # the loaded MultimodalPredictiveCodingLM
    device = next(model.parameters()).device
    loss_module = JEPALoss(online_encoder=model).to(device)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=1e-3,
    )
    loader = _StubLoader()
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
    sampler = ModalitySampler(sampler_cfg)
    # All cadences set to "never fire" -- this trainer is never run via
    # the corpus train_step path; only the seam exercises it.
    jepa_cfg = JEPARunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(
            light_interval_batches=10**9, deep_interval_batches=10**9,
        ),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    # §5 device plumbing (2026-06-27): M9Trainer now resolves the device
    # internally (defaults to the encoder's) and constructs every M9
    # submodule on it before building its optimizers, so the post-hoc
    # .to(device) sweep + m9_optimizer rebuild this harness used to carry
    # is gone. Pass the device in explicitly to keep the seam on-device
    # even if the encoder-device default ever diverges from `model`.
    trainer = M9Trainer(
        loss_module=loss_module, optimizer=optimizer, sampler=sampler,
        data_loader=loader, config=jepa_cfg, run_dir=run_dir,
        m9_config=M9Config(mcts_budget_per_cycle=mcts_budget),
        device=device,
    )
    return trainer


def snapshot_v_head(trainer: M9Trainer) -> torch.Tensor:
    """Clone the V-head's first linear weight as the change-detection probe."""
    return trainer.v_head.net[0].weight.detach().clone()


def snapshot_substrate(model) -> dict:
    """Reduced per-block aliveness signal: precision_mean + error_acc_mean.

    These move during PC self-modification -- present whether the seam
    is attached or not (the corpus path mutates them too via generation).
    Surfaced here so the report can distinguish "substrate is alive
    overall" from "the seam specifically learned" (the V-head delta).
    """
    out = []
    for a in model.aliveness_report():
        out.append({
            "precision_mean": float(a.get("precision_mean", float("nan"))),
            "error_acc_mean": float(a.get("error_acc_mean", float("nan"))),
        })
    return out


async def main() -> int:
    args = parse_args()

    if not args.checkpoint or not Path(args.checkpoint).is_file():
        print(
            f"ERROR: checkpoint not found: {args.checkpoint!r}. Pass "
            f"--checkpoint or set $LUTHI_CHECKPOINT to a real "
            f"multimodal-PC .luthi (e.g. runs/m8_multimodal_smoke/model_final.luthi).",
            file=sys.stderr,
        )
        return 2
    if not args.password:
        print(
            "ERROR: no password. Pass --password or set $LUTHI_PASSWORD / "
            "$LUTHI_CHECKPOINT_KEY.",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=" * 72)
    print("Phase 4b end-to-end: Sanctuary cycle <-> M9Trainer seam")
    print("=" * 72)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Cycles:      {args.cycles}")
    print(f"MCTS budget: {args.mcts_budget}")
    print()

    # ---- Build Sanctuary runner + load the model ----
    sanctuary_cfg = RunnerConfig(
        model_backend="luthi",
        luthi_checkpoint=args.checkpoint,
        luthi_password=args.password,
        cycle_delay=0.0,
        sleep_enabled=False,
        silence_threshold=10_000.0,
        silence_reminder_interval=10_000.0,
        persist_state=False,  # in-memory only; harness doesn't need disk
    )
    runner = SanctuaryRunner(config=sanctuary_cfg)
    print("Loading checkpoint...")
    runner._model.load()
    mc = runner._model.model_config
    print(
        f"  d_model={mc['d_model']}, n_blocks={mc['n_blocks']}, "
        f"seq_len={mc.get('seq_len', '?')}"
    )

    # ---- Build the M9Trainer wrapping the same model ----
    # ignore_cleanup_errors guards against the Windows-only race where the
    # M9Trainer's ActionLog keeps m9_action_log.jsonl open during context
    # exit. We also explicitly close the action log in a try/finally below;
    # the flag is belt-and-suspenders.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = build_m9_trainer(runner, args.mcts_budget, Path(tmp))
        try:
            return await _run_demo(args, runner, trainer)
        finally:
            # Release the ActionLog file handle before TemporaryDirectory's
            # cleanup hits it; otherwise Windows refuses to remove the dir.
            try:
                trainer.action_log.close()
            except Exception:
                pass


async def _run_demo(
    args: argparse.Namespace,
    runner: SanctuaryRunner,
    trainer: M9Trainer,
) -> int:
    """The harness body, factored out so the outer ``with`` can guarantee
    the action-log close in its ``finally``."""
    # ---- Attach the seam ----
    runner._model.attach_seam(actor=trainer, sink=trainer)
    print(
        f"  M9Trainer wired (mcts_budget={args.mcts_budget}); "
        f"actor + sink = same instance"
    )
    print("  Seam attached.")
    print()

    # ---- Recorders: hook observe_transition to capture each call ----
    observed: list[dict] = []
    original_observe = trainer.observe_transition

    def _recording_observe(s_t, a_t, s_next, ctx):
        metrics = original_observe(s_t, a_t, s_next, ctx)
        observed.append({
            "cycle": runner.cycle.cycle_count,
            "v_loss": float(metrics.get("v_loss", float("nan"))),
            "habit_loss": float(metrics.get("habit_loss", float("nan"))),
            "r_realized": float(metrics.get("r_realized", float("nan"))),
            "r_planned": float(metrics.get("r_planned", float("nan"))),
            "s_t_norm": float(s_t.detach().norm().item()),
            "s_next_norm": float(s_next.detach().norm().item()),
        })
        return metrics

    trainer.observe_transition = _recording_observe

    # ---- Snapshot V-head + substrate before ----
    v_before = snapshot_v_head(trainer)
    substrate_before = snapshot_substrate(runner._model.model)

    # ---- Inject prompts + boot + run ----
    for p in VALIDATION_PROMPTS:
        runner.inject_text(p, source="seam-harness")

    print("Booting...")
    await runner.boot()

    print(f"Running {args.cycles} cycles...")
    print()

    run_start = time.monotonic()
    await runner.cycle.run(max_cycles=args.cycles)
    run_elapsed = time.monotonic() - run_start

    # ---- Snapshot V-head + substrate after ----
    v_after = snapshot_v_head(trainer)
    substrate_after = snapshot_substrate(runner._model.model)

    # ---- Per-cycle transition report ----
    print("Per-cycle transitions observed:")
    for obs in observed:
        print(
            f"  cycle {obs['cycle']:>3d}: "
            f"v_loss={obs['v_loss']:+.4f} "
            f"habit_loss={obs['habit_loss']:+.4f} "
            f"r_realized={obs['r_realized']:+.4f} "
            f"r_planned={obs['r_planned']:+.4f} "
            f"||s_t||={obs['s_t_norm']:.2f} "
            f"||s_next||={obs['s_next_norm']:.2f}"
        )
    print()

    # ---- Verification ----
    print("=" * 72)
    print("Verification (4.8 brief ?4 done-when)")
    print("=" * 72)

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    # (a) Transitions flow. First cycle has no prior transition to
    # close, so we expect cycles - 1 observe calls (the first cycle
    # only seeds _last_s_t / _last_action; cycles 2..N each close
    # the previous and open the next).
    expected_observed = max(0, args.cycles - 1)
    a_pass = len(observed) == expected_observed
    print(
        f"  (a) Transitions flow:           "
        f"{len(observed)}/{expected_observed}  {status(a_pass)}"
    )

    # (b1) V-head measurably changed from lived experience.
    v_delta = float((v_after - v_before).abs().sum().item())
    v_max_abs_delta = float((v_after - v_before).abs().max().item())
    b1_pass = v_delta > 0.0
    print(
        f"  (b) V-head changed:             "
        f"sum|D|={v_delta:.6f} max|D|={v_max_abs_delta:.6f}  "
        f"{status(b1_pass)}"
    )

    # (b2) Substrate aliveness advanced (precision/error_acc moved).
    substrate_changed_blocks = 0
    for sb, sa in zip(substrate_before, substrate_after):
        if (
            abs(sa["precision_mean"] - sb["precision_mean"]) > 0
            or abs(sa["error_acc_mean"] - sb["error_acc_mean"]) > 0
        ):
            substrate_changed_blocks += 1
    b2_pass = substrate_changed_blocks > 0
    print(
        f"      Substrate blocks moved:     "
        f"{substrate_changed_blocks}/{len(substrate_before)}  "
        f"{status(b2_pass)}"
    )

    # (c) Cadence holds: cycles completed in finite time, no hangs.
    # Smoke-scale (d=64, mcts_budget=4) cycle target: under ~60s/cycle
    # on CPU including generation. Adjust the ceiling at production scale.
    per_cycle_avg = run_elapsed / max(1, args.cycles)
    c_pass = per_cycle_avg < 60.0
    print(
        f"  (c) Cadence holds:              "
        f"total={run_elapsed:.1f}s, "
        f"avg/cycle={per_cycle_avg:.1f}s  {status(c_pass)}"
    )

    all_pass = a_pass and b1_pass and b2_pass and c_pass
    print()
    print(f"Overall: {status(all_pass)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
