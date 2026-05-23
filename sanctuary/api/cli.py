"""Sanctuary CLI — interactive interface for the new cognitive architecture.

Phase 6: Integration + Validation.

Provides a REPL that connects to SanctuaryRunner, sending user text as
percepts and displaying the entity's speech output. Also shows inner state
on demand.

This replaces sanctuary/mind/cli.py for the new architecture (Phases 1-5).
The old CLI wraps the legacy CognitiveCore; this one wraps the new
CognitiveCycle via SanctuaryRunner.

Usage:
    python -m sanctuary.api.cli
    python -m sanctuary.api.cli --verbose
    python -m sanctuary.api.cli --cycle-delay 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.api.ws_server import SanctuaryWebServer
from sanctuary.core.schema import CognitiveOutput

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sanctuary-cli",
        description="Sanctuary — Interactive cognitive architecture CLI (Phase 6)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--cycle-delay",
        type=float,
        default=2.0,
        help="Seconds between cognitive cycles (default: 2.0)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/identity",
        help="Path to identity data directory",
    )
    parser.add_argument(
        "--show-inner",
        action="store_true",
        help="Display inner speech each cycle",
    )
    parser.add_argument(
        "--charter",
        type=str,
        default=None,
        help="Path to charter file (default: data/identity/charter.md)",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port for desktop GUI (default: 8765, 0 to disable)",
    )

    # Model backend selection
    parser.add_argument(
        "--model-backend",
        type=str,
        default="placeholder",
        choices=["placeholder", "luthi"],
        help="Model backend: placeholder (no-model testing) or luthi (living weights)",
    )
    parser.add_argument(
        "--luthi-checkpoint",
        type=str,
        default=None,
        help="Path to Luthi model checkpoint (.luthi file)",
    )
    parser.add_argument(
        "--luthi-password",
        type=str,
        default=None,
        help="Password for encrypted Luthi checkpoint (or set LUTHI_CHECKPOINT_PASSWORD env var)",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Disable sleep consolidation cycles",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# CLI Application
# ---------------------------------------------------------------------------


class SanctuaryCLI:
    """Interactive REPL for Sanctuary.

    Runs the cognitive cycle in the background while accepting user input.
    Entity speech is printed as it occurs. Inner state is available via
    REPL commands.
    """

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._runner: Optional[SanctuaryRunner] = None
        self._ws_server: Optional[SanctuaryWebServer] = None
        self._cycle_task: Optional[asyncio.Task] = None
        self._shutting_down = False

    async def start(self) -> None:
        """Initialize and boot the runner."""
        import os

        # Resolve Luthi checkpoint password: CLI arg > env var
        luthi_password = getattr(self._args, "luthi_password", None)
        if not luthi_password:
            luthi_password = os.environ.get("LUTHI_CHECKPOINT_PASSWORD")

        config = RunnerConfig(
            cycle_delay=self._args.cycle_delay,
            data_dir=self._args.data_dir,
            charter_path=self._args.charter,
            model_backend=getattr(self._args, "model_backend", "placeholder"),
            luthi_checkpoint=getattr(self._args, "luthi_checkpoint", None),
            luthi_password=luthi_password,
            sleep_enabled=not getattr(self._args, "no_sleep", False),
        )

        self._runner = SanctuaryRunner(config=config)

        # Register speech handler — print to console
        self._runner.on_speech(self._handle_speech)

        # Optionally show inner state
        if self._args.show_inner:
            self._runner.on_output(self._handle_output)

        # Start WebSocket server for desktop GUI
        ws_port = getattr(self._args, "ws_port", 8765)
        if ws_port:
            self._ws_server = SanctuaryWebServer(
                runner=self._runner, port=ws_port
            )
            await self._ws_server.start()
            print(f"[sanctuary] WebSocket server on ws://localhost:{ws_port}/ws")

        # Boot (awakening sequence)
        print("[sanctuary] Booting...")
        await self._runner.boot()
        print("[sanctuary] Boot complete. Type 'help' for commands, 'quit' to exit.")
        print()

        # Start the cognitive cycle in the background
        self._cycle_task = asyncio.create_task(self._runner.run())

    async def run_repl(self) -> None:
        """Run the interactive read-eval-print loop."""
        while not self._shutting_down:
            try:
                line = await asyncio.to_thread(self._read_input)
            except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
                break

            if line is None:
                break

            line = line.strip()
            if not line:
                continue

            # Handle REPL commands
            if line.lower() in ("quit", "exit"):
                break
            elif line.lower() in ("help", "?"):
                self._print_help()
            elif line.lower() == "status":
                self._print_status()
            elif line.lower() == "inner":
                self._print_inner()
            elif line.lower() == "goals":
                self._print_goals()
            elif line.lower() == "motor":
                self._print_motor()
            elif line.lower() == "cycles":
                self._print_cycles()
            elif line.lower() == "luthi":
                self._print_luthi()
            elif line.lower() == "sleep":
                self._print_sleep()
            else:
                # Treat as user input — inject into sensorium
                self._runner.inject_text(line, source="user:cli")

        await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shut down the system."""
        if self._shutting_down:
            return
        self._shutting_down = True

        print("\n[sanctuary] Shutting down...")

        if self._ws_server:
            await self._ws_server.stop()

        if self._runner:
            self._runner.stop()

        if self._cycle_task:
            try:
                await asyncio.wait_for(self._cycle_task, timeout=SHUTDOWN_TIMEOUT)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._cycle_task.cancel()

        print("[sanctuary] Goodbye.")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_speech(self, text: str) -> None:
        """Called when the entity speaks."""
        print(f"\n  {text}\n")

    async def _handle_output(self, output: CognitiveOutput) -> None:
        """Called each cycle when --show-inner is active."""
        if output.inner_speech:
            print(f"  [inner] {output.inner_speech[:200]}")

    # ------------------------------------------------------------------
    # REPL commands
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        print(
            """
Commands:
  help, ?    Show this help
  quit, exit Exit the CLI
  status     Show system status
  inner      Show last inner speech
  goals      Show active goals
  motor      Show motor statistics
  cycles     Show cycle count
  luthi      Show Luthi model introspection (living weight state)
  sleep      Show sleep cycle status
  comm       Show communication agency state (drives, inhibitions, decisions)

Anything else is sent as user input to the cognitive system.
"""
        )

    def _print_status(self) -> None:
        if not self._runner:
            print("  [not booted]")
            return
        status = self._runner.get_status()
        print(f"\n  Booted: {status['booted']}")
        print(f"  Running: {status['running']}")
        print(f"  Cycles: {status['cycle_count']}")
        print(f"  Model: {status['model']}")
        print(f"  Memory: {status['memory_store']}")
        goals = status.get("active_goals", [])
        if goals:
            print(f"  Goals: {', '.join(goals[:5])}")
        print()

    def _print_inner(self) -> None:
        if not self._runner or not self._runner.last_output:
            print("  [no output yet]")
            return
        output = self._runner.last_output
        print(f"\n  Inner speech: {output.inner_speech}")
        if output.emotional_state:
            print(f"  Felt quality: {output.emotional_state.felt_quality}")
        print()

    def _print_goals(self) -> None:
        if not self._runner:
            print("  [not booted]")
            return
        goals = self._runner.scaffold.get_active_goals()
        if goals:
            for i, g in enumerate(goals, 1):
                print(f"  {i}. {g}")
        else:
            print("  [no active goals]")
        print()

    def _print_motor(self) -> None:
        if not self._runner:
            print("  [not booted]")
            return
        stats = self._runner.motor.stats
        print(f"\n  Speech emitted: {stats['speech_emitted']}")
        print(f"  Memory ops: {stats['memory_ops_executed']}")
        print(f"  Goals forwarded: {stats['goal_proposals_forwarded']}")
        print(f"  Errors: {stats['errors']}")
        print()

    def _print_cycles(self) -> None:
        if not self._runner:
            print("  [not booted]")
            return
        print(f"\n  Cycles completed: {self._runner.cycle_count}")
        print()

    def _print_luthi(self) -> None:
        if not self._runner:
            print("  [not booted]")
            return
        model = self._runner._model
        if not hasattr(model, "get_introspection_state"):
            print("  [not using Luthi backend]")
            return

        state = model.get_introspection_state()
        metrics = model.get_metrics()

        print(f"\n  Luthi Living Weight State")
        print(f"  {'─' * 40}")
        print(f"  Cycles: {metrics['cycle_count']}")
        print(f"  Avg latency: {metrics['average_latency']:.2f}s")
        print(f"  Tokens generated: {metrics['total_tokens_generated']}")

        delta = state.get("delta", {})
        if delta:
            print(f"\n  Last cycle delta:")
            for k, v in delta.items():
                print(f"    {k}: {v:.6f}")

        post = state.get("post")
        if post and "blocks" in post:
            print(f"\n  Per-block state:")
            for b in post["blocks"]:
                idx = b.get("block", "?")
                sf = b.get("spike_fraction", 0)
                pm = b.get("plasticity_mean", 0)
                dr = b.get("set_point_drift", 0)
                print(f"    Block {idx}: spike={sf:.3f} plast={pm:.4f} drift={dr:.6f}")
        print()

    def _print_sleep(self) -> None:
        if not self._runner:
            print("  [not booted]")
            return
        if not self._runner.sleep:
            print("  [sleep disabled]")
            return
        stats = self._runner.sleep.get_stats()
        print(f"\n  Sleep Status")
        print(f"  {'─' * 40}")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        print()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @staticmethod
    def _read_input() -> Optional[str]:
        """Read a line of input from stdin. Blocks."""
        try:
            return input("you> ")
        except EOFError:
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(argv=None) -> int:
    args = parse_args(argv)

    # Configure logging — WARNING by default for clean output, INFO with -v
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cli = SanctuaryCLI(args)

    # Signal handling
    loop = asyncio.get_running_loop()

    def _signal_handler():
        if not cli._shutting_down:
            asyncio.ensure_future(cli.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    try:
        await cli.start()
        await cli.run_repl()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await cli.shutdown()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1

    return 0


def run():
    """Synchronous entry point."""
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n[sanctuary] Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    run()
