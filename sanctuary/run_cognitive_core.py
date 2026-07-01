"""Entry point for running the Sanctuary cognitive architecture.

Docker CMD entry point. It:
  1. Starts the health check HTTP server
  2. Starts the resource monitor
  3. Boots the SanctuaryRunner — state is restored from data_dir
     automatically at runner construction (journal, world graph,
     experiential layer, identity files all load from disk if
     present)
  4. Runs the cognitive cycle until shutdown
  5. On shutdown, calls runner.save_state() to flush the experiential
     layer (everything else auto-persists on each write)

Handles SIGTERM/SIGINT for graceful container shutdown.

Usage::

    python -m sanctuary.run_cognitive_core
    python -m sanctuary.run_cognitive_core --port 8000
    python -m sanctuary.run_cognitive_core --no-health-server
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sanctuary",
        description="Sanctuary — Cognitive architecture runner with health monitoring",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SANCTUARY_HEALTH_PORT", "8000")),
        help="Health check server port (default: 8000, or SANCTUARY_HEALTH_PORT env)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("SANCTUARY_HEALTH_HOST", "127.0.0.1"),
        help=(
            "Health check server host (default: 127.0.0.1, loopback only). "
            "The in-container HEALTHCHECK curls localhost, so this default "
            "keeps liveness working while /status and /metrics stay off the "
            "network. Set SANCTUARY_HEALTH_HOST=0.0.0.0 to expose externally."
        ),
    )
    parser.add_argument(
        "--no-health-server",
        action="store_true",
        help="Disable the health check HTTP server",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("SANCTUARY_IDENTITY_DIR", "data/identity"),
        help="Path to identity data directory",
    )
    parser.add_argument(
        "--cycle-delay",
        type=float,
        default=float(os.environ.get("SANCTUARY_CYCLE_DELAY", "2.0")),
        help="Seconds between cognitive cycles (default: 2.0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    """Main async entry point."""
    from sanctuary.api.health import HealthServer
    from sanctuary.api.resource_monitor import ResourceMonitor
    from sanctuary.api.runner import RunnerConfig, SanctuaryRunner

    # --- Resource monitor ---
    resource_monitor = ResourceMonitor()

    # --- Runner configuration ---
    config = RunnerConfig(
        cycle_delay=args.cycle_delay,
        data_dir=args.data_dir,
    )
    runner = SanctuaryRunner(config=config)

    # --- Health server ---
    health_server: HealthServer | None = None
    if not args.no_health_server:
        health_server = HealthServer(
            runner=runner,
            resource_monitor=resource_monitor,
            host=args.host,
            port=args.port,
        )
        await health_server.start()
        logger.info("Health server started on %s:%d", args.host, args.port)

    # --- Signal handling for graceful shutdown ---
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()
        runner.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    # --- Boot and run ---
    try:
        logger.info("Booting Sanctuary...")
        await runner.boot()
        logger.info("Boot complete — entering cognitive cycle")

        # Run cognitive cycle (blocks until stopped)
        await runner.run()

    except asyncio.CancelledError:
        logger.info("Cognitive cycle cancelled")
    except Exception as exc:
        logger.error("Fatal error in cognitive cycle: %s", exc, exc_info=True)
        return 1
    finally:
        # --- Shutdown sequence ---
        logger.info("Shutting down...")

        # Save runner state before exit. The journal, world graph, and
        # identity files auto-persist on every write; the CfC
        # experiential layer requires an explicit save call. The runner
        # handles all of that internally — call save_state once.
        try:
            runner.save_state()
        except Exception as exc:
            logger.error("save_state failed during shutdown: %s", exc)

        # Stop health server
        if health_server:
            await health_server.stop()

        logger.info("Shutdown complete")

    return 0


def main(argv=None) -> int:
    """Synchronous entry point (Docker CMD target)."""
    args = parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Received interrupt — exiting")
        return 0


if __name__ == "__main__":
    sys.exit(main())
