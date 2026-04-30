#!/usr/bin/env python3
"""
Run Sanctuary with Ollama — Interactive Chat (hardened).

This script boots the full Sanctuary cognitive architecture with a real
LLM (via Ollama) for language processing. It connects to your locally
running Ollama instance and uses whichever model you have pulled.

Includes:
  - Signal handling (SIGTERM / SIGINT) for clean container & terminal shutdown
  - Shutdown timeout to prevent indefinite hangs
  - Categorised error display (verbose mode shows full tracebacks)

Prerequisites:
    1. Install Python dependencies:
       pip install chromadb sentence-transformers torch pydantic numpy scikit-learn

    2. Install and start Ollama (https://ollama.com):
       ollama serve

    3. Pull a model:
       ollama pull gemma3:12b

Usage:
    python run_with_ollama.py                    # uses gemma3:12b by default
    python run_with_ollama.py --model llama3.2   # use a different model
    python run_with_ollama.py --mock-perception   # skip sentence-transformers
"""

import argparse
import asyncio
import logging
import signal
import sys
import traceback
from pathlib import Path

# Add the sanctuary package to the path
sys.path.insert(0, str(Path(__file__).parent / "sanctuary"))

from mind.client import SanctuaryAPI

# Maximum seconds to wait for graceful shutdown.
SHUTDOWN_TIMEOUT = 30.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sanctuary with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_with_ollama.py
  python run_with_ollama.py --model llama3.2
  python run_with_ollama.py --model gemma3:12b --ollama-url http://localhost:11434
  python run_with_ollama.py --mock-perception  (skip sentence-transformers requirement)
        """
    )
    parser.add_argument(
        "--model", default="gemma3:12b",
        help="Ollama model name (default: gemma3:12b)"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--mock-perception", action="store_true",
        help="Use mock perception (skips sentence-transformers/torch requirement)"
    )
    parser.add_argument(
        "--input-model", default=None,
        help="Separate Ollama model for input parsing (default: same as --model)"
    )
    parser.add_argument(
        "--cycle-rate", type=float, default=10.0,
        help="Cognitive cycle rate in Hz (default: 10.0)"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="LLM generation timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--shutdown-timeout", type=float, default=SHUTDOWN_TIMEOUT,
        help=f"Max seconds for graceful shutdown (default: {SHUTDOWN_TIMEOUT})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def build_config(args):
    """Build the CognitiveCore config from command-line arguments."""
    input_model = args.input_model or args.model

    config = {
        "cognitive_core": {
            "cycle_rate_hz": args.cycle_rate,
            "attention_budget": 100,
            "max_queue_size": 100,
            "log_interval_cycles": 50,

            # Perception — real embeddings by default, mock if requested
            "perception": {
                "mock_mode": args.mock_perception,
                "mock_embedding_dim": 384,
            },

            # LLM — both input and output go through Ollama
            "input_llm": {
                "use_real_model": True,
                "backend": "ollama",
                "model_name": input_model,
                "base_url": args.ollama_url,
                "temperature": 0.3,
                "max_tokens": 512,
                "timeout": args.timeout,
            },
            "output_llm": {
                "use_real_model": True,
                "backend": "ollama",
                "model_name": args.model,
                "base_url": args.ollama_url,
                "temperature": 0.7,
                "max_tokens": 500,
                "timeout": args.timeout,
            },

            # Checkpointing
            "checkpointing": {
                "enabled": True,
                "auto_save": False,
                "checkpoint_on_shutdown": True,
            },
        }
    }

    return config


def _format_error(exc: Exception, verbose: bool = False) -> str:
    """Return a user-friendly error string."""
    prefix = "Error"
    if isinstance(exc, ConnectionError):
        prefix = "Connection error"
    elif isinstance(exc, TimeoutError) or isinstance(exc, asyncio.TimeoutError):
        prefix = "Operation timed out"
    elif isinstance(exc, RuntimeError):
        prefix = "Runtime error"
    msg = f"{prefix}: {exc}"
    if verbose:
        msg += "\n" + traceback.format_exc()
    return msg


async def interactive_chat(api: SanctuaryAPI, model_name: str, shutdown_event: asyncio.Event, verbose: bool = False):
    """Run the interactive chat loop."""
    print()
    print("=" * 60)
    print("  SANCTUARY - Interactive Chat")
    print(f"  Model: {model_name} (via Ollama)")
    print("=" * 60)
    print()
    print("  Type a message and press Enter to chat.")
    print("  Type 'quit' or press Ctrl+C to exit.")
    print("  Type 'status' to see cognitive metrics.")
    print()

    while not shutdown_event.is_set():
        try:
            user_input = await asyncio.to_thread(input, "You: ")
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.lower() == "status":
            metrics = api.get_metrics()
            cog = metrics.get("cognitive_core", {})
            conv = metrics.get("conversation", {})
            print()
            print(f"  Cycles run:       {cog.get('total_cycles', '?')}")
            print(f"  Avg cycle time:   {cog.get('avg_cycle_time_ms', 0):.1f} ms")
            print(f"  Conversation turns: {conv.get('total_turns', '?')}")
            print()
            continue

        try:
            print()
            print("  (thinking...)")
            turn = await api.chat(user_input)
            print(f"\nSanctuary: {turn.system_response}")

            # Show emotional state if present
            if turn.emotional_state:
                emotions = turn.emotional_state
                valence = emotions.get("valence", 0)
                arousal = emotions.get("arousal", 0)
                mood = "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
                energy = "high" if arousal > 0.5 else "low" if arousal < -0.2 else "moderate"
                print(f"  [mood: {mood}, energy: {energy}]")
            print()

        except Exception as e:
            print(f"\n  {_format_error(e, verbose)}\n")


async def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    if not args.verbose:
        logging.getLogger("mind").setLevel(logging.WARNING)
        logging.getLogger("sanctuary").setLevel(logging.WARNING)
    else:
        logging.getLogger("mind").setLevel(logging.DEBUG)
        logging.getLogger("sanctuary").setLevel(logging.DEBUG)

    config = build_config(args)

    # ------- Signal handling -------
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler(sig, _frame=None):
        sig_name = signal.Signals(sig).name
        print(f"\n  Received {sig_name}, shutting down gracefully...")
        shutdown_event.set()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler, sig)
    else:
        signal.signal(signal.SIGINT, _signal_handler)

    print()
    print("Starting Sanctuary...")
    print(f"  Ollama URL:  {args.ollama_url}")
    print(f"  Model:       {args.model}")
    if args.input_model:
        print(f"  Input model: {args.input_model}")
    print(f"  Perception:  {'mock' if args.mock_perception else 'real (sentence-transformers)'}")
    print()

    # Quick Ollama connectivity check before booting the whole system
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(f"{args.ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json as _json
            data = _json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            print(f"  Ollama connected. Available models: {', '.join(models)}")
    except urllib.error.URLError:
        print(f"  WARNING: Cannot reach Ollama at {args.ollama_url}")
        print(f"  Make sure Ollama is running (ollama serve)")
        print(f"  Continuing anyway — will use mock responses as fallback.")
    print()

    print("  Booting cognitive architecture...")
    api = SanctuaryAPI(config)

    try:
        await api.start()
        print("  Cognitive loop running.")
        print()
        await interactive_chat(api, args.model, shutdown_event, args.verbose)
    except Exception as e:
        print(f"\n  {_format_error(e, args.verbose)}")

    finally:
        print("\nShutting down Sanctuary...")
        try:
            await asyncio.wait_for(
                api.stop(),
                timeout=args.shutdown_timeout,
            )
        except asyncio.TimeoutError:
            print(f"  Shutdown timed out after {args.shutdown_timeout}s — forcing exit")
        except Exception as stop_err:
            print(f"  Error during shutdown: {stop_err}")
        print("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
