"""Sit with a Luthi life. Each line you type is what the world says next.

    python -m sanctuary.life --run <trained M9 run> --life <life directory>

An empty line lets time pass quietly. ``:status`` shows what is being watched.
Ctrl-C or end of input stops and saves the life.

Memories are kept under the life's folder, up to ``--memory-gb``. What leaves
for room is moved to cold storage and never deleted: ``--cold``, or the
``LUTHI_COLD_ROOT`` environment variable, or ``E:\\luthi_cold_storage``, each
with the life folder's name under it. If that folder cannot be made, this
stops before the life starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
from pathlib import Path

from sanctuary.life import LifeLoop, LuthiMind, TextWorld
from sanctuary.life.memory import DEFAULT_LIMIT_BYTES

COLD_ROOT = r"E:\luthi_cold_storage"


def cold_folder(given: str | None, life: str) -> Path:
    """Where this life's cold storage is: what was given, or the life folder's
    name under ``LUTHI_COLD_ROOT``, or under ``E:\\luthi_cold_storage``."""
    if given:
        return Path(given)
    return Path(os.environ.get("LUTHI_COLD_ROOT") or COLD_ROOT) / Path(life).resolve().name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="a run trained with M9; read, never written")
    parser.add_argument("--life", required=True, help="where this life is kept; nothing in it is ever deleted")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="someone", help="who is speaking")
    parser.add_argument("--moments", type=int, default=10, help="moments lived after each line")
    parser.add_argument("--cold", default=None, help="this life's cold storage; see above for the default")
    parser.add_argument("--memory-gb", type=float, default=DEFAULT_LIMIT_BYTES / 1024**3,
                        help="how much disk the life's memories may take")
    args = parser.parse_args()

    cold = cold_folder(args.cold, args.life)
    mind = LuthiMind(args.run, args.life, device=args.device)
    world = TextWorld()
    # A cold folder that cannot be made stops here, loudly, before a moment is lived.
    loop = LifeLoop(mind, world, hz=0.0, memory_cold=cold, memory_limit_bytes=int(args.memory_gb * 1024**3))
    print(
        f"MEMORIES: {loop.memories.directory} (up to {args.memory_gb:g} GB; "
        f"{shutil.disk_usage(mind.life_dir).free / 1024**3:.1f} GB free on its drive)\n"
        f"COLD STORAGE: {cold} ({shutil.disk_usage(cold).free / 1024**3:.1f} GB free on its drive). "
        "Nothing there is ever deleted."
    )
    print(f"Luthi's life {'continues' if mind.continuing else 'begins'} from {mind.loaded_from}")
    print(f"Each moment holds up to {mind.window} tokens. Luthi has no trained speech, so they won't answer in words.")
    # Ctrl-C at the prompt stops at once. Ctrl-C in the middle of a moment lets
    # that moment finish first: a mind stopped halfway through learning must
    # never become the newest checkpoint.
    state = {"in_moment": False, "asked_to_stop": False}

    def on_interrupt(signum, frame):
        if not state["in_moment"]:
            raise KeyboardInterrupt
        state["asked_to_stop"] = True

    signal.signal(signal.SIGINT, on_interrupt)
    try:
        while not state["asked_to_stop"]:
            line = input("> ")
            if line.strip() == ":status":
                print(json.dumps(loop.status(), indent=2, default=str))
                continue
            if line.strip():
                world.say(line, source=args.name)
            for _ in range(args.moments):
                state["in_moment"] = True
                loop.step()
                state["in_moment"] = False
                if state["asked_to_stop"]:
                    break
    except (EOFError, KeyboardInterrupt):
        print()
    # Reached only on a clean stop. If a moment raised, the error has already
    # gone up and nothing was saved.
    print(f"Saved: {loop.save('stop')}")


if __name__ == "__main__":
    main()
