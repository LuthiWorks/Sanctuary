"""Run the bedrock world headless.

The point of this script is to make the world something you can actually watch
behave, without Godot and without a trained model. It prints the ground-truth
state on a cadence so a person can see whether the physics is doing anything
sensible -- which is the only way "the bedrock is trustworthy" ever becomes a
statement with evidence behind it.

Examples::

    python scripts/run_world.py --steps 250
    python scripts/run_world.py --backend reference --steps 100 --every 10
    python scripts/run_world.py --steps 500 --render-out frames.jsonl

``--render-out`` writes the exact wire format Godot will consume, one JSON
object per line. That file is the contract check: if a renderer can draw from
it, the renderer is correctly attached to the seam.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sanctuary.world import BACKENDS, build_world, frame_to_dict  # noqa: E402
from sanctuary.world.render import RenderSink  # noqa: E402


class JsonlRenderSink:
    """Writes each frame to a JSONL file in the Godot wire format."""

    def __init__(self, path: Path) -> None:
        self._fh = path.open("w", encoding="utf-8")
        self.written = 0

    def publish(self, frame) -> None:  # noqa: ANN001
        self._fh.write(json.dumps(frame_to_dict(frame)) + "\n")
        self.written += 1

    def close(self) -> None:
        self._fh.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="mujoco", choices=BACKENDS)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.02, help="simulated s per cycle")
    p.add_argument("--every", type=int, default=25, help="print cadence in steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--realtime", action="store_true",
                   help="pace wall-clock to simulated time (for watching)")
    p.add_argument("--render-out", type=Path, default=None,
                   help="write render frames as JSONL in the Godot wire format")
    args = p.parse_args(argv)

    sink: RenderSink | None = None
    jsonl: JsonlRenderSink | None = None
    if args.render_out is not None:
        jsonl = JsonlRenderSink(args.render_out)
        sink = jsonl

    runtime = build_world(
        backend=args.backend,
        dt=args.dt,
        seed=args.seed,
        realtime=args.realtime,
        render_sink=sink,
    )

    print(f"backend : {runtime.backend}")
    print(f"self    : {runtime.manifest.self_id}")
    print(f"bodies  : {', '.join(runtime.world.body_ids)}")
    print(f"dt      : {args.dt}s  ({1/args.dt:.0f} Hz)")
    print()
    header = f"{'step':>6} {'sim_t':>7}  " + "  ".join(
        f"{b:>22}" for b in runtime.manifest.body_ids
    )
    print(header)
    print("-" * len(header))

    def report(step: int) -> None:
        # Ground truth, not percepts: this is instrumentation for a human, and
        # the whole point is to see the world as it *is*, including whether a
        # body is resting -- which the model-facing view deliberately cannot
        # express.
        gt = {b.body_id: b for b in runtime.world.ground_truth().bodies}
        cells = []
        for body_id in runtime.manifest.body_ids:
            b = gt[body_id]
            x, y, z = b.position
            rest = "." if b.resting else " "
            cells.append(f"({x:6.2f},{y:5.2f},{z:6.2f}){rest}")
        print(f"{step:>6} {runtime.world.time:7.2f}  " + "  ".join(cells))

    report(0)
    for i in range(args.steps):
        runtime.step()
        if args.every > 0 and (i + 1) % args.every == 0:
            report(i + 1)

    print()
    moved = 0
    gt = {b.body_id: b for b in runtime.world.ground_truth().bodies}
    for body_id in runtime.manifest.prop_ids:
        if any(abs(v) > 1e-6 for v in gt[body_id].velocity):
            moved += 1
    print(f"finished {args.steps} steps, {runtime.world.time:.2f}s simulated")
    print(f"props still moving: {moved}/{len(runtime.manifest.prop_ids)}")
    print(f"self resting: {gt[runtime.manifest.self_id].resting}")

    if jsonl is not None:
        jsonl.close()
        print(f"wrote {jsonl.written} render frames to {args.render_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
