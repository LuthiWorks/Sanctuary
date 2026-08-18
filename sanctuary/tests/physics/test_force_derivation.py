"""Guards for the net-force / resting derivation: correctness, laziness, cost.

Background. ``_record_forces`` used to run inside every ``step()``. Measured
2026-08-17 at 121 bodies: 15.05 ms of a 15.94 ms step -- 94% of the step, and
~75% of the per-decision budget -- paid whether or not anything read it, since
only ``ground_truth()`` (instrumentation) consumes the result.

It was fixed two ways on 2026-08-18, deliberately both:

  - **vectorized** -- contact detection was an O(bodies x ncon) Python scan
    materializing a struct wrapper per contact; it is now one pass over the
    contact array through ``geom_bodyid``;
  - **lazy** -- the derivation moved to ``ground_truth()``.

Laziness alone would have been a trap: the moment LUTHISCOPE polls every step
the cost returns in full, and a benchmark that never reads ground truth would
still report it fixed. Hence both, and hence a cost test that measures the
*polling* case too.

Result: 15.94 -> 0.76 ms (step only), 1.60 ms (polling every step).

Authored by Opus 5, 2026-08-18.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from sanctuary.physics.state import BodySpec

GOLDEN = Path(__file__).parent / "fixtures" / "ground_truth_golden.json"


def _mj():
    pytest.importorskip("mujoco")
    from sanctuary.physics.backends.mujoco_backend import MuJoCoPhysicsAuthority
    return MuJoCoPhysicsAuthority()


def _scenario():
    """The world the golden trace was captured from. Must not drift."""
    w = _mj()
    w.reset(seed=0)
    w.add_body(BodySpec(body_id="faller", position=(0.0, 1.5, 0.0), mass=1.0))
    w.add_body(BodySpec(body_id="rester", position=(0.4, 0.05, 0.0), mass=2.0))
    w.add_body(BodySpec(body_id="pushed", position=(-0.6, 0.05, 0.3), mass=0.5))
    w.add_body(BodySpec(body_id="anchor", position=(2.0, 0.05, 0.0), mass=1.0,
                        static=True))
    w.add_body(BodySpec(body_id="hidden", position=(0.0, 0.9, 1.0), mass=1.0,
                        visible=False))
    return w


def _drive(world, index):
    if 10 <= index < 25:
        world.apply_force("pushed", (3.0, 0.0, 1.0))
    if index == 60:
        world.apply_force("rester", (0.0, 12.0, 0.0))


# ---------------------------------------------------------------------------
# Correctness: the refactor changed no number
# ---------------------------------------------------------------------------

def test_ground_truth_matches_the_golden_trace():
    """Captured from the eager implementation before the refactor.

    Covers a faller, a resting body, a pushed body, a static body and a hidden
    one, across 120 steps including contact and an impulse. If a future edit
    changes the physics or the derivation, this says so with a step number.
    """
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    world = _scenario()

    for index, expected in enumerate(golden):
        _drive(world, index)
        world.step(0.02)
        actual = {b.body_id: b for b in world.ground_truth().bodies}

        assert len(actual) == len(expected["bodies"]), f"body count at step {index}"
        for want in expected["bodies"]:
            body = actual[want["id"]]
            name = want["id"]
            for got, expect in zip(body.position, want["pos"]):
                assert got == pytest.approx(expect, abs=1e-8), f"{name} pos @{index}"
            for got, expect in zip(body.velocity, want["vel"]):
                assert got == pytest.approx(expect, abs=1e-8), f"{name} vel @{index}"
            for got, expect in zip(body.net_force, want["net"]):
                assert got == pytest.approx(expect, abs=1e-8), f"{name} net @{index}"
            assert body.resting is want["resting"], f"{name} resting @{index}"


def test_applied_force_survives_into_net_force():
    """step() clears pending forces; the derivation runs after.

    Without the snapshot the applied component would silently read as zero --
    net force would still look plausible (gravity + contact), which is exactly
    the kind of wrong number nothing downstream would flag.
    """
    world = _mj()
    world.reset(seed=0)
    world.add_body(BodySpec(body_id="b", position=(0.0, 2.0, 0.0), mass=1.0))
    world.step(0.02)

    world.apply_force("b", (7.0, 0.0, -3.0))
    world.step(0.02)

    net = {b.body_id: b for b in world.ground_truth().bodies}["b"].net_force
    assert net[0] == pytest.approx(7.0, abs=1e-6)
    assert net[2] == pytest.approx(-3.0, abs=1e-6)


def test_reading_ground_truth_twice_is_stable():
    """The second read must not re-derive from mutated state."""
    world = _scenario()
    world.step(0.02)
    first = {b.body_id: b.net_force for b in world.ground_truth().bodies}
    second = {b.body_id: b.net_force for b in world.ground_truth().bodies}
    assert first == second


def test_ground_truth_after_several_steps_reports_the_latest():
    world = _scenario()
    for _ in range(5):
        world.step(0.02)
    assert world.ground_truth().step == 5


# ---------------------------------------------------------------------------
# Laziness: checked through the mechanism, not the clock
# ---------------------------------------------------------------------------

def test_step_does_not_derive_forces():
    """The whole point of the change."""
    world = _scenario()
    world.step(0.02)
    assert world._forces_stale is True, "step() derived forces eagerly again"


def test_ground_truth_clears_the_staleness():
    world = _scenario()
    world.step(0.02)
    world.ground_truth()
    assert world._forces_stale is False


def test_a_further_step_marks_it_stale_again():
    """Otherwise ground_truth() would serve a stale derivation forever."""
    world = _scenario()
    world.step(0.02)
    world.ground_truth()
    world.step(0.02)
    assert world._forces_stale is True


# ---------------------------------------------------------------------------
# The index cache
# ---------------------------------------------------------------------------

def test_index_cache_is_populated_and_correct():
    import mujoco
    world = _scenario()
    world.step(0.02)
    for body_id, cached in world._index_of.items():
        assert cached == mujoco.mj_name2id(
            world._model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )


def test_index_cache_is_rebuilt_with_the_model():
    """A stale cache would return a wrong index silently -- worse than a miss."""
    import mujoco
    world = _scenario()
    world.step(0.02)
    world.add_body(BodySpec(body_id="late", position=(3.0, 1.0, 0.0)))
    world.step(0.02)
    assert "late" in world._index_of
    assert world._index_of["late"] == mujoco.mj_name2id(
        world._model, mujoco.mjtObj.mjOBJ_BODY, "late"
    )


# ---------------------------------------------------------------------------
# Cost: the instrument that catches the next regression
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_step_cost_stays_within_budget():
    """Generous ceilings -- catches an O(n^2) reintroduction, not jitter.

    The per-decision budget is 20 ms: EpisodeRunner calls world.step() exactly
    once per policy call. Before this fix, 121 bodies cost 15.94 ms of it.
    """
    world = _mj()
    world.reset(seed=0)
    for i in range(121):
        world.add_body(BodySpec(
            body_id=f"b{i}",
            position=((i % 20) * 0.7 - 7, 0.05, (i // 20) * 0.7 - 3),
        ))
    for _ in range(10):
        world.step(0.02)

    start = time.perf_counter()
    for _ in range(100):
        world.step(0.02)
    step_ms = (time.perf_counter() - start) / 100 * 1000

    start = time.perf_counter()
    for _ in range(100):
        world.step(0.02)
        world.ground_truth()
    polled_ms = (time.perf_counter() - start) / 100 * 1000

    assert step_ms < 4.0, f"step() at 121 bodies took {step_ms:.2f} ms (was 15.94)"
    assert polled_ms < 8.0, (
        f"step+ground_truth at 121 bodies took {polled_ms:.2f} ms -- the "
        f"derivation is no longer vectorized"
    )
