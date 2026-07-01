"""§4 async actor/learner boundary — queue, modes, coarse model_lock.

These tests exercise the concurrency machinery directly with a lightweight fake
sink + fake model, so they run anywhere (no luthi / no model load needed): §4's
correctness IS the queue/lock/threading, not the trainer internals (those are
covered by the lived-JEPA suite in LuthiModel and the e2e smoke harness).

Coverage mirrors the §4 brief:
  * drain mode: N produced == N consumed, FIFO, no loss; payloads detached;
  * threaded mode: no exceptions, consistent final theta-version under the lock,
    backpressure (blocking put, never silent loss), learner-error surfacing;
  * snapshot isolation: mutating live buffers after a clone leaves the clone
    unchanged (scale-step read-barrier infra, NOT active at smoke).
"""

from __future__ import annotations

import threading
import time
import types

import pytest
import torch

from sanctuary.core.async_learner import (
    AsyncLearner,
    Transition,
    assert_detached,
    clone_living_buffers,
    detach_context_obs,
)


D = 8


def _detached_transition(idx: int) -> Transition:
    """A fully-detached transition whose s_t[0] encodes its submit order."""
    s_t = torch.zeros(D)
    s_t[0] = float(idx)
    return Transition(
        s_t=s_t,
        a_t=torch.zeros(D),
        s_next=torch.ones(D),
        context_obs={"text_tokens": torch.zeros(1, 4, dtype=torch.long)},
    )


class _RecordingSink:
    """Records the order of transitions it observes; bumps a fake theta-version."""

    def __init__(self, model=None, work: float = 0.0):
        self.model = model
        self.work = work
        self.seen: list[int] = []
        self.calls = 0

    def observe_transition(self, s_t, a_t, s_next, ctx) -> dict:
        if self.work:
            time.sleep(self.work)
        self.seen.append(int(s_t[0].item()))
        self.calls += 1
        if self.model is not None:
            self.model.bump_nonatomic()
            return {"theta_version": self.model.theta_version}
        return {}


class _FakeModel:
    """Theta-version holder with a deliberately NON-ATOMIC bump, so the coarse
    model_lock is load-bearing: concurrent unlocked bumps would lose updates."""

    def __init__(self):
        self.theta_version = 0

    def bump_nonatomic(self) -> None:
        v = self.theta_version
        time.sleep(0.0005)  # widen the read-modify-write window
        self.theta_version = v + 1


# ----------------------------------------------------------------------
# Drain mode
# ----------------------------------------------------------------------
def test_drain_fifo_no_loss():
    sink = _RecordingSink()
    learner = AsyncLearner(sink, threading.Lock(), mode="drain")

    n = 25
    for i in range(n):
        learner.submit(_detached_transition(i))

    assert learner.produced == n
    assert learner.consumed == n
    assert sink.seen == list(range(n)), "drain mode must be FIFO with zero loss"


def test_drain_returns_metrics():
    model = _FakeModel()
    learner = AsyncLearner(_RecordingSink(model), threading.Lock(), mode="drain")
    metrics = learner.submit(_detached_transition(0))
    assert metrics == {"theta_version": 1}


# ----------------------------------------------------------------------
# Detached-payload guarantee (no autograd graph crosses the queue)
# ----------------------------------------------------------------------
def test_submit_rejects_graph_on_core_tensor():
    sink = _RecordingSink()
    learner = AsyncLearner(sink, threading.Lock(), mode="drain")
    bad = _detached_transition(0)
    bad.a_t = torch.zeros(D, requires_grad=True)
    with pytest.raises(AssertionError, match="autograd graph"):
        learner.submit(bad)
    assert sink.calls == 0, "a graph-carrying payload must never reach the sink"


def test_submit_rejects_graph_in_context_obs():
    t = _detached_transition(0)
    t.context_obs = {"text_tokens": torch.zeros(1, 4, requires_grad=True)}
    with pytest.raises(AssertionError, match="autograd graph"):
        assert_detached(t)


def test_submit_rejects_graph_in_plan_snapshot():
    t = _detached_transition(0)
    t.plan_snapshot = types.SimpleNamespace(
        visit_distribution=torch.zeros(3, requires_grad=True),
        candidate_actions=torch.zeros(3, D),
        r_best=0.0,
    )
    with pytest.raises(AssertionError, match="autograd graph"):
        assert_detached(t)


def test_detach_context_obs_severs_graph_and_aliasing():
    src = torch.ones(4, requires_grad=True)
    out = detach_context_obs({"text_tokens": src, "flag": 3})
    assert not out["text_tokens"].requires_grad
    assert out["flag"] == 3
    # A real clone: mutating the source must not touch the detached copy.
    with torch.no_grad():
        src.add_(1.0)
    assert torch.equal(out["text_tokens"], torch.ones(4))


# ----------------------------------------------------------------------
# Threaded mode
# ----------------------------------------------------------------------
def test_threaded_no_loss_and_consistent_theta_under_lock():
    """Learner + a concurrent actor-sim both bump a non-atomic theta-version
    UNDER the shared model_lock. If the lock serializes correctly, every bump
    lands: final theta == learner_bumps + actor_bumps, with zero lost updates."""
    model = _FakeModel()
    lock = threading.Lock()
    sink = _RecordingSink(model)
    learner = AsyncLearner(sink, lock, mode="threaded", maxsize=16)
    learner.start()

    n_learner = 30
    n_actor = 30

    def actor_sim():
        # Stand in for the actor's under-lock model mutation (its perception
        # critical section). Same lock the learner holds for observe.
        for _ in range(n_actor):
            with lock:
                model.bump_nonatomic()

    actor = threading.Thread(target=actor_sim, name="actor-sim")
    actor.start()
    for i in range(n_learner):
        learner.submit(_detached_transition(i))
    actor.join()
    learner.wait_until_drained()
    learner.stop()

    assert learner.errors == 0
    assert learner.produced == n_learner
    assert learner.consumed == n_learner, "no transition may be lost"
    assert sink.seen == list(range(n_learner)), "single consumer preserves FIFO"
    assert model.theta_version == n_learner + n_actor, (
        "lost updates -> the coarse model_lock failed to serialize "
        "actor/learner model mutation"
    )


def test_threaded_backpressure_blocks_never_drops():
    """A slow learner + a tiny queue forces the producer to block (backpressure);
    the overflow counter ticks, but nothing is dropped."""
    sink = _RecordingSink(work=0.003)
    learner = AsyncLearner(sink, threading.Lock(), mode="threaded", maxsize=2)
    learner.start()

    n = 20
    for i in range(n):
        learner.submit(_detached_transition(i))
    learner.wait_until_drained()
    learner.stop()

    assert learner.produced == n
    assert learner.consumed == n, "blocking put must never silently drop"
    assert learner.overflow_waits > 0, "a full queue must register backpressure"
    assert sink.seen == list(range(n))


def test_threaded_surfaces_learner_error_non_resilient():
    """A learner-thread exception must not vanish: stop() re-raises it."""
    class _Boom:
        def observe_transition(self, *a, **k):
            raise ValueError("boom in the learner")

    learner = AsyncLearner(_Boom(), threading.Lock(), mode="threaded", maxsize=4)
    learner.start()
    learner.submit(_detached_transition(0))
    # Give the learner thread a moment to consume + fail.
    for _ in range(200):
        if learner.errors:
            break
        time.sleep(0.005)
    assert learner.errors == 1
    with pytest.raises(ValueError, match="boom in the learner"):
        learner.stop()


def test_threaded_resilient_swallows_and_continues():
    class _SometimesBoom:
        def __init__(self):
            self.calls = 0

        def observe_transition(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("transient")
            return {}

    sink = _SometimesBoom()
    learner = AsyncLearner(
        sink, threading.Lock(), mode="threaded", maxsize=4, resilient=True,
    )
    learner.start()
    learner.submit(_detached_transition(0))  # raises -> swallowed
    learner.submit(_detached_transition(1))  # succeeds
    learner.wait_until_drained()
    learner.stop()  # must NOT raise in resilient mode
    assert learner.errors == 1
    assert learner.consumed == 1  # only the successful one counts as consumed


def test_bad_mode_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        AsyncLearner(_RecordingSink(), threading.Lock(), mode="bogus")


# ----------------------------------------------------------------------
# Snapshot isolation — SCALE-STEP INFRA, NOT ACTIVE AT SMOKE
# ----------------------------------------------------------------------
class _FakeLivingModule(torch.nn.Module):
    """Minimal stand-in carrying the living-state buffer names the snapshot
    helper looks for. Used only to prove the clone is a real read barrier."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.randn(4, 4))
        self.register_buffer("episode_contexts", torch.randn(3, 4))
        self.register_buffer("episode_outputs", torch.randn(3, 4))
        self.register_buffer("episode_saliences", torch.randn(3))
        self.register_buffer("episode_count", torch.tensor(2))


def test_clone_living_buffers_is_a_read_barrier():
    """Mutating the live buffers AFTER the snapshot leaves the snapshot
    unchanged -- proving the clone is a genuine read barrier.

    NOTE: scale-step infrastructure. §4 at smoke uses the coarse model_lock and
    does NOT consume these clones; this only locks in snapshot semantics for the
    future double-buffer / weight-override concurrency work. Do not read this as
    a live mechanism.
    """
    model = _FakeLivingModule()
    snap = clone_living_buffers(model)
    assert ".weight" in snap and ".episode_count" in snap

    with torch.no_grad():
        model.weight.add_(1.0)
        model.episode_contexts.mul_(0.0)
        model.episode_count.fill_(99)

    assert not torch.equal(snap[".weight"], model.weight)
    assert not torch.equal(snap[".episode_contexts"], model.episode_contexts)
    assert int(snap[".episode_count"].item()) == 2, "snapshot must be frozen"


# ----------------------------------------------------------------------
# Fix B regression: a dead learner must not hang its waiters
# ----------------------------------------------------------------------
def test_threaded_learner_death_drains_and_does_not_hang():
    """Non-resilient learner death must drain the items queued behind the
    failed one (so wait_until_drained()/stop() release) AND re-raise the error
    at stop(). Without the drain, queue.join() deadlocks forever -- the exact
    hang Window A's audit caught. Hang over silence."""
    submitted = threading.Event()

    class _BoomAfterAllQueued:
        def __init__(self):
            self.calls = 0

        def observe_transition(self, *a, **k):
            self.calls += 1
            # Block the first observe until the actor has queued everything, so
            # the items-behind-the-failure case is deterministic (no race).
            submitted.wait(timeout=5.0)
            raise ValueError("boom in the learner")

    learner = AsyncLearner(
        _BoomAfterAllQueued(), threading.Lock(), mode="threaded", maxsize=16,
    )
    learner.start()
    n = 6
    for i in range(n):
        learner.submit(_detached_transition(i))  # all queued (maxsize > n)
    submitted.set()  # unblock the first observe -> raise -> drain the rest

    learner.wait_until_drained()  # must NOT hang
    with pytest.raises(ValueError, match="boom in the learner"):
        learner.stop()
    assert learner.errors == 1


# ----------------------------------------------------------------------
# Minimal threaded smoke against the REAL M9Trainer (luthi-gated)
# ----------------------------------------------------------------------
# Window A: "A and B only actually bite against the real substrate." Unit tests
# above use a fake sink; this drives the real lived-JEPA observe_transition on
# the learner thread, with the actor's encode held under the SAME model_lock --
# the faithful coarse-lock smoke. Skips cleanly without LUTHI_PATH.
import os  # noqa: E402
import sys  # noqa: E402

_LUTHI_PATH = os.environ.get("LUTHI_PATH", "")
if _LUTHI_PATH and _LUTHI_PATH not in sys.path:
    sys.path.insert(0, _LUTHI_PATH)
try:
    import torch.optim as optim
    from luthi.sanctuary_interface import encode_state
    from luthi.v2.jepa_loss import JEPALoss
    from luthi.v2.jepa_runner import (
        CheckpointConfig, EpochConfig, KillCriteriaConfig, LoggingConfig,
        ModalitySampler, RunnerConfig, SamplerConfig,
    )
    from luthi.v2.m9.runner import M9Config, M9Trainer
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    _HAVE_LUTHI = True
except ImportError:
    _HAVE_LUTHI = False

requires_luthi = pytest.mark.skipif(
    not _HAVE_LUTHI, reason="luthi not importable; set LUTHI_PATH",
)

_VOCAB, _D, _SEQ, _B = 32, 32, 16, 2


class _TextLoader:
    def __init__(self, seed: int = 0):
        self.gen = torch.Generator().manual_seed(seed)

    def next_batch(self, modality: str) -> dict:
        assert modality == "text"
        return {
            "text_tokens": torch.randint(
                0, _VOCAB, (_B, _SEQ), generator=self.gen,
            )
        }

    def batch_token_count(self, modality, batch):
        return int(batch["text_tokens"].numel())

    def corpus_sizes_tokens(self):
        return {"text": 1000}


def _build_real_trainer(run_dir, *, seed: int = 7):
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=_VOCAB, d_model=_D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=_SEQ,
        max_audio_tokens=_SEQ, max_vision_tokens=_SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=1e-3,
    )
    loader = _TextLoader(seed=seed)
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
    cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(
            light_interval_batches=10**9, deep_interval_batches=10**9,
        ),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    return M9Trainer(
        loss_module=loss_module, optimizer=optimizer,
        sampler=ModalitySampler(sampler_cfg), data_loader=loader,
        config=cfg, run_dir=run_dir,
        # Isolate the lived gradient (replay off) so theta_version advances
        # exactly once per lived cycle -> a clean consistency assertion.
        m9_config=M9Config(
            mcts_budget_per_cycle=4, lived_lr=1e-2, corpus_replay_ratio=0.0,
        ),
    )


@requires_luthi
def test_threaded_real_trainer_smoke(tmp_path):
    """K lived cycles through the threaded learner against the REAL M9Trainer.
    The actor's encode runs UNDER model_lock (mimicking _think_sync); the
    learner's observe runs under the same lock. Asserts no NaN/divergence
    (errors==0), produced==consumed, and a consistent final theta-version
    (exactly +K with replay off -> no torn/lost lived updates under the lock)."""
    trainer = _build_real_trainer(tmp_path)
    enc = trainer.loss_module.online_encoder
    lock = threading.Lock()
    learner = AsyncLearner(trainer, lock, mode="threaded", maxsize=8)
    learner.start()

    theta_before = trainer.staleness.theta_version
    k = 6
    try:
        for i in range(k):
            # Actor perception writes living weights -> hold the lock, exactly
            # as the real cycle does. Build the detached transition, submit
            # OUTSIDE the lock (deadlock guard).
            with lock:
                s_t = encode_state(
                    enc, text_tokens=torch.randint(0, _VOCAB, (1, _SEQ)),
                    pool=True,
                )
                s_next = encode_state(
                    enc, text_tokens=torch.randint(0, _VOCAB, (1, _SEQ)),
                    pool=True,
                )
            learner.submit(Transition(
                s_t=s_t.detach(),
                a_t=torch.randn(1, _D),
                s_next=s_next.detach(),
                context_obs={"text_tokens": torch.randint(0, _VOCAB, (1, _SEQ))},
            ))
        learner.wait_until_drained()
    finally:
        learner.stop()  # re-raises any learner-thread error (NaN-kill etc.)
        trainer.action_log.close()  # Windows: release the handle before tmp rmtree

    assert learner.errors == 0
    assert learner.produced == k
    assert learner.consumed == k
    assert trainer.staleness.theta_version == theta_before + k, (
        "lived theta-version must advance exactly once per cycle -- a torn or "
        "lost update under the coarse lock would break this"
    )
    last = learner.last_metrics
    assert "lived_l_pred" in last
    assert last["lived_l_pred"] == last["lived_l_pred"], "lived_l_pred is NaN"
