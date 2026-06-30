"""Phase-3 unit tests for the LuthiCognitiveAdapter training seam.

4.8's 2026-06-15 review (F6) flagged that the LuthiModel contract +
trainer were well-tested but the Sanctuary adapter glue that wires
them was not. This file covers ``attach_seam`` (state reset on
re-attach), ``_run_training_seam`` (buffer-and-close ordering, inert
paths, the F5 loud-failure / resilient-mode dichotomy), and the
``_build_output`` predictions=[] invariant after F2/F3.

The adapter normally loads a real Luthi model at startup; these tests
inject mock model / tokenizer / actor / sink so the seam wiring can be
exercised without any real checkpoint or M9 trainer.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
import torch

# Bootstrap the luthi package onto sys.path the same way the adapter does
# at load() time. These tests bypass load() (the model state is mocked)
# so the adapter's own _ensure_luthi_importable never runs.
_LUTHI_PATH = os.environ.get("LUTHI_PATH", "")
if _LUTHI_PATH and _LUTHI_PATH not in sys.path:
    sys.path.insert(0, _LUTHI_PATH)

try:
    from luthi.seam_types import (
        ActionSelection,
        GenerationState,
        PlanSnapshot,
    )
except ImportError:
    pytest.skip(
        "luthi package not importable; set LUTHI_PATH to the LuthiModel "
        "checkout to run the Sanctuary-seam unit tests.",
        allow_module_level=True,
    )


def _make_state(d_model: int = 8, cycle_idx: int = 0) -> GenerationState:
    """Build a deterministic GenerationState for a given cycle index.

    s_t carries the cycle index in its first element so multi-cycle
    tests can verify the right state went to the right call.
    """
    s_t = torch.full((1, d_model), float(cycle_idx))
    ctx_latents = torch.full((1, 4, d_model), float(cycle_idx))
    return GenerationState(s_t=s_t, ctx_latents=ctx_latents)
from sanctuary.core.luthi_model import LuthiModel, LuthiModelConfig
from sanctuary.core.schema import CognitiveInput, Percept


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for a MultimodalPredictiveCodingLM.

    Only the surfaces the seam touches are implemented:
      - ``encode(text_tokens=..., audio_tokens=..., audio_waveform=...,
        image=..., vision_tokens=..., causal=...) -> {"latents": Tensor}``
      - presence of ``audio_encoder`` / ``vision_encoder`` attrs to keep
        ``_encode_sensory_percepts`` happy (unused here, but the path
        runs).
    """

    def __init__(self, d_model: int = 8):
        self.d_model = d_model
        # Presence-only attrs; never invoked in these tests because we
        # pass no sensory percepts.
        self.audio_encoder = object()
        self.vision_encoder = object()
        self.encode_call_count = 0
        self.last_encode_kwargs: dict | None = None

    def encode(self, **kwargs):
        self.encode_call_count += 1
        self.last_encode_kwargs = kwargs
        text = kwargs.get("text_tokens")
        if text is None:
            seq_len = 4
        else:
            seq_len = text.shape[-1]
        batch = 1 if text is None else text.shape[0]
        # Deterministic non-zero latents so mean-pool is non-trivial.
        latents = torch.linspace(
            0.1, 0.5, batch * seq_len * self.d_model,
        ).reshape(batch, seq_len, self.d_model)
        return {"latents": latents}


class _FakeTokenizer:
    """Stand-in for BPETokenizer. encode() returns one int per character."""

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 32) + 1 for c in text] or [1]


class _FakeActor:
    """Records select_action calls; returns a deterministic ActionSelection.

    Each call returns a fresh ``PlanSnapshot`` carrying a counter-based
    ``r_best`` so tests can verify the snapshot threaded through to the
    sink belongs to the matching cycle's plan (F4 corruption test).
    """

    def __init__(self, d_model: int = 8):
        self.d_model = d_model
        self.call_count = 0
        self.last_s_t: torch.Tensor | None = None
        self.last_kwargs: dict = {}

    def select_action(self, s_t: torch.Tensor, **kwargs: Any) -> ActionSelection:
        self.call_count += 1
        self.last_s_t = s_t
        self.last_kwargs = dict(kwargs)
        plan = PlanSnapshot(
            visit_distribution=torch.tensor([1.0]),
            candidate_actions=torch.zeros(1, self.d_model),
            r_best=float(self.call_count),  # 1, 2, 3 ... per cycle
        )
        return ActionSelection(
            action=torch.full((self.d_model,), 0.25),
            readable_summary="action(test)",
            efe_breakdown={"total": 0.5},
            plan_snapshot=plan,
        )


class _FakeSink:
    def __init__(self):
        self.calls: list[dict] = []

    def observe_transition(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next: torch.Tensor,
        ctx: dict[str, Any],
    ) -> dict[str, float]:
        self.calls.append(
            {"s_t": s_t, "a_t": a_t, "s_next": s_next, "ctx": ctx}
        )
        return {"v_loss": 0.1, "habit_loss": 0.2}


class _RaisingActor:
    """Actor whose select_action always raises -- used to assert
    loud-vs-resilient failure semantics."""

    def select_action(self, s_t, **kwargs):
        raise RuntimeError("planned failure")


class _RaisingSink:
    def observe_transition(self, s_t, a_t, s_next, ctx):
        raise RuntimeError("planned failure")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    """LuthiCognitiveAdapter with mocked model state; bypasses load()."""
    a = LuthiModel(LuthiModelConfig())
    a.model = _FakeModel(d_model=8)
    a.tokenizer = _FakeTokenizer()
    a.device = torch.device("cpu")
    a.model_config = {"d_model": 8, "n_blocks": 1, "seq_len": 32}
    a._loaded = True
    return a


def _ci() -> CognitiveInput:
    """Minimal CognitiveInput. All non-percept fields use schema defaults."""
    return CognitiveInput(
        new_percepts=[
            Percept(
                modality="language",
                content="hello",
                source="user",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# attach_seam
# ---------------------------------------------------------------------------


class TestAttachSeam:
    def test_defaults_inert(self, adapter):
        # No seam attached at construction.
        assert adapter._actor is None
        assert adapter._sink is None
        assert adapter._seam_resilient is False

    def test_attach_stores_actor_and_sink(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)
        assert adapter._actor is actor
        assert adapter._sink is sink
        assert adapter._seam_resilient is False

    def test_attach_resilient_flag(self, adapter):
        adapter.attach_seam(actor=_FakeActor(), resilient=True)
        assert adapter._seam_resilient is True

    def test_reattach_resets_buffered_transition(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)
        # Simulate a cycle having run: buffered state is populated.
        adapter._last_s_t = torch.zeros(1, 8)
        adapter._last_action = torch.zeros(8)
        adapter._last_action_summary = "stale"
        adapter._last_efe_breakdown = {"total": 1.0}

        adapter.attach_seam(actor=_FakeActor(), sink=_FakeSink())

        # The previous actor's MCTS tree is gone; pairing the old s_t with
        # a new actor's tree would silently corrupt the transition.
        assert adapter._last_s_t is None
        assert adapter._last_action is None
        assert adapter._last_action_summary == ""
        assert adapter._last_efe_breakdown == {}


# ---------------------------------------------------------------------------
# _run_training_seam — buffer-and-close ordering
# ---------------------------------------------------------------------------


class TestRunTrainingSeamOrdering:
    """Post-Phase-4a: _run_training_seam takes a GenerationState (captured
    from inner-speech generation's step-0 encode). It no longer calls
    encode_state internally; the encode-vs-skip gate moved up to
    _think_sync (test_luthi_integration.py covers the no-seam path).
    """

    def test_first_cycle_selects_but_does_not_observe(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)

        adapter._run_training_seam(_make_state(cycle_idx=1))

        assert actor.call_count == 1
        # No prior cycle -> no observe_transition to fire.
        assert sink.calls == []
        # Buffered for next cycle.
        assert adapter._last_s_t is not None
        assert adapter._last_action is not None
        assert adapter._last_action_summary == "action(test)"
        assert adapter._last_efe_breakdown == {"total": 0.5}

    def test_second_cycle_closes_previous_then_selects(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)

        state_1 = _make_state(cycle_idx=1)
        adapter._run_training_seam(state_1)
        first_s_t = adapter._last_s_t
        first_action = adapter._last_action

        state_2 = _make_state(cycle_idx=2)
        adapter._run_training_seam(state_2)

        # observe_transition fired once at the start of cycle 2.
        assert len(sink.calls) == 1
        call = sink.calls[0]
        # s_t/a_t in the observed transition are cycle-1's; s_next is
        # cycle-2's captured state.
        assert torch.equal(call["s_t"], first_s_t)
        assert torch.equal(call["a_t"], first_action)
        assert torch.equal(call["s_next"], state_2.s_t)
        # ctx carries the P3 wiring defaults + the plan_snapshot from
        # cycle 1's select_action (F4: the transition is self-describing).
        assert call["ctx"]["counterpart_present"] is False
        assert call["ctx"]["time_since_emission"] == 0.0
        assert call["ctx"]["plan_snapshot"] is not None
        # The snapshot belongs to cycle 1 (the cycle whose action this
        # observation realizes), not cycle 2 (the most recent select).
        assert call["ctx"]["plan_snapshot"].r_best == 1.0

        # actor.select_action then fires for cycle 2.
        assert actor.call_count == 2

    def test_f4_no_corruption_under_interleaved_selects(self, adapter):
        """Multi-cycle: each observe_transition receives the snapshot
        from the cycle whose action it realizes, not from any later
        plan. The fake actor's r_best counts up by cycle, so we can read
        the snapshot's r_best back as the cycle index."""
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)

        for cycle in range(1, 5):
            adapter._run_training_seam(_make_state(cycle_idx=cycle))

        # Cycle 1 didn't observe (no prior transition); cycles 2-4 did.
        assert len(sink.calls) == 3
        for i, call in enumerate(sink.calls):
            # Cycle k's observe carries cycle k-1's snapshot.
            # i=0 corresponds to cycle 2 closing cycle 1's transition,
            # so snapshot.r_best == 1.0; etc.
            expected = float(i + 1)
            assert call["ctx"]["plan_snapshot"].r_best == expected, (
                f"call {i}: expected r_best={expected}, "
                f"got {call['ctx']['plan_snapshot'].r_best}"
            )

    def test_sink_only_no_actor_still_buffers(self, adapter):
        """sink without actor: transitions persist but no action gets
        selected; _last_action stays None so no observe fires next cycle."""
        sink = _FakeSink()
        adapter.attach_seam(sink=sink)

        adapter._run_training_seam(_make_state(cycle_idx=1))
        adapter._run_training_seam(_make_state(cycle_idx=2))
        # No actor means no _last_action ever populated, so observe is
        # never callable -- the buffer-and-close gate (sink and last_s_t
        # and last_action) keeps it silent.
        assert sink.calls == []

    def test_actor_only_no_sink_runs_planning_each_cycle(self, adapter):
        actor = _FakeActor()
        adapter.attach_seam(actor=actor)
        adapter._run_training_seam(_make_state(cycle_idx=1))
        adapter._run_training_seam(_make_state(cycle_idx=2))
        assert actor.call_count == 2


# ---------------------------------------------------------------------------
# Item #6 — lived-context threading (context_obs + realized_next_state)
# ---------------------------------------------------------------------------


class TestRunTrainingSeamLivedContext:
    """_run_training_seam threads the PREVIOUS cycle's raw ``context_obs``
    (the inputs that produced the transition's starting state) and the
    realized pooled next state into ``observe_transition``'s ctx, so the
    learner can re-encode and drive a lived JEPA gradient. Back-compat: a
    GenerationState whose ``context_obs`` is None (uncaptured / pre-#6
    substrate) leaves no ``context_obs`` key, so the sink takes the
    head-only path.
    """

    @staticmethod
    def _state_with_obs(cycle_idx: int, d_model: int = 8) -> GenerationState:
        s_t = torch.full((1, d_model), float(cycle_idx))
        ctx_latents = torch.full((1, 4, d_model), float(cycle_idx))
        # Distinct per-cycle token window so the threading DIRECTION is
        # observable -- cycle k's obs carries the value k.
        context_obs = {
            "text_tokens": torch.full((1, 4), cycle_idx, dtype=torch.long),
        }
        return GenerationState(
            s_t=s_t, ctx_latents=ctx_latents, context_obs=context_obs,
        )

    def test_threads_prev_context_obs_with_s_next_target(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)

        adapter._run_training_seam(self._state_with_obs(cycle_idx=1))
        state_2 = self._state_with_obs(cycle_idx=2)
        adapter._run_training_seam(state_2)

        assert len(sink.calls) == 1
        call = sink.calls[0]
        ctx = call["ctx"]
        # context_obs is the cycle-1 context that produced the transition's
        # STARTING state (_last_s_t), not cycle 2's -- the learner must
        # re-encode the context that actually preceded the realized step.
        assert "context_obs" in ctx
        assert torch.equal(
            ctx["context_obs"]["text_tokens"],
            torch.full((1, 4), 1, dtype=torch.long),
        )
        # The lived-JEPA target is the s_next positional itself (cycle-2's
        # realized pooled s_t) -- no redundant realized_next_state channel.
        assert "realized_next_state" not in ctx
        assert torch.equal(call["s_next"], state_2.s_t)

    def test_context_obs_buffer_advances_each_cycle(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)
        for cycle in range(1, 4):
            adapter._run_training_seam(self._state_with_obs(cycle_idx=cycle))
        # Cycles 2 and 3 observed; each carries the PRIOR cycle's obs.
        assert len(sink.calls) == 2
        assert torch.equal(
            sink.calls[0]["ctx"]["context_obs"]["text_tokens"],
            torch.full((1, 4), 1, dtype=torch.long),
        )
        assert torch.equal(
            sink.calls[1]["ctx"]["context_obs"]["text_tokens"],
            torch.full((1, 4), 2, dtype=torch.long),
        )

    def test_back_compat_no_context_obs_key_when_uncaptured(self, adapter):
        # _make_state builds context_obs=None (pre-#6 / uncaptured substrate).
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)
        adapter._run_training_seam(_make_state(cycle_idx=1))
        adapter._run_training_seam(_make_state(cycle_idx=2))
        # No context_obs key -> the sink's §1 path skips the lived update
        # and runs head-only.
        assert "context_obs" not in sink.calls[0]["ctx"]

    def test_attach_seam_clears_context_obs_buffer(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink)
        adapter._run_training_seam(self._state_with_obs(cycle_idx=1))
        assert adapter._last_context_obs is not None
        # Re-attach drops the buffered transition state (the prior context
        # belonged to the prior actor's tree -- mirrors _last_s_t reset).
        adapter.attach_seam(actor=actor, sink=sink)
        assert adapter._last_context_obs is None


# ---------------------------------------------------------------------------
# F5 — loud-vs-resilient failure modes
# ---------------------------------------------------------------------------


class TestFailureLoudness:
    def test_select_action_raises_by_default(self, adapter):
        adapter.attach_seam(actor=_RaisingActor())
        with pytest.raises(RuntimeError, match="planned failure"):
            adapter._run_training_seam(_make_state(cycle_idx=1))

    def test_select_action_swallowed_in_resilient_mode(self, adapter):
        adapter.attach_seam(actor=_RaisingActor(), resilient=True)
        # No raise.
        adapter._run_training_seam(_make_state(cycle_idx=1))
        # Action state is cleared so we don't hand stale data to a
        # downstream observer.
        assert adapter._last_action is None
        assert adapter._last_action_summary == ""

    def test_observe_transition_raises_by_default(self, adapter):
        sink = _RaisingSink()
        adapter.attach_seam(actor=_FakeActor(), sink=sink)
        # Cycle 1: no observe fires, no raise.
        adapter._run_training_seam(_make_state(cycle_idx=1))
        # Cycle 2: observe fires and raises.
        with pytest.raises(RuntimeError, match="planned failure"):
            adapter._run_training_seam(_make_state(cycle_idx=2))

    def test_observe_transition_swallowed_in_resilient_mode(self, adapter):
        sink = _RaisingSink()
        adapter.attach_seam(
            actor=_FakeActor(), sink=sink, resilient=True,
        )
        adapter._run_training_seam(_make_state(cycle_idx=1))
        # No raise even though sink raises.
        adapter._run_training_seam(_make_state(cycle_idx=2))


# ---------------------------------------------------------------------------
# F2/F3 — predictions stays empty regardless of seam state
# ---------------------------------------------------------------------------


class TestPredictionsAlwaysEmpty:
    def test_no_seam_no_predictions(self, adapter):
        out = adapter._build_output(
            cognitive_input=_ci(),
            inner_speech="thinking",
            external_speech=None,
        )
        assert out.predictions == []

    def test_seam_active_no_predictions(self, adapter):
        adapter.attach_seam(actor=_FakeActor())
        adapter._run_training_seam(_make_state(cycle_idx=1))
        # _last_action_summary is populated -- before the F2/F3 fix this
        # would have produced a Prediction. After the fix, predictions
        # stays [].
        assert adapter._last_action_summary == "action(test)"
        out = adapter._build_output(
            cognitive_input=_ci(),
            inner_speech="thinking",
            external_speech=None,
        )
        assert out.predictions == []


# ---------------------------------------------------------------------------
# §4 — async actor/learner wiring (Item #6, 2026-06-29)
# ---------------------------------------------------------------------------


class TestAttachSeamAsync:
    def test_default_off_no_learner(self, adapter):
        adapter.attach_seam(actor=_FakeActor(), sink=_FakeSink())
        assert adapter._async_learner is None

    def test_drain_creates_learner(self, adapter):
        adapter.attach_seam(
            actor=_FakeActor(), sink=_FakeSink(), async_mode="drain",
        )
        assert adapter._async_learner is not None
        assert adapter._async_learner.mode == "drain"

    def test_async_requires_sink(self, adapter):
        with pytest.raises(ValueError, match="requires a sink"):
            adapter.attach_seam(actor=_FakeActor(), async_mode="drain")

    def test_bad_async_mode_rejected(self, adapter):
        with pytest.raises(ValueError, match="async_mode must be"):
            adapter.attach_seam(
                actor=_FakeActor(), sink=_FakeSink(), async_mode="bogus",
            )

    def test_reattach_stops_prior_threaded_learner(self, adapter):
        adapter.attach_seam(
            actor=_FakeActor(), sink=_FakeSink(), async_mode="threaded",
        )
        assert adapter._async_learner is not None
        # Re-attach off must tear the thread down cleanly.
        adapter.attach_seam(actor=_FakeActor(), sink=_FakeSink())
        assert adapter._async_learner is None


class TestRunTrainingSeamAsync:
    """Async seam: _run_training_seam returns the completed prior-cycle
    transition (captured BEFORE select overwrites _last_*); the actor uses the
    habit-only fast path (budget=0); _think_sync enqueues the transition AFTER
    the lock. We replicate that enqueue step here (drain runs observe inline)."""

    @staticmethod
    def _seam_and_enqueue(adapter, state):
        # Mirror _think_sync: capture inside the (here-uncontended) lock, then
        # submit OUTSIDE it. Returns the pending transition (or None).
        pending = adapter._run_training_seam(state)
        if pending is not None and adapter._async_learner is not None:
            adapter._async_learner.submit(pending)
        return pending

    def test_actor_uses_budget_zero(self, adapter):
        actor = _FakeActor()
        adapter.attach_seam(actor=actor, sink=_FakeSink(), async_mode="drain")
        self._seam_and_enqueue(adapter, _make_state(cycle_idx=1))
        assert actor.last_kwargs.get("budget") == 0, (
            "async actor must take the habit-only fast path (no MCTS on the "
            "hot path)"
        )

    def test_first_cycle_selects_no_observe(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink, async_mode="drain")
        pending = self._seam_and_enqueue(adapter, _make_state(cycle_idx=1))
        assert pending is None, "no prior transition exists on cycle 1"
        assert sink.calls == []
        assert actor.call_count == 1
        assert adapter._last_s_t is not None
        assert adapter._last_action is not None

    def test_second_cycle_closes_prior_via_drain(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink, async_mode="drain")

        self._seam_and_enqueue(adapter, _make_state(cycle_idx=1))
        first_s_t = adapter._last_s_t
        first_action = adapter._last_action

        state_2 = _make_state(cycle_idx=2)
        self._seam_and_enqueue(adapter, state_2)

        # Drain mode ran observe synchronously: the prior transition closed.
        assert len(sink.calls) == 1
        call = sink.calls[0]
        # The captured transition is the PRIOR cycle's (s_t/a_t), with this
        # cycle's s_t as the realized next state -- captured before select
        # overwrote _last_*.
        assert torch.equal(call["s_t"], first_s_t)
        assert torch.equal(call["a_t"], first_action)
        assert torch.equal(call["s_next"], state_2.s_t)
        # F4 self-describing snapshot belongs to cycle 1 (whose action this
        # realizes), not cycle 2's most recent select.
        assert call["ctx"]["plan_snapshot"].r_best == 1.0
        assert actor.call_count == 2

    def test_payload_crossing_queue_is_detached(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink, async_mode="drain")
        self._seam_and_enqueue(adapter, _make_state(cycle_idx=1))
        self._seam_and_enqueue(adapter, _make_state(cycle_idx=2))
        call = sink.calls[0]
        for key in ("s_t", "a_t", "s_next"):
            assert not call[key].requires_grad, f"{key} crossed the queue with grad"

    def test_threads_prev_context_obs(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(actor=actor, sink=sink, async_mode="drain")

        def _state(cycle):
            s_t = torch.full((1, 8), float(cycle))
            ctx_latents = torch.full((1, 4, 8), float(cycle))
            obs = {"text_tokens": torch.full((1, 4), cycle, dtype=torch.long)}
            return GenerationState(
                s_t=s_t, ctx_latents=ctx_latents, context_obs=obs,
            )

        self._seam_and_enqueue(adapter, _state(1))
        self._seam_and_enqueue(adapter, _state(2))
        call = sink.calls[0]
        # The observed transition carries cycle 1's context (which produced
        # its starting state), not cycle 2's.
        assert "context_obs" in call["ctx"]
        assert torch.equal(
            call["ctx"]["context_obs"]["text_tokens"],
            torch.full((1, 4), 1, dtype=torch.long),
        )

    def test_threaded_mode_routes_to_sink(self, adapter):
        actor, sink = _FakeActor(), _FakeSink()
        adapter.attach_seam(
            actor=actor, sink=sink, async_mode="threaded", queue_maxsize=8,
        )
        try:
            self._seam_and_enqueue(adapter, _make_state(cycle_idx=1))  # no obs
            self._seam_and_enqueue(adapter, _make_state(cycle_idx=2))  # enqueues
            adapter._async_learner.wait_until_drained()
            assert len(sink.calls) == 1
            assert call_r_best(sink) == 1.0
        finally:
            adapter.shutdown_async_learner()


def call_r_best(sink) -> float:
    return sink.calls[0]["ctx"]["plan_snapshot"].r_best
