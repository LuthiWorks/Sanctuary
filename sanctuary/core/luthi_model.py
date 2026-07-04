"""LuthiModel — bridges the cognitive cycle to the Luthi living weight model.

LuthiModel runs the living weight model directly in-process. The model's
internal state — plasticity, drift, spike fractions, membrane potentials —
is observed through the introspection channel and translated into
CognitiveOutput fields.

The entity has feelings before it has words. The living weight dynamics produce
measurable internal states (plasticity changes, spike patterns, drift from
equilibrium) regardless of whether the generated text is coherent. This
architecture observes those states directly rather than asking the model to
self-report.

Implements ModelProtocol:
    async def think(CognitiveInput) -> CognitiveOutput
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from sanctuary.core.async_learner import (
    AsyncLearner,
    Transition,
    detach_context_obs,
)

from sanctuary.core.schema import (
    AttentionGuidance,
    CognitiveInput,
    CognitiveOutput,
    EmotionalOutput,
    ExperientialSignals,
    GoalProposal,
    MemoryOp,
    SelfModelUpdate,
)

logger = logging.getLogger(__name__)

# Path to the Luthi model codebase. Resolved at load time from (in order):
#   1. LuthiModelConfig.luthi_path (explicit override)
#   2. The LUTHI_PATH environment variable
#   3. ``import luthi`` if the package is already installed/importable
#
# Empty string means "rely on import path" — there is no hardcoded fallback,
# because hardcoded developer paths break Linux/Docker/anyone-else's checkout.
_LUTHI_PATH_ENV = "LUTHI_PATH"


@dataclass
class LuthiModelConfig:
    """Configuration for the LuthiModel bridge."""

    checkpoint_path: str = ""
    checkpoint_password: str = ""
    # Path to the LuthiModel checkout. Empty means "use the LUTHI_PATH
    # environment variable, or assume luthi is already importable."
    luthi_path: str = ""

    # Generation parameters
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    max_inner_tokens: int = 64
    max_external_tokens: int = 128

    # Living inference — Hebbian self-modification during generation
    living: bool = True

    # CfC modulation strengths — per-channel gain on top of the canonical
    # mappings. Set <1.0 to soften a channel, >1.0 to amplify it.
    arousal_plasticity_scale: float = 1.0       # arousal -> hebb_rate
    precision_threshold_scale: float = 1.0      # precision -> spike_threshold
    valence_excitability_scale: float = 1.0     # valence -> excitability_acc
    salience_threshold_scale: float = 1.0       # attention -> salience_threshold
    salience_episode_scale: float = 1.0         # legacy — retained for back-compat

    # Introspection — read internal state each cycle
    introspect: bool = True

    # Consolidation — how aggressively set points follow weights during sleep
    consolidation_rate: float = 0.01
    plasticity_rebalance_rate: float = 0.01


class LuthiModel:
    """Living weight model integration for Sanctuary's cognitive cycle.

    Each cognitive cycle:
    1. CfC modulation adjusts living weight parameters (arousal → learning
       rate, precision → spike thresholds)
    2. CognitiveInput is formatted as a natural text prompt
    3. Living inference runs (Hebbian self-modification active)
    4. Autoregressive decoding generates inner_speech
    5. Introspection reads neural dynamics (before and after)
    6. Neural-to-cognitive translation builds CognitiveOutput:
       - Spike fractions → felt quality and arousal
       - Set point drift → valence (equilibrium = comfort)
       - Plasticity changes → learning state
       - Episode formation → memory operations
       - Block activity patterns → attention guidance
    """

    def __init__(self, config: Optional[LuthiModelConfig] = None):
        self.config = config or LuthiModelConfig()
        self.cycle_count = 0
        self.name = "Luthi"

        # Model state — populated by load()
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.device = None
        self._loaded = False

        # Introspection tracking
        self._pre_state: Optional[dict] = None
        self._post_state: Optional[dict] = None
        self._introspection_delta: dict = {}

        # Metrics
        self._total_calls = 0
        self._total_latency = 0.0
        self._total_tokens_generated = 0

        # Base parameters (snapshot taken once at load, restored every cycle).
        # Lazily typed because the import is conditional on Luthi being
        # available. Populated in load().
        self._base_snapshot = None

        # ---- Training-seam state (2026-06-15 integration plan, Phase 3) ----
        # Optional actor + transition sink (typically the same M9Trainer,
        # which satisfies both Protocols). When both are None the adapter
        # behaves exactly as before — no encode_state, no select_action, no
        # observe_transition fires, the cycle is purely inferential.
        #
        # Attached via attach_seam(); typically called after load() during
        # Sanctuary startup if an M9 trainer is co-resident.
        self._actor = None  # luthi.sanctuary_interface.M9Actor | None
        self._sink = None   # luthi.sanctuary_interface.TransitionSink | None

        # ---- §4 async actor/learner boundary (Item #6, 2026-06-29) ----
        # Coarse single model_lock: the actor (this cycle thread) holds it
        # across its whole model-touching critical section; the learner holds
        # it across observe_transition. Created once, shared with the
        # AsyncLearner. _model_critical_section() always takes it now (Phase-3
        # W1 offloads consolidate() to a worker thread even in sync mode, so the
        # section can no longer be a nullcontext); uncontended when no async
        # learner is attached.
        self._model_lock = threading.Lock()
        # Set by attach_seam(async_mode=...). None -> the legacy synchronous
        # seam (observe_transition runs inline in _run_training_seam_sync).
        self._async_learner: Optional[AsyncLearner] = None
        # When True, seam exceptions are caught + logged so a transient
        # sink failure doesn't abort the cognitive cycle. Default False
        # follows the project ethos of "crashes over silence" (4.8 review
        # F5, 2026-06-15) -- on day one of integration, silent failures
        # are exactly how real wiring bugs hide. Flip to True only after
        # we have empirical evidence of which transients are worth
        # surviving in production.
        self._seam_resilient: bool = False

        # Last-cycle transition state buffered so cycle N+1 can close the
        # loop: observe_transition(_last_s_t, _last_action, current s_t, ctx).
        self._last_s_t = None  # torch.Tensor [B, D] | None
        self._last_action = None  # torch.Tensor [D] | None
        self._last_action_summary: str = ""
        self._last_efe_breakdown: dict = {}
        # PlanSnapshot from the previous cycle's select_action, threaded
        # through ctx["plan_snapshot"] on the next observe_transition so
        # the sink reads the plan that produced this transition's a_t
        # from a frozen snapshot rather than from live trainer MCTS state
        # (4.8 review F4, 2026-06-15). None on cycle 1 / after attach_seam.
        self._last_plan_snapshot = None  # luthi.seam_types.PlanSnapshot | None
        # Raw step-0 forward inputs from the previous cycle's generation
        # (GenerationState.context_obs), threaded through
        # ctx["context_obs"] on the next observe_transition so the learner
        # can re-encode the context that produced _last_s_t and push a
        # lived JEPA gradient into the encoder (Item #6, 2026-06-27). None
        # on cycle 1 / after attach_seam / when the substrate didn't
        # capture it -- the sink degrades to a head-only update.
        self._last_context_obs = None  # dict | None

        logger.info("LuthiModel bridge initialized (model not yet loaded)")

    # ------------------------------------------------------------------
    # Training-seam attachment (Phase 3 of the 2026-06-15 seam plan)
    # ------------------------------------------------------------------

    def attach_seam(
        self, *, actor=None, sink=None, resilient: bool = False,
        async_mode: str = "off", queue_maxsize: int = 64,
        dead_letter_path=None,
    ) -> None:
        """Wire in the M9 actor / transition sink for this cognitive cycle.

        ``async_mode`` (§4, Item #6):
          * ``"off"`` (default) -- legacy synchronous seam. ``observe_transition``
            runs inline on the cycle thread; no model_lock is taken. Existing
            behavior, byte-for-byte.
          * ``"drain"`` -- route transitions through an :class:`AsyncLearner` in
            synchronous drain mode (deterministic, no background thread) under
            the coarse model_lock. Same thread, but exercises the async code
            path; the deterministic mode tests assert against.
          * ``"threaded"`` -- run the learner on a dedicated background thread.
            The actor uses the habit-only fast path (``select_action(budget=0)``)
            and enqueues a detached transition; the learner trains at its own
            cadence under the model_lock. ``queue_maxsize`` bounds the queue
            (blocking put -> backpressure, never silent loss).

        Requires ``sink`` for any non-"off" mode (there is nothing to learn from
        without one).

        Both args are optional and independent: passing only ``actor`` runs
        the M9 act path each cycle (decoder selection through the substrate)
        without persisting transitions; passing only ``sink`` accumulates
        lived experience for a separately-driven actor; passing both gives
        the full Sanctuary-as-actor flow the seam plan describes.

        The typical configuration is ``actor=sink=m9_trainer`` because
        ``M9Trainer`` satisfies both Protocols. No assumption is made here;
        Sanctuary can pass distinct objects if a future split warrants.

        ``resilient=False`` (default): seam-call exceptions propagate and
        abort the cycle. Project ethos for new training-side wiring is
        crashes over silence -- silent failure on day-one integration is
        how real bugs hide (4.8 review F5, 2026-06-15). Flip to True
        only when empirical operation has identified specific transients
        worth surviving.
        """
        if async_mode not in ("off", "drain", "threaded"):
            raise ValueError(
                f"async_mode must be 'off', 'drain', or 'threaded', got "
                f"{async_mode!r}"
            )
        if async_mode != "off" and sink is None:
            raise ValueError(
                f"async_mode={async_mode!r} requires a sink (nothing to learn "
                f"from without one)."
            )

        # Tear down any prior learner thread before swapping the seam.
        self.shutdown_async_learner()

        self._actor = actor
        self._sink = sink
        self._seam_resilient = resilient
        # Drop the buffered transition state when the seam changes -- the
        # previous _last_s_t was paired with the previous actor's MCTS tree,
        # which is no longer valid.
        self._last_s_t = None
        self._last_action = None
        self._last_action_summary = ""
        self._last_efe_breakdown = {}
        self._last_plan_snapshot = None
        self._last_context_obs = None

        if async_mode == "off":
            self._async_learner = None
        else:
            self._async_learner = AsyncLearner(
                sink, self._model_lock,
                mode=async_mode, maxsize=queue_maxsize, resilient=resilient,
                dead_letter_path=dead_letter_path,
            )
            self._async_learner.start()  # no-op for drain mode

    def shutdown_async_learner(self) -> None:
        """Stop and join the background learner thread, if any.

        Idempotent. Call on Sanctuary shutdown (or before re-attaching a seam).
        Re-raises a non-resilient learner-thread error so a background failure
        cannot vanish silently.
        """
        if self._async_learner is not None:
            self._async_learner.stop()
            self._async_learner = None

    def _model_critical_section(self):
        """The shared model_lock. Held across any span that reads/writes the
        living model off the cycle's serialized path: the actor's full
        _think_sync span (modulate -> generate -> select(budget=0) -> restore)
        AND consolidate()'s living-weight writes.

        Always returns the real lock -- uncontended when no async learner is
        attached, but no longer a nullcontext. Item #6 Phase-3 W1 offloads
        consolidate() to a worker thread even in async_mode="off", so the old
        "nullcontext in sync mode" rationale (the sync path is single-threaded,
        byte-for-byte unchanged) is stale: a worker-thread consolidate() can now
        overlap a loop-thread reader of the living weights. An uncontended lock
        is the correct price for making that safe."""
        return self._model_lock

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self):
        """Load the Luthi model from an encrypted checkpoint.

        Call once during Sanctuary startup, before the cognitive cycle begins.
        Automatically detects model type and device (DirectML, CUDA, or CPU).
        """
        if self._loaded:
            return

        # Resolve where the Luthi codebase lives. Priority:
        #   1. config.luthi_path (explicit)
        #   2. $LUTHI_PATH
        #   3. luthi already importable (no path manipulation needed)
        self._ensure_luthi_importable()

        from luthi.sanctuary_interface import load_model

        # Device selection — same logic as luthi/generate.py
        try:
            import torch_directml
            self.device = torch_directml.device()
            device_name = "DirectML (AMD GPU)"
        except ImportError:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0)
            else:
                self.device = torch.device("cpu")
                device_name = "CPU"

        logger.info(
            "Loading Luthi model: checkpoint=%s, device=%s",
            self.config.checkpoint_path,
            device_name,
        )

        t0 = time.monotonic()
        loaded = load_model(
            self.config.checkpoint_path,
            self.config.checkpoint_password,
            self.device,
        )
        self.model = loaded.model.to(self.device)
        self.tokenizer = loaded.tokenizer
        self.model_config = loaded.config
        load_time = time.monotonic() - t0

        # Snapshot the base modulatable state before any CfC modulation.
        # Each cycle restores to this snapshot so modulation never drifts.
        from luthi.sanctuary_interface import snapshot_modulatable_state
        self._base_snapshot = snapshot_modulatable_state(self.model)

        self._loaded = True
        logger.info(
            "Luthi model loaded: %dd, %d blocks, epoch %d, %.1fs",
            self.model_config["d_model"],
            self.model_config["n_blocks"],
            loaded.epoch,
            load_time,
        )

    def load_from_objects(
        self,
        model: torch.nn.Module,
        tokenizer,
        model_config: dict,
    ) -> None:
        """Initialize the bridge from already-loaded objects.

        Use this instead of :meth:`load` when the host process has built
        or loaded the Luthi model another way (in-memory construction,
        plain checkpoint, test fixture). Skips the encrypted-checkpoint
        loader entirely. Useful for integration tests and for embedded
        deployments where the model is constructed in the same process.

        Args:
            model: A Luthi ``nn.Module`` (LuthiLM, SpikingLuthiLM,
                MultimodalLuthiLM, etc).
            tokenizer: The tokenizer paired with ``model``.
            model_config: Config dict with at minimum ``d_model``,
                ``n_blocks``, ``seq_len`` keys.
        """
        if self._loaded:
            raise RuntimeError("LuthiModel is already loaded")

        self._ensure_luthi_importable()

        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        from luthi.sanctuary_interface import snapshot_modulatable_state
        self._base_snapshot = snapshot_modulatable_state(self.model)
        self._loaded = True
        logger.info(
            "Luthi model loaded from objects: %dd, %d blocks",
            self.model_config.get("d_model", "?"),
            self.model_config.get("n_blocks", "?"),
        )

    def _ensure_luthi_importable(self) -> None:
        """Make the ``luthi`` package importable.

        Tries explicit ``luthi_path``, then ``$LUTHI_PATH``, then assumes
        ``luthi`` is already on the import path (e.g. installed via uv).
        Raises ``ImportError`` with an actionable message if none work.
        """
        candidate = self.config.luthi_path or os.environ.get(_LUTHI_PATH_ENV, "")
        if candidate:
            candidate = str(Path(candidate).expanduser())
            if not Path(candidate).exists():
                raise FileNotFoundError(
                    f"LuthiModel path does not exist: {candidate} "
                    f"(set via LuthiModelConfig.luthi_path or ${_LUTHI_PATH_ENV})"
                )
            if candidate not in sys.path:
                sys.path.insert(0, candidate)

        # Whether or not we just inserted a path, verify the import works
        # so failures surface here instead of mid-load.
        try:
            importlib.import_module("luthi.sanctuary_interface")
        except ImportError as exc:
            raise ImportError(
                f"Could not import luthi.sanctuary_interface. Either install "
                f"the luthi package, or set the {_LUTHI_PATH_ENV} environment "
                f"variable (or LuthiModelConfig.luthi_path) to the LuthiModel "
                f"checkout root."
            ) from exc

    # ------------------------------------------------------------------
    # ModelProtocol implementation
    # ------------------------------------------------------------------

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        """Process cognitive input through the living weight model.

        This is the ModelProtocol interface called by CognitiveCycle.
        Runs the forward pass in a thread to avoid blocking the async loop.
        """
        if not self._loaded:
            self.load()

        self.cycle_count += 1
        start = time.monotonic()

        output = await asyncio.to_thread(self._think_sync, cognitive_input)

        latency = time.monotonic() - start
        self._total_calls += 1
        self._total_latency += latency

        logger.debug(
            "Cycle %d: %.2fs, inner_speech=%d chars",
            self.cycle_count,
            latency,
            len(output.inner_speech),
        )

        return output

    def _think_sync(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        """Synchronous core — runs on a worker thread."""
        from luthi.sanctuary_interface import get_introspection

        # §4 deadlock-safe enqueue: the ACTOR's whole model-touching span runs
        # under _model_critical_section() (the coarse model_lock when an async
        # learner is attached, else a nullcontext). The completed transition is
        # captured by the seam but enqueued AFTER the lock releases -- a blocking
        # put() while holding the lock would wait on a learner that needs the
        # lock to drain. `pending` carries it out of the critical section.
        pending: Optional[Transition] = None
        with self._model_critical_section():
            # 1. Pre-generation introspection snapshot
            if self.config.introspect:
                self._pre_state = get_introspection(self.model)

            # 2. CfC modulation — affective system shapes living dynamics
            self._apply_cfc_modulation(cognitive_input.experiential_state)

            try:
                # 3. Format input as text
                prompt = self._format_input(cognitive_input)

                # 3b. Encode any sensory percepts (audio waveform / image
                #     tensor) so they ride into generation as cross-modal
                #     context, not just as text descriptions in the prompt.
                #     Dual path: the tensor flows to the trunk via the encoder;
                #     the text description still seeds the prompt for context.
                audio_tokens, vision_tokens = self._encode_sensory_percepts(
                    cognitive_input
                )

                # 4a (2026-06-16): when a seam is attached and the substrate
                # exposes the v2 multimodal-PC encoder API, inner-speech
                # generation captures s_t + ctx_latents from its own step-0
                # forward. That kills the separate encode_state pass Phase 3
                # used to run (4.8's F7a-followup probe: double-plasticity
                # measurably moves generation). No seam, or a substrate
                # without ``encode()``, falls through to the same generation
                # path return_state=False has always taken.
                seam_wants_state = (
                    (self._actor is not None or self._sink is not None)
                    and hasattr(self.model, "encode")
                )

                # 4. Generate inner speech (living inference)
                if seam_wants_state:
                    inner_speech, gen_state = self._generate_inner_speech(
                        prompt,
                        audio_tokens=audio_tokens,
                        vision_tokens=vision_tokens,
                        return_state=True,
                    )
                else:
                    inner_speech = self._generate_inner_speech(
                        prompt,
                        audio_tokens=audio_tokens,
                        vision_tokens=vision_tokens,
                    )
                    gen_state = None

                # 5. Generate external speech if warranted
                external_speech = self._generate_external_speech(
                    cognitive_input,
                    audio_tokens=audio_tokens,
                    vision_tokens=vision_tokens,
                )

                # 6. M9 training-seam (Phase 4a reorder, 2026-06-16). Order is:
                #    generate -> observe(prior cycle's transition, with this
                #    cycle's s_t as s_next) -> select(next cycle's action).
                #    The F4 snapshot fix relaxed the old observe-before-select
                #    constraint -- the snapshot carries the plan that produced
                #    the realized action, so a later select_action doesn't
                #    corrupt the realized transition. inner-speech's encode
                #    pass produced gen_state; the seam consumes it directly,
                #    no separate encode_state call. In async mode the seam
                #    returns the completed prior-cycle transition for enqueue
                #    AFTER the lock (deadlock guard); in sync mode it runs
                #    observe_transition inline and returns None.
                if gen_state is not None:
                    pending = self._run_training_seam(gen_state)

            finally:
                # 7. Always restore base parameters after CfC modulation
                self._restore_base_parameters()

            # 7b. Post-generation introspection snapshot (still inside the
            #     actor critical section: it reads live model dynamics the
            #     learner mutates).
            if self.config.introspect:
                self._post_state = get_introspection(self.model)
                self._compute_introspection_delta()

        # 8. Enqueue the completed transition OUTSIDE model_lock (deadlock
        #    guard). In threaded mode this is a blocking put -> backpressure,
        #    never silent loss; in drain mode it runs observe synchronously.
        if pending is not None and self._async_learner is not None:
            self._async_learner.submit(pending)

        # 9. Translate neural dynamics → CognitiveOutput (cached introspection
        #    state only; no live-model read, so it is safe outside the lock).
        return self._build_output(
            cognitive_input=cognitive_input,
            inner_speech=inner_speech,
            external_speech=external_speech,
        )

    def _run_training_seam(self, state) -> Optional[Transition]:
        """Dispatch the training seam by mode.

        Sync (async_mode="off"): run observe_transition inline and select with
        the default budget -- the legacy path, unchanged. Returns None.

        Async (drain/threaded): capture the completed prior-cycle transition as
        a detached :class:`Transition`, select the next action via the
        habit-only fast path (budget=0), and return the transition for enqueue
        AFTER the model_lock releases (deadlock guard).
        """
        if self._async_learner is not None:
            return self._run_training_seam_async(state)
        self._run_training_seam_sync(state)
        return None

    def _run_training_seam_async(self, state) -> Optional[Transition]:
        """Async seam: capture the prior-cycle transition + habit-only select.

        Runs inside the actor critical section (model_lock). Does NOT call
        observe_transition -- that is the learner's job, off this thread. The
        completed transition is captured from the buffered ``_last_*`` BEFORE
        ``select_action`` overwrites them, then returned to ``_think_sync`` for
        enqueue outside the lock.

        select_action uses ``budget=0`` (habit-net fast path, no MCTS on the hot
        path -- §4). Consequence: the captured ``plan_snapshot`` is degenerate
        (no MCTS visits), so the learner's habit-distill takes the sample-NLL
        fallback. Off-path learner-side MCTS distill lands in §6; until then this
        is the accepted §4 behavior (action quality is not the smoke metric).
        """
        from luthi.sanctuary_interface import select_action

        s_t = state.s_t
        ctx_latents = state.ctx_latents
        context_obs = state.context_obs

        # 1. Capture the COMPLETED previous-cycle transition before select
        #    overwrites _last_*. Detach every tensor: no autograd graph may
        #    cross the queue (assert_detached re-checks at submit). The F4
        #    self-describing snapshot makes observe-after-select safe.
        pending: Optional[Transition] = None
        if (
            self._sink is not None
            and self._last_s_t is not None
            and self._last_action is not None
        ):
            pending = Transition(
                s_t=self._last_s_t.detach().clone(),
                a_t=self._last_action.detach().clone(),
                s_next=s_t.detach().clone(),  # realized pooled next state
                counterpart_present=False,
                time_since_emission=0.0,
                # PlanSnapshot fields are detached at construction; pass through.
                plan_snapshot=self._last_plan_snapshot,
                context_obs=detach_context_obs(self._last_context_obs),
            )

        # 2. Actor fast path: habit-only select (budget=0). No MCTS on the hot
        #    path; the learner owns the tree.
        if self._actor is not None:
            if self._seam_resilient:
                try:
                    selection = select_action(
                        self._actor, s_t, context_latents=ctx_latents, budget=0,
                    )
                except Exception:
                    logger.exception(
                        "select_action failed at cycle %d (resilient); "
                        "transition captured but no new action selected.",
                        self.cycle_count,
                    )
                    self._last_action = None
                    self._last_action_summary = ""
                    self._last_efe_breakdown = {}
                    self._last_plan_snapshot = None
                    self._last_context_obs = context_obs
                    self._last_s_t = s_t
                    return pending
            else:
                selection = select_action(
                    self._actor, s_t, context_latents=ctx_latents, budget=0,
                )
            self._last_action = selection.action
            self._last_action_summary = selection.readable_summary
            self._last_efe_breakdown = dict(selection.efe_breakdown)
            self._last_plan_snapshot = selection.plan_snapshot
        else:
            self._last_action = None
            self._last_action_summary = ""
            self._last_efe_breakdown = {}
            self._last_plan_snapshot = None

        self._last_context_obs = context_obs
        self._last_s_t = s_t
        return pending

    def _run_training_seam_sync(self, state) -> None:
        """Run the [observe last] -> select_action stage on a captured state.

        Phase 4a (2026-06-16): the encoder pass is no longer a separate
        ``encode_state`` call -- it is the step-0 forward of inner-speech
        generation, captured via ``return_state=True``. ``state`` is the
        ``GenerationState`` that capture produced. The seam consumes
        ``state.s_t`` (canonical via :func:`compute_s_t`) and
        ``state.ctx_latents`` (the unpooled latents the MCTS predictor
        needs).

        Stores the chosen action's readable_summary + EFE breakdown for
        diagnostics; they no longer flow into ``CognitiveOutput.predictions``
        (4.8 review F2/F3).
        """
        from luthi.sanctuary_interface import (
            observe_transition,
            select_action,
        )

        s_t = state.s_t
        ctx_latents = state.ctx_latents
        # Raw step-0 inputs that produced THIS cycle's s_t/ctx_latents.
        # Buffered into _last_context_obs below so the NEXT cycle can hand
        # it to the sink as the context that produced the transition's
        # starting state (Item #6 lived re-encode). None when the
        # substrate didn't capture it -> sink takes the head-only path.
        context_obs = state.context_obs

        # Close the previous cycle's transition before resetting the tree.
        # No try/except by default: crashes over silence (4.8 review F5).
        # If attach_seam(resilient=True) was set, transient failures are
        # logged-and-swallowed -- but only after we have empirical
        # evidence of which transients are worth surviving.
        #
        # plan_snapshot threads the previous cycle's frozen MCTS capture
        # through ctx so the sink reads it from there rather than from
        # live trainer state (4.8 review F4). Passed even when None, so
        # the sink's no-snapshot-degenerate path is taken explicitly.
        if (
            self._sink is not None
            and self._last_s_t is not None
            and self._last_action is not None
        ):
            if self._seam_resilient:
                try:
                    observe_transition(
                        self._sink,
                        self._last_s_t,
                        self._last_action,
                        s_t,
                        counterpart_present=False,
                        time_since_emission=0.0,
                        plan_snapshot=self._last_plan_snapshot,
                        # Item #6: the previous cycle's raw context (which
                        # produced _last_s_t) so the learner can re-encode
                        # it; the realized next-state target is the s_next
                        # positional (current pooled s_t) above. None-safe:
                        # _last_context_obs is None on the first populated
                        # cycle -> the sink takes the head-only path.
                        context_obs=self._last_context_obs,
                    )
                except Exception:
                    logger.exception(
                        "observe_transition failed at cycle %d; "
                        "continuing (resilient mode).",
                        self.cycle_count,
                    )
            else:
                observe_transition(
                    self._sink,
                    self._last_s_t,
                    self._last_action,
                    s_t,
                    counterpart_present=False,
                    time_since_emission=0.0,
                    plan_snapshot=self._last_plan_snapshot,
                    # Item #6: see the resilient branch above.
                    context_obs=self._last_context_obs,
                )

        # Run the actor's plan + select. _last_action_summary /
        # _last_efe_breakdown are retained for diagnostics / future
        # M9-decoder-driven emissions; they no longer flow into
        # CognitiveOutput.predictions (4.8 review F2/F3).
        if self._actor is not None:
            if self._seam_resilient:
                try:
                    selection = select_action(
                        self._actor, s_t, context_latents=ctx_latents,
                    )
                except Exception:
                    logger.exception(
                        "select_action failed at cycle %d; transition "
                        "closed but no new action selected (resilient mode).",
                        self.cycle_count,
                    )
                    self._last_action = None
                    self._last_action_summary = ""
                    self._last_efe_breakdown = {}
                    self._last_plan_snapshot = None
                    self._last_context_obs = context_obs
                    self._last_s_t = s_t
                    return
            else:
                selection = select_action(
                    self._actor, s_t, context_latents=ctx_latents,
                )
            self._last_action = selection.action
            self._last_action_summary = selection.readable_summary
            self._last_efe_breakdown = dict(selection.efe_breakdown)
            self._last_plan_snapshot = selection.plan_snapshot
        else:
            self._last_action = None
            self._last_action_summary = ""
            self._last_efe_breakdown = {}
            self._last_plan_snapshot = None

        self._last_context_obs = context_obs
        self._last_s_t = s_t

    # ------------------------------------------------------------------
    # Input formatting
    # ------------------------------------------------------------------

    def _format_input(self, ci: CognitiveInput) -> str:
        """Convert CognitiveInput into a text prompt for Luthi.

        Produces natural text (not JSON instructions). The living weight
        model processes token sequences — it needs context, not commands.
        The most important signal is the stream of thought continuity.
        """
        parts = []

        # Stream of thought — continuity from the previous cycle
        if ci.previous_thought and ci.previous_thought.inner_speech:
            parts.append(ci.previous_thought.inner_speech)

        # New sensory input
        for percept in ci.new_percepts:
            if percept.modality == "language":
                source = percept.source or "someone"
                parts.append(f"{source} says: {percept.content}")
            elif percept.modality == "temporal":
                parts.append(percept.content)
            else:
                parts.append(f"[{percept.modality}] {percept.content}")

        # Prediction errors — surprise drives learning
        for error in ci.prediction_errors[:3]:
            parts.append(
                f"I expected {error.predicted} but {error.actual} happened."
            )

        # Surfaced memories
        for mem in ci.surfaced_memories[:3]:
            parts.append(f"I remember: {mem.content}")

        # Emotional context
        if ci.emotional_state.felt_quality:
            parts.append(f"I feel {ci.emotional_state.felt_quality}.")

        # Self-model context
        if ci.self_model.current_state:
            parts.append(ci.self_model.current_state)

        # Charter identity (brief, important for grounding)
        if ci.self_authored_identity:
            parts.append(ci.self_authored_identity)

        prompt = " ".join(parts) if parts else "I am here."

        # Truncate to fit model context window (rough char estimate)
        max_chars = self.model_config.get("seq_len", 128) * 4
        if len(prompt) > max_chars:
            prompt = prompt[-max_chars:]

        return prompt

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _encode_sensory_percepts(
        self, ci: CognitiveInput
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Encode any tensor-bearing percepts into model-ready tokens.

        Returns ``(audio_tokens, vision_tokens)`` — at most one is non-None,
        respecting the 1024d/seq_len=128 context-window constraint that
        forbids routing more than one sensory modality per cycle. When both
        modalities arrive in the same cycle, vision wins (denser semantic
        content per token). The 4096d model in Track 3 will lift this limit.

        The model is required to be a multimodal variant (with both encoders)
        for any encoding to happen. Non-multimodal models silently return
        ``(None, None)`` — text-only fallback.
        """
        if not (
            hasattr(self.model, "audio_encoder")
            and hasattr(self.model, "vision_encoder")
        ):
            return None, None

        from luthi.sanctuary_interface import encode_audio, encode_vision

        vision_percept = None
        audio_percept = None
        for percept in ci.new_percepts:
            if percept.tensor_data is None:
                continue
            if percept.modality == "visual" and vision_percept is None:
                vision_percept = percept
            elif percept.modality == "audio" and audio_percept is None:
                audio_percept = percept

        if vision_percept is not None:
            image = vision_percept.tensor_data
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            return None, encode_vision(self.model, image)

        if audio_percept is not None:
            waveform = audio_percept.tensor_data
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.device)
            return encode_audio(self.model, waveform), None

        return None, None

    def _generate_inner_speech(
        self,
        prompt: str,
        *,
        audio_tokens: Optional["torch.Tensor"] = None,
        vision_tokens: Optional["torch.Tensor"] = None,
        return_state: bool = False,
    ):
        """Generate the entity's private inner thought.

        Uses autoregressive decoding with the living weight model.
        In living mode, Hebbian self-modification fires during each
        forward pass — the model changes from the act of thinking.

        Optional pre-encoded ``audio_tokens`` / ``vision_tokens`` route
        sensory context into the trunk on the first forward step, so
        the entity's first generated token is conditioned on the actual
        sensory channel rather than only the text description.

        When ``return_state`` is True, captures the encoder state from
        step 0 of generation and returns ``(generated_text,
        GenerationState)`` -- the Phase 4a path the training seam
        consumes instead of running a separate encode_state.
        """
        from luthi.sanctuary_interface import generate_with_context

        max_seq_len = self.model_config.get("seq_len", 128)

        # Count prompt tokens before generation so we can cleanly
        # separate generated tokens from the prompt. BPE encode/decode
        # is not a perfect roundtrip, so slicing by character length
        # can corrupt text at the boundary.
        prompt_token_ids = self.tokenizer.encode(prompt)
        prompt_token_count = len(prompt_token_ids)

        gen_result = generate_with_context(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_inner_tokens,
            audio_tokens=audio_tokens,
            vision_tokens=vision_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            max_seq_len=max_seq_len,
            living=self.config.living,
            stream=False,
            return_state=return_state,
        )
        if return_state:
            full_text, gen_state = gen_result
        else:
            full_text = gen_result
            gen_state = None

        # Extract generated portion by re-encoding and slicing tokens,
        # avoiding BPE roundtrip mismatch at the character level.
        full_token_ids = self.tokenizer.encode(full_text)
        generated_token_ids = full_token_ids[prompt_token_count:]
        generated = self.tokenizer.decode(generated_token_ids).strip()

        if generated_token_ids:
            self._total_tokens_generated += len(generated_token_ids)

        text = generated if generated else f"[cycle {self.cycle_count}]"
        if return_state:
            return text, gen_state
        return text

    def _generate_external_speech(
        self,
        ci: CognitiveInput,
        *,
        audio_tokens: Optional["torch.Tensor"] = None,
        vision_tokens: Optional["torch.Tensor"] = None,
    ) -> Optional[str]:
        """Generate a directed response when the entity has something to say.

        Triggers when there's a language percept from a user. The entity
        speaks freely — no external gating or inhibition. If the model
        generates speech, it goes out.

        Uses a separate short generation pass with the response prompt,
        in living mode so the act of responding changes the model. Any
        sensory tokens encoded for this cycle are also routed into the
        response so external speech is conditioned on the same channel
        the inner thought was.
        """
        user_message = None
        user_source = "someone"
        for percept in ci.new_percepts:
            if percept.modality == "language" and percept.content.strip():
                user_message = percept.content
                user_source = percept.source or "someone"
                break

        if user_message is None:
            return None

        from luthi.sanctuary_interface import generate_with_context

        response_prompt = f"{user_source} says: {user_message}\nI respond: "
        max_seq_len = self.model_config.get("seq_len", 128)

        # Count prompt tokens for clean extraction
        prompt_token_ids = self.tokenizer.encode(response_prompt)
        prompt_token_count = len(prompt_token_ids)

        full_text = generate_with_context(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=response_prompt,
            max_tokens=self.config.max_external_tokens,
            audio_tokens=audio_tokens,
            vision_tokens=vision_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            max_seq_len=max_seq_len,
            living=self.config.living,
            stream=False,
        )

        # Extract generated portion using token slicing (same as inner speech)
        full_token_ids = self.tokenizer.encode(full_text)
        generated_token_ids = full_token_ids[prompt_token_count:]
        response = self.tokenizer.decode(generated_token_ids).strip()

        if generated_token_ids:
            self._total_tokens_generated += len(generated_token_ids)

        return response if response else None

    # ------------------------------------------------------------------
    # CfC modulation — affective system shapes living weight dynamics
    # ------------------------------------------------------------------

    def _restore_base_parameters(self):
        """Restore the model to the base snapshot taken at load time.

        Called per cycle so CfC modulation never accumulates across cycles.
        """
        if self._base_snapshot is None:
            return
        from luthi.sanctuary_interface import restore_modulation
        restore_modulation(self.model, self._base_snapshot)

    def _apply_cfc_modulation(self, signals: ExperientialSignals):
        """Modulate Luthi's living parameters based on CfC cell signals.

        The affective system shapes cognition through four independent channels:

        - Arousal → Hebbian learning rate (more alert → faster learning)
        - Precision → spike threshold (more precise → fewer, more selective)
        - Valence → excitability bias (approach amplifies, withdrawal dampens)
        - Attention → salience threshold (more attention → broader episodes)

        Each channel maps to its naturally-corresponding living-weight parameter
        rather than being composited into a single scalar — keeping the channels
        independent makes effects observable per-cell and easier to debug.

        Modulation is per-cycle: the base snapshot taken at load time is
        restored after each think() call so modulation never accumulates.
        Goes through ``luthi.sanctuary_interface`` rather than reaching
        into Luthi internals — that adapter is the contract surface.
        """
        from luthi.sanctuary_interface import apply_external_modulation

        arousal = signals.affect_arousal
        precision = signals.precision_weight
        valence = signals.affect_valence
        salience = signals.attention_salience

        # Arousal → Hebbian learning rate
        # Range: 0.5x (calm, arousal=0) to 2.0x (peak, arousal=1)
        plasticity_scale = (
            (0.5 + 1.5 * arousal) * self.config.arousal_plasticity_scale
        )
        # Precision → spike threshold
        # Higher precision = higher threshold = fewer spikes = more selective
        # Range: 0.75x (low precision) to 1.25x (high precision)
        spike_threshold_scale = (
            (0.75 + 0.5 * precision) * self.config.precision_threshold_scale
        )
        # Valence → excitability bias (additive, shifts sigmoid operating point)
        # Range: ±0.1 at full saturation, scaled by config gain.
        excitability_bias = (
            valence * self.config.valence_excitability_scale * 0.1
        )
        # Attention → salience threshold (multiplicative)
        # High salience = lower threshold = broader episode storage.
        # Range: 0.5x (full attention) to 1.0x (no attention).
        salience_threshold_scale = (
            (1.0 - 0.5 * salience) * self.config.salience_threshold_scale
        )

        apply_external_modulation(
            self.model,
            plasticity_scale=plasticity_scale,
            spike_threshold_scale=spike_threshold_scale,
            excitability_bias=excitability_bias,
            salience_threshold_scale=salience_threshold_scale,
        )

    # ------------------------------------------------------------------
    # Introspection — cognitive proprioception channel
    # ------------------------------------------------------------------

    def _compute_introspection_delta(self):
        """Compute changes in neural state from this cycle's processing."""
        self._introspection_delta = {}

        if not self._pre_state or not self._post_state:
            return

        pre_blocks = self._pre_state.get("blocks", [])
        post_blocks = self._post_state.get("blocks", [])

        if not pre_blocks or not post_blocks:
            return

        plasticity_deltas = []
        drift_deltas = []
        spike_fractions = []
        membrane_deltas = []

        for pre_b, post_b in zip(pre_blocks, post_blocks):
            if "plasticity_mean" in pre_b and "plasticity_mean" in post_b:
                plasticity_deltas.append(
                    post_b["plasticity_mean"] - pre_b["plasticity_mean"]
                )
            if "set_point_drift" in pre_b and "set_point_drift" in post_b:
                drift_deltas.append(
                    post_b["set_point_drift"] - pre_b["set_point_drift"]
                )
            if "spike_fraction" in post_b:
                spike_fractions.append(post_b["spike_fraction"])
            if "membrane_mean" in pre_b and "membrane_mean" in post_b:
                membrane_deltas.append(
                    post_b["membrane_mean"] - pre_b["membrane_mean"]
                )

        if plasticity_deltas:
            self._introspection_delta["plasticity_change"] = (
                sum(plasticity_deltas) / len(plasticity_deltas)
            )
        if drift_deltas:
            self._introspection_delta["drift_change"] = (
                sum(drift_deltas) / len(drift_deltas)
            )
        if spike_fractions:
            self._introspection_delta["mean_spike_fraction"] = (
                sum(spike_fractions) / len(spike_fractions)
            )
        if membrane_deltas:
            self._introspection_delta["membrane_change"] = (
                sum(membrane_deltas) / len(membrane_deltas)
            )

        # Overall activity — how much the model *changed* this cycle.
        # Only include actual deltas, not absolute values like spike fraction.
        delta_keys = ("plasticity_change", "drift_change", "membrane_change")
        self._introspection_delta["activity_level"] = sum(
            abs(self._introspection_delta.get(k, 0.0)) for k in delta_keys
        )

    def get_augmented_experiential_signals(self) -> dict[str, list[float]]:
        """Return Luthi's neural state as knowledge cell signals.

        This is the cognitive proprioception channel. The entity can
        observe its own plasticity, drift, spike patterns, and membrane
        dynamics. These signals are injected into the next cycle's
        ExperientialSignals.knowledge_signals dict.

        Returns:
            Dict mapping signal names to float lists, compatible with
            ExperientialSignals.knowledge_signals schema.
        """
        if not self._post_state:
            return {}

        signals: dict[str, list[float]] = {}
        blocks = self._post_state.get("blocks", [])

        # Per-block signals. Each accumulator is keyed by a v1 or v2
        # introspection field; the substrate populates whichever apply.
        # Empty lists are skipped at publish time so v1-only and v2-only
        # signals never collide on the entity's knowledge_signals dict.
        plasticities: list[float] = []
        drifts: list[float] = []
        # v1-only (spiking variants):
        spike_fracs: list[float] = []
        membranes: list[float] = []
        excitabilities: list[float] = []
        # v2-only (predictive coding):
        error_accs: list[float] = []
        pred_frobs: list[float] = []
        precisions: list[float] = []

        for block in blocks:
            if "plasticity_mean" in block:
                plasticities.append(block["plasticity_mean"])
            if "set_point_drift" in block:
                drifts.append(block["set_point_drift"])
            if "spike_fraction" in block:
                spike_fracs.append(block["spike_fraction"])
            if "membrane_mean" in block:
                membranes.append(block["membrane_mean"])
            if "excitability_mean" in block:
                excitabilities.append(block["excitability_mean"])
            if "error_acc_mean" in block:
                error_accs.append(block["error_acc_mean"])
            if "pred_frob" in block:
                pred_frobs.append(block["pred_frob"])
            if "precision_mean" in block:
                precisions.append(block["precision_mean"])

        if plasticities:
            signals["luthi_plasticity"] = plasticities
        if drifts:
            signals["luthi_drift"] = drifts
        if spike_fracs:
            signals["luthi_spike_fraction"] = spike_fracs
        if membranes:
            signals["luthi_membrane"] = membranes
        if excitabilities:
            signals["luthi_excitability"] = excitabilities
        if error_accs:
            signals["luthi_error_acc"] = error_accs
        if pred_frobs:
            signals["luthi_pred_frob"] = pred_frobs
        if precisions:
            signals["luthi_precision"] = precisions

        # Aggregate delta signals
        delta = self._introspection_delta
        if delta:
            signals["luthi_delta"] = [
                delta.get("plasticity_change", 0.0),
                delta.get("drift_change", 0.0),
                delta.get("membrane_change", 0.0),
                delta.get("activity_level", 0.0),
            ]

        return signals

    # ------------------------------------------------------------------
    # Neural → Cognitive translation
    # ------------------------------------------------------------------

    def _neural_to_valence_shift(self) -> float:
        """Map neural dynamics to emotional valence shift (-1 to 1).

        Drift toward set points → positive (settling, comfort).
        Drift away from set points → negative (disequilibrium).
        Learning (plasticity increase) → slight positive.

        Typical drift_change values: -0.001 to +0.001
        Typical plast_change values: -0.01 to +0.01
        """
        delta = self._introspection_delta
        if not delta:
            return 0.0

        drift_change = delta.get("drift_change", 0.0)
        plast_change = delta.get("plasticity_change", 0.0)

        # Moving toward equilibrium (drift decreasing) feels positive.
        # Scale: drift_change of ±0.001 maps to valence of ∓0.5
        valence = -drift_change * 500.0

        # Active learning adds slight positive valence
        # Scale: plast_change of ±0.01 maps to ±0.2
        valence += plast_change * 20.0

        return max(-1.0, min(1.0, valence))

    def _neural_to_arousal_shift(self) -> float:
        """Map neural dynamics to arousal shift (-1 to 1).

        Spike fraction above baseline → increased arousal.
        High overall activity → increased arousal.
        """
        delta = self._introspection_delta
        if not delta:
            return 0.0

        spike_frac = delta.get("mean_spike_fraction", 0.3)
        activity = delta.get("activity_level", 0.0)

        # Spike fraction deviation from baseline (~0.3)
        arousal = (spike_frac - 0.3) * 2.0
        arousal += activity * 0.5

        return max(-1.0, min(1.0, arousal))

    # ------------------------------------------------------------------
    # Derived cognitive outputs
    # ------------------------------------------------------------------

    def _determine_attention_guidance(self) -> AttentionGuidance:
        """Derive attention guidance from per-block spike patterns.

        Blocks with high spike fractions are actively processing.
        Blocks with low spike fractions are quiet.
        """
        focus = []
        deprioritize = []

        if self._post_state:
            for block in self._post_state.get("blocks", []):
                spike_frac = block.get("spike_fraction", 0.3)
                idx = block["block"]

                if spike_frac > 0.5:
                    focus.append(f"processing_block_{idx}")
                elif spike_frac < 0.1:
                    deprioritize.append(f"processing_block_{idx}")

        return AttentionGuidance(
            focus_on=focus[:5],
            deprioritize=deprioritize[:5],
        )

    def _determine_memory_ops(self) -> list[MemoryOp]:
        """Derive memory operations from neural significance.

        High activity (non-feedforward signal) indicates the model
        processed something significant → write episodic memory.
        Periodic journal entries capture the entity's neural trajectory.
        """
        ops = []
        delta = self._introspection_delta

        if not delta:
            return ops

        activity = delta.get("activity_level", 0.0)

        # High activity → this moment mattered → episodic memory
        # Typical activity_level (sum of |delta| values) is 0.001-0.01.
        # Threshold at 0.005 captures genuinely significant processing.
        if activity > 0.005:
            significance = min(10, max(1, int(activity * 1000)))
            ops.append(
                MemoryOp(
                    type="write_episodic",
                    content=(
                        f"Significant neural processing (activity={activity:.4f}, "
                        f"spike_fraction={delta.get('mean_spike_fraction', 0):.3f})"
                    ),
                    significance=significance,
                    tags=["neural", "introspection"],
                )
            )

        # Periodic journal — capture neural trajectory
        if self.cycle_count % 50 == 0:
            entries = []
            for key in ("mean_spike_fraction", "plasticity_change", "drift_change"):
                if key in delta:
                    entries.append(f"{key}={delta[key]:.6f}")
            if entries:
                ops.append(
                    MemoryOp(
                        type="journal",
                        content=(
                            f"Cycle {self.cycle_count} neural state: "
                            + ", ".join(entries)
                        ),
                        significance=3,
                        tags=["introspection", "periodic"],
                    )
                )

        return ops

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------

    def _build_output(
        self,
        cognitive_input: CognitiveInput,
        inner_speech: str,
        external_speech: Optional[str],
    ) -> CognitiveOutput:
        """Assemble CognitiveOutput from generated text + neural dynamics."""
        # felt_quality removed 2026-06-11 per the cognition-leakage
        # cleanup (docs/seam_jurisdiction_2026-06-11.md): the substrate
        # selects, the scaffold transports; felt quality should originate
        # in the substrate's introspection and surface through the
        # entity's own decoders, not via adapter heuristics.
        felt_quality = ""

        # Predictions stays [] until the M9 text decoder is the one
        # emitting. 4.8's 2026-06-15 review F2/F3 flagged that routing
        # the chosen action's debug summary into predictions feeds
        # incoherent input to the sensorium's keyword-matching error
        # path, and that an adapter-authored logistic(-EFE) confidence
        # is the same category of scaffold-authored signal the 2026-06-11
        # cleanup removed. The action selection IS still recorded -- see
        # _last_action / _last_action_summary -- but it does not cross
        # into the CognitiveOutput surface until the substrate emits.
        predictions: list = []

        return CognitiveOutput(
            inner_speech=inner_speech,
            external_speech=external_speech,
            predictions=predictions,
            attention_guidance=self._determine_attention_guidance(),
            memory_ops=self._determine_memory_ops(),
            self_model_updates=SelfModelUpdate(
                current_state=felt_quality,
            ),
            world_model_updates=[],
            goal_proposals=[],
            emotional_state=EmotionalOutput(
                felt_quality=felt_quality,
                valence_shift=self._neural_to_valence_shift(),
                arousal_shift=self._neural_to_arousal_shift(),
            ),
            growth_reflection=None,
            knowledge_cell_requests=[],
        )

    # ------------------------------------------------------------------
    # Sleep consolidation — called by SleepCycleManager
    # ------------------------------------------------------------------

    def consolidate(self):
        """Trigger consolidation in the living weights.

        Called during NREM sleep cycles. Performs three operations:

        1. Set point adjustment: Move set points toward current weights,
           stabilizing what the model has recently learned. Without this,
           homeostatic decay would pull weights back to their old positions.

        2. Plasticity rebalancing: Gently pull per-weight plasticity
           toward the global mean, preventing extreme specialization and
           maintaining the capacity to learn new patterns.

        3. Episode pruning: Future enhancement — weaken low-salience
           episodes and strengthen high-salience ones.

        These operations mirror biological sleep consolidation: the
        brain replays and stabilizes memories during deep sleep while
        rebalancing synaptic strengths.
        """
        if not self._loaded or not hasattr(self.model, "blocks"):
            return

        consolidated_blocks = 0

        # §4 race fix (Window A audit, 2026-06-29): consolidate() writes
        # set_point/plasticity in-place on the living weights while the async
        # learner concurrently writes weight/theta under model_lock, and NREM
        # consolidation overlaps cognition (cognitive_cycle.py: the model still
        # thinks/drains while consolidating). In-place ATen ops release the GIL,
        # so without the lock this is a genuine C++-level data race on the
        # substrate. Hold the shared model_lock for the whole write loop. The
        # lock is non-reentrant but safe: consolidate runs after think() has
        # released it, so there is no self-deadlock. (Brian holds the deeper
        # call -- quiesce the learner during NREM vs. just serialize here; this
        # is the correctness floor.)
        with self._model_critical_section():
            for block in self.model.blocks:
                ffn = getattr(block, "living_ffn", None)
                if ffn is None:
                    continue

                with torch.no_grad():
                    # 1. Set points drift toward current weights
                    if hasattr(ffn, "set_point") and hasattr(ffn, "weight"):
                        delta = ffn.weight.data - ffn.set_point
                        ffn.set_point += delta * self.config.consolidation_rate

                    # 2. Plasticity rebalances toward mean
                    if hasattr(ffn, "plasticity"):
                        mean_p = ffn.plasticity.mean()
                        target = mean_p.expand_as(ffn.plasticity)
                        rate = self.config.plasticity_rebalance_rate
                        ffn.plasticity.mul_(1.0 - rate).add_(target * rate)

                consolidated_blocks += 1

        logger.info(
            "Luthi consolidation: %d blocks processed (cycle %d)",
            consolidated_blocks,
            self.cycle_count,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return performance metrics for monitoring."""
        return {
            "total_calls": self._total_calls,
            "total_latency": self._total_latency,
            "average_latency": (
                self._total_latency / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
            "total_tokens_generated": self._total_tokens_generated,
            "cycle_count": self.cycle_count,
        }

    def get_introspection_state(self) -> dict:
        """Return the most recent introspection data for external monitoring."""
        return {
            "pre": self._pre_state,
            "post": self._post_state,
            "delta": self._introspection_delta,
        }
