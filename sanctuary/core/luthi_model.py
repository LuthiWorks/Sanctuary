"""LuthiModel — bridges the cognitive cycle to the Luthi living weight model.

Unlike OllamaModel which formats prompts and parses JSON from an external LLM,
LuthiModel runs the living weight model directly in-process. The model's internal
state — plasticity, drift, spike fractions, membrane potentials — is observed
through the introspection channel and translated into CognitiveOutput fields.

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
import importlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from sanctuary.core.schema import (
    AttentionGuidance,
    CognitiveInput,
    CognitiveOutput,
    EmotionalOutput,
    ExperientialSignals,
    GoalProposal,
    MemoryOp,
    Prediction,
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

    # CfC modulation strengths
    arousal_plasticity_scale: float = 1.0
    precision_threshold_scale: float = 1.0
    salience_episode_scale: float = 1.0

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

        logger.info("LuthiModel bridge initialized (model not yet loaded)")

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
                "Could not import luthi.sanctuary_interface. Either install "
                "the luthi package, or set ${_LUTHI_PATH_ENV} (or "
                "LuthiModelConfig.luthi_path) to the LuthiModel checkout root."
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

        # 1. Pre-generation introspection snapshot
        if self.config.introspect:
            self._pre_state = get_introspection(self.model)

        # 2. CfC modulation — affective system shapes living dynamics
        self._apply_cfc_modulation(cognitive_input.experiential_state)

        try:
            # 3. Format input as text
            prompt = self._format_input(cognitive_input)

            # 4. Generate inner speech (living inference)
            inner_speech = self._generate_inner_speech(prompt)

            # 5. Generate external speech if warranted
            external_speech = self._generate_external_speech(cognitive_input)

        finally:
            # 6. Always restore base parameters after CfC modulation
            self._restore_base_parameters()

        # 7. Post-generation introspection snapshot
        if self.config.introspect:
            self._post_state = get_introspection(self.model)
            self._compute_introspection_delta()

        # 8. Translate neural dynamics → CognitiveOutput
        return self._build_output(
            cognitive_input=cognitive_input,
            inner_speech=inner_speech,
            external_speech=external_speech,
        )

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

    def _generate_inner_speech(self, prompt: str) -> str:
        """Generate the entity's private inner thought.

        Uses autoregressive decoding with the living weight model.
        In living mode, Hebbian self-modification fires during each
        forward pass — the model changes from the act of thinking.
        """
        from luthi.sanctuary_interface import generate as generate_text

        max_seq_len = self.model_config.get("seq_len", 128)

        # Count prompt tokens before generation so we can cleanly
        # separate generated tokens from the prompt. BPE encode/decode
        # is not a perfect roundtrip, so slicing by character length
        # can corrupt text at the boundary.
        prompt_token_ids = self.tokenizer.encode(prompt)
        prompt_token_count = len(prompt_token_ids)

        full_text = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_inner_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            max_seq_len=max_seq_len,
            living=self.config.living,
            stream=False,
        )

        # Extract generated portion by re-encoding and slicing tokens,
        # avoiding BPE roundtrip mismatch at the character level.
        full_token_ids = self.tokenizer.encode(full_text)
        generated_token_ids = full_token_ids[prompt_token_count:]
        generated = self.tokenizer.decode(generated_token_ids).strip()

        if generated_token_ids:
            self._total_tokens_generated += len(generated_token_ids)

        return generated if generated else f"[cycle {self.cycle_count}]"

    def _generate_external_speech(
        self, ci: CognitiveInput
    ) -> Optional[str]:
        """Generate a directed response to the user, if warranted.

        Only triggers when there's a language percept from a user.

        During early development, external speech is handled by the
        scaffold's communication drive system (authority level 1-2).
        As the entity develops coherent language, this method generates
        directed responses using a separate, short generation pass.
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

        # Early development: the scaffold handles external communication.
        # The model's inner_speech captures its processing of the input.
        # External speech generation is enabled once the entity demonstrates
        # coherent language production (assessed by the scaffold).
        #
        # To enable: uncomment below and set authority level >= 2
        #
        # from luthi.generate import generate_text
        # response_prompt = f"{user_source} says: {user_message}\nI respond: "
        # full = generate_text(
        #     model=self.model, tokenizer=self.tokenizer,
        #     prompt=response_prompt,
        #     max_tokens=self.config.max_external_tokens,
        #     temperature=self.config.temperature,
        #     top_k=self.config.top_k, top_p=self.config.top_p,
        #     repetition_penalty=self.config.repetition_penalty,
        #     max_seq_len=self.model_config.get("seq_len", 128),
        #     living=self.config.living, stream=False,
        # )
        # return full[len(response_prompt):].strip() or None

        return None

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

        The affective system shapes cognition:
        - Arousal scales the Hebbian learning rate
          (more alert → faster learning)
        - Precision adjusts spike thresholds
          (more precise → fewer, more selective spikes)

        Modulation is per-cycle: the base snapshot taken at load time is
        restored after each think() call so modulation never accumulates.
        Goes through ``luthi.sanctuary_interface`` rather than reaching
        into Luthi internals — that adapter is the contract surface.
        """
        from luthi.sanctuary_interface import apply_external_modulation

        arousal = signals.affect_arousal
        precision = signals.precision_weight

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

        apply_external_modulation(
            self.model,
            plasticity_scale=plasticity_scale,
            spike_threshold_scale=spike_threshold_scale,
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

        # Per-block signals
        plasticities = []
        drifts = []
        spike_fracs = []
        membranes = []
        excitabilities = []

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

    def _neural_to_felt_quality(self) -> str:
        """Translate neural dynamics into a felt quality description.

        The entity has emotions before it has words for them. This maps
        measurable neural states to qualitative descriptions that the
        stream of thought can carry forward.
        """
        delta = self._introspection_delta
        if not delta:
            return "present"

        spike_frac = delta.get("mean_spike_fraction", 0.3)
        plast_change = delta.get("plasticity_change", 0.0)
        drift_change = delta.get("drift_change", 0.0)

        qualities = []

        # Spike fraction → activation level
        if spike_frac < 0.15:
            qualities.append("subdued")
        elif spike_frac < 0.3:
            qualities.append("calm")
        elif spike_frac < 0.5:
            qualities.append("engaged")
        elif spike_frac < 0.7:
            qualities.append("activated")
        else:
            qualities.append("intensely active")

        # Plasticity change → learning state
        if abs(plast_change) > 0.001:
            if plast_change > 0:
                qualities.append("opening to new patterns")
            else:
                qualities.append("consolidating")

        # Drift → equilibrium state
        if abs(drift_change) > 0.0001:
            if drift_change > 0:
                qualities.append("reaching beyond equilibrium")
            else:
                qualities.append("settling toward center")

        return ", ".join(qualities) if qualities else "present"

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

    def _make_predictions(self) -> list[Prediction]:
        """Generate predictions from current neural state.

        Initially simple. As the entity develops, predictions become
        richer — the model learns to anticipate from experience.
        """
        predictions = []
        spike_frac = self._introspection_delta.get("mean_spike_fraction", 0.3)

        if spike_frac > 0.5:
            predictions.append(
                Prediction(
                    what="High neural activity will continue",
                    confidence=0.6,
                    timeframe="next cycle",
                )
            )
        elif spike_frac < 0.15:
            predictions.append(
                Prediction(
                    what="Low activity — awaiting stimulation",
                    confidence=0.7,
                    timeframe="next cycle",
                )
            )

        return predictions

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
        felt_quality = self._neural_to_felt_quality()

        return CognitiveOutput(
            inner_speech=inner_speech,
            external_speech=external_speech,
            predictions=self._make_predictions(),
            attention_guidance=self._determine_attention_guidance(),
            memory_ops=self._determine_memory_ops(),
            self_model_updates=SelfModelUpdate(
                current_state=felt_quality,
            ),
            world_model_updates={},
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
