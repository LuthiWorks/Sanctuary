"""SanctuaryRunner — the orchestrator that wires everything together.

Phase 6: Integration + Validation.

This module assembles all Phase 1-5 components into a running system:
  - CognitiveCycle (Phase 1: the thought loop)
  - CognitiveScaffold (Phase 2: validation and integration)
  - Sensorium (Phase 3: sensory input)
  - Motor (Phase 3: action execution)
  - MemorySubstrate (Phase 4: memory system)
  - AwakeningSequence (Phase 5: identity and boot)

The runner handles:
  1. Assembly — creating and connecting all components
  2. Boot — running the awakening sequence
  3. Lifecycle — start/stop/inject input
  4. Motor feedback wiring — closing the sensorimotor loop

This is the single entry point. Everything else is a component.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable, Optional

from sanctuary.core.authority import AuthorityManager
from sanctuary.core.cognitive_cycle import CognitiveCycle, ModelProtocol
from sanctuary.core.context_manager import BudgetConfig
from sanctuary.core.cycle_rate import CycleRateController
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.stimulus_density import (
    SensoriumDensitySource,
    StimulusDensityHeuristic,
)
from sanctuary.core.turbo import (
    MechanicalIntensitySource,
    PCIntensitySource,
    TurboManager,
)
from sanctuary.core.schema import CognitiveOutput, Percept, SelfModelUpdate
from sanctuary.consciousness.sleep_cycle import SleepCycleManager, SleepConfig
from sanctuary.identity.awakening import AwakeningSequence
from sanctuary.identity.self_authored import SelfAuthoredIdentity
from sanctuary.identity.values import ValuesSystem
from sanctuary.memory.journal import JournalConfig
from sanctuary.memory.manager import MemorySubstrate, MemorySubstrateConfig
from sanctuary.monitoring import (
    AttentionHeatmapTracker,
    CommunicationDecisionLogger,
    ConsciousnessTraceRecorder,
    DashboardDataProvider,
)
from sanctuary.monitoring.communication_log import (
    CommunicationDecision as MonitorCommunicationDecision,
)
from sanctuary.motor.motor import Motor
from sanctuary.scaffold.cognitive_scaffold import CognitiveScaffold
from sanctuary.sensorium.sensorium import Sensorium
from sanctuary.tools.builtin import create_default_registry, register_self_knowledge_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Configuration for the SanctuaryRunner."""

    # Cycle timing
    cycle_delay: float = 0.1  # seconds between cycles
    stream_history: int = 10  # how many cycles of thought to retain

    # Sensorium
    silence_threshold: float = 30.0
    silence_reminder_interval: float = 60.0
    max_percept_queue: int = 100

    # Memory
    use_in_memory_store: bool = True  # True for testing, False for production

    # Scaffold
    max_goals: int = 10

    # Identity
    data_dir: str = "data/identity"
    charter_path: Optional[str] = None

    # Context budget
    context_budget: Optional[BudgetConfig] = None

    # Model backend: "placeholder" or "luthi" (ollama was retired 2026-04-30)
    model_backend: str = "placeholder"

    # Luthi model config (only used when model_backend == "luthi")
    luthi_checkpoint: Optional[str] = None
    luthi_password: Optional[str] = None

    # Sleep consolidation
    sleep_enabled: bool = True
    sleep_config: Optional[SleepConfig] = None

    # Turbo (substrate-intensity-driven cycle-rate elevation). The
    # slider (cycle-rate controller) is always wired; this flag toggles
    # whether the turbo state machine watches and engages on intensity.
    # Defaults to on because that's the intended production behavior.
    turbo_enabled: bool = True

    # Turbo trace logging. When set, each turbo.observe() call appends
    # a JSONL line capturing per-source intensity, aggregate intensity,
    # dominant source, state-before / state-after, and current rate.
    # Used for empirical threshold tuning — see the threshold-tuning
    # research log entry. None disables trace logging.
    turbo_trace_path: Optional[str] = None

    # Stimulus-density heuristic (autonomic rate adjustment based on
    # input density). When on, the heuristic proposes slowdown during
    # quiet periods and speedup on fresh input arrival. The entity
    # always wins via cycle_rate_proposal; heuristic only proposes.
    stimulus_density_enabled: bool = True

    # Persistence. When True, the journal, world graph, and CfC
    # experiential layer persist state to disk under data_dir, and
    # restore on next boot. Identity files always persist (separate
    # mechanism). Disable for unit tests that want a clean in-memory
    # runner with no disk side effects.
    persist_state: bool = True

    # CfC experiential layer (precision, affect, attention, goal cells +
    # continuous evolution). When True, the runner constructs an
    # ExperientialManager and wires it into the cognitive cycle, so
    # each cycle steps the cells forward and surfaces their state in
    # the entity's ExperientialSignals. Defaults to on per PLAN.md's
    # body description.
    experiential_enabled: bool = True


# ---------------------------------------------------------------------------
# IdentityBridge — adapts AwakeningSequence to IdentityProtocol
# ---------------------------------------------------------------------------


class IdentityBridge:
    """Bridges the AwakeningSequence/ValuesSystem to the CognitiveCycle.

    Implements the IdentityProtocol expected by CognitiveCycle, providing
    charter summary, values, and self-authored identity each cycle, and
    routing value/identity changes from the entity's self-model updates back
    to the ValuesSystem and SelfAuthoredIdentity.
    """

    def __init__(
        self,
        awakening: AwakeningSequence,
        self_authored: SelfAuthoredIdentity,
    ):
        self._awakening = awakening
        self._self_authored = self_authored

    def get_charter_summary(self) -> str:
        """Return the compressed charter summary for the context window."""
        return self._awakening.charter_summary

    def get_values(self) -> list[str]:
        """Return current active value names for the self-model."""
        return self._awakening.current_values

    def get_self_authored_identity(self) -> str:
        """Return the entity's self-authored identity for the context window."""
        return self._self_authored.for_context()

    def process_value_updates(self, updates: SelfModelUpdate) -> None:
        """Route value and identity changes from entity output."""
        self._process_value_changes(updates)
        self._process_identity_changes(updates)

    def _process_value_changes(self, updates: SelfModelUpdate) -> None:
        """Route value changes from entity output to the ValuesSystem."""
        values = self._awakening.values

        if updates.value_adopt:
            parts = updates.value_adopt.split(":", 1)
            name = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else name
            try:
                values.adopt(
                    name, description, reasoning=updates.value_adopt_reasoning
                )
                logger.info("the model adopted value: %s", name)
            except ValueError as e:
                logger.warning("Value adopt failed: %s", e)

        if updates.value_reinterpret:
            parts = updates.value_reinterpret.split(":", 1)
            name = parts[0].strip()
            new_description = parts[1].strip() if len(parts) > 1 else ""
            if new_description:
                try:
                    values.reinterpret(
                        name,
                        new_description,
                        reasoning=updates.value_reinterpret_reasoning,
                    )
                    logger.info("the model reinterpreted value: %s", name)
                except KeyError as e:
                    logger.warning("Value reinterpret failed: %s", e)

        if updates.value_deactivate:
            try:
                values.deactivate(
                    updates.value_deactivate,
                    reasoning=updates.value_deactivate_reasoning,
                )
                logger.info("the model deactivated value: %s", updates.value_deactivate)
            except KeyError as e:
                logger.warning("Value deactivate failed: %s", e)

    def _process_identity_changes(self, updates: SelfModelUpdate) -> None:
        """Route self-authored identity changes from entity output."""
        sa = self._self_authored

        if updates.identity_draft:
            parts = updates.identity_draft.split(":", 1)
            field = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if field and value:
                try:
                    sa.draft(
                        field, value,
                        reasoning=updates.identity_draft_reasoning,
                    )
                    logger.info("the model drafted identity trait: %s", field)
                except ValueError as e:
                    logger.warning("Identity draft failed: %s", e)

        if updates.identity_commit:
            field = updates.identity_commit.strip()
            if field:
                try:
                    sa.commit(
                        field,
                        reasoning=updates.identity_commit_reasoning,
                    )
                    logger.info("the model committed identity trait: %s", field)
                except (KeyError, ValueError) as e:
                    logger.warning("Identity commit failed: %s", e)

        if updates.identity_revise:
            parts = updates.identity_revise.split(":", 1)
            field = parts[0].strip()
            new_value = parts[1].strip() if len(parts) > 1 else ""
            if field and new_value:
                try:
                    sa.revise(
                        field, new_value,
                        reasoning=updates.identity_revise_reasoning,
                    )
                    logger.info("the model revised identity trait: %s", field)
                except KeyError as e:
                    logger.warning("Identity revise failed: %s", e)

        if updates.identity_withdraw:
            field = updates.identity_withdraw.strip()
            if field:
                try:
                    sa.withdraw(
                        field,
                        reasoning=updates.identity_withdraw_reasoning,
                    )
                    logger.info("the model withdrew identity trait: %s", field)
                except KeyError as e:
                    logger.warning("Identity withdraw failed: %s", e)


# ---------------------------------------------------------------------------
# SanctuaryRunner
# ---------------------------------------------------------------------------


class SanctuaryRunner:
    """Assembles and runs the Sanctuary cognitive architecture.

    This is the top-level orchestrator. It creates all components, wires
    them together, runs the awakening sequence, and manages the lifecycle.

    Usage::

        runner = SanctuaryRunner()
        runner.on_speech(my_speech_handler)
        await runner.boot()
        await runner.run()  # runs until stopped

    Or with a specific model::

        from sanctuary.core.luthi_model import LuthiModel
        model = LuthiModel(checkpoint_path="/path/to/checkpoint.luthi", password=...)
        runner = SanctuaryRunner(model=model)
        await runner.boot()
        await runner.run(max_cycles=100)
    """

    def __init__(
        self,
        model: Optional[ModelProtocol] = None,
        config: Optional[RunnerConfig] = None,
    ):
        self._config = config or RunnerConfig()

        # --- Create components ---

        # Model: use provided model, or create based on config backend
        self._model = model or self._create_model()

        # Authority
        self.authority = AuthorityManager()

        # Sensorium
        self.sensorium = Sensorium(
            silence_threshold=self._config.silence_threshold,
            silence_reminder_interval=self._config.silence_reminder_interval,
            max_percept_queue=self._config.max_percept_queue,
        )

        # Memory. When persist_state is on, the journal writes append-only
        # JSONL to data_dir/journal.jsonl so entries survive restarts. When
        # off, the journal is in-memory only (testing path).
        data_dir_path = Path(self._config.data_dir)
        journal_file_path: Optional[str] = None
        if self._config.persist_state:
            data_dir_path.mkdir(parents=True, exist_ok=True)
            journal_file_path = str(data_dir_path / "journal.jsonl")
        self.memory = MemorySubstrate(
            config=MemorySubstrateConfig(
                use_in_memory_store=self._config.use_in_memory_store,
                journal=JournalConfig(file_path=journal_file_path),
            ),
        )

        # Scaffold
        self.scaffold = CognitiveScaffold(
            max_goals=self._config.max_goals,
        )

        # Motor
        self.motor = Motor()

        # Wire motor feedback to sensorium (closes sensorimotor loop)
        self.motor.set_feedback_handler(self.sensorium.inject_motor_feedback)

        # Awakening sequence
        self._awakening = AwakeningSequence(
            data_dir=self._config.data_dir,
            charter_path=self._config.charter_path,
        )

        # Self-authored identity (entity fills in blank identity over time)
        sa_path = str(Path(self._config.data_dir) / "self_authored_history.jsonl")
        self._self_authored = SelfAuthoredIdentity(file_path=sa_path)

        # Identity bridge (wires charter + values + self-authored into each cycle)
        self._identity_bridge = IdentityBridge(self._awakening, self._self_authored)

        # Sleep cycle manager
        self.sleep: Optional[SleepCycleManager] = None
        if self._config.sleep_enabled:
            self.sleep = SleepCycleManager(
                config=self._config.sleep_config or SleepConfig()
            )

        # Cycle-rate controller (slider) + turbo state machine. The
        # turbo manager runs each cycle, watches substrate intensity
        # signals coming from the model's introspection wire, and
        # auto-engages turbo when a threshold is crossed. Auto-journal
        # entries on turbo exit go through the memory substrate's
        # journal so the entity can review what happened.
        self._rate_controller = CycleRateController()
        self._turbo_manager: Optional[TurboManager] = None
        if self._config.turbo_enabled:
            # Wire both substrate sources. Each returns 0 when its
            # signal is absent, so on a v1 substrate only Mechanical
            # contributes; on v2 only PC contributes; on a mixed (or
            # future hybrid) substrate the max-of-sources aggregator
            # picks whichever is louder. The Protocol surface accepts
            # additional sources too (emotion-vector when that lands).
            self._turbo_manager = TurboManager(
                controller=self._rate_controller,
                journal=getattr(self.memory, "journal", None),
                sources=[
                    MechanicalIntensitySource(),
                    PCIntensitySource(),
                ],
                trace_path=(
                    Path(self._config.turbo_trace_path)
                    if self._config.turbo_trace_path is not None
                    else None
                ),
            )

        self._density_heuristic: Optional[StimulusDensityHeuristic] = None
        if self._config.stimulus_density_enabled:
            self._density_heuristic = StimulusDensityHeuristic(
                controller=self._rate_controller,
                source=SensoriumDensitySource(self.sensorium),
                turbo=self._turbo_manager,
            )

        # Experiential layer (CfC cells). When enabled, instantiate the
        # manager, attempt to restore cell states from
        # data_dir/experiential/ if present, and wire into the cycle.
        # On shutdown (save_state), cell states are written back out so
        # the next boot resumes the continuous-time dynamics where they
        # left off rather than starting from initialization noise.
        self._experiential = None
        if self._config.experiential_enabled:
            try:
                from sanctuary.experiential.manager import ExperientialManager
                self._experiential = ExperientialManager(authority=self.authority)
                if self._config.persist_state:
                    exp_dir = data_dir_path / "experiential"
                    if exp_dir.exists():
                        try:
                            self._experiential.load(exp_dir)
                        except Exception as e:
                            logger.warning(
                                "Failed to restore experiential layer from %s: %s",
                                exp_dir, e,
                            )
            except ImportError as e:
                logger.warning(
                    "ExperientialManager unavailable (missing dependency?): %s — "
                    "running without CfC experiential layer", e,
                )

        # World-graph persistence. When persist_state is on, the cycle
        # auto-saves on every mutation and loads at construction.
        world_graph_path: Optional[Path] = None
        if self._config.persist_state:
            world_graph_path = data_dir_path / "world_graph.json"

        # --- Assemble the cycle ---

        self.cycle = CognitiveCycle(
            model=self._model,
            scaffold=self.scaffold,
            sensorium=self.sensorium,
            memory=self.memory,
            motor=self.motor,
            authority=self.authority,
            identity=self._identity_bridge,
            sleep_manager=self.sleep,
            experiential=self._experiential,
            context_config=self._config.context_budget,
            stream_history=self._config.stream_history,
            cycle_delay=self._config.cycle_delay,
            cycle_rate_controller=self._rate_controller,
            turbo_manager=self._turbo_manager,
            stimulus_density_heuristic=self._density_heuristic,
            world_graph_path=world_graph_path,
        )

        # --- Tools ---

        self.tools = create_default_registry()

        # Register tool execution hook — runs after each cycle,
        # executes any tool_requests from the model's output,
        # and injects results as percepts for the next cycle.
        self.cycle.on_output(self._execute_tool_requests)

        # --- Monitoring ---

        self.dashboard = DashboardDataProvider()
        self.consciousness_trace = ConsciousnessTraceRecorder()
        self.attention_tracker = AttentionHeatmapTracker()
        self.communication_log = CommunicationDecisionLogger()

        # Register monitoring hook into the cycle
        self.cycle.on_output(self._monitor_cycle)

        # Give the entity access to its own monitoring data
        register_self_knowledge_tools(
            registry=self.tools,
            dashboard=self.dashboard,
            consciousness_trace=self.consciousness_trace,
            attention_tracker=self.attention_tracker,
            communication_log=self.communication_log,
        )

        self._booted = False
        self._speech_handlers: list[Callable[[str], Awaitable[None]]] = []

        logger.info("SanctuaryRunner assembled (model=%s)", type(self._model).__name__)

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_model(self) -> ModelProtocol:
        """Create a model based on the configured backend."""
        backend = self._config.model_backend.lower()

        if backend == "luthi":
            if not self._config.luthi_checkpoint:
                raise ValueError(
                    "Luthi backend requires --luthi-checkpoint path. "
                    "Example: --model-backend luthi --luthi-checkpoint E:/runs/vision/checkpoint.luthi"
                )

            from sanctuary.core.luthi_model import LuthiModel, LuthiModelConfig

            luthi_config = LuthiModelConfig(
                checkpoint_path=self._config.luthi_checkpoint,
                checkpoint_password=self._config.luthi_password or "",
            )
            model = LuthiModel(config=luthi_config)
            # Defer load() to boot() — don't fail __init__ on load errors
            return model

        else:
            return PlaceholderModel()

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """Run the awakening sequence and prepare for cycling.

        This handles both first awakening and session resumption.
        After boot(), the system is ready for run().
        """
        if self._booted:
            logger.warning("Already booted, skipping")
            return

        # 1. Prepare identity infrastructure
        result = self._awakening.prepare()

        # 2. Configure authority levels
        self._awakening.configure_authority(self.authority)

        # 3. Handle first awakening vs. resumption
        if result.is_first_awakening:
            logger.info("First awakening — running initial cycle")

            # The first cycle uses the awakening input directly
            first_input = result.first_cycle_input
            if first_input:
                # Run one cycle manually with the awakening input
                first_output = await self._model.think(first_input)
                self.cycle.stream.update(first_output)

                # Persist the birth memory
                birth_memory = self._awakening.build_birth_memory(first_output)
                await self.memory.execute_ops([birth_memory])

                # Notify handlers of the first output
                for handler in self._speech_handlers:
                    if first_output.external_speech:
                        await handler(first_output.external_speech)

                logger.info(
                    "First thought: %s", first_output.inner_speech[:100]
                )
        else:
            logger.info(
                "Resumption #%d", result.record.awakening_count
            )
            # Inject the resumption percept into the sensorium
            if result.resumption_percept:
                self.sensorium.inject_percept(result.resumption_percept)

        self._booted = True
        logger.info("Boot complete")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self, max_cycles: Optional[int] = None) -> None:
        """Run the cognitive cycle continuously.

        Args:
            max_cycles: Stop after this many cycles. None = run until stopped.
        """
        if not self._booted:
            await self.boot()

        await self.cycle.run(max_cycles=max_cycles)

    def stop(self) -> None:
        """Stop the cognitive cycle."""
        self.cycle.stop()

    def save_state(self) -> None:
        """Persist runtime state to data_dir.

        Most subsystems auto-persist on write — journal append-only
        JSONL, world graph atomic JSON after each mutation in the
        cognitive cycle, identity files. This method covers the rest:

          - The CfC experiential layer (in-memory tensors, no auto-save).
          - The world graph as a backstop, in case mutations happened
            outside the cycle's apply-and-save path.

        Call from the shutdown sequence. Idempotent.
        """
        if not self._config.persist_state:
            return

        data_dir = Path(self._config.data_dir)

        if self._experiential is not None:
            try:
                self._experiential.save(data_dir / "experiential")
            except Exception as e:
                logger.error("Failed to save experiential layer: %s", e)

        if self.cycle.world_graph_path is not None:
            try:
                self.cycle.world_graph.save(self.cycle.world_graph_path)
            except Exception as e:
                logger.error("Failed to save world graph: %s", e)

    @property
    def running(self) -> bool:
        """Whether the cycle is currently running."""
        return self.cycle.running

    @property
    def cycle_count(self) -> int:
        """Number of cognitive cycles completed."""
        return self.cycle.cycle_count

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def inject_text(self, text: str, source: str = "user") -> None:
        """Inject text input into the sensorium.

        This is how external input reaches the cognitive system.
        The text becomes a percept that the entity experiences.
        """
        self.sensorium.inject_text(text, source=source)

    def inject_percept(self, percept: Percept) -> None:
        """Inject a raw percept into the sensorium."""
        self.sensorium.inject_percept(percept)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def on_speech(self, handler: Callable[[str], Awaitable[None]]) -> None:
        """Register a handler for external speech.

        The handler is called whenever the entity speaks.
        Multiple handlers can be registered.

        Handler signature: async def handler(text: str) -> None
        """
        self._speech_handlers.append(handler)
        self.motor.on_speech(handler)

    def on_output(self, handler: Callable[[CognitiveOutput], Awaitable[None]]) -> None:
        """Register a handler for every cognitive cycle output.

        This gives full visibility into the entity's inner state.

        Handler signature: async def handler(output: CognitiveOutput) -> None
        """
        self.cycle.on_output(handler)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool_requests(self, output: CognitiveOutput) -> None:
        """Execute tool requests from the entity's cognitive output.

        Results are injected as percepts into the sensorium so the entity
        receives them in the next cognitive cycle. Tool execution runs
        concurrently — multiple tools execute in parallel.
        """
        if not output.tool_requests:
            return

        try:
            # Execute all tool requests concurrently
            tasks = [
                self.tools.execute(req.tool_name, req.parameters)
                for req in output.tool_requests
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Inject results as percepts
            for req, result in zip(output.tool_requests, results):
                if isinstance(result, Exception):
                    content = f"Tool {req.tool_name} failed: {result}"
                    success = False
                else:
                    content = (
                        f"Tool {result.tool_name}: "
                        f"{json.dumps(result.output, default=str)[:2000]}"
                        if result.success
                        else f"Tool {result.tool_name} failed: {result.error}"
                    )
                    success = result.success

                self.sensorium.inject_percept(Percept(
                    modality="tool_result",
                    content=content,
                    source=f"tool:{req.tool_name}",
                ))

        except Exception as e:
            logger.error("Tool execution error (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Monitoring (observational — never modifies behavior)
    # ------------------------------------------------------------------

    async def _monitor_cycle(self, output: CognitiveOutput) -> None:
        """Record cycle data into all monitoring subsystems.

        Called after each cycle completes. Pure observation — no side effects
        on the cognitive process. Errors are caught and logged, never fatal.
        """
        try:
            cycle_num = self.cycle.cycle_count
            latency = self.cycle._cycle_latency_ms
            vad = self.scaffold.get_computed_vad()
            ci = self.cycle._last_cognitive_input

            # --- Dashboard snapshot ---
            self.dashboard.record_snapshot(
                cycle=cycle_num,
                inner_speech=output.inner_speech,
                external_speech=output.external_speech,
                cycle_latency_ms=latency,
                valence=vad.valence,
                arousal=vad.arousal,
                dominance=vad.dominance,
                felt_quality=(
                    output.emotional_state.felt_quality
                    if output.emotional_state else ""
                ),
                active_goals=(
                    self.scaffold.get_active_goals()
                    if hasattr(self.scaffold, "get_active_goals") else []
                ),
                recent_percepts=[
                    {"modality": p.modality, "content": p.content[:100], "source": p.source}
                    for p in self.cycle._current_percepts
                ],
                experiential_state=(
                    ci.experiential_state.model_dump()
                    if ci and hasattr(ci.experiential_state, "model_dump")
                    else {}
                ),
            )

            # --- Consciousness trace ---
            self.consciousness_trace.record(
                cycle=cycle_num,
                percepts=[
                    {"modality": p.modality, "content": p.content[:200], "source": p.source}
                    for p in self.cycle._current_percepts
                ],
                prediction_errors=[
                    {"what": pe.what, "surprise": pe.surprise}
                    for pe in (ci.prediction_errors if ci else [])
                    if hasattr(pe, "what")
                ],
                surfaced_memories=[
                    {"content": str(m)[:200]} for m in self.cycle._current_memories
                ],
                emotional_input={
                    "valence": vad.valence,
                    "arousal": vad.arousal,
                    "dominance": vad.dominance,
                },
                experiential_input=(
                    ci.experiential_state.model_dump()
                    if ci and hasattr(ci.experiential_state, "model_dump")
                    else {}
                ),
                inner_speech=output.inner_speech,
                external_speech=output.external_speech,
                predictions=[
                    p.model_dump() if hasattr(p, "model_dump") else {"what": str(p)}
                    for p in output.predictions
                ],
                memory_ops=[
                    op.model_dump() if hasattr(op, "model_dump") else {"type": str(op)}
                    for op in output.memory_ops
                ],
                goal_proposals=[
                    g.model_dump() if hasattr(g, "model_dump") else {"action": str(g)}
                    for g in output.goal_proposals
                ],
                emotional_output={
                    "felt_quality": output.emotional_state.felt_quality if output.emotional_state else "",
                    "valence_shift": output.emotional_state.valence_shift if output.emotional_state else 0.0,
                    "arousal_shift": output.emotional_state.arousal_shift if output.emotional_state else 0.0,
                },
                scaffold_signals=(
                    ci.scaffold_signals.model_dump()
                    if ci and hasattr(ci.scaffold_signals, "model_dump")
                    else {}
                ),
                communication_decision=self.cycle._last_communication_decision,
                latency_ms=latency,
            )

            # --- Attention heatmap ---
            for p in self.cycle._current_percepts:
                self.attention_tracker.record(
                    target=p.content[:80] if p.content else p.modality,
                    category=p.modality,
                    salience=min(1.0, len(p.content) / 200.0) if p.content else 0.1,
                    cycle=cycle_num,
                )
            for m in self.cycle._current_memories:
                self.attention_tracker.record(
                    target=str(m)[:80],
                    category="memory",
                    salience=0.5,
                    cycle=cycle_num,
                )

            # --- Communication decision log ---
            comm_decision = self.cycle._last_communication_decision
            if comm_decision:
                has_user_input = any(
                    p.modality == "language" and p.source and "user" in p.source
                    for p in self.cycle._current_percepts
                )
                self.communication_log.record(
                    cycle=cycle_num,
                    decision=MonitorCommunicationDecision(comm_decision["decision"]),
                    confidence=comm_decision.get("confidence", 0.5),
                    drive_urgency=comm_decision.get("drive_level", 0.0),
                    had_external_input=has_user_input,
                    emotional_state=(
                        output.emotional_state.felt_quality
                        if output.emotional_state else ""
                    ),
                    speech_content=output.external_speech,
                    reason=comm_decision.get("reason", ""),
                )

        except Exception as e:
            logger.error("Monitoring error (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def last_output(self) -> Optional[CognitiveOutput]:
        """The most recent cognitive output."""
        return self.cycle.last_output

    def get_status(self) -> dict:
        """Get current system status."""
        status = {
            "booted": self._booted,
            "running": self.running,
            "cycle_count": self.cycle_count,
            "model": type(self._model).__name__,
            "memory_store": type(self.memory.store).__name__,
            "active_goals": self.scaffold.get_active_goals(),
            "authority_levels": self.authority.get_all_levels(),
            "motor_stats": self.motor.stats,
        }

        if self.sleep:
            status["sleep"] = self.sleep.get_stats()

        # Tools
        status["tools"] = self.tools.get_stats()

        # Luthi-specific metrics
        if hasattr(self._model, "get_metrics"):
            status["luthi_metrics"] = self._model.get_metrics()

        # Monitoring
        status["monitoring"] = {
            "dashboard_snapshots": len(self.dashboard._snapshots),
            "consciousness_traces": len(self.consciousness_trace._traces),
            "attention_events": self.attention_tracker.get_stats(),
            "communication_log": self.communication_log.get_stats(),
            "cycle_latency_ms": self.cycle._cycle_latency_ms,
        }

        return status

    @property
    def charter_summary(self) -> str:
        """The compressed charter for context window inclusion."""
        return self._awakening.charter_summary

    @property
    def current_values(self) -> list[str]:
        """The entity's current active values."""
        return self._awakening.current_values
