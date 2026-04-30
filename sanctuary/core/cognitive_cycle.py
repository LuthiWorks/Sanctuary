"""The cognitive cycle — the continuous stream of thought.

Each cycle: assemble input -> LLM processes -> scaffold integrates ->
execute actions -> broadcast -> update predictions.

The LLM's output from cycle N becomes part of its input for cycle N+1.
The scaffold provides defaults, validation, and anomaly detection.
The authority manager governs how much weight the LLM's signals carry
versus the scaffold's defaults.

This is the heart of Sanctuary. The cycle IS active inference:
predict -> perceive -> compute error -> update model -> act.

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

from sanctuary.consciousness.sleep_cycle import SleepCycleManager, SleepStage
from sanctuary.core.authority import AuthorityManager
from sanctuary.core.context_manager import BudgetConfig, ContextManager
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    EmotionalInput,
    ExperientialSignals,
    ComputedVAD,
    Percept,
    ScaffoldSignals,
    TemporalContext,
)
from sanctuary.core.stream_of_thought import StreamOfThought
from sanctuary.motor.motor import Motor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for subsystem interfaces (will be implemented in later phases)
# ---------------------------------------------------------------------------


class ModelProtocol(Protocol):
    """Interface for the experiential core (LLM or placeholder)."""

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput: ...


class ScaffoldProtocol(Protocol):
    """Interface for the cognitive scaffold (Phase 2).

    The scaffold validates LLM output and integrates it with existing
    subsystems. Until Phase 2, this is a passthrough.
    """

    async def integrate(
        self, output: CognitiveOutput, authority: AuthorityManager
    ) -> CognitiveOutput: ...

    def get_signals(self) -> ScaffoldSignals: ...

    async def broadcast(self, output: CognitiveOutput) -> None: ...

    def get_computed_vad(self) -> ComputedVAD: ...


class SensoriumProtocol(Protocol):
    """Interface for the sensorium (Phase 3).

    Provides percepts, temporal context, and prediction error tracking.
    Until Phase 3, percepts are injected manually.
    """

    async def drain_percepts(self) -> list[Percept]: ...

    def get_prediction_errors(self) -> list: ...

    def get_temporal_context(self) -> TemporalContext: ...

    def update_predictions(self, predictions: list) -> None: ...


class MemoryProtocol(Protocol):
    """Interface for the memory system (Phase 3).

    Surfaces relevant memories, queues retrievals, and executes
    memory operations from the LLM's cognitive output.
    """

    async def surface(self, context: str) -> list: ...

    async def queue_retrieval(self, query: str) -> None: ...

    async def execute_ops(
        self, ops: list, emotional_tone: str = ""
    ) -> list[str]: ...

    def tick(self) -> None: ...


class MotorProtocol(Protocol):
    """Interface for the motor system (Phase 5).

    Executes actions from CognitiveOutput and produces feedback percepts.
    """

    async def execute(
        self,
        output: CognitiveOutput,
        memory: ...,
        goal_integrator: ...,
        authority: ...,
    ) -> None: ...


class IdentityProtocol(Protocol):
    """Interface for the identity system (Phase 5 integration).

    Provides charter summary, values, and self-authored identity for each
    cycle, and processes value/identity changes from the LLM's self-model updates.
    """

    def get_charter_summary(self) -> str: ...

    def get_values(self) -> list[str]: ...

    def get_self_authored_identity(self) -> str: ...

    def process_value_updates(self, updates) -> None: ...


# ---------------------------------------------------------------------------
# Null implementations (stand-ins until later phases)
# ---------------------------------------------------------------------------


class NullScaffold:
    """Passthrough scaffold — no validation, no integration.

    Used during Phase 1 testing. All LLM output passes through unchanged.
    """

    async def integrate(
        self, output: CognitiveOutput, authority: AuthorityManager
    ) -> CognitiveOutput:
        return output

    def get_signals(self) -> ScaffoldSignals:
        return ScaffoldSignals()

    async def broadcast(self, output: CognitiveOutput) -> None:
        pass

    def get_computed_vad(self) -> ComputedVAD:
        return ComputedVAD()


class NullSensorium:
    """Minimal sensorium — percepts injected manually, no prediction tracking."""

    def __init__(self):
        self._percept_queue: list[Percept] = []

    def inject_percept(self, percept: Percept):
        self._percept_queue.append(percept)

    async def drain_percepts(self) -> list[Percept]:
        percepts = list(self._percept_queue)
        self._percept_queue.clear()
        return percepts

    def get_prediction_errors(self) -> list:
        return []

    def get_temporal_context(self) -> TemporalContext:
        return TemporalContext(
            time_of_day=datetime.now().strftime("%H:%M"),
        )

    def update_predictions(self, predictions: list) -> None:
        pass


class NullMemory:
    """Minimal memory — no surfacing, no retrieval, no persistence."""

    async def surface(self, context: str) -> list:
        return []

    async def queue_retrieval(self, query: str) -> None:
        pass

    async def execute_ops(
        self, ops: list, emotional_tone: str = ""
    ) -> list[str]:
        return []

    def tick(self) -> None:
        pass


class NullIdentity:
    """Minimal identity — no charter, no values, no self-authored traits."""

    def get_charter_summary(self) -> str:
        return ""

    def get_values(self) -> list[str]:
        return []

    def get_self_authored_identity(self) -> str:
        return ""

    def process_value_updates(self, updates) -> None:
        pass


# ---------------------------------------------------------------------------
# The Cognitive Cycle
# ---------------------------------------------------------------------------


class CognitiveCycle:
    """The continuous stream of thought.

    Each cycle:
    1. Assemble input from all sources (including scaffold signals)
    2. Compress to fit context budget
    3. LLM processes (experiential cognition happens here)
    4. Scaffold validates and integrates
    5. Update stream of thought (continuity — always from LLM, never scaffold)
    6. Execute actions (only those that passed scaffold validation)
    7. Broadcast to all subsystems (GWT ignition)
    8. Update prediction tracking for next cycle
    """

    def __init__(
        self,
        model: ModelProtocol,
        scaffold: Optional[ScaffoldProtocol] = None,
        sensorium: Optional[SensoriumProtocol] = None,
        memory: Optional[MemoryProtocol] = None,
        motor: Optional[Motor] = None,
        authority: Optional[AuthorityManager] = None,
        identity: Optional[IdentityProtocol] = None,
        sleep_manager: Optional[SleepCycleManager] = None,
        communication=None,
        experiential=None,
        growth=None,
        environment=None,
        context_config: Optional[BudgetConfig] = None,
        stream_history: int = 10,
        cycle_delay: float = 0.1,
    ):
        self.model = model
        self.scaffold = scaffold or NullScaffold()
        self.sensorium = sensorium or NullSensorium()
        self.memory = memory or NullMemory()
        self.motor = motor
        self.authority = authority or AuthorityManager()
        self.identity = identity or NullIdentity()
        self.sleep = sleep_manager  # Optional SleepCycleManager
        self.experiential = experiential  # Optional ExperientialManager
        self.growth = growth  # Optional GrowthProcessor
        self.environment = environment  # Optional EnvironmentIntegration
        self.context_mgr = ContextManager(context_config)
        self.stream = StreamOfThought(max_history=stream_history)

        self.running = False
        self.cycle_count = 0
        self._cycle_delay = cycle_delay
        self._output_handlers: list = []
        self._last_output: Optional[CognitiveOutput] = None
        # Stashed per-cycle for communication agency (needs them after think())
        self._current_percepts: list = []
        self._current_memories: list = []

        # Stashed per-cycle for monitoring hooks
        self._cycle_latency_ms: float = 0.0
        self._last_cognitive_input: Optional[CognitiveInput] = None
        self._last_communication_decision: Optional[Dict[str, Any]] = None

    # -- Public API --

    def inject_percept(self, percept: Percept):
        """Inject a percept into the sensorium for the next cycle."""
        if isinstance(self.sensorium, NullSensorium):
            self.sensorium.inject_percept(percept)
        else:
            # Real sensorium will have its own input method
            self.sensorium.inject_percept(percept)

    def on_output(self, handler):
        """Register a callback for each cycle's output.

        Handler signature: async def handler(output: CognitiveOutput) -> None
        """
        self._output_handlers.append(handler)

    async def run(self, max_cycles: Optional[int] = None):
        """Run the cognitive cycle continuously.

        Args:
            max_cycles: Stop after this many cycles (None = run until stopped).
        """
        self.running = True
        cycles = 0

        while self.running:
            # Per AGENTS.md: no top-level crash boundary until Phase 9 (First
            # Awakening). During development and testing, let cycle failures
            # surface so they can be diagnosed.
            await self._cycle()

            cycles += 1

            if max_cycles is not None and cycles >= max_cycles:
                self.running = False
                break

            await asyncio.sleep(self._cycle_delay)

    def stop(self):
        """Stop the cognitive cycle."""
        self.running = False

    @property
    def last_output(self) -> Optional[CognitiveOutput]:
        """The most recent cognitive output."""
        return self._last_output

    # -- The cycle --

    async def _cycle(self):
        """Execute one cycle of cognition."""
        cycle_start = time.perf_counter()
        self._last_communication_decision = None

        # 0. Advance sleep cycle (if enabled)
        if self.sleep is not None:
            self.sleep.tick(self.cycle_count)

        # 1. Assemble input from all sources
        cognitive_input = await self._assemble_input()
        self._last_cognitive_input = cognitive_input

        # 2. Compress to fit context budget
        compressed_input = self.context_mgr.compress(cognitive_input)

        # 3. LLM processes
        cognitive_output = await self.model.think(compressed_input)

        # 3a. Record speech for monitoring (no gating — entity speaks freely)
        if cognitive_output.external_speech:
            self._last_communication_decision = {
                "decision": "speak",
                "confidence": 1.0,
                "reason": "Entity chose to speak",
                "drive_level": 1.0,
                "inhibition_level": 0.0,
            }

        # 3b. Sleep consolidation — during NREM, trigger living weight
        #     consolidation (set point adjustment, plasticity rebalancing).
        #     The model still thinks during sleep, but we also consolidate.
        if (
            self.sleep is not None
            and self.sleep.stage == SleepStage.NREM
            and hasattr(self.model, "consolidate")
        ):
            try:
                self.model.consolidate()
            except Exception as e:
                logger.error("Luthi consolidation error (non-fatal): %s", e)

        # 4. Scaffold validates and integrates
        integrated = await self.scaffold.integrate(
            cognitive_output, self.authority
        )

        # 5. Update stream of thought (always from raw LLM output, not scaffold)
        self.stream.update(cognitive_output)

        # 5b. Route value updates to identity system
        if cognitive_output.self_model_updates:
            try:
                self.identity.process_value_updates(cognitive_output.self_model_updates)
            except Exception as e:
                logger.error("Identity value update error (non-fatal): %s", e)

        # 6. Execute actions
        try:
            await self._execute(integrated)
        except Exception as e:
            logger.error("Action execution error (non-fatal): %s", e)

        # 7. Broadcast
        try:
            await self.scaffold.broadcast(integrated)
        except Exception as e:
            logger.error("Broadcast error (non-fatal): %s", e)

        # 8. Update prediction tracking
        self.sensorium.update_predictions(cognitive_output.predictions)

        # 6b. Process environment actions (before bookkeeping, so percepts
        #     land in the queue for the next cycle)
        if self.environment is not None:
            try:
                self.environment.process_output(integrated)
            except Exception as e:
                logger.error("Environment processing error (non-fatal): %s", e)

        # Bookkeeping
        self._cycle_latency_ms = (time.perf_counter() - cycle_start) * 1000.0
        self.cycle_count += 1
        self._last_output = cognitive_output

        # Notify handlers
        for handler in self._output_handlers:
            await handler(cognitive_output)

        # 9. Growth processing (after all other processing, errors never crash cycle)
        if self.growth is not None:
            try:
                await self.growth.process_cycle(cognitive_output, self.cycle_count)
            except Exception as e:
                logger.error("Growth processing error (non-fatal): %s", e)

    async def _assemble_input(self) -> CognitiveInput:
        """Gather everything the LLM needs for this moment of thought."""

        percepts = await self.sensorium.drain_percepts()

        # Sleep sensory gating — reduce percept throughput during sleep.
        # Urgent percepts (user input) still trigger forced wake.
        if self.sleep is not None and self.sleep.is_sleeping:
            gate = self.sleep.get_sensory_gate()
            if gate < 1.0:
                # Check for urgent percepts that should wake us
                urgent = [
                    p for p in percepts
                    if p.modality == "language" and p.source and "user" in p.source
                ]
                if urgent:
                    self.sleep.wake()
                    logger.info("User input during sleep — waking")
                else:
                    # Gate: keep only a fraction of percepts proportional to gate
                    keep = max(1, math.ceil(len(percepts) * gate))
                    percepts = percepts[:keep]

        # Compute prediction errors by comparing predictions to actual percepts
        if hasattr(self.sensorium, "compute_prediction_errors"):
            self.sensorium.compute_prediction_errors(percepts)

        prediction_errors = self.sensorium.get_prediction_errors()
        temporal = self.sensorium.get_temporal_context()
        temporal.interactions_this_session = self.cycle_count

        try:
            surfaced = await self.memory.surface(
                context=self.stream.get_recent_context()
            )
        except Exception as e:
            logger.error("Memory surfacing failed (cycle continues without memories): %s", e)
            surfaced = []

        # Step the experiential layer (CfC cells evolve)
        experiential_signals = ExperientialSignals()
        if self.experiential is not None:
            vad = self.scaffold.get_computed_vad()
            # Compute average prediction error for CfC input
            avg_error = 0.0
            if prediction_errors:
                avg_error = sum(
                    pe.surprise if hasattr(pe, "surprise") else 0.0
                    for pe in prediction_errors
                ) / len(prediction_errors)

            state = self.experiential.step(
                arousal=vad.arousal,
                prediction_error=avg_error,
                base_precision=0.5,
                scaffold_precision=0.5,
                scaffold_vad=(vad.valence, vad.arousal, vad.dominance),
            )
            experiential_signals = ExperientialSignals(
                precision_weight=state.precision_weight,
                affect_valence=max(-1.0, min(1.0, state.affect_vad[0])),
                affect_arousal=max(0.0, min(1.0, state.affect_vad[1])),
                affect_dominance=max(0.0, min(1.0, state.affect_vad[2])),
                attention_salience=max(0.0, min(1.0, state.attention_salience)),
                goal_adjustment=max(-1.0, min(1.0, state.goal_adjustment)),
                cells_active=state.cell_active,
            )

        # Inject Luthi's introspection signals (cognitive proprioception).
        # The model's neural dynamics — plasticity, drift, spike fractions,
        # membrane potentials — flow into knowledge_signals so the entity
        # can observe its own internal state changing in real time.
        if hasattr(self.model, "get_augmented_experiential_signals"):
            try:
                luthi_signals = self.model.get_augmented_experiential_signals()
                if luthi_signals:
                    experiential_signals.knowledge_signals.update(luthi_signals)
            except Exception as e:
                logger.error("Luthi introspection injection error (non-fatal): %s", e)

        # Populate self-model with current values from identity system
        self_model = self.stream.get_self_model()
        identity_values = self.identity.get_values()
        if identity_values:
            self_model.values = identity_values

        # Inject environment location context into world model
        world_model = self.stream.get_world_model()
        if self.environment is not None:
            location_ctx = self.environment.get_location_context()
            if location_ctx:
                world_model.environment.update(location_ctx)

        # Stash percepts and memories for the communication evaluation step
        # (evaluate_speech runs after model.think() and needs these)
        self._current_percepts = percepts
        self._current_memories = surfaced

        return CognitiveInput(
            previous_thought=self.stream.get_previous(),
            new_percepts=percepts,
            prediction_errors=prediction_errors,
            surfaced_memories=surfaced,
            emotional_state=EmotionalInput(
                computed=self.scaffold.get_computed_vad(),
                felt_quality=self.stream.get_felt_quality(),
            ),
            temporal_context=temporal,
            self_model=self_model,
            world_model=world_model,
            scaffold_signals=self.scaffold.get_signals(),
            experiential_state=experiential_signals,
            charter_summary=self.identity.get_charter_summary(),
            self_authored_identity=self.identity.get_self_authored_identity(),
        )

    async def _execute(self, output: CognitiveOutput):
        """Execute actions from the integrated output.

        When a Motor is available, it handles all execution and produces
        feedback percepts back to the sensorium. Without a motor, falls
        back to direct memory execution (Phase 3 behavior).
        """
        if self.motor is not None:
            # Motor handles everything: speech, memory, goals, feedback
            goal_integrator = getattr(self.scaffold, "goals", None)
            await self.motor.execute(
                output,
                memory=self.memory,
                goal_integrator=goal_integrator,
                authority=self.authority,
            )
        else:
            # Fallback: direct execution without motor feedback loop
            if output.memory_ops:
                felt = ""
                if output.emotional_state:
                    felt = output.emotional_state.felt_quality
                await self.memory.execute_ops(
                    output.memory_ops, emotional_tone=felt
                )
            self.memory.tick()
