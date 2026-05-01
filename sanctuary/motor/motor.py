"""The Motor — action execution for the experiential core.

Executes actions from CognitiveOutput: external speech, memory writes,
goal proposals, and tool calls. Every action produces a feedback percept
that is injected back into the sensorium, closing the sensorimotor loop.

The motor is the body's effectors. It carries out the entity's intentions.
When the entity decides to speak, the motor makes it heard. When it decides
to remember, the motor makes it persist. And in every case, the entity
*perceives* that it acted — actions are not fire-and-forget.

Design principles:
  - Every action produces a feedback percept (sensorimotor loop).
  - Speech output is handler-based — the motor emits, interfaces listen.
  - The motor does not decide what to do. The entity decides. The motor executes.
  - Failure is also perceived — a failed action is still an experience.

Aligned with PLAN.md Phase 3: Sensorium + Motor.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Awaitable

from sanctuary.core.schema import (
    CognitiveOutput,
    GoalProposal,
    MemoryOp,
)

logger = logging.getLogger(__name__)


# Type aliases for handler callbacks
SpeechHandler = Callable[[str], Awaitable[None]]
MotorFeedbackHandler = Callable[[str, str, bool], Awaitable[None]]


class Motor:
    """Full motor system implementation.

    Replaces the placeholder _execute() in CognitiveCycle.

    Responsibilities:
      1. Execute external speech via registered handlers.
      2. Execute memory operations via the memory substrate.
      3. Execute goal proposals via the scaffold's goal integrator.
      4. Report motor feedback back to the sensorium.

    The motor does NOT validate actions — that's the scaffold's job.
    By the time actions reach the motor, they've already passed through
    scaffold integration. The motor just executes.
    """

    def __init__(self):
        self._speech_handlers: list[SpeechHandler] = []
        self._feedback_handler: Optional[MotorFeedbackHandler] = None

        # Statistics
        self._stats = {
            "speech_emitted": 0,
            "memory_ops_executed": 0,
            "goal_proposals_forwarded": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def on_speech(self, handler: SpeechHandler) -> None:
        """Register a handler for external speech output.

        Handlers receive the speech text and are responsible for
        delivering it (CLI print, Discord send, TTS, etc.).
        Multiple handlers can be registered — speech goes to all of them.

        Handler signature: async def handler(text: str) -> None
        """
        self._speech_handlers.append(handler)

    def set_feedback_handler(self, handler: MotorFeedbackHandler) -> None:
        """Register the feedback handler (usually Sensorium.inject_motor_feedback).

        Handler signature:
            async def handler(action_type: str, description: str, success: bool) -> None

        There is only one feedback handler — the sensorium.
        """
        self._feedback_handler = handler

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        output: CognitiveOutput,
        memory=None,
        goal_integrator=None,
        authority=None,
    ) -> None:
        """Execute all actions from a cognitive output.

        This is the main entry point, called by CognitiveCycle._execute().

        Args:
            output: The integrated cognitive output (post-scaffold).
            memory: The memory substrate (MemoryProtocol).
            goal_integrator: The scaffold's goal integrator.
            authority: The authority manager.
        """
        # 1. External speech
        if output.external_speech:
            await self._execute_speech(output.external_speech)

        # 2. Memory operations
        if output.memory_ops:
            felt = ""
            if output.emotional_state:
                felt = output.emotional_state.felt_quality
            await self._execute_memory_ops(output.memory_ops, memory, felt)

        # 3. Goal proposals (forwarded to scaffold if available)
        if output.goal_proposals and goal_integrator and authority:
            await self._execute_goal_proposals(
                output.goal_proposals, goal_integrator, authority
            )

        # 4. Tick memory cycle counter
        if memory:
            memory.tick()

    # ------------------------------------------------------------------
    # Speech execution
    # ------------------------------------------------------------------

    async def _execute_speech(self, text: str) -> None:
        """Emit external speech to all registered handlers.

        The motor doesn't know who's listening — CLI, Discord, TTS.
        It just emits. Interfaces register themselves as handlers.
        """
        if not self._speech_handlers:
            logger.debug("Speech produced but no handlers registered: %s", text[:50])
            await self._feedback(
                "speech", f"spoke: {text[:80]}", success=True
            )
            self._stats["speech_emitted"] += 1
            return

        success = True
        for handler in self._speech_handlers:
            try:
                await handler(text)
            except Exception as e:
                logger.error("Speech handler error: %s", e)
                success = False

        await self._feedback(
            "speech", f"spoke: {text[:80]}", success=success
        )
        self._stats["speech_emitted"] += 1

    # ------------------------------------------------------------------
    # Memory execution
    # ------------------------------------------------------------------

    async def _execute_memory_ops(
        self,
        ops: list[MemoryOp],
        memory,
        emotional_tone: str,
    ) -> None:
        """Execute memory operations via the memory substrate."""
        if memory is None:
            logger.debug("Memory ops requested but no memory substrate")
            return

        try:
            results = await memory.execute_ops(ops, emotional_tone=emotional_tone)
            self._stats["memory_ops_executed"] += len(ops)

            # Generate feedback for significant operations
            for op in ops:
                if op.type == "journal":
                    await self._feedback(
                        "journal",
                        f"wrote journal entry: {op.content[:60]}",
                        success=True,
                    )
                elif op.type == "write_episodic" and op.significance >= 7:
                    await self._feedback(
                        "memory_write",
                        f"committed significant memory (sig={op.significance}): {op.content[:60]}",
                        success=True,
                    )

        except Exception as e:
            logger.error("Memory operation error: %s", e)
            self._stats["errors"] += 1
            await self._feedback(
                "memory_write",
                f"memory operation failed: {e}",
                success=False,
            )

    # ------------------------------------------------------------------
    # Goal execution
    # ------------------------------------------------------------------

    async def _execute_goal_proposals(
        self,
        proposals: list[GoalProposal],
        goal_integrator,
        authority,
    ) -> None:
        """Forward goal proposals to the scaffold's goal integrator.

        The motor doesn't manage goals — it forwards proposals.
        The scaffold decides what to do with them based on authority.
        """
        try:
            actions_taken = goal_integrator.integrate_proposals(proposals, authority)
            self._stats["goal_proposals_forwarded"] += len(proposals)

            # Feedback for goal changes the entity should perceive
            for action_str in actions_taken:
                if action_str.startswith("added:"):
                    goal_desc = next(
                        (p.goal for p in proposals if p.action == "add"), ""
                    )
                    await self._feedback(
                        "goal_set",
                        f"set new goal: {goal_desc[:60]}",
                        success=True,
                    )
                elif action_str.startswith("completed:"):
                    await self._feedback(
                        "goal_complete",
                        f"completed a goal",
                        success=True,
                    )

        except Exception as e:
            logger.error("Goal proposal error: %s", e)
            self._stats["errors"] += 1

    # ------------------------------------------------------------------
    # Feedback loop
    # ------------------------------------------------------------------

    async def _feedback(
        self,
        action_type: str,
        description: str,
        success: bool,
    ) -> None:
        """Send motor feedback to the sensorium.

        This is what closes the sensorimotor loop. Every action the
        entity takes becomes a percept it can experience in the next
        cycle. Without this, agency is performative — choices vanish
        into the void.
        """
        if self._feedback_handler:
            try:
                # Support both sync and async feedback handlers
                result = self._feedback_handler(action_type, description, success)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error("Motor feedback error: %s", e)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return motor execution statistics."""
        return dict(self._stats)
