"""SanctuaryAPI — programmatic interface for the cognitive architecture.

Phase 6: Integration + Validation.

Provides a clean async API for embedding Sanctuary in applications,
bots, or test harnesses. Wraps SanctuaryRunner with a simpler
request-response interface while the cognitive cycle runs continuously
in the background.

Usage::

    api = SanctuaryAPI()
    await api.start()

    response = await api.send("Hello, how are you?")
    print(response)  # entity's speech output

    status = api.status()
    await api.stop()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.cognitive_cycle import ModelProtocol
from sanctuary.core.supervision import supervise
from sanctuary.core.schema import CognitiveOutput

logger = logging.getLogger(__name__)


class SanctuaryAPI:
    """Programmatic interface for Sanctuary.

    The cognitive cycle runs continuously in the background. Calling
    send() injects a percept and waits for the next cycle to produce
    a response. If no speech is produced, returns None.

    This is the recommended API for integrations (Discord bots, web
    interfaces, test harnesses).
    """

    def __init__(
        self,
        model: Optional[ModelProtocol] = None,
        config: Optional[RunnerConfig] = None,
    ):
        self._runner = SanctuaryRunner(model=model, config=config)
        self._cycle_task: Optional[asyncio.Task] = None
        self._speech_queue: asyncio.Queue[str] = asyncio.Queue()
        self._started = False

        # Wire speech to the queue
        self._runner.on_speech(self._capture_speech)

    async def start(self) -> None:
        """Boot and start the cognitive cycle."""
        if self._started:
            return

        await self._runner.boot()

        def _on_runner_death(exc: BaseException) -> None:
            # The cognitive loop is dead; reflect that so a restart is possible
            # and health checks stop reporting a running mind.
            self._started = False

        self._cycle_task = supervise(
            asyncio.create_task(self._runner.run()),
            name="cognitive_runner",
            fatal=True,
            on_death=_on_runner_death,
        )
        self._started = True
        logger.info("SanctuaryAPI started")

    async def stop(self) -> None:
        """Stop the cognitive cycle and clean up."""
        if not self._started:
            return

        self._runner.stop()

        if self._cycle_task:
            try:
                await asyncio.wait_for(self._cycle_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._cycle_task.cancel()

        self._started = False
        logger.info("SanctuaryAPI stopped")

    async def send(self, text: str, timeout: float = 5.0) -> Optional[str]:
        """Send a message and wait for the entity's response.

        Injects the text as a percept, then waits up to `timeout` seconds
        for the entity to produce speech. Returns the speech text, or
        None if the entity chose silence.

        Args:
            text: The message to send.
            timeout: How long to wait for a response (seconds).

        Returns:
            The entity's speech, or None if it chose silence.
        """
        # Drain any stale speech from previous cycles
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Inject the message
        self._runner.inject_text(text, source="user:api")

        # Wait for speech output
        try:
            response = await asyncio.wait_for(
                self._speech_queue.get(), timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            return None

    def inject_percept_raw(self, modality: str, content: str, source: str = "") -> None:
        """Inject a raw percept without waiting for a response."""
        from sanctuary.core.schema import Percept

        self._runner.inject_percept(
            Percept(modality=modality, content=content, source=source)
        )

    @property
    def last_output(self) -> Optional[CognitiveOutput]:
        """The most recent full cognitive output."""
        return self._runner.last_output

    def status(self) -> dict:
        """Get system status."""
        return self._runner.get_status()

    @property
    def cycle_count(self) -> int:
        return self._runner.cycle_count

    @property
    def runner(self) -> SanctuaryRunner:
        """Direct access to the runner for advanced usage."""
        return self._runner

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _capture_speech(self, text: str) -> None:
        """Capture speech output into the queue."""
        await self._speech_queue.put(text)
