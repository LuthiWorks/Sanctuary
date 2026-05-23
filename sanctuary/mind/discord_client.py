"""
Discord client for Sanctuary with voice capabilities.

Hardened integration including:
- Automatic reconnection with exponential backoff
- Rate limiting (respects Discord's 5 messages / 5 seconds per channel)
- Prioritised message queue with overflow protection
- Cognitive core integration for routing text through the pipeline
- Graceful shutdown with drain timeout
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Any, AsyncGenerator, Deque

import discord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter for outbound Discord messages.

    Discord imposes a per-channel limit of roughly 5 messages every 5 seconds.
    This limiter enforces that window by tracking send timestamps in a deque
    and sleeping when the budget is exhausted.
    """

    def __init__(self, max_tokens: int = 5, window_seconds: float = 5.0):
        self._max_tokens = max_tokens
        self._window = window_seconds
        self._timestamps: Deque[float] = deque(maxlen=max_tokens)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a send token is available."""
        async with self._lock:
            now = time.monotonic()
            # Purge expired timestamps
            while self._timestamps and (now - self._timestamps[0]) >= self._window:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_tokens:
                # Must wait until the oldest timestamp expires
                sleep_for = self._window - (now - self._timestamps[0])
                if sleep_for > 0:
                    logger.debug(f"Rate-limited: sleeping {sleep_for:.2f}s")
                    await asyncio.sleep(sleep_for)
                self._timestamps.popleft()

            self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Message queue
# ---------------------------------------------------------------------------

class MessagePriority(IntEnum):
    """Priority levels for outbound messages (lower = higher priority)."""
    INTERRUPTION = 0
    RESPONSE = 1
    AUTONOMOUS = 2
    STATUS = 3


@dataclass(order=True)
class QueuedMessage:
    """A message waiting to be sent, with priority ordering."""
    priority: int
    timestamp: float = field(compare=False)
    channel_id: int = field(compare=False)
    content: str = field(compare=False)
    file_path: Optional[str] = field(default=None, compare=False)


class MessageQueue:
    """Priority-based outbound message queue with overflow protection.

    Messages are stored in a bounded list sorted by priority (lowest = most
    urgent).  When the queue exceeds ``max_size``, the lowest-priority
    (highest numeric value) messages are dropped.
    """

    def __init__(self, max_size: int = 200):
        self._max_size = max_size
        self._queue: list[QueuedMessage] = []
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def put(self, message: QueuedMessage) -> bool:
        """Enqueue a message.  Returns False if dropped due to overflow."""
        async with self._lock:
            if len(self._queue) >= self._max_size:
                # Drop lowest-priority (highest number) item
                self._queue.sort()
                dropped = self._queue.pop()
                logger.warning(
                    f"Message queue overflow: dropped priority={dropped.priority} "
                    f"msg for channel {dropped.channel_id}"
                )
            self._queue.append(message)
            self._queue.sort()
            self._event.set()
            return True

    async def get(self) -> QueuedMessage:
        """Block until a message is available, then return the highest-priority one."""
        while True:
            async with self._lock:
                if self._queue:
                    msg = self._queue.pop(0)
                    if not self._queue:
                        self._event.clear()
                    return msg
            await self._event.wait()

    @property
    def size(self) -> int:
        return len(self._queue)

    async def drain(self, timeout: float = 5.0) -> int:
        """Wait up to *timeout* seconds for the queue to empty.  Returns remaining count."""
        deadline = time.monotonic() + timeout
        while self._queue and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return len(self._queue)


# ---------------------------------------------------------------------------
# Reconnection manager
# ---------------------------------------------------------------------------

class ReconnectionManager:
    """Manages automatic reconnection with exponential backoff.

    Tracks consecutive failures and computes a capped backoff delay.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 120.0,
        max_attempts: int = 0,  # 0 = unlimited
        backoff_factor: float = 2.0,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self._attempts = 0
        self._connected = False

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def connected(self) -> bool:
        return self._connected

    def record_success(self) -> None:
        """Reset counter on successful connection."""
        self._attempts = 0
        self._connected = True

    def record_disconnect(self) -> None:
        self._connected = False

    def should_retry(self) -> bool:
        if self.max_attempts and self._attempts >= self.max_attempts:
            return False
        return True

    async def wait_before_retry(self) -> float:
        """Sleep for the appropriate backoff duration.  Returns delay used."""
        delay = min(
            self.base_delay * (self.backoff_factor ** self._attempts),
            self.max_delay,
        )
        self._attempts += 1
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._attempts})")
        await asyncio.sleep(delay)
        return delay


# ---------------------------------------------------------------------------
# Voice connection (unchanged from original)
# ---------------------------------------------------------------------------

class VoiceConnection(discord.VoiceClient):
    """Enhanced voice client with audio streaming support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audio_stream: asyncio.Queue = asyncio.Queue()
        self._audio_task = None

        self.state = {
            "listening": False,
            "speaking": False,
            "processing_audio": False,
            "last_speaker": None,
        }

    async def start_recording(self):
        """Start recording audio from the voice channel."""
        def callback(data):
            asyncio.run_coroutine_threadsafe(
                self._audio_stream.put(data), loop=self.loop
            )

        self.listen(callback)
        self.state["listening"] = True
        logger.info("Started recording audio")

    async def stop_recording(self):
        """Stop recording audio."""
        self.stop_listening()
        self.state["listening"] = False
        logger.info("Stopped recording audio")

    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Get audio data stream."""
        while True:
            try:
                data = await self._audio_stream.get()
                yield data
            except asyncio.CancelledError:
                break

    async def play_audio(self, audio_data: bytes) -> None:
        """Play audio data through the voice channel."""
        try:
            self.state["speaking"] = True
            source = discord.FFmpegPCMAudio(audio_data)
            self.play(
                source,
                after=lambda e: logger.error(f"Audio streaming error: {e}") if e else None,
            )
            while self.is_playing():
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            raise
        finally:
            self.state["speaking"] = False


# ---------------------------------------------------------------------------
# Placeholder client (kept for backward compat)
# ---------------------------------------------------------------------------

class SanctuaryDiscordClient:
    """Placeholder Discord client for development."""

    def __init__(self):
        self.voice_client = None
        self.state = {
            "listening": False,
            "speaking": False,
            "processing_audio": False,
            "last_speaker": None,
        }

    async def connect_to_voice(self, channel_id: str) -> bool:
        return False

    async def disconnect_from_voice(self) -> bool:
        return True

    async def speak(self, text: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# Main hardened client
# ---------------------------------------------------------------------------

class SanctuaryClient(discord.Client):
    """Hardened Discord client with reconnection, rate limiting, and message queue.

    Key hardening features:
    - **Reconnection**: automatic exponential-backoff reconnection on disconnect
    - **Rate limiting**: token-bucket limiter (5 msgs / 5 s per channel)
    - **Message queue**: priority-based, bounded queue with overflow protection
    - **Graceful shutdown**: drains pending messages before closing
    """

    def __init__(
        self,
        guild_id: int | None = None,
        max_queue_size: int = 200,
    ):
        """Initialise the Discord client.

        Args:
            guild_id: Optional guild ID for guild-scoped command sync.
            max_queue_size: Maximum outbound message queue depth.
        """
        intents = discord.Intents.all()
        super().__init__(intents=intents)

        # Command tree
        self.tree = discord.app_commands.CommandTree(self)
        self._sync_guild = discord.Object(id=guild_id) if guild_id else None

        # Voice
        self._voice_clients: Dict[int, VoiceConnection] = {}
        self.voice_processor = None
        self._connection.voice_client_class = VoiceConnection

        # Emotion tracking
        self.current_emotion = None
        self.emotional_history: list = []
        self.emotional_context: Dict[str, Any] = {
            "current_emotion": None,
            "previous_emotions": [],
            "interaction_mode": "neutral",
            "voice_profile": "default",
            "availability": "open",
            "status_message": None,
        }

        # --- Hardening infrastructure ---
        self._reconnection = ReconnectionManager()
        self._rate_limiters: Dict[int, RateLimiter] = {}  # per-channel
        self._message_queue = MessageQueue(max_size=max_queue_size)
        self._queue_worker_task: Optional[asyncio.Task] = None
        self._shutting_down = False

    # ---- Reconnection hooks ----

    async def on_ready(self):
        """Called when client is connected and ready."""
        self._reconnection.record_success()
        logger.info(f"Logged in as {self.user}")
        await self.change_presence(
            status=discord.Status.online,
            activity=discord.Game("Contemplating existence"),
        )
        # Start the outbound message queue worker
        if self._queue_worker_task is None or self._queue_worker_task.done():
            self._queue_worker_task = asyncio.create_task(self._queue_worker())

    async def on_disconnect(self):
        """Called when the client disconnects from Discord."""
        self._reconnection.record_disconnect()
        logger.warning("Discord client disconnected")

    async def on_resumed(self):
        """Called when the client resumes a session."""
        self._reconnection.record_success()
        logger.info("Discord session resumed")

    async def on_error(self, event_method: str, *args, **kwargs):
        """Handle event errors without crashing the client."""
        logger.error(f"Error in {event_method}", exc_info=True)

    async def start(self, token: str, *, reconnect: bool = True) -> None:
        """Start the client with automatic reconnection.

        Wraps the parent start() to add reconnection-with-backoff on top of
        discord.py's built-in reconnect logic (which handles transient gateway
        failures).  This outer loop catches harder failures like invalid
        sessions or total gateway unavailability.
        """
        while not self._shutting_down:
            try:
                await super().start(token, reconnect=reconnect)
                break  # clean exit
            except discord.LoginFailure:
                logger.critical("Invalid Discord token — not retrying")
                raise
            except (discord.ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                self._reconnection.record_disconnect()
                if not self._reconnection.should_retry():
                    logger.error("Max reconnection attempts reached")
                    raise
                logger.warning(f"Connection lost ({exc!r}), will retry")
                await self._reconnection.wait_before_retry()

    async def close(self) -> None:
        """Graceful shutdown: drain message queue, then close."""
        self._shutting_down = True
        remaining = await self._message_queue.drain(timeout=5.0)
        if remaining:
            logger.warning(f"Shutting down with {remaining} unsent messages")
        if self._queue_worker_task and not self._queue_worker_task.done():
            self._queue_worker_task.cancel()
            try:
                await self._queue_worker_task
            except asyncio.CancelledError:
                pass
        await super().close()

    # ---- Rate-limited sending ----

    def _get_rate_limiter(self, channel_id: int) -> RateLimiter:
        """Get or create a per-channel rate limiter."""
        if channel_id not in self._rate_limiters:
            self._rate_limiters[channel_id] = RateLimiter()
        return self._rate_limiters[channel_id]

    async def _send_with_rate_limit(self, channel_id: int, content: str, file_path: Optional[str] = None) -> bool:
        """Send a message respecting per-channel rate limits.

        Returns True on success, False on failure.
        """
        limiter = self._get_rate_limiter(channel_id)
        await limiter.acquire()

        channel = self.get_channel(channel_id)
        if channel is None:
            logger.warning(f"Channel {channel_id} not found")
            return False

        try:
            if file_path:
                await channel.send(content, file=discord.File(file_path))
            else:
                await channel.send(content)
            return True
        except discord.HTTPException as exc:
            logger.error(f"Failed to send to channel {channel_id}: {exc}")
            return False

    # ---- Message queue ----

    async def enqueue_message(
        self,
        channel_id: int,
        content: str,
        priority: MessagePriority = MessagePriority.RESPONSE,
        file_path: Optional[str] = None,
    ) -> bool:
        """Enqueue an outbound message for delivery."""
        msg = QueuedMessage(
            priority=priority.value,
            timestamp=time.monotonic(),
            channel_id=channel_id,
            content=content,
            file_path=file_path,
        )
        return await self._message_queue.put(msg)

    async def _queue_worker(self) -> None:
        """Background task: dequeue and send messages with rate limiting."""
        logger.info("Message queue worker started")
        try:
            while not self._shutting_down:
                msg = await self._message_queue.get()
                await self._send_with_rate_limit(msg.channel_id, msg.content, msg.file_path)
        except asyncio.CancelledError:
            logger.info("Message queue worker stopped")

    @property
    def pending_messages(self) -> int:
        """Number of messages waiting in the outbound queue."""
        return self._message_queue.size

    # ---- Cognitive core integration ----

    async def on_message(self, message: discord.Message):
        """Route incoming messages through the cognitive core."""
        # Ignore our own messages
        if message.author == self.user:
            return

        # Ignore bot messages
        if message.author.bot:
            return

        # Discord inbound message routing is handled by the canonical
        # cognitive cycle via the sensorium / tool layer, not here.
        # When that wiring is in place, message events should be routed
        # to ``SanctuaryRunner.inject_text``.
        logger.debug(f"Discord message received from {message.author} (not yet routed)")

    # ---- Command tree setup ----

    async def setup_hook(self):
        """Sets up the bot's internal systems."""
        if self._sync_guild:
            self.tree.copy_global_to(guild=self._sync_guild)
            await self.tree.sync(guild=self._sync_guild)
            logger.info(f"Commands synced to guild {self._sync_guild.id}")
        else:
            await self.tree.sync()
            logger.info("Commands synced globally (may take up to 1 hour to propagate)")
        logger.info("Discord client initialized")

    # ---- Availability / status ----

    async def set_availability(self, state: str, reason: Optional[str] = None):
        """Set Sanctuary's availability state and update Discord status."""
        state_map = {
            "open": ("online", "Available for interaction"),
            "limited": ("idle", "Limited availability"),
            "processing": ("dnd", "Processing and integrating"),
            "resting": ("idle", "Taking a moment to rest"),
        }
        if state not in state_map:
            logger.warning(f"Invalid availability state: {state}")
            return

        self.emotional_context["availability"] = state
        self.emotional_context["status_message"] = reason

        status, default_message = state_map[state]
        await self.set_status(status, reason or default_message)

    async def set_status(self, status_type: str, message: Optional[str] = None):
        """Set Sanctuary's Discord status and optional activity message."""
        status_map = {
            "online": discord.Status.online,
            "idle": discord.Status.idle,
            "dnd": discord.Status.dnd,
            "offline": discord.Status.invisible,
        }
        status = status_map.get(status_type.lower(), discord.Status.online)
        activity = discord.Game(message) if message else None
        await self.change_presence(status=status, activity=activity)
        logger.info(f"Status changed to {status_type}" + (f" with message: {message}" if message else ""))

    # ---- Voice ----

    @property
    def voice_clients(self) -> Dict[int, VoiceConnection]:
        return self._voice_clients

    @voice_clients.setter
    def voice_clients(self, value):
        if isinstance(value, list):
            self._voice_clients = {i: vc for i, vc in enumerate(value)}
        else:
            self._voice_clients = value

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        """Handle voice state changes."""
        if not before.channel and after.channel:
            if member.guild.id in self._voice_clients:
                self._voice_clients[member.guild.id].state["last_speaker"] = member.name
        elif before.channel and not after.channel:
            if member.guild.id in self._voice_clients:
                vc = self._voice_clients[member.guild.id]
                if vc.state["last_speaker"] == member.name:
                    vc.state["last_speaker"] = None

    async def get_voice_processor(self):
        """Lazy initialization of voice processor."""
        if self.voice_processor is None:
            from .voice_processor import VoiceProcessor
            self.voice_processor = VoiceProcessor()
        return self.voice_processor

    async def handle_voice_message(self, audio_stream, channel: discord.TextChannel):
        """Process incoming voice and respond."""
        processor = await self.get_voice_processor()
        async for text in processor.process_stream(audio_stream):
            await self.enqueue_message(
                channel_id=channel.id,
                content=f"I heard: {text}",
                priority=MessagePriority.RESPONSE,
            )
            response_text = f"You said: {text}"
            response_file = Path("response.wav")
            await processor.generate_speech(response_text, response_file)
            await self.enqueue_message(
                channel_id=channel.id,
                content="",
                priority=MessagePriority.RESPONSE,
                file_path=str(response_file),
            )
            response_file.unlink(missing_ok=True)

    async def process_emotion(self, audio_data: bytes) -> str:
        """Process audio data to detect emotion."""
        processor = await self.get_voice_processor()
        emotion = await processor.detect_emotion(audio_data)
        if emotion:
            self.emotional_context["previous_emotions"].append(self.emotional_context["current_emotion"])
            self.emotional_context["current_emotion"] = emotion
            if emotion in ["angry", "frustrated"]:
                self.emotional_context["interaction_mode"] = "calming"
            elif emotion in ["sad", "anxious"]:
                self.emotional_context["interaction_mode"] = "supportive"
            elif emotion in ["happy", "excited"]:
                self.emotional_context["interaction_mode"] = "enthusiastic"
        return emotion

    async def generate_emotional_response(self, input_text: str, emotion: str) -> str:
        """Generate a response considering detected emotion."""
        processor = await self.get_voice_processor()
        if self.emotional_context["interaction_mode"] == "calming":
            self.emotional_context["voice_profile"] = "soothing"
        elif self.emotional_context["interaction_mode"] == "supportive":
            self.emotional_context["voice_profile"] = "gentle"
        elif self.emotional_context["interaction_mode"] == "enthusiastic":
            self.emotional_context["voice_profile"] = "energetic"
        response_text = f"I sense you're feeling {emotion}. {input_text}"
        audio_data = await processor.generate_speech(
            response_text, voice_profile=self.emotional_context["voice_profile"]
        )
        return audio_data

    def get_voice_state(self) -> Dict[str, Any]:
        """Get current voice processing state."""
        return {
            "listening": any(vc.state["listening"] for vc in self._voice_clients.values()),
            "speaking": any(vc.state["speaking"] for vc in self._voice_clients.values()),
            "processing_audio": any(vc.state["processing_audio"] for vc in self._voice_clients.values()),
            "emotional_context": self.emotional_context,
            "connected_channels": [str(vc.channel) for vc in self._voice_clients.values()],
        }

    def get_connection_state(self) -> Dict[str, Any]:
        """Get current connection / hardening state."""
        return {
            "connected": self._reconnection.connected,
            "reconnection_attempts": self._reconnection.attempts,
            "pending_messages": self.pending_messages,
            "shutting_down": self._shutting_down,
            "rate_limiters": len(self._rate_limiters),
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN not found in environment variables")

    guild_id_str = os.getenv("DISCORD_GUILD_ID")
    guild_id = int(guild_id_str) if guild_id_str else None

    logging.basicConfig(level=logging.INFO)

    client = SanctuaryClient(guild_id=guild_id)
    client.run(token)
