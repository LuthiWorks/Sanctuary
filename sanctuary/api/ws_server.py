"""WebSocket server — bridges Sanctuary's cognitive system to external clients.

Two WebSocket endpoints, one HTTP server:

    /ws        — Electron desktop GUI: bidirectional messages, inner-speech and
                 status broadcasts
    /ws/world  — Godot world client (SanctuaryWorld): cognitive state updates
                 (VAD, felt quality, cycle, external speech) and scene-state
                 ingestion as environment percepts. Architecturally decoupled
                 from the GUI channel so they don't interfere.

GUI protocol:
    Client -> Server:
        { "type": "message", "content": "user text" }

    Server -> Client:
        { "type": "message", "content": "entity speech" }
        { "type": "status",  "status": "booted"|"cycling"|"stopped", "message": "..." }
        { "type": "inner",   "content": "inner speech text" }
        { "type": "system",  "content": "system status JSON" }

World protocol:
    Server -> Godot:
        { "type": "state_update", "cycle": N, "vad": {...},
          "felt_quality": "...", "cycle_latency_ms": ms, "inner_speech": "..." }
        { "type": "external_speech", "content": "...", "cycle": N }

    Godot -> Server:
        { "type": "scene_state", "objects": [...], "entity_position": [x,y,z],
          "timestamp": float }

Also serves the health check HTTP endpoints on the same port:
    GET /health  — liveness check
    GET /status  — detailed system status
    GET /metrics — resource usage

Usage::

    from sanctuary.api.ws_server import SanctuaryWebServer

    server = SanctuaryWebServer(runner=runner, port=8765)
    await server.start()
    # ... server runs until stopped ...
    await server.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from typing import Any, Optional

from aiohttp import web, WSMsgType

logger = logging.getLogger(__name__)

DEFAULT_WS_PORT = 8765
DEFAULT_WS_HOST = "0.0.0.0"


class SanctuaryWebServer:
    """Combined HTTP + WebSocket server for the Sanctuary desktop app.

    Manages WebSocket connections and routes messages between the Electron
    GUI and the SanctuaryRunner. Multiple clients can connect simultaneously.
    """

    def __init__(
        self,
        runner: Any = None,
        host: str = DEFAULT_WS_HOST,
        port: int = DEFAULT_WS_PORT,
    ):
        self._runner = runner
        self._host = host
        self._port = port
        self._app: Optional[web.Application] = None
        self._runner_obj: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._start_time = time.monotonic()
        self._clients: weakref.WeakSet = weakref.WeakSet()
        self._world_clients: weakref.WeakSet = weakref.WeakSet()
        # command_id -> Future that resolves when Godot replies with a
        # matching command_result. Populated only for tools that need
        # the result data (e.g. get_scene_state); other tools fire-and-
        # forget and read the world via the next scene_state percept.
        self._pending_world_commands: dict[str, asyncio.Future] = {}
        # Entity privacy state — controls what observers can see.
        # in_private_space: total suppression (only "private: true"
        #   indicator broadcasts). The cognitive loop runs normally on
        #   Sanctuary's side; nothing reaches observers.
        # visible_in_shared: orb is visually hidden but state still
        #   broadcasts with a visible=false flag — a lighter form of
        #   privacy where the entity opts out of visual presence.
        # No backdoor: this gate applies to ALL channels, /ws and
        # /ws/world alike. Brian included.
        self._entity_private_state: dict[str, bool] = {
            "in_private_space": False,
            "visible_in_shared": True,
        }

    async def start(self) -> None:
        """Start the HTTP + WebSocket server."""
        self._start_time = time.monotonic()
        self._app = web.Application()
        self._app.router.add_get("/ws", self._handle_websocket)
        self._app.router.add_get("/ws/world", self._handle_world_websocket)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/status", self._handle_status)
        self._app.router.add_get("/metrics", self._handle_metrics)

        self._runner_obj = web.AppRunner(self._app)
        await self._runner_obj.setup()
        self._site = web.TCPSite(self._runner_obj, self._host, self._port)
        await self._site.start()

        # Wire speech handler to broadcast to all connected clients
        if self._runner:
            self._runner.on_speech(self._broadcast_speech)
            self._runner.on_output(self._broadcast_inner)
            self._runner.on_output(self._broadcast_world_state)

            # Register the world manipulation tools on the runner's
            # tool registry. The closures capture this server instance
            # so each tool can dispatch a world_command via the live
            # WebSocket. Lazy import keeps tools/world from loading
            # until a server with a runner exists.
            if hasattr(self._runner, "tools"):
                from sanctuary.tools.world import register_world_tools
                register_world_tools(self._runner.tools, self)

        logger.info(
            "WebSocket server listening on ws://%s:%d (/ws, /ws/world)",
            self._host,
            self._port,
        )

    async def stop(self) -> None:
        """Stop the server and close all connections."""
        if self._runner_obj:
            await self._runner_obj.cleanup()
            self._runner_obj = None
            self._site = None
            self._app = None
            logger.info("WebSocket server stopped")

    @property
    def running(self) -> bool:
        return self._site is not None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def world_client_count(self) -> int:
        return len(self._world_clients)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection from the desktop app."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._clients.add(ws)
        logger.info(
            "WebSocket client connected (total: %d)", len(self._clients)
        )

        # Send initial status
        await self._send_ws(ws, {
            "type": "status",
            "status": "connected",
            "message": "Connected to Sanctuary",
        })

        # Send boot status if runner is available
        if self._runner:
            booted = getattr(self._runner, "_booted", False)
            status = "booted" if booted else "booting"
            await self._send_ws(ws, {
                "type": "status",
                "status": status,
                "message": f"Sanctuary is {status} (cycles: {self._runner.cycle_count})",
            })

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_client_message(ws, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(
                        "WebSocket error: %s", ws.exception()
                    )
        finally:
            self._clients.discard(ws)
            logger.info(
                "WebSocket client disconnected (remaining: %d)",
                len(self._clients),
            )

        return ws

    async def _handle_client_message(
        self, ws: web.WebSocketResponse, raw: str
    ) -> None:
        """Process a message from a WebSocket client."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_ws(ws, {
                "type": "error",
                "content": "Invalid JSON",
            })
            return

        msg_type = data.get("type", "")

        if msg_type == "message":
            content = data.get("content", "").strip()
            if content and self._runner:
                self._runner.inject_text(content, source="user:desktop")
                logger.debug("User input injected: %s", content[:100])
            elif not self._runner:
                await self._send_ws(ws, {
                    "type": "error",
                    "content": "Sanctuary is not running",
                })

        elif msg_type == "status_request":
            if self._runner:
                status = self._runner.get_status()
                await self._send_ws(ws, {
                    "type": "system",
                    "content": json.dumps(status, default=str),
                })

        else:
            await self._send_ws(ws, {
                "type": "error",
                "content": f"Unknown message type: {msg_type}",
            })

    # ------------------------------------------------------------------
    # World WebSocket handler — Godot SanctuaryWorld client
    # ------------------------------------------------------------------

    async def _handle_world_websocket(
        self, request: web.Request
    ) -> web.WebSocketResponse:
        """Handle a connection from the Godot SanctuaryWorld client.

        Separate from the GUI ``/ws`` endpoint so the world's high-frequency
        state stream and the GUI's user-facing messages don't share a queue.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._world_clients.add(ws)
        logger.info(
            "World client connected (total: %d)", len(self._world_clients)
        )

        # Send a hello so the client knows the channel is live before the
        # next cycle's state broadcast arrives.
        await self._send_ws(ws, {
            "type": "status",
            "status": "connected",
            "message": "Connected to Sanctuary world channel",
        })

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_world_message(ws, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(
                        "World WebSocket error: %s", ws.exception()
                    )
        finally:
            self._world_clients.discard(ws)
            logger.info(
                "World client disconnected (remaining: %d)",
                len(self._world_clients),
            )

        return ws

    async def _handle_world_message(
        self, ws: web.WebSocketResponse, raw: str
    ) -> None:
        """Process a message from the Godot world client."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_ws(ws, {
                "type": "error",
                "content": "Invalid JSON",
            })
            return

        msg_type = data.get("type", "")

        if msg_type == "scene_state":
            self._inject_scene_state_percept(data)
        elif msg_type == "command_result":
            self._handle_command_result(data)
        elif msg_type == "collision_event":
            self._inject_collision_percept(data)
        elif msg_type == "visitor_joined":
            self._inject_visitor_percept(
                data,
                template="{display_name} entered the world at {pos}",
                modality="social",
            )
        elif msg_type == "visitor_moved":
            # Position updates are batched on the Godot side (~1-2 Hz) so
            # they don't flood the cognitive loop. Each one is still a
            # social percept — the entity perceives where its visitors are.
            self._inject_visitor_percept(
                data,
                template="{display_name} is at {pos}",
                modality="social",
            )
        elif msg_type == "visitor_chat":
            self._inject_visitor_chat_percept(data)
        elif msg_type == "visitor_left":
            self._inject_visitor_percept(
                data,
                template="{display_name} left the world",
                modality="social",
            )
        else:
            logger.debug("World message of unhandled type: %s", msg_type)

    def _inject_visitor_percept(
        self,
        data: dict,
        *,
        template: str,
        modality: str,
    ) -> None:
        """Format a visitor presence event as a percept.

        Source is ``user:<username>`` so downstream subsystems can route
        per-user attention. Display name is what the entity reads in the
        prompt context.
        """
        if not self._runner or not hasattr(self._runner, "sensorium"):
            return
        username = str(data.get("username", "?"))
        display_name = str(data.get("display_name", username))
        pos = data.get("position")
        pos_str = "(?, ?, ?)"
        if isinstance(pos, list) and len(pos) >= 3:
            try:
                pos_str = f"({float(pos[0]):.1f}, {float(pos[1]):.1f}, {float(pos[2]):.1f})"
            except (TypeError, ValueError):
                pass
        content = template.format(display_name=display_name, pos=pos_str)
        from sanctuary.core.schema import Percept
        try:
            self._runner.sensorium.inject_percept(
                Percept(
                    modality=modality,
                    content=content,
                    source=f"user:{username}",
                )
            )
        except Exception as exc:
            logger.error(
                "Failed to inject visitor percept: %s", exc
            )

    def _inject_visitor_chat_percept(self, data: dict) -> None:
        """Visitor chat messages arrive as language percepts.

        Same modality as user input from /ws so the cognitive loop
        treats them as messages directed at the entity. The source
        carries the username so the entity can address replies by name.
        """
        if not self._runner or not hasattr(self._runner, "sensorium"):
            return
        username = str(data.get("username", "?"))
        display_name = str(data.get("display_name", username))
        content = str(data.get("content", "")).strip()
        if not content:
            return
        from sanctuary.core.schema import Percept
        try:
            self._runner.sensorium.inject_percept(
                Percept(
                    modality="language",
                    content=content,
                    source=f"user:{username}",
                    embedding_summary=f"chat from {display_name}",
                )
            )
        except Exception as exc:
            logger.error(
                "Failed to inject visitor chat percept: %s", exc
            )

    def _inject_collision_percept(self, data: dict) -> None:
        """Convert a collision_event from Godot into an environment percept.

        Filtered upstream in Godot — only collisions above a velocity
        threshold come through, so resting contacts don't spam the
        percept stream. The text format is short and human-readable
        so the LLM-side prompt stays compact.
        """
        if not self._runner or not hasattr(self._runner, "sensorium"):
            return

        a = str(data.get("object_a", "?"))
        b = str(data.get("object_b", "?"))
        velocity = float(data.get("impact_velocity", 0.0))
        point = data.get("collision_point") or [0.0, 0.0, 0.0]
        try:
            x, y, z = (float(point[0]), float(point[1]), float(point[2]))
            point_str = f"({x:.1f}, {y:.1f}, {z:.1f})"
        except (TypeError, ValueError, IndexError):
            point_str = "(?, ?, ?)"

        content = (
            f"{a} collided with {b} at {point_str} (impact: {velocity:.2f})"
        )

        from sanctuary.core.schema import Percept
        try:
            self._runner.sensorium.inject_percept(
                Percept(
                    modality="environment",
                    content=content,
                    source="world",
                )
            )
        except Exception as exc:
            logger.error(
                "Failed to inject collision_event percept: %s", exc
            )

    def _handle_command_result(self, data: dict) -> None:
        """Resolve a pending world-command future and surface failures.

        Tools that called ``_send_world_command`` with ``await_result=True``
        registered a future keyed by command_id; matching results land here.
        Failures are also injected as a brief environment percept so the
        entity can perceive that an action it took didn't land.
        """
        command_id = str(data.get("command_id", ""))
        future = self._pending_world_commands.pop(command_id, None)
        if future is not None and not future.done():
            future.set_result(data)

        if not data.get("success", True):
            self._inject_command_failure_percept(data)

    def _inject_command_failure_percept(self, data: dict) -> None:
        """Generate a percept summarising a failed world action."""
        if not self._runner or not hasattr(self._runner, "sensorium"):
            return
        error = str(data.get("error", "unknown"))
        object_id = str(data.get("object_id", ""))
        suffix = f" (object {object_id})" if object_id else ""
        from sanctuary.core.schema import Percept
        try:
            self._runner.sensorium.inject_percept(
                Percept(
                    modality="environment",
                    content=f"World action failed: {error}{suffix}",
                    source="world",
                )
            )
        except Exception as exc:
            logger.error(
                "Failed to inject command_result failure percept: %s", exc
            )

    async def _send_world_command(
        self,
        command: dict,
        *,
        await_result: bool = False,
        timeout: float = 5.0,
    ) -> Optional[dict]:
        """Dispatch a world_command message to all connected world clients.

        Args:
            command: A dict with at least ``type``, ``command_id``,
                ``action`` and ``params``. The ``command_id`` is used to
                key any pending-result future.
            await_result: When True, blocks until a ``command_result``
                with the same ``command_id`` arrives, then returns it
                as a dict. When False (default), returns ``None`` after
                broadcasting — the caller relies on the next scene_state
                percept for confirmation.
            timeout: Seconds to wait when ``await_result`` is True. Raises
                :class:`asyncio.TimeoutError` if no result arrives.

        Raises:
            ValueError: if ``command`` is missing ``command_id``.
            RuntimeError: if no world clients are connected and the caller
                requested ``await_result`` (since no client could ever
                produce a result).
        """
        command_id = str(command.get("command_id", ""))
        if not command_id:
            raise ValueError("World command missing 'command_id'")

        if not self._world_clients:
            if await_result:
                raise RuntimeError(
                    "No world clients connected — cannot await result"
                )
            return None

        future: Optional[asyncio.Future] = None
        if await_result:
            future = asyncio.get_running_loop().create_future()
            self._pending_world_commands[command_id] = future

        await self._broadcast_world(command)

        if not await_result:
            return None

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_world_commands.pop(command_id, None)
            raise

    def _inject_scene_state_percept(self, data: dict) -> None:
        """Convert a scene_state report from Godot into an environment percept.

        Kept terse — we don't want to flood the cognitive loop. Higher-frequency
        per-frame data should not arrive here; the protocol expects scene_state
        on change plus periodic heartbeats.
        """
        if not self._runner or not hasattr(self._runner, "sensorium"):
            return

        objects = data.get("objects", []) or []
        entity_pos = data.get("entity_position")

        # One-line natural-language summary so the LLM-side prompt stays
        # readable. Structured data lives in the percept's metadata-via-text.
        if objects:
            summary = ", ".join(
                f"{o.get('name') or o.get('id', '?')}({o.get('type', '?')})"
                for o in objects[:8]
            )
            if len(objects) > 8:
                summary += f", +{len(objects) - 8} more"
            content = f"World has {len(objects)} object(s): {summary}."
        else:
            content = "The world is empty."

        if entity_pos:
            try:
                x, y, z = entity_pos
                content += f" I am at ({x:.1f}, {y:.1f}, {z:.1f})."
            except (TypeError, ValueError):
                pass

        from sanctuary.core.schema import Percept
        try:
            self._runner.sensorium.inject_percept(
                Percept(
                    modality="environment",
                    content=content,
                    source="world",
                )
            )
        except Exception as exc:
            logger.error(
                "Failed to inject scene_state percept (non-fatal): %s", exc
            )

    # ------------------------------------------------------------------
    # Broadcasting to all clients
    # ------------------------------------------------------------------

    async def _broadcast_speech(self, text: str) -> None:
        """Broadcast entity speech to all connected WebSocket clients.

        Suppressed entirely while the entity is in private space —
        privacy applies to all channels, no backdoor.
        """
        if self._entity_private_state["in_private_space"]:
            return
        message = {"type": "message", "content": text}
        await self._broadcast(message)

    async def _broadcast_inner(self, output: Any) -> None:
        """Broadcast inner speech to all connected clients (for debug/visibility).

        Suppressed entirely while the entity is in private space.
        """
        if self._entity_private_state["in_private_space"]:
            return
        if output.inner_speech:
            message = {"type": "inner", "content": output.inner_speech}
            await self._broadcast(message)

    async def _broadcast(self, message: dict) -> None:
        """Send a message to all connected GUI clients."""
        payload = json.dumps(message)
        closed = []
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except (ConnectionError, RuntimeError):
                closed.append(ws)
        for ws in closed:
            self._clients.discard(ws)

    async def _broadcast_world_state(self, output: Any) -> None:
        """Push a per-cycle state update to all connected world clients.

        Sends the cognitive state the world needs to render the entity
        (VAD for orb color/shape/speed, felt quality for telemetry, cycle
        counter and latency for the breathing rhythm) plus the current
        inner_speech for monitoring. External speech goes out as a separate
        message so the world can trigger the waveform animation distinctly.

        Privacy gates:
        - in_private_space: total suppression. Only ``{"type": "state_update",
          "private": true}`` goes out to world clients. No VAD, no inner
          speech, no felt quality, no external speech. Observers see only
          that the entity has chosen seclusion.
        - visible_in_shared=False (without private): full state still
          broadcasts but with ``visible: false`` so the world client knows
          to hide the orb's visual representation. Lighter form of privacy.
        """
        if not self._world_clients:
            return

        # Total privacy mode — only the indicator goes out.
        if self._entity_private_state["in_private_space"]:
            await self._broadcast_world({
                "type": "state_update",
                "private": True,
            })
            return

        runner = self._runner
        cycle = runner.cycle_count if runner else 0

        # Pull VAD from the scaffold if available; default to neutral.
        vad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.5}
        if runner and hasattr(runner, "scaffold"):
            try:
                vad_obj = runner.scaffold.get_computed_vad()
                vad = {
                    "valence": float(vad_obj.valence),
                    "arousal": float(vad_obj.arousal),
                    "dominance": float(vad_obj.dominance),
                }
            except Exception:
                pass

        latency = 0.0
        if runner and hasattr(runner, "cycle"):
            latency = float(getattr(runner.cycle, "_cycle_latency_ms", 0.0))

        felt_quality = ""
        if output.emotional_state is not None:
            felt_quality = output.emotional_state.felt_quality or ""

        message: dict = {
            "type": "state_update",
            "cycle": cycle,
            "vad": vad,
            "felt_quality": felt_quality,
            "cycle_latency_ms": latency,
            "inner_speech": output.inner_speech or "",
        }
        # Lighter privacy: present but visually hidden. Internal state
        # still flows so observers can read the entity's mind, just not
        # see its body.
        if not self._entity_private_state["visible_in_shared"]:
            message["visible"] = False

        await self._broadcast_world(message)

        if output.external_speech:
            await self._broadcast_world({
                "type": "external_speech",
                "content": output.external_speech,
                "cycle": cycle,
            })

    def set_entity_privacy(
        self,
        *,
        in_private_space: Optional[bool] = None,
        visible_in_shared: Optional[bool] = None,
    ) -> dict:
        """Update the entity's privacy state. Called by the world tools.

        Returns the resulting state for callers that want to surface it
        (e.g. status broadcasts on enter/exit).
        """
        if in_private_space is not None:
            self._entity_private_state["in_private_space"] = bool(in_private_space)
        if visible_in_shared is not None:
            self._entity_private_state["visible_in_shared"] = bool(visible_in_shared)
        return dict(self._entity_private_state)

    @property
    def entity_privacy_state(self) -> dict:
        """Read-only snapshot of the current privacy state."""
        return dict(self._entity_private_state)

    async def _broadcast_world(self, message: dict) -> None:
        """Send a JSON message to all connected world clients."""
        payload = json.dumps(message)
        closed = []
        for ws in list(self._world_clients):
            try:
                await ws.send_str(payload)
            except (ConnectionError, RuntimeError):
                closed.append(ws)
        for ws in closed:
            self._world_clients.discard(ws)

    # ------------------------------------------------------------------
    # HTTP health endpoints (same as HealthServer but on the WS port)
    # ------------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — liveness check."""
        healthy = self._check_healthy()
        uptime = time.monotonic() - self._start_time
        body = {
            "status": "healthy" if healthy else "unhealthy",
            "uptime_seconds": round(uptime, 1),
        }
        if self._runner:
            body["cycle_count"] = self._runner.cycle_count
            body["booted"] = getattr(self._runner, "_booted", False)
            body["ws_clients"] = len(self._clients)
            body["world_clients"] = len(self._world_clients)

        return web.json_response(body, status=200 if healthy else 503)

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /status — detailed system status."""
        status: dict[str, Any] = {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "ws_clients": len(self._clients),
            "world_clients": len(self._world_clients),
        }
        if self._runner:
            try:
                runner_status = self._runner.get_status()
                sanitized = {}
                for k, v in runner_status.items():
                    try:
                        json.dumps(v)
                        sanitized[k] = v
                    except (TypeError, ValueError):
                        sanitized[k] = str(v)
                status["runner"] = sanitized
            except Exception as exc:
                status["runner_error"] = str(exc)
        return web.json_response(status)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics — resource usage metrics."""
        metrics: dict[str, Any] = {"ws_clients": len(self._clients)}
        if self._runner:
            metrics["cycle_count"] = self._runner.cycle_count
        return web.json_response(metrics)

    def _check_healthy(self) -> bool:
        """Determine if the system is healthy."""
        if not self._runner:
            return True
        booted = getattr(self._runner, "_booted", False)
        if not booted:
            uptime = time.monotonic() - self._start_time
            return uptime < 120.0
        return getattr(self._runner, "running", False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _send_ws(ws: web.WebSocketResponse, data: dict) -> None:
        """Send a JSON message to a single WebSocket client."""
        try:
            await ws.send_json(data)
        except (ConnectionError, RuntimeError):
            pass
