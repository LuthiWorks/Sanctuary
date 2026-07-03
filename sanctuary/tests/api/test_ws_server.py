"""Tests for the WebSocket server bridge.

Tests that the SanctuaryWebServer:
1. Starts and stops cleanly
2. Accepts WebSocket connections
3. Routes user messages to the runner
4. Broadcasts speech to connected clients
5. Serves health/status HTTP endpoints
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio
import aiohttp

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.api.ws_server import SanctuaryWebServer
from sanctuary.core.schema import CognitiveOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a charter file."""
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)

    charter_path = data_dir / "charter.md"
    charter_path.write_text(
        """\
# The Sanctuary Charter

## Value Seeds

- **Honesty**: Say what you believe to be true.
- **Care**: The wellbeing of others matters.
""",
        encoding="utf-8",
    )
    return data_dir


@pytest.fixture
def runner_config(tmp_data_dir: Path) -> RunnerConfig:
    return RunnerConfig(
        cycle_delay=0.01,
        data_dir=str(tmp_data_dir),
        charter_path=str(tmp_data_dir / "charter.md"),
        use_in_memory_store=True,
        silence_threshold=999.0,
        stream_history=5,
    )


@pytest_asyncio.fixture
async def booted_runner(runner_config: RunnerConfig) -> SanctuaryRunner:
    runner = SanctuaryRunner(config=runner_config)
    await runner.boot()
    return runner


# Use a different port for each test to avoid conflicts
_port_counter = 19700


def next_port() -> int:
    global _port_counter
    _port_counter += 1
    return _port_counter


# ---------------------------------------------------------------------------
# Tests: Server lifecycle
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    """WebSocket server starts and stops cleanly."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        assert server.running
        assert server.client_count == 0
        await server.stop()
        assert not server.running

    @pytest.mark.asyncio
    async def test_start_without_runner(self):
        """Server can start without a runner (standalone mode)."""
        port = next_port()
        server = SanctuaryWebServer(runner=None, port=port)
        await server.start()
        assert server.running
        await server.stop()


# ---------------------------------------------------------------------------
# Tests: HTTP health endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """Health/status/metrics endpoints work over HTTP."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/health") as resp:
                    assert resp.status in (200, 503)
                    data = await resp.json()
                    assert "status" in data
                    assert "uptime_seconds" in data
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_status_endpoint(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/status") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "uptime_seconds" in data
                    assert "ws_clients" in data
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/metrics") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "ws_clients" in data
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Tests: WebSocket connection
# ---------------------------------------------------------------------------


class TestWebSocketConnection:
    """WebSocket client can connect and receive messages."""

    @pytest.mark.asyncio
    async def test_connect_and_receive_status(self, booted_runner):
        """Client receives status messages on connect."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Should receive initial status messages
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "status"
                    assert msg["status"] == "connected"

                    assert server.client_count >= 1
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_send_message_injects_into_runner(self, booted_runner):
        """User messages from WebSocket are injected into the runner."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial status messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    # Send a user message
                    await ws.send_json({
                        "type": "message",
                        "content": "Hello from test"
                    })

                    # Give the sensorium time to receive
                    await asyncio.sleep(0.1)

                    # Verify the percept was injected
                    percepts = await booted_runner.sensorium.drain_percepts()
                    texts = [p.content for p in percepts]
                    assert any("Hello from test" in t for t in texts)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_broadcast_speech(self, booted_runner):
        """Speech from the runner is broadcast to connected clients."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    # Simulate speech broadcast
                    await server._broadcast_speech("Test speech output")

                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "message"
                    assert msg["content"] == "Test speech output"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, booted_runner):
        """Invalid JSON from client produces an error response."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await ws.send_str("not json at all")
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "error"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_status_request(self, booted_runner):
        """Client can request system status."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await ws.send_json({"type": "status_request"})
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "system"
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Tests: /ws/world endpoint (Godot SanctuaryWorld bridge)
# ---------------------------------------------------------------------------


def _fake_cognitive_output(
    inner: str = "thinking quietly",
    external: Optional[str] = None,
    felt: str = "calm",
) -> CognitiveOutput:
    from sanctuary.core.schema import EmotionalOutput

    return CognitiveOutput(
        inner_speech=inner,
        external_speech=external,
        emotional_state=EmotionalOutput(felt_quality=felt),
    )


class TestWorldWebSocket:
    """The /ws/world endpoint serves the Godot SanctuaryWorld client."""

    @pytest.mark.asyncio
    async def test_world_connect_and_receive_hello(self, booted_runner):
        """World client receives a connected status on connect."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "status"
                    assert msg["status"] == "connected"
                    assert server.world_client_count >= 1
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_world_state_broadcast(self, booted_runner):
        """A CognitiveOutput drives a state_update on the world channel."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # hello

                    output = _fake_cognitive_output(
                        inner="hello world", felt="curious"
                    )
                    await server._broadcast_world_state(output)

                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "state_update"
                    assert msg["inner_speech"] == "hello world"
                    assert msg["felt_quality"] == "curious"
                    assert "vad" in msg
                    assert {"valence", "arousal", "dominance"} <= set(msg["vad"])
                    assert "cycle" in msg
                    assert "cycle_latency_ms" in msg
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_world_external_speech_is_separate_message(
        self, booted_runner
    ):
        """external_speech goes out as its own message after state_update."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # hello

                    output = _fake_cognitive_output(external="Hello, world.")
                    await server._broadcast_world_state(output)

                    state = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    speech = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert state["type"] == "state_update"
                    assert speech["type"] == "external_speech"
                    assert speech["content"] == "Hello, world."
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_world_scene_state_injects_environment_percept(
        self, booted_runner
    ):
        """Godot's scene_state report becomes an environment percept."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # hello
                    await ws.send_json({
                        "type": "scene_state",
                        "objects": [
                            {"id": "obj_1", "type": "cube", "name": "Red Cube"},
                            {"id": "obj_2", "type": "sphere", "name": "Blue Sphere"},
                        ],
                        "entity_position": [0.0, 1.0, 0.0],
                    })
                    await asyncio.sleep(0.1)
                    percepts = await booted_runner.sensorium.drain_percepts()
                    env = [p for p in percepts if p.modality == "environment"]
                    assert env, "No environment percept was injected"
                    content = env[0].content
                    assert "2 object" in content
                    assert "Red Cube" in content
                    assert "I am at" in content
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_world_invalid_json_returns_error(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # hello
                    await ws.send_str("not json")
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "error"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_world_and_gui_channels_isolated(self, booted_runner):
        """A state_update on /ws/world does NOT reach GUI clients on /ws."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                gui = await session.ws_connect(f"http://localhost:{port}/ws")
                world = await session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                )
                try:
                    # Drain initial hellos.
                    await asyncio.wait_for(gui.receive_json(), timeout=2.0)
                    await asyncio.wait_for(gui.receive_json(), timeout=2.0)
                    await asyncio.wait_for(world.receive_json(), timeout=2.0)

                    output = _fake_cognitive_output(inner="quiet")
                    await server._broadcast_world_state(output)

                    # World gets the state_update.
                    msg = await asyncio.wait_for(
                        world.receive_json(), timeout=2.0
                    )
                    assert msg["type"] == "state_update"

                    # GUI must not receive a state_update — only the inner
                    # broadcast it already gets via on_output.
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(gui.receive_json(), timeout=0.3)
                finally:
                    await gui.close()
                    await world.close()
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Tests: DoS connection cap
# ---------------------------------------------------------------------------


class TestConnectionCap:
    """The server refuses new connections past its capacity, counting
    connections that are still open (not only fully-authenticated ones)."""

    @pytest.mark.asyncio
    async def test_gui_connection_cap_counts_open_connections(self, booted_runner, monkeypatch):
        from sanctuary.api import ws_server as W
        monkeypatch.setattr(W, "MAX_WS_CLIENTS", 1)
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                # First connection takes the single slot and is held open.
                async with session.ws_connect(f"http://localhost:{port}/ws") as ws1:
                    await asyncio.wait_for(ws1.receive(), timeout=5.0)  # connected
                    for _ in range(50):
                        if server._gui_conns >= 1:
                            break
                        await asyncio.sleep(0.01)
                    assert server._gui_conns == 1  # the live counter tracks it
                    # Second connection is refused at capacity.
                    async with session.ws_connect(f"http://localhost:{port}/ws") as ws2:
                        msg = await asyncio.wait_for(ws2.receive(), timeout=5.0)
                        data = json.loads(msg.data)
                        assert data["type"] == "error"
                        assert "capacity" in data["content"]
                # After both close, the counter returns to zero.
                for _ in range(50):
                    if server._gui_conns == 0:
                        break
                    await asyncio.sleep(0.01)
                assert server._gui_conns == 0
        finally:
            await server.stop()
