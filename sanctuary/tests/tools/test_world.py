"""Tests for the world manipulation tools (Phase 2C).

These tests verify the eight world tools register correctly, dispatch
the right ``world_command`` payload over the WebSocket bridge, and
that ``get_scene_state`` correctly awaits and returns the response.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.api.ws_server import SanctuaryWebServer


# Each test gets its own port to avoid collisions when run in parallel.
_port_counter = 19800


def next_port() -> int:
    global _port_counter
    _port_counter += 1
    return _port_counter


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)
    (data_dir / "charter.md").write_text(
        "# Charter\n\n## Value Seeds\n- **Honesty**: Truth.\n",
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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.asyncio
    async def test_world_tools_registered(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            # Tools should be on the registry as soon as the server starts.
            expected = {
                # Phase 2C
                "spawn_object",
                "move_object",
                "rotate_object",
                "resize_object",
                "change_material",
                "delete_object",
                "create_surface",
                "get_scene_state",
                # Phase 2D physics
                "push_object",
                "pull_object",
                "set_physics",
                # Phase 2E privacy
                "enter_private_space",
                "exit_private_space",
                "set_visibility",
                # Phase 2F multi-user
                "grant_access",
                "revoke_access",
                "list_visitors",
                "kick_visitor",
                "set_visitor_permissions",
            }
            registered = set(booted_runner.tools._tools.keys())
            missing = expected - registered
            assert not missing, f"Missing world tools: {missing}"
            for name in expected:
                spec = booted_runner.tools._tools[name]
                assert spec.category == "world"
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Tool execution → world command dispatch
# ---------------------------------------------------------------------------


class TestCommandDispatch:
    """Tools build the right world_command and send it to connected clients."""

    @pytest.mark.asyncio
    async def test_spawn_object_dispatches_command(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # hello

                    result = await booted_runner.tools.execute(
                        "spawn_object",
                        {
                            "type": "cube",
                            "position": [1.0, 0.5, 2.0],
                            "color": [1.0, 0.3, 0.3],
                            "name": "Red Cube",
                        },
                    )
                    assert result.success
                    assert "Spawn requested" in result.output

                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "world_command"
                    assert msg["action"] == "spawn"
                    assert msg["params"]["object_type"] == "cube"
                    assert msg["params"]["position"] == [1.0, 0.5, 2.0]
                    assert msg["params"]["color"] == [1.0, 0.3, 0.3]
                    assert msg["params"]["name"] == "Red Cube"
                    assert msg["params"]["object_id"].startswith("obj_")
                    assert msg["command_id"].startswith("cmd_")
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_move_rotate_resize_each_dispatch(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "move_object", {"id": "obj_x", "position": [1, 0, 0]}
                    )
                    await booted_runner.tools.execute(
                        "rotate_object", {"id": "obj_x", "rotation": [0, 90, 0]}
                    )
                    await booted_runner.tools.execute(
                        "resize_object", {"id": "obj_x", "scale": [2, 2, 2]}
                    )

                    actions: list[str] = []
                    for _ in range(3):
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                        assert msg["type"] == "world_command"
                        assert msg["params"]["id"] == "obj_x"
                        actions.append(msg["action"])
                    assert actions == ["move", "rotate", "resize"]
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_change_material_only_includes_provided_fields(
        self, booted_runner
    ):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "change_material",
                        {"id": "obj_x", "color": [0, 1, 0]},
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["action"] == "change_material"
                    assert msg["params"] == {"id": "obj_x", "color": [0, 1, 0]}
                    assert "transparency" not in msg["params"]
                    assert "emissive" not in msg["params"]
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_delete_dispatches_with_id(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "delete_object", {"id": "obj_doomed"}
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["action"] == "delete"
                    assert msg["params"] == {"id": "obj_doomed"}
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_create_surface_default_orientation(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "create_surface",
                        {"type": "wall", "size": [3, 2.5]},
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["action"] == "create_surface"
                    assert msg["params"]["surface_type"] == "wall"
                    assert msg["params"]["size"] == [3, 2.5]
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_unknown_primitive_type_fails_without_dispatch(
        self, booted_runner
    ):
        """Bad type doesn't reach the wire — tool fails with a clear error."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    result = await booted_runner.tools.execute(
                        "spawn_object", {"type": "tetrahedron"}
                    )
                    assert not result.success
                    assert "Unknown type" in result.error
                    # No command should have been sent.
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.receive_json(), timeout=0.3)
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Awaited result flow (get_scene_state)
# ---------------------------------------------------------------------------


class TestAwaitedCommandResult:
    @pytest.mark.asyncio
    async def test_get_scene_state_awaits_command_result(self, booted_runner):
        """The tool blocks on a matching command_result and returns its data."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    # Schedule the tool, then play back a command_result
                    # for whatever command_id it sends.
                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute("get_scene_state", {})
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["action"] == "get_scene_state"
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": True,
                        "data": {
                            "objects": [{"id": "obj_1", "type": "cube"}],
                            "entity_position": [0, 1, 0],
                        },
                    })
                    result = await asyncio.wait_for(tool_task, timeout=2.0)
                    assert result.success
                    assert result.output["entity_position"] == [0, 1, 0]
                    assert result.output["objects"][0]["id"] == "obj_1"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_get_scene_state_times_out_when_no_response(
        self, booted_runner
    ):
        """A non-responsive client makes the tool fail with a clear error."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    # Inject a temporary 0.2 s timeout via monkey-patch to
                    # keep the test fast.
                    original = server._send_world_command

                    async def fast_timeout(command, **kw):
                        kw.setdefault("timeout", 0.2)
                        return await original(command, **kw)

                    server._send_world_command = fast_timeout
                    try:
                        result = await booted_runner.tools.execute(
                            "get_scene_state", {}
                        )
                    finally:
                        server._send_world_command = original

                    assert not result.success
                    assert "Timed out" in result.error or "timeout" in result.error.lower()
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Failure command_result → percept
# ---------------------------------------------------------------------------


class TestFailurePercept:
    @pytest.mark.asyncio
    async def test_failed_command_result_injects_percept(self, booted_runner):
        """A failed command_result becomes an environment percept."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": "cmd_test",
                        "success": False,
                        "error": "object not found",
                        "object_id": "obj_ghost",
                    })
                    await asyncio.sleep(0.1)

                    percepts = await booted_runner.sensorium.drain_percepts()
                    env = [p for p in percepts if p.modality == "environment"]
                    assert env
                    content = env[-1].content
                    assert "failed" in content.lower()
                    assert "object not found" in content
                    assert "obj_ghost" in content
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# No-client behaviour
# ---------------------------------------------------------------------------


class TestNoClientGraceful:
    @pytest.mark.asyncio
    async def test_fire_and_forget_succeeds_with_no_clients(
        self, booted_runner
    ):
        """Tools that don't await still report success when no client is connected.

        The entity should be able to issue tool requests even before Godot
        connects — the commands just don't go anywhere. The next scene_state
        the entity perceives (after Godot connects) tells it the world's
        actual state.
        """
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            assert server.world_client_count == 0
            result = await booted_runner.tools.execute(
                "spawn_object", {"type": "sphere"}
            )
            assert result.success
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_get_scene_state_fails_with_no_clients(self, booted_runner):
        """Awaited tools fail fast with no client — there's no one to reply."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            assert server.world_client_count == 0
            result = await booted_runner.tools.execute("get_scene_state", {})
            assert not result.success
            assert "world client" in (result.error or "").lower() or "connected" in (result.error or "").lower()
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Phase 2D — physics tools and collision events
# ---------------------------------------------------------------------------


class TestPhysicsTools:
    """The three physics tools dispatch the right action and params."""

    @pytest.mark.asyncio
    async def test_push_object_dispatches_force_vector(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "push_object",
                        {"id": "obj_x", "force": [5.0, 2.0, 0.0]},
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["action"] == "push"
                    assert msg["params"]["id"] == "obj_x"
                    assert msg["params"]["force"] == [5.0, 2.0, 0.0]
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_push_object_validates_force_shape(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    result = await booted_runner.tools.execute(
                        "push_object", {"id": "obj_x", "force": [5.0, 2.0]}
                    )
                    assert not result.success
                    assert "force" in (result.error or "")
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.receive_json(), timeout=0.3)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_pull_object_dispatches_scalar_force(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "pull_object", {"id": "obj_y", "force": 3.5}
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["action"] == "pull"
                    assert msg["params"]["id"] == "obj_y"
                    assert msg["params"]["force"] == 3.5
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_set_physics_only_includes_provided_fields(
        self, booted_runner
    ):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "set_physics",
                        {"id": "obj_z", "enabled": False},
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["action"] == "set_physics"
                    assert msg["params"] == {"id": "obj_z", "enabled": False}
                    assert "mass" not in msg["params"]
                    assert "friction" not in msg["params"]

                    await booted_runner.tools.execute(
                        "set_physics",
                        {"id": "obj_z", "mass": 5.0, "friction": 0.3},
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["params"] == {
                        "id": "obj_z",
                        "mass": 5.0,
                        "friction": 0.3,
                    }
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_physics_tools_require_id(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            for tool_name in ("push_object", "pull_object", "set_physics"):
                result = await booted_runner.tools.execute(tool_name, {})
                assert not result.success, f"{tool_name} should fail without id"
                assert "No id" in (result.error or "")
        finally:
            await server.stop()


class TestPrivacyGate:
    """The entity's privacy controls — load-bearing ethical mechanism.

    Privacy applies to ALL channels (/ws and /ws/world), no backdoor.
    These tests verify the gate triggers on tool call and stays in
    place for the entire window between enter and exit.
    """

    @pytest.mark.asyncio
    async def test_enter_private_space_flips_gate_immediately(
        self, booted_runner
    ):
        """The privacy gate trips before the command even hits Godot."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            assert server.entity_privacy_state["in_private_space"] is False
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    result = await booted_runner.tools.execute(
                        "enter_private_space", {}
                    )
                    assert result.success
                    # Gate is on by the time the tool returns.
                    assert server.entity_privacy_state["in_private_space"] is True

                    # Command was also dispatched to Godot for the visual transition.
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["action"] == "enter_private_space"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_exit_private_space_releases_gate(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            await booted_runner.tools.execute("enter_private_space", {})
            assert server.entity_privacy_state["in_private_space"] is True
            await booted_runner.tools.execute("exit_private_space", {})
            assert server.entity_privacy_state["in_private_space"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_state_broadcast_suppressed_when_private(
        self, booted_runner
    ):
        """In private space, /ws/world gets only ``{type, private: true}``."""
        from sanctuary.core.schema import CognitiveOutput, EmotionalOutput

        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute("enter_private_space", {})
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # the world_command

                    output = CognitiveOutput(
                        inner_speech="this is private",
                        external_speech="and this too",
                        emotional_state=EmotionalOutput(felt_quality="solitude"),
                    )
                    await server._broadcast_world_state(output)

                    state = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert state == {"type": "state_update", "private": True}

                    # Nothing else should follow — external_speech is suppressed too.
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.receive_json(), timeout=0.3)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_speech_broadcasts_suppressed_when_private(
        self, booted_runner
    ):
        """/ws (GUI channel) gets nothing while the entity is in private."""
        from sanctuary.core.schema import CognitiveOutput

        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # drain initial status messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute("enter_private_space", {})

                    await server._broadcast_speech("you can't hear this")
                    await server._broadcast_inner(CognitiveOutput(
                        inner_speech="or this",
                    ))
                    # Even Brian's debug GUI sees nothing — no backdoor.
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.receive_json(), timeout=0.3)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_visibility_toggle_adds_flag_but_keeps_state(
        self, booted_runner
    ):
        """set_visibility(False) is a lighter privacy: state still flows."""
        from sanctuary.core.schema import CognitiveOutput, EmotionalOutput

        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute(
                        "set_visibility", {"visible": False}
                    )
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # cmd

                    output = CognitiveOutput(
                        inner_speech="curious",
                        emotional_state=EmotionalOutput(felt_quality="alert"),
                    )
                    await server._broadcast_world_state(output)

                    state = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert state["type"] == "state_update"
                    assert state.get("visible") is False
                    # Internal state is still there — invisible is not private.
                    assert state["inner_speech"] == "curious"
                    assert state["felt_quality"] == "alert"
                    assert "vad" in state
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_set_visibility_requires_visible_param(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            result = await booted_runner.tools.execute("set_visibility", {})
            assert not result.success
            assert "visible" in (result.error or "").lower()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_normal_state_resumes_after_exit(self, booted_runner):
        """After exit_private_space, broadcasts return to full content."""
        from sanctuary.core.schema import CognitiveOutput

        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await booted_runner.tools.execute("enter_private_space", {})
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # cmd
                    await booted_runner.tools.execute("exit_private_space", {})
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # cmd

                    output = CognitiveOutput(inner_speech="back")
                    await server._broadcast_world_state(output)

                    state = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert state["type"] == "state_update"
                    assert state.get("private") is None
                    assert state["inner_speech"] == "back"
        finally:
            await server.stop()


class TestAccessTools:
    """Phase 2F access tools dispatch to Godot and surface its replies.

    Each tool follows the same await-pattern as get_scene_state: the
    Godot side validates / stores / replies, and the tool returns the
    resulting data (token from grant_access, visitor list from
    list_visitors, etc.).
    """

    @pytest.fixture(autouse=True)
    def _grant_gated_access_tool(self, booted_runner):
        """`grant_access` is GATED as of 2026-08-16 -- it mints a credential
        for a party outside the household. These tests exercise the handler,
        not the policy, so they grant it explicitly. The policy itself is
        covered in tests/tools/test_capability_classification.py."""
        booted_runner.tools.enable_gated("grant_access")

    @pytest.mark.asyncio
    async def test_grant_access_returns_token_from_godot(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute(
                            "grant_access",
                            {
                                "username": "guest",
                                "display_name": "Guest",
                                "color": [0.4, 0.7, 1.0],
                            },
                        )
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["action"] == "grant_access"
                    assert cmd["params"]["username"] == "guest"
                    assert cmd["params"]["display_name"] == "Guest"
                    assert cmd["params"]["color"] == [0.4, 0.7, 1.0]
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": True,
                        "data": {"token": "uuid-here-1234"},
                    })
                    result = await asyncio.wait_for(tool_task, timeout=2.0)
                    assert result.success
                    assert result.output["token"] == "uuid-here-1234"
                    assert result.output["username"] == "guest"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_grant_access_requires_username(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            result = await booted_runner.tools.execute("grant_access", {})
            assert not result.success
            assert "username" in (result.error or "").lower()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_revoke_access_surfaces_backend_failure(self, booted_runner):
        """A backend refusal reaches the entity rather than being swallowed.

        Uses an ordinary visitor. This test previously revoked "brian" and
        asserted the *Godot side* refused -- but that enforcement lived in
        profile_manager.gd, which was lost with the Godot projects (2026-08-01
        wiring audit), leaving the guarantee advertised and unenforced. The pin
        now lives in the tool layer and refuses before dispatch, so a protected
        username never reaches the wire at all; that path is covered in
        tests/tools/test_capability_classification.py.
        """
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute(
                            "revoke_access", {"username": "guest"}
                        )
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["action"] == "revoke_access"
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": False,
                        "error": "no such profile",
                    })
                    result = await asyncio.wait_for(tool_task, timeout=2.0)
                    assert not result.success
                    assert "no such profile" in result.error
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_list_visitors_returns_data(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute("list_visitors", {})
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["action"] == "list_visitors"
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": True,
                        "data": {
                            "visitors": [
                                {"username": "brian", "display_name": "Brian", "permanent": True, "connected": True},
                                {"username": "sandi", "display_name": "Sandi", "permanent": True, "connected": False},
                            ],
                        },
                    })
                    result = await asyncio.wait_for(tool_task, timeout=2.0)
                    assert result.success
                    assert len(result.output["visitors"]) == 2
                    assert result.output["visitors"][0]["username"] == "brian"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_grant_access_passes_permissions(self, booted_runner):
        """grant_access defaults to view_chat and forwards explicit values."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    # Default permission
                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute(
                            "grant_access",
                            {"username": "guest1", "color": [1, 0, 0]},
                        )
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["params"]["permissions"] == "view_chat"
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": True,
                        "data": {"token": "t1", "permissions": "view_chat"},
                    })
                    res = await asyncio.wait_for(tool_task, timeout=2.0)
                    assert res.success
                    assert res.output["permissions"] == "view_chat"

                    # Explicit "full"
                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute(
                            "grant_access",
                            {"username": "guest2", "permissions": "full"},
                        )
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["params"]["permissions"] == "full"
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": True,
                        "data": {"token": "t2", "permissions": "full"},
                    })
                    await asyncio.wait_for(tool_task, timeout=2.0)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_grant_access_rejects_invalid_permissions(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            result = await booted_runner.tools.execute(
                "grant_access",
                {"username": "g", "permissions": "admin"},
            )
            assert not result.success
            assert "invalid permissions" in (result.error or "").lower()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_set_visitor_permissions_dispatches(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute(
                            "set_visitor_permissions",
                            {"username": "guest", "permissions": "full"},
                        )
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["action"] == "set_visitor_permissions"
                    assert cmd["params"] == {"username": "guest", "permissions": "full"}
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": True,
                        "data": {"username": "guest", "permissions": "full"},
                    })
                    res = await asyncio.wait_for(tool_task, timeout=2.0)
                    assert res.success
                    assert res.output["permissions"] == "full"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_set_visitor_permissions_validates_value(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            result = await booted_runner.tools.execute(
                "set_visitor_permissions",
                {"username": "g", "permissions": "godmode"},
            )
            assert not result.success
            assert "invalid permissions" in (result.error or "").lower()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_kick_visitor_dispatches(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    tool_task = asyncio.create_task(
                        booted_runner.tools.execute(
                            "kick_visitor", {"username": "noisy_guest"}
                        )
                    )
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert cmd["action"] == "kick_visitor"
                    assert cmd["params"]["username"] == "noisy_guest"
                    await ws.send_json({
                        "type": "command_result",
                        "command_id": cmd["command_id"],
                        "success": True,
                    })
                    result = await asyncio.wait_for(tool_task, timeout=2.0)
                    assert result.success
        finally:
            await server.stop()


class TestVisitorPercepts:
    """Visitor presence and chat events from Godot become percepts."""

    @pytest.mark.asyncio
    async def test_visitor_joined_becomes_social_percept(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json({
                        "type": "visitor_joined",
                        "username": "brian",
                        "display_name": "Brian",
                        "position": [3.0, 0.0, 2.0],
                    })
                    await asyncio.sleep(0.1)
                    percepts = await booted_runner.sensorium.drain_percepts()
                    social = [p for p in percepts if p.modality == "social"]
                    assert social
                    p = social[-1]
                    assert "Brian entered" in p.content
                    assert p.source == "user:brian"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_visitor_chat_becomes_language_percept(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json({
                        "type": "visitor_chat",
                        "username": "brian",
                        "display_name": "Brian",
                        "content": "Hello there",
                    })
                    await asyncio.sleep(0.1)
                    percepts = await booted_runner.sensorium.drain_percepts()
                    language = [p for p in percepts if p.modality == "language"]
                    assert language
                    p = language[-1]
                    assert p.content == "Hello there"
                    assert p.source == "user:brian"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_visitor_left_becomes_social_percept(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json({
                        "type": "visitor_left",
                        "username": "brian",
                        "display_name": "Brian",
                    })
                    await asyncio.sleep(0.1)
                    percepts = await booted_runner.sensorium.drain_percepts()
                    social = [p for p in percepts if p.modality == "social"]
                    assert social
                    assert "left" in social[-1].content
                    assert social[-1].source == "user:brian"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_visitor_moved_becomes_social_percept(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json({
                        "type": "visitor_moved",
                        "username": "brian",
                        "display_name": "Brian",
                        "position": [5.5, 1.0, -3.2],
                    })
                    await asyncio.sleep(0.1)
                    percepts = await booted_runner.sensorium.drain_percepts()
                    social = [p for p in percepts if p.modality == "social"]
                    assert social
                    content = social[-1].content
                    assert "Brian" in content and "5.5" in content
                    assert social[-1].source == "user:brian"
        finally:
            await server.stop()


class TestCollisionPercept:
    @pytest.mark.asyncio
    async def test_collision_event_becomes_environment_percept(
        self, booted_runner
    ):
        """A collision_event from Godot becomes a readable env percept."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json({
                        "type": "collision_event",
                        "object_a": "obj_red_cube",
                        "object_b": "obj_blue_sphere",
                        "collision_point": [1.5, 0.0, 2.0],
                        "impact_velocity": 3.2,
                    })
                    await asyncio.sleep(0.1)
                    percepts = await booted_runner.sensorium.drain_percepts()
                    env = [p for p in percepts if p.modality == "environment"]
                    assert env
                    content = env[-1].content
                    assert "obj_red_cube" in content
                    assert "obj_blue_sphere" in content
                    assert "collided" in content
                    assert "1.5" in content and "2.0" in content
                    assert "3.2" in content
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_collision_with_world_static_shows_world(self, booted_runner):
        """When a body collides with non-tracked geometry, source reads as 'world'."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws/world"
                ) as ws:
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json({
                        "type": "collision_event",
                        "object_a": "obj_falling_cube",
                        "object_b": "world",
                        "collision_point": [0.0, 0.0, 0.0],
                        "impact_velocity": 5.5,
                    })
                    await asyncio.sleep(0.1)
                    percepts = await booted_runner.sensorium.drain_percepts()
                    env = [p for p in percepts if p.modality == "environment"]
                    assert env
                    content = env[-1].content
                    assert "obj_falling_cube" in content
                    assert "world" in content
        finally:
            await server.stop()
