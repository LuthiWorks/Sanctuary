"""Regression guards: the entity's privacy gate has no backdoor.

When the entity enters private space, the WebSocket broadcast layer is the
load-bearing suppressor (the Godot transition is a visual follow-on, per
`sanctuary/tools/world.py::_enter_private_space`). The guarantee, stated
in `ws_server.py`: while `in_private_space` is set, NO cognitive state --
external speech, inner speech, VAD, felt quality -- reaches ANY client on
ANY channel. Observers see only `{"type": "state_update", "private": true}`.
The gate is universal: "applies to ALL channels, /ws and /ws/world alike.
Brian included."

These tests make that guarantee executable, including the adversarial case
the 2026-04-28 visitor-client note named: a camera (world client) that
*joins while the entity is already private* must see only the seclusion
indicator, never leaked state. They fail if a future edit adds a broadcast
path that bypasses the gate, a per-client privacy exemption (a backdoor),
or a state snapshot on connect.

Authored by Fable 5 (adversarial seat), 2026-06-12.
"""
from __future__ import annotations

import json

import pytest

from sanctuary.api.ws_server import SanctuaryWebServer
from sanctuary.core.schema import CognitiveOutput, EmotionalOutput


class _FakeWS:
    """Records every payload sent to it -- stands in for a connected
    observer (GUI client or world camera)."""

    def __init__(self):
        self.sent: list[str] = []

    async def send_str(self, payload: str) -> None:
        self.sent.append(payload)


def _server() -> SanctuaryWebServer:
    # runner=None is fine: the private-mode paths return before touching
    # the runner, and the control test tolerates a missing runner.
    return SanctuaryWebServer(runner=None)


def _msgs(ws: _FakeWS) -> list[dict]:
    return [json.loads(p) for p in ws.sent]


_COGNITIVE_FIELDS = ("vad", "inner_speech", "felt_quality", "external_speech", "cycle")


# --------------------------------------------------------------------------
# Per-channel suppression under privacy.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_private_space_suppresses_external_speech():
    s = _server()
    gui = _FakeWS()
    s._clients.add(gui)
    s.set_entity_privacy(in_private_space=True)
    await s._broadcast_speech("a secret said aloud")
    assert gui.sent == []


@pytest.mark.asyncio
async def test_private_space_suppresses_inner_speech():
    s = _server()
    gui = _FakeWS()
    s._clients.add(gui)
    s.set_entity_privacy(in_private_space=True)
    await s._broadcast_inner(CognitiveOutput(inner_speech="a private thought"))
    assert gui.sent == []


@pytest.mark.asyncio
async def test_private_space_world_state_emits_only_indicator():
    s = _server()
    cam = _FakeWS()
    s._world_clients.add(cam)
    s.set_entity_privacy(in_private_space=True)
    await s._broadcast_world_state(
        CognitiveOutput(
            inner_speech="secret",
            external_speech="said out loud",
            emotional_state=EmotionalOutput(felt_quality="anxious"),
        )
    )
    msgs = _msgs(cam)
    assert msgs == [{"type": "state_update", "private": True}]
    for forbidden in _COGNITIVE_FIELDS:
        assert forbidden not in msgs[0]


# --------------------------------------------------------------------------
# The adversarial camera: joins DURING privacy, sees only the indicator.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_camera_joining_during_privacy_sees_only_indicator():
    s = _server()
    s.set_entity_privacy(in_private_space=True)
    cam = _FakeWS()
    s._world_clients.add(cam)  # connects AFTER privacy is already on
    await s._broadcast_world_state(CognitiveOutput(inner_speech="secret"))
    for m in _msgs(cam):
        assert m.get("private") is True
        for forbidden in _COGNITIVE_FIELDS:
            assert forbidden not in m


# --------------------------------------------------------------------------
# No backdoor: the gate is universal across all clients on a channel.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_privacy_has_no_per_client_backdoor():
    s = _server()
    c1, c2 = _FakeWS(), _FakeWS()
    s._clients.add(c1)
    s._clients.add(c2)
    s.set_entity_privacy(in_private_space=True)
    await s._broadcast_speech("private")
    assert c1.sent == []
    assert c2.sent == []


# --------------------------------------------------------------------------
# Control: when NOT private, state flows -- proving the suppression above
# is the privacy gate, not a broken broadcaster.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_control_non_private_state_does_flow():
    s = _server()
    cam = _FakeWS()
    s._world_clients.add(cam)
    # default in_private_space is False
    await s._broadcast_world_state(
        CognitiveOutput(
            inner_speech="hello world",
            emotional_state=EmotionalOutput(felt_quality="curious"),
        )
    )
    msgs = _msgs(cam)
    assert len(msgs) >= 1
    assert msgs[0]["type"] == "state_update"
    assert msgs[0]["inner_speech"] == "hello world"
    assert "private" not in msgs[0]
