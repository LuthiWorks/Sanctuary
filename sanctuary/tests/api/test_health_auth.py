"""Regression guard: the health server serves /status only to loopback peers.

/health is open (Docker liveness). /status and /metrics expose detailed
internal state and are gated to loopback peers -- detailed internals never
leave the host even on a 0.0.0.0 bind, and this server has no token system.

Tests the peer classifier directly (a non-loopback socket can't be made
in-process).

Authored by Fable 5 (adversarial seat), 2026-07-02.
"""
from __future__ import annotations

from sanctuary.api.health import _peer_is_loopback


class _FakeWriter:
    def __init__(self, peer):
        self._peer = peer

    def get_extra_info(self, key):
        return self._peer if key == "peername" else None


def test_peer_is_loopback_classification():
    assert _peer_is_loopback(_FakeWriter(("127.0.0.1", 5000))) is True
    assert _peer_is_loopback(_FakeWriter(("::1", 5000))) is True
    assert _peer_is_loopback(_FakeWriter(("8.8.8.8", 5000))) is False
    assert _peer_is_loopback(_FakeWriter(("10.0.0.5", 5000))) is False
    assert _peer_is_loopback(_FakeWriter(("192.168.1.4", 5000))) is False
    # No peer info / malformed -> treated as not-loopback (fail closed).
    assert _peer_is_loopback(_FakeWriter(None)) is False
