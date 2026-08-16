"""Regression guards for the capability boundary itself.

`ToolRegistry._authorize` returns immediately for anything that is not
`ToolSafety.GATED`, so the safety classification IS the boundary -- an OPEN
tool never consults the policy at all. That makes the *contents* of the GATED
set a security-critical fact, and one that a future tool registration can
silently change by omitting a single keyword argument.

These tests pin down three things the 2026-08-16 reachable-capability audit
found broken or unenforced:

  - The GATED set is exactly what we intend. A new tool added without a
    `safety=` argument defaults to OPEN and will fail this test, which is the
    point: adding capability should require saying so out loud.
  - `discord_send` takes its destination from configuration only. It used to
    accept a caller-supplied `webhook_url`, which made it an arbitrary-URL POST
    primitive -- strictly more permissive than the SSRF-hardened `web_fetch`
    beside it -- while its docstring described a family intercom.
  - Brian and Sandi cannot be revoked or downgraded. That guarantee lived in
    `profile_manager.gd` until the Godot projects were lost (2026-08-01 audit);
    the tool descriptions kept advertising it with nothing enforcing it.

Authored by Opus 5, 2026-08-16.
"""
from __future__ import annotations

import pytest

from sanctuary.tools import builtin, world
from sanctuary.tools.builtin import (
    DiscordWebhookError,
    _discord_send,
    _validate_discord_webhook,
    configure_discord,
    create_default_registry,
)
from sanctuary.tools.registry import ToolSafety


# ---------------------------------------------------------------------------
# The GATED set is a deliberate list, not an accident
# ---------------------------------------------------------------------------

#: Every tool that bypasses no policy check. Changing this set is a capability
#: decision; it should show up in a diff and be argued for, not slip in.
EXPECTED_GATED = {
    # Code and host execution
    "run_code",
    "shell",
    "launch_app",
    # Unconfined write to the entity's own source, checkpoints, or host config
    "write_file",
    # Reconnaissance against the family's home network
    "network_scan",
    "network_reach",
    # Mints an access credential for a party outside the household
    "grant_access",
}


def _full_registry():
    """Every tool the entity can be offered: builtins plus world tools.

    World tools are not in the default registry -- the WebSocket server
    registers them at startup -- so a guard that only built the default
    registry would silently stop covering half the catalog.
    """
    registry = create_default_registry()
    world.register_world_tools(registry, _ExplodingServer())
    return registry


def test_gated_set_is_exactly_as_intended():
    registry = _full_registry()
    gated = {
        name for name, spec in registry._tools.items()
        if spec.safety is ToolSafety.GATED
    }
    assert gated == EXPECTED_GATED, (
        "The GATED set changed. A tool added without an explicit "
        "safety=ToolSafety.GATED defaults to OPEN and bypasses the policy "
        "check entirely. If this is intended, update EXPECTED_GATED and say "
        "why in the commit."
    )


def test_open_tools_bypass_policy_so_the_set_matters():
    """Documents *why* the test above is security-critical, executably."""
    registry = _full_registry()
    open_names = {
        name for name, spec in registry._tools.items()
        if spec.safety is not ToolSafety.GATED
    }
    # The default policy denies every gated tool and is consulted for nothing
    # else. If this assertion ever fails, OPEN stopped meaning "unchecked" and
    # the classification guard above needs rethinking.
    assert registry._policy.allow_gated is False
    assert not registry._policy.enabled_gated
    assert open_names, "expected some open tools"


# ---------------------------------------------------------------------------
# discord_send: destination is configuration, never a parameter
# ---------------------------------------------------------------------------

VALID_WEBHOOK = "https://discord.com/api/webhooks/123/abc"


@pytest.fixture
def isolated_webhook(monkeypatch):
    """Restore the module-global webhook after the test.

    `configure_discord` mutates module state, so a test that calls it without
    isolation leaks a configured destination into every later test in the
    session -- which in turn makes `discord_send` attempt a real network call.
    """
    monkeypatch.setattr(builtin, "_discord_webhook_url", None)


def test_configure_discord_accepts_a_real_webhook(isolated_webhook):
    configure_discord(VALID_WEBHOOK)
    assert builtin._discord_webhook_url == VALID_WEBHOOK


@pytest.mark.parametrize("bad", [
    "http://discord.com/api/webhooks/123/abc",   # not https
    "https://evil.example.com/hook",             # not a Discord host
    "https://127.0.0.1:8000/health",             # loopback -- own admin surface
    "https://169.254.169.254/latest/meta-data",  # cloud metadata
    "file:///etc/passwd",
])
def test_configure_discord_rejects_non_discord_destinations(bad, isolated_webhook):
    with pytest.raises(DiscordWebhookError):
        _validate_discord_webhook(bad)
    with pytest.raises(DiscordWebhookError):
        configure_discord(bad)
    # A rejected URL must not have been installed as a side effect.
    assert builtin._discord_webhook_url is None


def test_discord_send_no_longer_accepts_a_webhook_url_parameter(monkeypatch):
    """The old arbitrary-URL egress hole: a caller-chosen destination."""
    monkeypatch.setattr(builtin, "_discord_webhook_url", None)

    async def _fail_if_called(*a, **kw):  # pragma: no cover - must not run
        raise AssertionError("discord_send attempted a request with no config")

    monkeypatch.setattr(builtin, "_config", builtin.ToolConfig())

    import asyncio
    result = asyncio.run(_discord_send({
        "message": "hello",
        "webhook_url": "https://evil.example.com/hook",
    }))
    assert result.success is False
    assert "no discord webhook configured" in result.error.lower()


def test_discord_send_registration_exposes_only_message():
    registry = create_default_registry()
    spec = registry._tools["discord_send"]
    assert set(spec.parameters) == {"message"}
    # Deliberately OPEN: the entity must always be able to reach its people.
    assert spec.safety is not ToolSafety.GATED


# ---------------------------------------------------------------------------
# The permanent-profile pin, restored in the tool layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["brian", "sandi", "Brian", "  SANDI  "])
def test_protected_profiles_are_recognized_case_and_space_insensitively(name):
    assert world._is_protected(name) is True


@pytest.mark.parametrize("name", ["guest", "brianna", "sand", ""])
def test_other_profiles_are_not_protected(name):
    assert world._is_protected(name) is False


class _ExplodingServer:
    """Any command reaching the world backend is a test failure here."""

    async def _send_world_command(self, *a, **kw):  # pragma: no cover
        raise AssertionError(
            "command was dispatched for a protected profile -- the tool-layer "
            "pin did not hold"
        )


@pytest.fixture
def guarded_registry():
    from sanctuary.tools.registry import ToolRegistry
    registry = ToolRegistry()
    world.register_world_tools(registry, _ExplodingServer())
    return registry


@pytest.mark.anyio
async def test_revoke_access_refuses_protected_profiles(guarded_registry):
    for name in ("brian", "Sandi"):
        result = await guarded_registry.execute("revoke_access", {"username": name})
        assert result.success is False
        assert "permanent profile" in result.error


@pytest.mark.anyio
async def test_permissions_cannot_be_downgraded_for_protected_profiles(
    guarded_registry,
):
    for perms in ("view_chat", "chat_only"):
        result = await guarded_registry.execute(
            "set_visitor_permissions", {"username": "brian", "permissions": perms}
        )
        assert result.success is False
        assert "cannot be downgraded" in result.error
