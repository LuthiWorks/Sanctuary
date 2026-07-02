"""Regression guards: the tool layer's graduated-capability gate is real.

The registry's ``ToolSafety.GATED`` flag used to be cosmetic -- ``execute``
ran every tool regardless of safety, so "gated" tools (code execution, shell,
file writes, app launch) had zero friction. These tests pin the gate down:

  - GATED tools are DENIED BY DEFAULT (deny-all ``ToolPolicy()``): the call
    never reaches the executor and returns a failed ToolResult.
  - OPEN tools are unaffected.
  - A gated tool runs only when deliberately granted -- by name
    (``enabled_gated`` / ``enable_gated``), by the wide ``allow_gated`` switch,
    or by a confirm hook (sync or async).
  - ``write_file`` is gated (it can overwrite the entity's own source), while
    ``read_file`` / ``list_directory`` stay open.

They fail if a future edit removes the authorization check, flips the default
to allow, or reclassifies a dangerous tool back to open.

Authored by Fable 5 (adversarial seat), 2026-07-02.
"""
from __future__ import annotations

import pytest

from sanctuary.tools.builtin import create_default_registry
from sanctuary.tools.registry import (
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSafety,
)


def _make_registry(policy=None) -> tuple[ToolRegistry, dict]:
    """A registry with one OPEN and one GATED probe tool that record calls."""
    calls = {"open": 0, "gated": 0}

    async def open_exec(params):
        calls["open"] += 1
        return ToolResult(tool_name="probe_open", success=True, output="ran")

    async def gated_exec(params):
        calls["gated"] += 1
        return ToolResult(tool_name="probe_gated", success=True, output="ran")

    reg = ToolRegistry(policy=policy)
    reg.register("probe_open", "open probe", {}, open_exec, category="test")
    reg.register(
        "probe_gated", "gated probe", {}, gated_exec,
        safety=ToolSafety.GATED, category="test",
    )
    return reg, calls


# ---------------------------------------------------------------------------
# Default policy: gated denied, open allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_tool_runs_by_default():
    reg, calls = _make_registry()
    result = await reg.execute("probe_open", {})
    assert result.success
    assert calls["open"] == 1


@pytest.mark.asyncio
async def test_gated_tool_denied_by_default():
    reg, calls = _make_registry()
    result = await reg.execute("probe_gated", {})
    assert result.success is False
    assert "blocked by tool policy" in result.error
    # The executor must NEVER have run.
    assert calls["gated"] == 0


@pytest.mark.asyncio
async def test_denied_call_is_recorded_in_history():
    reg, _ = _make_registry()
    await reg.execute("probe_gated", {})
    stats = reg.get_stats()
    assert stats["by_tool"].get("probe_gated") == 1
    assert stats["success_rate"] == 0.0


# ---------------------------------------------------------------------------
# Granting capability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gated_tool_runs_when_named():
    reg, calls = _make_registry(ToolPolicy(enabled_gated=frozenset({"probe_gated"})))
    result = await reg.execute("probe_gated", {})
    assert result.success
    assert calls["gated"] == 1


@pytest.mark.asyncio
async def test_enable_gated_method_grants_at_runtime():
    reg, calls = _make_registry()
    assert (await reg.execute("probe_gated", {})).success is False
    reg.enable_gated("probe_gated")
    assert (await reg.execute("probe_gated", {})).success is True
    assert calls["gated"] == 1


@pytest.mark.asyncio
async def test_allow_gated_switch_grants_all():
    reg, calls = _make_registry(ToolPolicy(allow_gated=True))
    assert (await reg.execute("probe_gated", {})).success
    assert calls["gated"] == 1


@pytest.mark.asyncio
async def test_named_grant_is_specific_not_global():
    # Granting one gated tool must not open a different one.
    reg, calls = _make_registry()

    async def other_exec(params):
        calls.setdefault("other", 0)
        calls["other"] += 1
        return ToolResult(tool_name="probe_gated2", success=True)

    reg.register(
        "probe_gated2", "another gated", {}, other_exec,
        safety=ToolSafety.GATED, category="test",
    )
    reg.enable_gated("probe_gated")
    assert (await reg.execute("probe_gated", {})).success is True
    assert (await reg.execute("probe_gated2", {})).success is False
    assert calls.get("other", 0) == 0


# ---------------------------------------------------------------------------
# Confirm hook (human-in-the-loop), sync and async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_hook_sync_approves_and_denies():
    for decision, expect in ((True, True), (False, False)):
        reg, calls = _make_registry(ToolPolicy(confirm=lambda spec, params: decision))
        result = await reg.execute("probe_gated", {})
        assert result.success is expect
        assert calls["gated"] == (1 if expect else 0)


@pytest.mark.asyncio
async def test_confirm_hook_async_is_awaited():
    async def approve(spec, params):
        return True

    reg, calls = _make_registry(ToolPolicy(confirm=approve))
    assert (await reg.execute("probe_gated", {})).success
    assert calls["gated"] == 1


@pytest.mark.asyncio
async def test_confirm_hook_receives_spec_and_params():
    seen = {}

    def spy(spec, params):
        seen["name"] = spec.name
        seen["params"] = params
        return True

    reg, _ = _make_registry(ToolPolicy(confirm=spy))
    await reg.execute("probe_gated", {"x": 1})
    assert seen["name"] == "probe_gated"
    assert seen["params"] == {"x": 1}


# ---------------------------------------------------------------------------
# with_enabled is immutable-style
# ---------------------------------------------------------------------------


def test_with_enabled_does_not_mutate_original():
    base = ToolPolicy()
    widened = base.with_enabled("shell")
    assert base.enabled_gated == frozenset()
    assert widened.enabled_gated == frozenset({"shell"})


# ---------------------------------------------------------------------------
# The real default registry: which tools are gated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_registry_gates_write_file_but_not_read():
    reg = create_default_registry()  # default deny-all policy
    # write_file is gated -> denied without a grant
    w = await reg.execute("write_file", {"path": "/tmp/should_not_write", "content": "x"})
    assert w.success is False
    assert "blocked by tool policy" in w.error
    # read_file stays open (missing file is a normal failure, NOT a policy block)
    r = await reg.execute("read_file", {"path": "/nonexistent/path/xyz"})
    assert "blocked by tool policy" not in (r.error or "")


@pytest.mark.asyncio
async def test_default_registry_gates_shell_and_run_code():
    reg = create_default_registry()
    for name, params in (
        ("shell", {"command": "echo hi"}),
        ("run_code", {"code": "print(1)", "language": "python"}),
        ("launch_app", {"app": "calc"}),
    ):
        result = await reg.execute(name, params)
        assert result.success is False, name
        assert "blocked by tool policy" in result.error, name


@pytest.mark.asyncio
async def test_default_registry_can_grant_via_create():
    reg = create_default_registry(policy=ToolPolicy(enabled_gated=frozenset({"shell"})))
    # shell granted; run_code still denied
    assert "blocked by tool policy" not in (
        (await reg.execute("shell", {"command": "echo hi"})).error or ""
    )
    assert (await reg.execute("run_code", {"code": "x", "language": "python"})).success is False


# ---------------------------------------------------------------------------
# environment search bypass is closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_environment_search_cannot_leak_secrets(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_API_KEY", "sk-leakme")
    monkeypatch.setenv("PATH", "/usr/bin")
    reg = create_default_registry()
    # Searching for the secret must return nothing (search is scoped to the
    # curated safe set, which never contains secrets).
    result = await reg.execute("environment", {"search": "secret"})
    assert result.success
    assert "SUPER_SECRET_API_KEY" not in result.output
    # And a broad substring that would have matched everything before.
    result2 = await reg.execute("environment", {"search": "key"})
    assert "SUPER_SECRET_API_KEY" not in result2.output
