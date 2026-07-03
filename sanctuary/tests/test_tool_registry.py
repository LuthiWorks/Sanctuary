"""Comprehensive tests for the ToolRegistry and built-in tools.

Tests edge cases, error handling, concurrent execution, and
cross-platform compatibility (Linux target deployment).
"""

import asyncio
import os
import platform
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from sanctuary.tools.registry import ToolRegistry, ToolResult, ToolSafety, ToolSpec
from sanctuary.tools.builtin import (
    create_default_registry,
    register_self_knowledge_tools,
    _read_file,
    _write_file,
    _list_directory,
    _clock,
    _system_info,
    _web_search,
    _web_fetch,
    _wikipedia,
    _run_code_docker,
    _shell_command,
    _network_scan,
    _network_reach,
    _view_dashboard,
    _view_emotional_timeline,
    _view_consciousness_trace,
    _view_attention_heatmap,
    _view_communication_patterns,
    _git_status,
    _git_log,
    _git_diff,
    _home_info,
    _list_processes,
    _launch_app,
    _environment,
    _workspace,
    _discord_send,
    _looks_like_ip,
    _find_mac_in_parts,
    _get_proxy_dict,
    ToolConfig,
    _config,
)


# ============================================================================
# ToolRegistry Core
# ============================================================================


class TestToolRegistry:
    """Test ToolRegistry registration and execution."""

    def test_empty_registry(self):
        registry = ToolRegistry()
        assert registry.tool_count == 0
        assert registry.get_catalog() == []
        assert registry.get_categories() == {}

    def test_register_tool(self):
        registry = ToolRegistry()

        async def dummy(params):
            return ToolResult(tool_name="dummy", success=True, output="ok")

        registry.register(
            name="dummy",
            description="A test tool",
            parameters={"x": "A parameter"},
            execute=dummy,
            category="test",
        )
        assert registry.tool_count == 1
        assert registry.has_tool("dummy")
        assert not registry.has_tool("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_registered_tool(self):
        registry = ToolRegistry()

        async def echo(params):
            return ToolResult(tool_name="echo", success=True, output=params.get("msg"))

        registry.register(
            name="echo", description="Echo", parameters={"msg": "Message"},
            execute=echo, category="test",
        )
        result = await registry.execute("echo", {"msg": "hello"})
        assert result.success
        assert result.output == "hello"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_that_raises(self):
        registry = ToolRegistry()

        async def bad_tool(params):
            raise ValueError("kaboom")

        registry.register(
            name="bad", description="Fails", parameters={},
            execute=bad_tool, category="test",
        )
        result = await registry.execute("bad", {})
        assert not result.success
        assert "kaboom" in result.error

    def test_catalog_sorted_by_category(self):
        registry = ToolRegistry()

        async def noop(params):
            return ToolResult(tool_name="x", success=True)

        registry.register(name="z_tool", description="Z", parameters={}, execute=noop, category="beta")
        registry.register(name="a_tool", description="A", parameters={}, execute=noop, category="alpha")
        registry.register(name="m_tool", description="M", parameters={}, execute=noop, category="alpha")

        catalog = registry.get_catalog()
        assert catalog[0]["name"] == "a_tool"
        assert catalog[1]["name"] == "m_tool"
        assert catalog[2]["name"] == "z_tool"

    def test_categories(self):
        registry = ToolRegistry()

        async def noop(params):
            return ToolResult(tool_name="x", success=True)

        registry.register(name="t1", description="", parameters={}, execute=noop, category="fs")
        registry.register(name="t2", description="", parameters={}, execute=noop, category="fs")
        registry.register(name="t3", description="", parameters={}, execute=noop, category="web")

        cats = registry.get_categories()
        assert sorted(cats["fs"]) == ["t1", "t2"]
        assert cats["web"] == ["t3"]

    @pytest.mark.asyncio
    async def test_history_tracking(self):
        registry = ToolRegistry()

        async def ok(params):
            return ToolResult(tool_name="ok", success=True)

        registry.register(name="ok", description="", parameters={}, execute=ok, category="test")

        await registry.execute("ok", {})
        await registry.execute("ok", {})
        await registry.execute("missing", {})

        stats = registry.get_stats()
        assert stats["total_executions"] == 3
        assert stats["success_rate"] == 2 / 3
        assert stats["by_tool"]["ok"] == 2
        assert stats["by_tool"]["missing"] == 1

    @pytest.mark.asyncio
    async def test_history_bounded(self):
        registry = ToolRegistry()
        registry._max_history = 5

        async def ok(params):
            return ToolResult(tool_name="ok", success=True)

        registry.register(name="ok", description="", parameters={}, execute=ok, category="test")

        for _ in range(20):
            await registry.execute("ok", {})

        assert len(registry._history) == 5

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Multiple tools should execute concurrently."""
        registry = ToolRegistry()
        call_order = []

        async def slow_tool(params):
            call_order.append(f"start_{params['id']}")
            await asyncio.sleep(0.05)
            call_order.append(f"end_{params['id']}")
            return ToolResult(tool_name="slow", success=True, output=params["id"])

        registry.register(name="slow", description="", parameters={}, execute=slow_tool, category="test")

        # Execute 3 concurrently
        tasks = [registry.execute("slow", {"id": i}) for i in range(3)]
        results = await asyncio.gather(*tasks)

        assert all(r.success for r in results)
        # All starts should happen before all ends (concurrent, not serial)
        starts = [i for i, x in enumerate(call_order) if x.startswith("start")]
        ends = [i for i, x in enumerate(call_order) if x.startswith("end")]
        assert max(starts) < max(ends)

    def test_safety_levels(self):
        registry = create_default_registry()
        catalog = registry.get_catalog()

        open_tools = [t for t in catalog if t["safety"] == "open"]
        gated_tools = [t for t in catalog if t["safety"] == "gated"]

        assert len(open_tools) >= 5  # filesystem + information tools
        assert len(gated_tools) >= 2  # run_code + shell


# ============================================================================
# Default Registry
# ============================================================================


class TestDefaultRegistry:
    """Test the default registry created by create_default_registry()."""

    def test_creates_all_tools(self):
        registry = create_default_registry()
        assert registry.tool_count == 21

        expected = [
            "read_file", "write_file", "list_directory",
            "clock", "system_info", "web_search", "web_fetch", "wikipedia",
            "run_code", "shell",
            "network_scan", "network_reach",
            "git_status", "git_log", "git_diff",
            "home_info", "list_processes", "launch_app", "environment", "workspace",
            "discord_send",
        ]
        for name in expected:
            assert registry.has_tool(name), f"Missing tool: {name}"

    def test_creates_with_proxy_config(self):
        from sanctuary.tools.builtin import ToolConfig
        config = ToolConfig(
            http_proxy="http://192.168.1.100:8080",
            https_proxy="http://192.168.1.100:8080",
            local_network="192.168.1.0/24",
        )
        registry = create_default_registry(config)
        assert registry.tool_count == 21


# ============================================================================
# Filesystem Tools
# ============================================================================


@pytest.fixture
def _grant_tmp_root(tmp_path, monkeypatch):
    """Grant the filesystem sandbox access to this test's tmp_path.

    The file tools fail closed (no roots -> no access); these tests exercise
    tool behavior within an allowed root, so we grant tmp_path.
    """
    from sanctuary.tools import builtin
    monkeypatch.setattr(builtin._config, "filesystem_roots", (str(tmp_path),))
    yield


class TestReadFile:
    """Test read_file tool."""

    @pytest.fixture(autouse=True)
    def _sandbox(self, _grant_tmp_root):
        yield

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        result = await _read_file({"path": str(f)})
        assert result.success
        assert result.output == "hello world"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        # Inside the allowed root but missing -> "not found" (not a sandbox
        # denial, which is asserted separately in test_filesystem_sandbox).
        result = await _read_file({"path": str(tmp_path / "file.txt")})
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_no_path(self):
        result = await _read_file({})
        assert not result.success
        assert "No path" in result.error

    @pytest.mark.asyncio
    async def test_read_empty_path(self):
        result = await _read_file({"path": ""})
        assert not result.success

    @pytest.mark.asyncio
    async def test_read_directory_not_file(self, tmp_path):
        result = await _read_file({"path": str(tmp_path)})
        assert not result.success
        assert "Not a file" in result.error

    @pytest.mark.asyncio
    async def test_read_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 10_000, encoding="utf-8")
        result = await _read_file({"path": str(f), "max_bytes": 100})
        assert result.success
        assert len(result.output) < 200
        assert "truncated" in result.output

    @pytest.mark.asyncio
    async def test_read_unicode(self, tmp_path):
        f = tmp_path / "unicode.txt"
        content = "Hello 世界 🌍 привет мир"
        f.write_text(content, encoding="utf-8")
        result = await _read_file({"path": str(f)})
        assert result.success
        assert result.output == content

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = await _read_file({"path": str(f)})
        assert result.success
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_read_binary_file_graceful(self, tmp_path):
        """Binary files should not crash — errors='replace' handles them."""
        f = tmp_path / "binary.bin"
        f.write_bytes(bytes(range(256)))
        result = await _read_file({"path": str(f)})
        assert result.success  # Should not crash


class TestWriteFile:
    """Test write_file tool."""

    @pytest.fixture(autouse=True)
    def _sandbox(self, _grant_tmp_root):
        yield

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        result = await _write_file({"path": str(f), "content": "hello"})
        assert result.success
        assert f.read_text(encoding="utf-8") == "hello"

    @pytest.mark.asyncio
    async def test_write_creates_parents(self, tmp_path):
        f = tmp_path / "a" / "b" / "c" / "deep.txt"
        result = await _write_file({"path": str(f), "content": "deep"})
        assert result.success
        assert f.read_text(encoding="utf-8") == "deep"

    @pytest.mark.asyncio
    async def test_write_overwrite(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("old", encoding="utf-8")
        result = await _write_file({"path": str(f), "content": "new"})
        assert result.success
        assert f.read_text(encoding="utf-8") == "new"

    @pytest.mark.asyncio
    async def test_write_no_path(self):
        result = await _write_file({"content": "hello"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_write_unicode(self, tmp_path):
        f = tmp_path / "unicode.txt"
        content = "日本語テスト 🎌"
        result = await _write_file({"path": str(f), "content": content})
        assert result.success
        assert f.read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_write_empty_content(self, tmp_path):
        f = tmp_path / "empty.txt"
        result = await _write_file({"path": str(f), "content": ""})
        assert result.success
        assert f.read_text(encoding="utf-8") == ""


class TestListDirectory:
    """Test list_directory tool."""

    @pytest.fixture(autouse=True)
    def _sandbox(self, _grant_tmp_root):
        yield

    @pytest.mark.asyncio
    async def test_list_with_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("bb")
        (tmp_path / "subdir").mkdir()
        result = await _list_directory({"path": str(tmp_path)})
        assert result.success
        names = [e["name"] for e in result.output]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names
        # Check types
        types = {e["name"]: e["type"] for e in result.output}
        assert types["a.txt"] == "file"
        assert types["subdir"] == "dir"

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = await _list_directory({"path": str(d)})
        assert result.success
        assert result.output == []

    @pytest.mark.asyncio
    async def test_list_nonexistent(self, tmp_path):
        result = await _list_directory({"path": str(tmp_path / "nonexistent")})
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_file_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = await _list_directory({"path": str(f)})
        assert not result.success
        assert "Not a directory" in result.error


# ============================================================================
# Information Tools
# ============================================================================


class TestClock:
    """Test clock tool."""

    @pytest.mark.asyncio
    async def test_clock_returns_time(self):
        result = await _clock({})
        assert result.success
        assert "local" in result.output
        assert "utc" in result.output
        assert "date" in result.output
        assert "time" in result.output


class TestSystemInfo:
    """Test system_info tool."""

    @pytest.mark.asyncio
    async def test_system_info_basic(self):
        result = await _system_info({})
        assert result.success
        assert "platform" in result.output
        assert "python" in result.output
        assert "cpu_count" in result.output
        assert result.output["cpu_count"] > 0


# ============================================================================
# Code Execution (Docker)
# ============================================================================


class TestRunCode:
    """Test Docker code execution tool."""

    @pytest.mark.asyncio
    async def test_no_code_provided(self):
        result = await _run_code_docker({"language": "python"})
        assert not result.success
        assert "No code" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        result = await _run_code_docker({"code": "print('hi')", "language": "cobol"})
        assert not result.success
        assert "Unsupported language" in result.error

    @pytest.mark.asyncio
    async def test_rust_not_direct(self):
        result = await _run_code_docker({"code": "fn main() {}", "language": "rust"})
        assert not result.success
        assert "not supported" in result.error.lower()


# ============================================================================
# Shell Tool
# ============================================================================


class TestShell:
    """Test shell command tool."""

    @pytest.mark.asyncio
    async def test_no_command(self):
        result = await _shell_command({})
        assert not result.success
        assert "No command" in result.error

    @pytest.mark.asyncio
    async def test_simple_command(self):
        # echo / exit are shell builtins, so exercise the explicit shell path.
        result = await _shell_command({"command": "echo hello", "use_shell": True})
        assert result.success
        assert "hello" in result.output["stdout"]

    @pytest.mark.asyncio
    async def test_failing_command(self):
        result = await _shell_command({"command": "exit 1", "use_shell": True})
        assert not result.success
        assert result.output["returncode"] == 1


# ============================================================================
# Web Tools (network-dependent — graceful failure)
# ============================================================================


class TestWebSearch:
    """Test web_search tool — graceful if no network."""

    @pytest.mark.asyncio
    async def test_no_query(self):
        result = await _web_search({})
        assert not result.success
        assert "No query" in result.error


class TestWebFetch:
    """Test web_fetch tool — graceful if no network."""

    @pytest.mark.asyncio
    async def test_no_url(self):
        result = await _web_fetch({})
        assert not result.success
        assert "No URL" in result.error


class TestWikipedia:
    """Test wikipedia tool — graceful if package missing."""

    @pytest.mark.asyncio
    async def test_no_topic(self):
        result = await _wikipedia({})
        assert not result.success
        assert "No topic" in result.error


# ============================================================================
# Cross-platform path handling
# ============================================================================


class TestCrossPlatform:
    """Ensure tools work with both forward and backslash paths."""

    @pytest.fixture(autouse=True)
    def _sandbox(self, _grant_tmp_root):
        yield

    @pytest.mark.asyncio
    async def test_forward_slash_paths(self, tmp_path):
        """Forward slashes should work on all platforms."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        # Use forward slashes explicitly
        forward_path = str(f).replace("\\", "/")
        result = await _read_file({"path": forward_path})
        assert result.success
        assert result.output == "content"

    @pytest.mark.asyncio
    async def test_pathlib_compatibility(self, tmp_path):
        """Path objects should work when stringified."""
        f = tmp_path / "sub" / "test.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("nested")
        result = await _read_file({"path": str(f)})
        assert result.success

    @pytest.mark.asyncio
    async def test_write_with_forward_slashes(self, tmp_path):
        forward_path = str(tmp_path / "output.txt").replace("\\", "/")
        result = await _write_file({"path": forward_path, "content": "works"})
        assert result.success


# ============================================================================
# Network Tools
# ============================================================================


class TestNetworkScan:
    """Test network_scan tool."""

    @pytest.mark.asyncio
    async def test_scan_returns_devices(self):
        result = await _network_scan({})
        assert result.success
        assert "devices" in result.output
        assert "count" in result.output
        assert isinstance(result.output["devices"], list)


class TestNetworkReach:
    """Test network_reach tool."""

    @pytest.mark.asyncio
    async def test_reach_no_host(self):
        result = await _network_reach({})
        assert not result.success
        assert "No host" in result.error

    @pytest.mark.asyncio
    async def test_reach_localhost(self):
        result = await _network_reach({"host": "127.0.0.1"})
        assert result.success
        assert result.output["reachable"] is True

    @pytest.mark.asyncio
    async def test_reach_unreachable(self):
        """Unreachable host should return reachable=False, not error."""
        result = await _network_reach({"host": "192.0.2.1"})  # RFC 5737 TEST-NET
        assert result.success  # Tool itself succeeds
        assert result.output["reachable"] is False


# ============================================================================
# Proxy and Network Helpers
# ============================================================================


class TestProxyHelpers:
    """Test proxy configuration helpers."""

    def test_looks_like_ip_valid(self):
        assert _looks_like_ip("192.168.1.1")
        assert _looks_like_ip("10.0.0.1")
        assert _looks_like_ip("0.0.0.0")
        assert _looks_like_ip("255.255.255.255")

    def test_looks_like_ip_invalid(self):
        assert not _looks_like_ip("not_an_ip")
        assert not _looks_like_ip("256.1.1.1")
        assert not _looks_like_ip("1.2.3")
        assert not _looks_like_ip("1.2.3.4.5")
        assert not _looks_like_ip("")

    def test_looks_like_ip_with_parens(self):
        """Windows ARP output wraps IPs in parentheses."""
        assert _looks_like_ip("(192.168.1.1)")

    def test_find_mac_colon_format(self):
        assert _find_mac_in_parts(["00:1a:2b:3c:4d:5e"]) == "00:1a:2b:3c:4d:5e"

    def test_find_mac_dash_format(self):
        assert _find_mac_in_parts(["00-1A-2B-3C-4D-5E"]) == "00:1a:2b:3c:4d:5e"

    def test_find_mac_none(self):
        assert _find_mac_in_parts(["not", "a", "mac"]) is None

    def test_get_proxy_dict_none(self):
        """No proxy configured returns None."""
        import sanctuary.tools.builtin as mod
        old = mod._config
        mod._config = ToolConfig()
        assert _get_proxy_dict() is None
        mod._config = old

    def test_get_proxy_dict_configured(self):
        import sanctuary.tools.builtin as mod
        old = mod._config
        mod._config = ToolConfig(
            http_proxy="http://proxy:8080",
            https_proxy="http://proxy:8443",
        )
        result = _get_proxy_dict()
        assert result == {"http://": "http://proxy:8080", "https://": "http://proxy:8443"}
        mod._config = old


# ============================================================================
# Self-Knowledge Tools
# ============================================================================


class TestSelfKnowledgeToolsWithoutMonitoring:
    """Test self-knowledge tools when monitoring is not available.

    The self-knowledge tools read module-level singleton refs in
    ``sanctuary.tools.builtin`` that get populated by
    ``register_self_knowledge_tools()`` whenever a SanctuaryRunner is
    instantiated. Other test files (test_ws_server, test_health_server,
    test_world) instantiate the runner and leave those globals populated,
    so we explicitly clear them before each test in this class —
    otherwise the "unavailable" path never executes.
    """

    @pytest.fixture(autouse=True)
    def reset_module_refs(self):
        import sanctuary.tools.builtin as mod
        old = (
            mod._dashboard_ref,
            mod._consciousness_trace_ref,
            mod._attention_tracker_ref,
            mod._communication_log_ref,
        )
        mod._dashboard_ref = None
        mod._consciousness_trace_ref = None
        mod._attention_tracker_ref = None
        mod._communication_log_ref = None
        yield
        (
            mod._dashboard_ref,
            mod._consciousness_trace_ref,
            mod._attention_tracker_ref,
            mod._communication_log_ref,
        ) = old

    @pytest.mark.asyncio
    async def test_dashboard_unavailable(self):
        result = await _view_dashboard({})
        assert not result.success
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_timeline_unavailable(self):
        result = await _view_emotional_timeline({})
        assert not result.success

    @pytest.mark.asyncio
    async def test_trace_unavailable(self):
        result = await _view_consciousness_trace({})
        assert not result.success

    @pytest.mark.asyncio
    async def test_heatmap_unavailable(self):
        result = await _view_attention_heatmap({})
        assert not result.success

    @pytest.mark.asyncio
    async def test_communication_unavailable(self):
        result = await _view_communication_patterns({})
        assert not result.success


class TestSelfKnowledgeToolsWithMonitoring:
    """Test self-knowledge tools with real monitoring objects."""

    @pytest.fixture(autouse=True)
    def setup_monitoring(self):
        from sanctuary.monitoring import (
            DashboardDataProvider,
            ConsciousnessTraceRecorder,
            AttentionHeatmapTracker,
            CommunicationDecisionLogger,
        )
        import sanctuary.tools.builtin as mod
        # Save originals
        old_dash = mod._dashboard_ref
        old_trace = mod._consciousness_trace_ref
        old_attn = mod._attention_tracker_ref
        old_comm = mod._communication_log_ref

        # Inject real monitoring objects
        mod._dashboard_ref = DashboardDataProvider()
        mod._consciousness_trace_ref = ConsciousnessTraceRecorder()
        mod._attention_tracker_ref = AttentionHeatmapTracker()
        mod._communication_log_ref = CommunicationDecisionLogger()

        # Record some data
        mod._dashboard_ref.record_snapshot(
            cycle=1, inner_speech="thinking", valence=0.3, arousal=0.5,
        )
        mod._consciousness_trace_ref.record(
            cycle=1, inner_speech="thinking", latency_ms=1.5,
        )
        mod._attention_tracker_ref.record(target="hello", category="language", salience=0.8, cycle=1)

        yield

        # Restore
        mod._dashboard_ref = old_dash
        mod._consciousness_trace_ref = old_trace
        mod._attention_tracker_ref = old_attn
        mod._communication_log_ref = old_comm

    @pytest.mark.asyncio
    async def test_dashboard_with_data(self):
        result = await _view_dashboard({})
        assert result.success
        assert result.output["cycle"] == 1
        assert result.output["valence"] == 0.3

    @pytest.mark.asyncio
    async def test_emotional_timeline_with_data(self):
        result = await _view_emotional_timeline({"cycles": 5})
        assert result.success
        assert isinstance(result.output, list)

    @pytest.mark.asyncio
    async def test_consciousness_trace_with_data(self):
        result = await _view_consciousness_trace({"cycles": 3})
        assert result.success
        assert len(result.output) >= 1
        assert result.output[0]["cycle"] == 1

    @pytest.mark.asyncio
    async def test_attention_with_data(self):
        result = await _view_attention_heatmap({})
        assert result.success
        assert result.output["total_events"] >= 1

    @pytest.mark.asyncio
    async def test_communication_with_data(self):
        result = await _view_communication_patterns({})
        assert result.success
        assert "total_entries" in result.output

    @pytest.mark.asyncio
    async def test_dashboard_no_data(self):
        """Dashboard with no snapshots returns empty status."""
        import sanctuary.tools.builtin as mod
        from sanctuary.monitoring import DashboardDataProvider
        mod._dashboard_ref = DashboardDataProvider()  # Fresh, empty
        result = await _view_dashboard({})
        assert result.success
        assert result.output["status"] == "No data yet"


# ============================================================================
# Git Tools
# ============================================================================


# Resolve the Sanctuary repo root once for the git-tool tests below.
# Path-relative discovery so the tests work on any developer's checkout
# (Linux, Docker, CI), not just the original author's machine.
_SANCTUARY_REPO_ROOT = str(Path(__file__).resolve().parents[2])


class TestGitStatus:
    """Test git_status tool."""

    @pytest.mark.asyncio
    async def test_status_in_repo(self):
        """Should work in Sanctuary's own repo.

        Asserts the tool produces git's short-status header (a line
        starting with ``##``) — not a specific branch name. Branch
        names vary by environment: ``main`` on the canonical repo,
        a feature branch during development, ``HEAD (no branch)`` on
        GitHub Actions PR checkouts (detached HEAD).
        """
        result = await _git_status({"repo": _SANCTUARY_REPO_ROOT})
        assert result.success
        assert "##" in result.output

    @pytest.mark.asyncio
    async def test_status_not_a_repo(self, tmp_path):
        result = await _git_status({"repo": str(tmp_path)})
        assert not result.success


class TestGitLog:
    """Test git_log tool."""

    @pytest.mark.asyncio
    async def test_log_with_count(self):
        result = await _git_log({
            "repo": _SANCTUARY_REPO_ROOT,
            "count": 3,
        })
        assert result.success
        lines = result.output.strip().split("\n")
        assert len(lines) <= 3


class TestGitDiff:
    """Test git_diff tool."""

    @pytest.mark.asyncio
    async def test_diff_summary(self):
        result = await _git_diff({
            "repo": _SANCTUARY_REPO_ROOT,
        })
        assert result.success  # May be empty if no changes


# ============================================================================
# Home Interaction Tools
# ============================================================================


class TestHomeInfo:
    """Test home_info tool."""

    @pytest.mark.asyncio
    async def test_returns_info(self):
        result = await _home_info({})
        assert result.success
        assert "home_directory" in result.output
        assert "hostname" in result.output
        assert "os" in result.output


class TestListProcesses:
    """Test list_processes tool."""

    @pytest.mark.asyncio
    async def test_returns_processes(self):
        result = await _list_processes({})
        assert result.success
        # Should have at least some processes
        if isinstance(result.output, dict):
            assert result.output["count"] > 0


class TestLaunchApp:
    """Test launch_app tool."""

    @pytest.mark.asyncio
    async def test_no_app(self):
        result = await _launch_app({})
        assert not result.success
        assert "No app" in result.error

    @pytest.mark.asyncio
    async def test_nonexistent_app(self):
        result = await _launch_app({"app": "nonexistent_app_xyz_12345"})
        assert not result.success
        assert "not found" in result.error.lower()


class TestEnvironment:
    """Test environment tool."""

    @pytest.mark.asyncio
    async def test_default_returns_safe_vars(self):
        result = await _environment({})
        assert result.success
        assert "total_env_vars" in result.output

    @pytest.mark.asyncio
    async def test_search_filter(self):
        result = await _environment({"search": "PATH"})
        assert result.success
        # Should find at least PATH
        assert len(result.output) >= 1


class TestWorkspace:
    """Test workspace tool."""

    @pytest.mark.asyncio
    async def test_init_workspace(self, tmp_path):
        import sanctuary.tools.builtin as mod
        old_home = mod._home_dir
        mod._home_dir = str(tmp_path)

        result = await _workspace({"action": "init"})
        assert result.success
        assert "initialized" in result.output["status"]
        # Verify directories created
        ws = tmp_path / "sanctuary_workspace"
        assert (ws / "journal").is_dir()
        assert (ws / "projects").is_dir()
        assert (ws / "experiments").is_dir()
        assert (ws / "notes").is_dir()

        mod._home_dir = old_home

    @pytest.mark.asyncio
    async def test_workspace_info(self, tmp_path):
        import sanctuary.tools.builtin as mod
        old_home = mod._home_dir
        mod._home_dir = str(tmp_path)

        # Init first
        await _workspace({"action": "init"})
        result = await _workspace({"action": "info"})
        assert result.success
        assert "contents" in result.output

        mod._home_dir = old_home

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path):
        import sanctuary.tools.builtin as mod
        old_home = mod._home_dir
        mod._home_dir = str(tmp_path)
        await _workspace({"action": "init"})

        result = await _workspace({"action": "destroy"})
        assert not result.success
        assert "Unknown action" in result.error

        mod._home_dir = old_home


# ============================================================================
# Discord Tools
# ============================================================================


class TestDiscordSend:
    """Test discord_send tool."""

    @pytest.mark.asyncio
    async def test_no_message(self):
        result = await _discord_send({})
        assert not result.success
        assert "No message" in result.error

    @pytest.mark.asyncio
    async def test_no_webhook(self):
        result = await _discord_send({"message": "hello"})
        assert not result.success
        assert "webhook" in result.error.lower()


class TestSelfKnowledgeRegistration:
    """Test the register_self_knowledge_tools function."""

    def test_registers_all_tools(self):
        from sanctuary.monitoring import (
            DashboardDataProvider,
            ConsciousnessTraceRecorder,
            AttentionHeatmapTracker,
            CommunicationDecisionLogger,
        )
        registry = ToolRegistry()
        register_self_knowledge_tools(
            registry,
            dashboard=DashboardDataProvider(),
            consciousness_trace=ConsciousnessTraceRecorder(),
            attention_tracker=AttentionHeatmapTracker(),
            communication_log=CommunicationDecisionLogger(),
        )
        assert registry.has_tool("view_dashboard")
        assert registry.has_tool("view_emotional_timeline")
        assert registry.has_tool("view_consciousness_trace")
        assert registry.has_tool("view_attention_heatmap")
        assert registry.has_tool("view_communication_patterns")
        assert registry.tool_count == 5


# ============================================================================
# Integration with CognitiveOutput schema
# ============================================================================


class TestToolRequestSchema:
    """Test that ToolRequest works in CognitiveOutput."""

    def test_tool_request_in_output(self):
        from sanctuary.core.schema import CognitiveOutput, ToolRequest

        output = CognitiveOutput(
            inner_speech="I want to check the time",
            tool_requests=[
                ToolRequest(tool_name="clock", parameters={}),
                ToolRequest(tool_name="read_file", parameters={"path": "/some/file"}),
            ],
        )
        assert len(output.tool_requests) == 2
        assert output.tool_requests[0].tool_name == "clock"
        assert output.tool_requests[1].parameters == {"path": "/some/file"}

    def test_empty_tool_requests(self):
        from sanctuary.core.schema import CognitiveOutput

        output = CognitiveOutput(inner_speech="nothing to do")
        assert output.tool_requests == []

    def test_tool_request_serialization(self):
        from sanctuary.core.schema import ToolRequest

        req = ToolRequest(tool_name="web_search", parameters={"query": "IWMT paper"})
        data = req.model_dump()
        assert data["tool_name"] == "web_search"
        assert data["parameters"]["query"] == "IWMT paper"

        # Round-trip
        req2 = ToolRequest.model_validate(data)
        assert req2.tool_name == req.tool_name
        assert req2.parameters == req.parameters
