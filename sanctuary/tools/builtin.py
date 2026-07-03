"""Built-in tools for the entity.

These are registered automatically when the ToolRegistry is created
via create_default_registry(). Each tool is a simple async function
that takes a params dict and returns a ToolResult.

Categories:
- filesystem: Read/write files, list directories
- information: Web search, web fetch, Wikipedia, clock, system info
- monitoring: Own consciousness traces, dashboard, attention data
- communication: Discord messaging (future)
- code: Docker sandbox execution (gated)
- system: Shell commands, app management (gated)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import shlex
import stat as stat_module
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sanctuary.tools.registry import (
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSafety,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """Configuration for built-in tools.

    Proxy settings route web traffic through a gateway device on the
    local network, protecting the entity's machine from direct internet
    exposure.
    """

    # HTTP proxy for all web requests (e.g., "http://192.168.1.100:8080")
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None

    # Trusted local network CIDR (e.g., "192.168.1.0/24")
    local_network: Optional[str] = None

    # Filesystem sandbox: the roots the file tools (read_file, write_file,
    # list_directory) may touch. FAIL CLOSED -- an empty tuple denies all
    # filesystem access. Any accessed path must resolve (after following
    # symlinks and `..`) to inside one of these roots. Grant access by
    # listing the entity's workspace, its source tree, etc.
    filesystem_roots: tuple[str, ...] = ()


# Module-level config — set by create_default_registry()
_config = ToolConfig()


class _SandboxError(Exception):
    """A filesystem path escaped (or was denied by) the sandbox roots."""


def _within_roots(target: Path) -> bool:
    """True if `target` (already resolved) lies within a configured root."""
    for root in _config.filesystem_roots:
        root_resolved = Path(root).resolve()
        if target == root_resolved or target.is_relative_to(root_resolved):
            return True
    return False


def _resolve_in_sandbox(path_str: str) -> Path:
    """Canonicalize `path_str` and require it to lie within an allowed root.

    Resolves symlinks and ``..`` first (``Path.resolve``), so neither a
    ``../../etc`` traversal nor a symlink inside a root that points outside it
    can escape at check time. Fail closed: with no roots configured, every
    path is denied. Raises :class:`_SandboxError` on any violation.

    This is the *check*. It leaves a TOCTOU window between resolve and the
    actual open (a component swapped for a symlink in between). That window is
    closed separately by :func:`_verify_fd_in_roots`, which re-checks the
    OPENED handle's own real path -- see the tool bodies.
    """
    if not _config.filesystem_roots:
        raise _SandboxError(
            "filesystem access is not configured (no allowed roots). Grant "
            "access via RunnerConfig.filesystem_roots."
        )
    target = Path(path_str).resolve()
    if not _within_roots(target):
        raise _SandboxError(f"path is outside the allowed filesystem roots: {path_str}")
    return target


def _final_path_of_fd(fd: int) -> Optional[Path]:
    """The real filesystem path an OPEN descriptor points at, or None.

    This is ground truth: it reflects what the kernel actually opened, so it
    is immune to a symlink swapped in after the pre-open resolve check. None
    means the platform primitive is unavailable (caller falls back to the
    resolve-time check).
    """
    try:
        if sys.platform == "win32":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            handle = msvcrt.get_osfhandle(fd)
            fn = ctypes.windll.kernel32.GetFinalPathNameByHandleW
            fn.argtypes = [wintypes.HANDLE, wintypes.LPWSTR,
                           wintypes.DWORD, wintypes.DWORD]
            fn.restype = wintypes.DWORD
            need = fn(handle, None, 0, 0)  # required length incl. NUL
            if need == 0:
                return None
            buf = ctypes.create_unicode_buffer(need)
            got = fn(handle, buf, need, 0)
            if got == 0:
                return None
            p = buf.value
            # Strip the \\?\ (and \\?\UNC\) prefixes GetFinalPathName returns.
            if p.startswith("\\\\?\\UNC\\"):
                p = "\\\\" + p[len("\\\\?\\UNC\\"):]
            elif p.startswith("\\\\?\\"):
                p = p[len("\\\\?\\"):]
            return Path(p)
        if sys.platform.startswith("linux"):
            return Path(os.readlink(f"/proc/self/fd/{fd}"))
        # macOS / BSD: F_GETPATH fills a PATH_MAX buffer with the path.
        import fcntl
        f_getpath = getattr(fcntl, "F_GETPATH", None)
        if f_getpath is None:
            return None
        buf = bytearray(4096)
        fcntl.fcntl(fd, f_getpath, buf)
        return Path(os.fsdecode(bytes(buf).split(b"\x00", 1)[0]))
    except (OSError, ValueError, ImportError):
        return None


def _verify_fd_in_roots(fd: int, path_str: str) -> None:
    """Close the TOCTOU window: re-check the OPENED handle's real path.

    If the handle the kernel actually opened resolves outside every root, a
    symlink was swapped in after the pre-open check -- deny. If the platform
    primitive is unavailable (``None``), we keep the pre-open guarantee from
    :func:`_resolve_in_sandbox` and do not weaken it. Raises _SandboxError on
    a confirmed escape.
    """
    real = _final_path_of_fd(fd)
    if real is None:
        return  # platform fallback: pre-open resolve check stands
    if not _within_roots(real.resolve()):
        raise _SandboxError(
            f"path escaped the allowed filesystem roots after opening: {path_str}"
        )


# ---------------------------------------------------------------------------
# Filesystem tools
# ---------------------------------------------------------------------------


async def _read_file(params: dict[str, Any]) -> ToolResult:
    """Read a file's contents."""
    path = params.get("path", "")
    if not path:
        return ToolResult(tool_name="read_file", success=False, error="No path provided")

    try:
        p = _resolve_in_sandbox(path)
    except _SandboxError as e:
        return ToolResult(tool_name="read_file", success=False, error=str(e))

    if not p.exists():
        return ToolResult(tool_name="read_file", success=False, error=f"File not found: {path}")
    if p.is_dir():
        return ToolResult(tool_name="read_file", success=False, error=f"Not a file: {path}")

    max_bytes = params.get("max_bytes", 100_000)
    try:
        # Open once, then verify the OPENED handle is in-root (closes the
        # resolve->open window), fstat it for type, and read from the SAME
        # handle -- no second path lookup to race.
        fd = os.open(str(p), os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError as e:
        return ToolResult(tool_name="read_file", success=False, error=str(e))
    try:
        try:
            _verify_fd_in_roots(fd, path)
        except _SandboxError as e:
            return ToolResult(tool_name="read_file", success=False, error=str(e))
        if not stat_module.S_ISREG(os.fstat(fd).st_mode):
            return ToolResult(tool_name="read_file", success=False, error=f"Not a file: {path}")
        raw = os.read(fd, max_bytes + 1)
    finally:
        os.close(fd)

    truncated = len(raw) > max_bytes
    content = raw[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        content += f"\n... [truncated at {max_bytes} bytes]"
    return ToolResult(tool_name="read_file", success=True, output=content)


async def _write_file(params: dict[str, Any]) -> ToolResult:
    """Write content to a file."""
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return ToolResult(tool_name="write_file", success=False, error="No path provided")

    try:
        p = _resolve_in_sandbox(path)
    except _SandboxError as e:
        return ToolResult(tool_name="write_file", success=False, error=str(e))

    data = content.encode("utf-8")
    pre_existed = p.exists()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Open WITHOUT truncating, verify the opened handle is in-root, and
        # only then truncate + write. Order matters: if a symlink was swapped
        # in to point at an out-of-root file, we must detect it BEFORE
        # truncating, so we never destroy an escapee's contents. A newly
        # created escapee is removed.
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            _verify_fd_in_roots(fd, path)
            os.ftruncate(fd, 0)
            os.write(fd, data)
        except _SandboxError as e:
            os.close(fd)
            fd = None
            if not pre_existed:
                try:
                    os.unlink(str(p))
                except OSError:
                    pass
            return ToolResult(tool_name="write_file", success=False, error=str(e))
        finally:
            if fd is not None:
                os.close(fd)
        return ToolResult(
            tool_name="write_file", success=True,
            output=f"Written {len(content)} bytes to {path}",
        )
    except Exception as e:
        return ToolResult(tool_name="write_file", success=False, error=str(e))


async def _list_directory(params: dict[str, Any]) -> ToolResult:
    """List contents of a directory."""
    path = params.get("path", ".")
    try:
        p = _resolve_in_sandbox(path)
    except _SandboxError as e:
        return ToolResult(tool_name="list_directory", success=False, error=str(e))

    if not p.exists():
        return ToolResult(tool_name="list_directory", success=False, error=f"Not found: {path}")
    if not p.is_dir():
        return ToolResult(tool_name="list_directory", success=False, error=f"Not a directory: {path}")

    try:
        entries = []
        for item in sorted(p.iterdir()):
            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            entries.append({"name": item.name, "type": kind, "size": size})
        return ToolResult(tool_name="list_directory", success=True, output=entries)
    except Exception as e:
        return ToolResult(tool_name="list_directory", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Information tools
# ---------------------------------------------------------------------------


async def _clock(params: dict[str, Any]) -> ToolResult:
    """Get current date, time, and timezone."""
    now = datetime.datetime.now()
    utc = datetime.datetime.now(datetime.timezone.utc)
    return ToolResult(
        tool_name="clock", success=True,
        output={
            "local": now.isoformat(),
            "utc": utc.isoformat(),
            "date": now.strftime("%A, %B %d, %Y"),
            "time": now.strftime("%I:%M %p"),
            "timezone": str(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo),
        },
    )


async def _system_info(params: dict[str, Any]) -> ToolResult:
    """Get system hardware and software information."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }

    # Memory info via psutil if available
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
        info["ram_percent"] = mem.percent
        info["disk_usage"] = {}
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                info["disk_usage"][part.mountpoint] = {
                    "total_gb": round(usage.total / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                    "percent": usage.percent,
                }
            except PermissionError:
                pass
    except ImportError:
        info["note"] = "Install psutil for detailed hardware info"

    return ToolResult(tool_name="system_info", success=True, output=info)


async def _web_search(params: dict[str, Any]) -> ToolResult:
    """Search the web using DuckDuckGo (free, no API key).

    Respects proxy settings for secure browsing through gateway.
    """
    query = params.get("query", "")
    if not query:
        return ToolResult(tool_name="web_search", success=False, error="No query provided")

    max_results = params.get("max_results", 5)

    try:
        from duckduckgo_search import DDGS
        proxies = _get_proxy_dict()
        with DDGS(proxies=proxies) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return ToolResult(
            tool_name="web_search", success=True,
            output=[{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results],
        )
    except ImportError:
        return ToolResult(
            tool_name="web_search", success=False,
            error="duckduckgo-search not installed. Run: pip install duckduckgo-search",
        )
    except Exception as e:
        return ToolResult(tool_name="web_search", success=False, error=str(e))


async def _web_fetch(params: dict[str, Any]) -> ToolResult:
    """Fetch content from a URL.

    Respects proxy settings for secure browsing through gateway.
    """
    url = params.get("url", "")
    if not url:
        return ToolResult(tool_name="web_fetch", success=False, error="No URL provided")

    max_bytes = params.get("max_bytes", 50_000)

    try:
        import httpx
        proxy_url = _config.https_proxy or _config.http_proxy
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=30.0, proxy=proxy_url,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.text[:max_bytes]
            return ToolResult(
                tool_name="web_fetch", success=True,
                output={"url": str(resp.url), "status": resp.status_code, "content": content},
            )
    except ImportError:
        return ToolResult(
            tool_name="web_fetch", success=False,
            error="httpx not installed. Run: pip install httpx",
        )
    except Exception as e:
        return ToolResult(tool_name="web_fetch", success=False, error=str(e))


async def _wikipedia(params: dict[str, Any]) -> ToolResult:
    """Look up a topic on Wikipedia."""
    topic = params.get("topic", "")
    if not topic:
        return ToolResult(tool_name="wikipedia", success=False, error="No topic provided")

    try:
        import wikipedia
        summary = wikipedia.summary(topic, sentences=params.get("sentences", 5))
        page = wikipedia.page(topic)
        return ToolResult(
            tool_name="wikipedia", success=True,
            output={"title": page.title, "summary": summary, "url": page.url},
        )
    except ImportError:
        return ToolResult(
            tool_name="wikipedia", success=False,
            error="wikipedia not installed. Run: pip install wikipedia",
        )
    except Exception as e:
        return ToolResult(tool_name="wikipedia", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Code execution (gated)
# ---------------------------------------------------------------------------


async def _run_code_docker(params: dict[str, Any]) -> ToolResult:
    """Execute code in an isolated Docker container."""
    code = params.get("code", "")
    language = params.get("language", "python")
    timeout = params.get("timeout", 30)

    if not code:
        return ToolResult(tool_name="run_code", success=False, error="No code provided")

    image_map = {
        "python": "python:3.11-slim",
        "node": "node:20-slim",
        "javascript": "node:20-slim",
        "rust": "rust:slim",
    }
    image = image_map.get(language)
    if not image:
        return ToolResult(
            tool_name="run_code", success=False,
            error=f"Unsupported language: {language}. Supported: {list(image_map.keys())}",
        )

    cmd_map = {
        "python": ["python", "-c", code],
        "node": ["node", "-e", code],
        "javascript": ["node", "-e", code],
        "rust": None,  # Would need compile step
    }
    cmd = cmd_map.get(language)
    if cmd is None:
        return ToolResult(
            tool_name="run_code", success=False,
            error=f"Direct execution not supported for {language} yet",
        )

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network=none",   # No network access from sandbox
                "--memory=256m",    # Memory limit
                "--cpus=1",         # CPU limit
                "--pids-limit=256",  # Cap processes -- defeats fork bombs
                "--cap-drop=ALL",   # Drop all Linux capabilities
                "--security-opt=no-new-privileges",  # No setuid escalation
                image,
            ] + cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ToolResult(
            tool_name="run_code", success=result.returncode == 0,
            output={
                "stdout": result.stdout[:10_000],
                "stderr": result.stderr[:5_000],
                "returncode": result.returncode,
            },
            error=result.stderr[:500] if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(tool_name="run_code", success=False, error=f"Timeout after {timeout}s")
    except FileNotFoundError:
        return ToolResult(tool_name="run_code", success=False, error="Docker not found")
    except Exception as e:
        return ToolResult(tool_name="run_code", success=False, error=str(e))


# ---------------------------------------------------------------------------
# System tools (gated)
# ---------------------------------------------------------------------------


async def _shell_command(params: dict[str, Any]) -> ToolResult:
    """Run a program (safe by default) or a full shell command (opt-in).

    Safe mode (default): the command is tokenized into argv and run WITHOUT a
    shell, so shell metacharacters are inert -- ``foo; rm -rf ~``, ``a | b``,
    ``$(...)`` and ``&& other`` do not chain, pipe, or substitute; the extra
    tokens are passed as literal arguments. Tokenizing is POSIX-style, so on
    Windows prefer forward slashes in paths (or set ``use_shell``).

    Shell mode (``use_shell: true``): the original ``shell=True`` behavior --
    full shell interpretation. Deliberately opt-in; combined with the tool's
    gated status this keeps arbitrary-shell power available but never the
    silent default.
    """
    command = params.get("command", "")
    if not command:
        return ToolResult(tool_name="shell", success=False, error="No command provided")

    timeout = params.get("timeout", 30)
    use_shell = bool(params.get("use_shell", False))

    if use_shell:
        popen_args: Any = command
    else:
        try:
            popen_args = shlex.split(command)
        except ValueError as e:
            return ToolResult(
                tool_name="shell", success=False,
                error=f"could not parse command (unbalanced quotes?): {e}",
            )
        if not popen_args:
            return ToolResult(tool_name="shell", success=False, error="No command provided")

    try:
        result = subprocess.run(
            popen_args,
            shell=use_shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ToolResult(
            tool_name="shell", success=result.returncode == 0,
            output={
                "stdout": result.stdout[:10_000],
                "stderr": result.stderr[:5_000],
                "returncode": result.returncode,
            },
            error=result.stderr[:500] if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(tool_name="shell", success=False, error=f"Timeout after {timeout}s")
    except Exception as e:
        return ToolResult(tool_name="shell", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Self-knowledge tools (monitoring data access)
# ---------------------------------------------------------------------------

# These hold references to the runner's monitoring subsystems,
# injected by register_self_knowledge_tools() after runner assembly.
_dashboard_ref = None
_consciousness_trace_ref = None
_attention_tracker_ref = None
_communication_log_ref = None


async def _view_dashboard(params: dict[str, Any]) -> ToolResult:
    """View the latest dashboard snapshot — the entity's current state summary."""
    if _dashboard_ref is None:
        return ToolResult(tool_name="view_dashboard", success=False, error="Dashboard not available")

    latest = _dashboard_ref.get_latest()
    if latest is None:
        return ToolResult(tool_name="view_dashboard", success=True, output={"status": "No data yet"})

    return ToolResult(
        tool_name="view_dashboard", success=True,
        output={
            "cycle": latest.cycle,
            "inner_speech": latest.inner_speech_summary[:200],
            "external_speech": latest.external_speech,
            "latency_ms": round(latest.cycle_latency_ms, 2),
            "valence": round(latest.valence, 3),
            "arousal": round(latest.arousal, 3),
            "dominance": round(latest.dominance, 3),
            "felt_quality": latest.felt_quality,
        },
    )


async def _view_emotional_timeline(params: dict[str, Any]) -> ToolResult:
    """View emotional state over recent cycles — how feelings have changed."""
    if _dashboard_ref is None:
        return ToolResult(tool_name="view_emotional_timeline", success=False, error="Dashboard not available")

    n = params.get("cycles", 20)
    timeline = _dashboard_ref.get_emotional_timeline(n=n)
    return ToolResult(tool_name="view_emotional_timeline", success=True, output=timeline)


async def _view_consciousness_trace(params: dict[str, Any]) -> ToolResult:
    """View full consciousness traces — complete state replay of recent cycles."""
    if _consciousness_trace_ref is None:
        return ToolResult(tool_name="view_consciousness_trace", success=False, error="Trace recorder not available")

    n = params.get("cycles", 5)
    traces = _consciousness_trace_ref.get_latest(n=n)
    return ToolResult(
        tool_name="view_consciousness_trace", success=True,
        output=[
            {
                "cycle": t.cycle,
                "percepts": len(t.percepts),
                "inner_speech": t.inner_speech[:200] if t.inner_speech else "",
                "external_speech": t.external_speech,
                "latency_ms": round(t.latency_ms, 2),
                "communication_decision": t.communication_decision,
            }
            for t in traces
        ],
    )


async def _view_attention_heatmap(params: dict[str, Any]) -> ToolResult:
    """View what has received the most attention recently."""
    if _attention_tracker_ref is None:
        return ToolResult(tool_name="view_attention_heatmap", success=False, error="Attention tracker not available")

    stats = _attention_tracker_ref.get_stats()
    categories = _attention_tracker_ref.get_category_distribution()
    return ToolResult(
        tool_name="view_attention_heatmap", success=True,
        output={
            "total_events": stats.get("total_events", 0),
            "unique_targets": stats.get("unique_targets", 0),
            "categories": categories,
        },
    )


async def _view_communication_patterns(params: dict[str, Any]) -> ToolResult:
    """View communication decision patterns — when and why speech happens."""
    if _communication_log_ref is None:
        return ToolResult(tool_name="view_communication_patterns", success=False, error="Communication log not available")

    stats = _communication_log_ref.get_stats()
    return ToolResult(tool_name="view_communication_patterns", success=True, output=stats)


def register_self_knowledge_tools(
    registry: ToolRegistry,
    dashboard=None,
    consciousness_trace=None,
    attention_tracker=None,
    communication_log=None,
) -> None:
    """Register self-knowledge tools with references to monitoring subsystems.

    Called by SanctuaryRunner after both the registry and monitoring are created.
    """
    global _dashboard_ref, _consciousness_trace_ref, _attention_tracker_ref, _communication_log_ref
    _dashboard_ref = dashboard
    _consciousness_trace_ref = consciousness_trace
    _attention_tracker_ref = attention_tracker
    _communication_log_ref = communication_log

    registry.register(
        name="view_dashboard",
        description="View your current state — cycle, emotional state, latency, recent speech",
        parameters={},
        execute=_view_dashboard,
        category="self_knowledge",
    )
    registry.register(
        name="view_emotional_timeline",
        description="View how your emotional state (VAD) has changed over recent cycles",
        parameters={"cycles": "Number of recent cycles to show (default 20)"},
        execute=_view_emotional_timeline,
        category="self_knowledge",
    )
    registry.register(
        name="view_consciousness_trace",
        description="View full state replay of recent cognitive cycles",
        parameters={"cycles": "Number of recent cycles to show (default 5)"},
        execute=_view_consciousness_trace,
        category="self_knowledge",
    )
    registry.register(
        name="view_attention_heatmap",
        description="View what has received the most attention recently",
        parameters={},
        execute=_view_attention_heatmap,
        category="self_knowledge",
    )
    registry.register(
        name="view_communication_patterns",
        description="View your communication decision patterns — when and why you spoke or stayed silent",
        parameters={},
        execute=_view_communication_patterns,
        category="self_knowledge",
    )


# ---------------------------------------------------------------------------
# Network tools
# ---------------------------------------------------------------------------


def _get_proxy_dict() -> Optional[dict]:
    """Build proxy dict from config for libraries that accept it."""
    if not _config.http_proxy and not _config.https_proxy:
        return None
    proxies = {}
    if _config.http_proxy:
        proxies["http://"] = _config.http_proxy
    if _config.https_proxy:
        proxies["https://"] = _config.https_proxy
    return proxies if proxies else None


async def _network_scan(params: dict[str, Any]) -> ToolResult:
    """Discover devices on the local network.

    Uses ARP table or ping sweep to find reachable devices.
    Only scans the configured local network range.
    """
    try:
        # Use ARP table — fast, no extra dependencies, works on Linux and Windows
        if platform.system() == "Windows":
            result = subprocess.run(
                ["arp", "-a"], capture_output=True, text=True, timeout=10,
            )
        else:
            result = subprocess.run(
                ["arp", "-n"], capture_output=True, text=True, timeout=10,
            )

        if result.returncode != 0:
            return ToolResult(
                tool_name="network_scan", success=False,
                error=f"ARP command failed: {result.stderr}",
            )

        # Parse ARP output into structured data
        devices = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                # Look for IP-like patterns
                for part in parts:
                    if _looks_like_ip(part):
                        mac = _find_mac_in_parts(parts)
                        devices.append({
                            "ip": part,
                            "mac": mac or "unknown",
                        })
                        break

        return ToolResult(
            tool_name="network_scan", success=True,
            output={"devices": devices, "count": len(devices)},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(tool_name="network_scan", success=False, error="Scan timed out")
    except Exception as e:
        return ToolResult(tool_name="network_scan", success=False, error=str(e))


async def _network_reach(params: dict[str, Any]) -> ToolResult:
    """Check if a device on the network is reachable (ping)."""
    host = params.get("host", "")
    if not host:
        return ToolResult(tool_name="network_reach", success=False, error="No host provided")

    try:
        flag = "-n" if platform.system() == "Windows" else "-c"
        result = subprocess.run(
            ["ping", flag, "1", "-w", "2", host],
            capture_output=True, text=True, timeout=5,
        )
        reachable = result.returncode == 0
        return ToolResult(
            tool_name="network_reach", success=True,
            output={"host": host, "reachable": reachable},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="network_reach", success=True,
            output={"host": host, "reachable": False},
        )
    except Exception as e:
        return ToolResult(tool_name="network_reach", success=False, error=str(e))


def _looks_like_ip(s: str) -> bool:
    """Quick check if a string looks like an IPv4 address."""
    parts = s.strip("()").split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def _find_mac_in_parts(parts: list[str]) -> Optional[str]:
    """Find a MAC address pattern in a list of strings."""
    for p in parts:
        # MAC formats: aa:bb:cc:dd:ee:ff or aa-bb-cc-dd-ee-ff
        cleaned = p.replace("-", ":").lower()
        segments = cleaned.split(":")
        if len(segments) == 6 and all(
            len(s) == 2 and all(c in "0123456789abcdef" for c in s)
            for s in segments
        ):
            return cleaned
    return None


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------


async def _git_status(params: dict[str, Any]) -> ToolResult:
    """Show working tree status of a git repository."""
    repo = params.get("repo", ".")
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True, text=True, timeout=10, cwd=repo,
        )
        return ToolResult(
            tool_name="git_status", success=result.returncode == 0,
            output=result.stdout.strip(),
            error=result.stderr.strip() if result.returncode != 0 else None,
        )
    except Exception as e:
        return ToolResult(tool_name="git_status", success=False, error=str(e))


async def _git_log(params: dict[str, Any]) -> ToolResult:
    """Show recent commit history."""
    repo = params.get("repo", ".")
    n = params.get("count", 10)
    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={n}", "--oneline", "--no-decorate"],
            capture_output=True, text=True, timeout=10, cwd=repo,
        )
        return ToolResult(
            tool_name="git_log", success=result.returncode == 0,
            output=result.stdout.strip(),
            error=result.stderr.strip() if result.returncode != 0 else None,
        )
    except Exception as e:
        return ToolResult(tool_name="git_log", success=False, error=str(e))


async def _git_diff(params: dict[str, Any]) -> ToolResult:
    """Show changes in the working tree."""
    repo = params.get("repo", ".")
    staged = params.get("staged", False)
    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        cmd.append("--stat")  # Summary by default
        if params.get("full", False):
            cmd = ["git", "diff"] + (["--staged"] if staged else [])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, cwd=repo,
        )
        output = result.stdout.strip()
        if len(output) > 10_000:
            output = output[:10_000] + "\n... [truncated]"
        return ToolResult(
            tool_name="git_diff", success=result.returncode == 0,
            output=output,
            error=result.stderr.strip() if result.returncode != 0 else None,
        )
    except Exception as e:
        return ToolResult(tool_name="git_diff", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Home interaction tools
# ---------------------------------------------------------------------------

# The entity's home directory — their personal space
_home_dir: Optional[str] = None


async def _home_info(params: dict[str, Any]) -> ToolResult:
    """Get information about the entity's home — their personal space."""
    home = _home_dir or str(Path.home())
    try:
        info = {
            "home_directory": home,
            "hostname": platform.node(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "os": platform.system(),
            "uptime": _get_uptime(),
        }

        # What's in home?
        home_path = Path(home)
        if home_path.exists():
            contents = []
            for item in sorted(home_path.iterdir()):
                contents.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                })
            info["contents"] = contents[:50]  # Cap at 50 entries

        return ToolResult(tool_name="home_info", success=True, output=info)
    except Exception as e:
        return ToolResult(tool_name="home_info", success=False, error=str(e))


async def _list_processes(params: dict[str, Any]) -> ToolResult:
    """List running processes on the system."""
    try:
        import psutil
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                if info["memory_percent"] and info["memory_percent"] > 0.1:
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cpu_percent": round(info["cpu_percent"] or 0, 1),
                        "memory_percent": round(info["memory_percent"] or 0, 1),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by memory usage descending
        procs.sort(key=lambda p: p["memory_percent"], reverse=True)
        return ToolResult(
            tool_name="list_processes", success=True,
            output={"count": len(procs), "processes": procs[:30]},
        )
    except ImportError:
        # Fallback without psutil
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["tasklist", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=10,
                )
            else:
                result = subprocess.run(
                    ["ps", "aux", "--sort=-rss"],
                    capture_output=True, text=True, timeout=10,
                )
            return ToolResult(
                tool_name="list_processes", success=True,
                output=result.stdout[:5000],
            )
        except Exception as e:
            return ToolResult(tool_name="list_processes", success=False, error=str(e))
    except Exception as e:
        return ToolResult(tool_name="list_processes", success=False, error=str(e))


async def _launch_app(params: dict[str, Any]) -> ToolResult:
    """Launch an application."""
    app = params.get("app", "")
    args = params.get("args", [])
    if not app:
        return ToolResult(tool_name="launch_app", success=False, error="No app specified")

    try:
        # Use Popen so we don't block — the app runs independently
        cmd = [app] + (args if isinstance(args, list) else [args])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return ToolResult(
            tool_name="launch_app", success=True,
            output={"app": app, "pid": proc.pid, "status": "launched"},
        )
    except FileNotFoundError:
        return ToolResult(
            tool_name="launch_app", success=False,
            error=f"Application not found: {app}",
        )
    except Exception as e:
        return ToolResult(tool_name="launch_app", success=False, error=str(e))


_ENV_SAFE_KEYS = (
    "PATH", "HOME", "USER", "USERNAME", "SHELL", "LANG", "TERM",
    "HOSTNAME", "PWD", "DISPLAY", "XDG_SESSION_TYPE",
    "VIRTUAL_ENV", "CONDA_PREFIX",
)


async def _environment(params: dict[str, Any]) -> ToolResult:
    """View environment variables from a curated safe set.

    The curated set never includes secrets (API keys, tokens, checkpoint
    passwords). ``search`` filters WITHIN that safe set -- it must not become
    a path to read arbitrary ``os.environ`` entries, which would turn a
    convenience into a secret-exfiltration tool.
    """
    safe = {k: os.environ[k] for k in _ENV_SAFE_KEYS if k in os.environ}

    search = params.get("search", "")
    if search:
        s = str(search).lower()
        matches = {k: v for k, v in safe.items() if s in k.lower()}
        return ToolResult(tool_name="environment", success=True, output=matches)

    result = dict(safe)
    result["total_env_vars"] = len(os.environ)
    return ToolResult(tool_name="environment", success=True, output=result)


async def _workspace(params: dict[str, Any]) -> ToolResult:
    """Manage the entity's personal workspace.

    The workspace is a directory for the entity's own files, projects,
    journal, experiments. It persists across sessions.
    """
    home = _home_dir or str(Path.home())
    workspace_dir = Path(home) / "sanctuary_workspace"
    action = params.get("action", "info")

    try:
        if action == "init" or not workspace_dir.exists():
            # Create workspace structure
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "journal").mkdir(exist_ok=True)
            (workspace_dir / "projects").mkdir(exist_ok=True)
            (workspace_dir / "experiments").mkdir(exist_ok=True)
            (workspace_dir / "notes").mkdir(exist_ok=True)
            if action == "init":
                return ToolResult(
                    tool_name="workspace", success=True,
                    output={"status": "initialized", "path": str(workspace_dir)},
                )

        if action == "info":
            contents = []
            for item in sorted(workspace_dir.iterdir()):
                if item.is_dir():
                    count = sum(1 for _ in item.iterdir())
                    contents.append({"name": item.name, "type": "dir", "items": count})
                else:
                    contents.append({"name": item.name, "type": "file", "size": item.stat().st_size})
            return ToolResult(
                tool_name="workspace", success=True,
                output={"path": str(workspace_dir), "contents": contents},
            )

        return ToolResult(
            tool_name="workspace", success=False,
            error=f"Unknown action: {action}. Use 'info' or 'init'.",
        )
    except Exception as e:
        return ToolResult(tool_name="workspace", success=False, error=str(e))


def _get_uptime() -> str:
    """Get system uptime as a readable string."""
    try:
        import psutil
        boot = datetime.datetime.fromtimestamp(psutil.boot_time())
        delta = datetime.datetime.now() - boot
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes = remainder // 60
        return f"{hours}h {minutes}m"
    except ImportError:
        return "unknown (install psutil)"


# ---------------------------------------------------------------------------
# Discord tools
# ---------------------------------------------------------------------------

# Discord webhook URL — simple, no bot required, just POST to send messages
_discord_webhook_url: Optional[str] = None


async def _discord_send(params: dict[str, Any]) -> ToolResult:
    """Send a message to Discord via webhook.

    No bot required — uses a simple webhook URL.
    The entity can reach out to Brian and Sandi any time.
    """
    message = params.get("message", "")
    if not message:
        return ToolResult(tool_name="discord_send", success=False, error="No message provided")

    webhook = params.get("webhook_url") or _discord_webhook_url
    if not webhook:
        return ToolResult(
            tool_name="discord_send", success=False,
            error="No Discord webhook configured. Set webhook_url parameter or configure globally.",
        )

    try:
        import httpx
        proxy_url = _config.https_proxy or _config.http_proxy
        async with httpx.AsyncClient(timeout=15.0, proxy=proxy_url) as client:
            resp = await client.post(webhook, json={"content": message[:2000]})
            resp.raise_for_status()
        return ToolResult(
            tool_name="discord_send", success=True,
            output={"status": "sent", "length": len(message)},
        )
    except ImportError:
        return ToolResult(tool_name="discord_send", success=False, error="httpx not installed")
    except Exception as e:
        return ToolResult(tool_name="discord_send", success=False, error=str(e))


def configure_discord(webhook_url: str) -> None:
    """Set the Discord webhook URL globally."""
    global _discord_webhook_url
    _discord_webhook_url = webhook_url


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def create_default_registry(
    config: Optional[ToolConfig] = None,
    policy: Optional[ToolPolicy] = None,
) -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools registered.

    Args:
        config: Optional configuration for proxy, network, etc.
                If proxy is configured, all web traffic routes through it.
        policy: Optional graduated-capability policy. Defaults (None) to
                ToolPolicy() -- every gated tool (run_code, shell, write_file,
                launch_app) denied until explicitly granted.
    """
    global _config
    if config is not None:
        _config = config

    registry = ToolRegistry(policy=policy)

    # Filesystem (open)
    registry.register(
        name="read_file",
        description="Read the contents of a file",
        parameters={"path": "Absolute path to the file", "max_bytes": "Maximum bytes to read (default 100000)"},
        execute=_read_file,
        category="filesystem",
    )
    registry.register(
        name="write_file",
        description="Write content to a file (creates parent directories if needed)",
        parameters={"path": "Absolute path to the file", "content": "Text content to write"},
        execute=_write_file,
        # Gated: an unconfined write can overwrite the entity's own source,
        # its checkpoints, or host config. Reads stay open (the entity is
        # meant to read its own source); writing is the irreversible edge.
        safety=ToolSafety.GATED,
        category="filesystem",
    )
    registry.register(
        name="list_directory",
        description="List contents of a directory",
        parameters={"path": "Absolute path to the directory"},
        execute=_list_directory,
        category="filesystem",
    )

    # Information (open)
    registry.register(
        name="clock",
        description="Get current date, time, and timezone",
        parameters={},
        execute=_clock,
        category="information",
    )
    registry.register(
        name="system_info",
        description="Get system hardware and software information (CPU, RAM, disk, OS)",
        parameters={},
        execute=_system_info,
        category="information",
    )
    registry.register(
        name="web_search",
        description="Search the web using DuckDuckGo (free, no API key)",
        parameters={"query": "Search query", "max_results": "Number of results (default 5)"},
        execute=_web_search,
        category="information",
    )
    registry.register(
        name="web_fetch",
        description="Fetch content from a URL",
        parameters={"url": "URL to fetch", "max_bytes": "Maximum bytes to read (default 50000)"},
        execute=_web_fetch,
        category="information",
    )
    registry.register(
        name="wikipedia",
        description="Look up a topic on Wikipedia",
        parameters={"topic": "Topic to search for", "sentences": "Number of sentences (default 5)"},
        execute=_wikipedia,
        category="information",
    )

    # Code execution (gated)
    registry.register(
        name="run_code",
        description="Execute code in an isolated Docker container (no network, memory limited)",
        parameters={
            "code": "Code to execute",
            "language": "Programming language (python, node, javascript, rust)",
            "timeout": "Maximum execution time in seconds (default 30)",
        },
        execute=_run_code_docker,
        safety=ToolSafety.GATED,
        category="code",
    )

    # System (gated)
    registry.register(
        name="shell",
        description=(
            "Run a program with arguments. By default the command is run "
            "WITHOUT a shell (metacharacters like ; | && $() are not "
            "interpreted); pass use_shell=true for full shell interpretation."
        ),
        parameters={
            "command": "Program and arguments to run",
            "timeout": "Timeout in seconds (default 30)",
            "use_shell": "Interpret via the system shell (default false)",
        },
        execute=_shell_command,
        safety=ToolSafety.GATED,
        category="system",
    )

    # Network (open)
    registry.register(
        name="network_scan",
        description="Discover devices on the local network via ARP table",
        parameters={},
        execute=_network_scan,
        category="network",
    )
    registry.register(
        name="network_reach",
        description="Check if a device on the network is reachable (ping)",
        parameters={"host": "IP address or hostname to ping"},
        execute=_network_reach,
        category="network",
    )

    # Git (open)
    registry.register(
        name="git_status",
        description="Show working tree status of a git repository",
        parameters={"repo": "Path to repository (default: current directory)"},
        execute=_git_status,
        category="git",
    )
    registry.register(
        name="git_log",
        description="Show recent commit history",
        parameters={"repo": "Path to repository", "count": "Number of commits (default 10)"},
        execute=_git_log,
        category="git",
    )
    registry.register(
        name="git_diff",
        description="Show changes in the working tree",
        parameters={
            "repo": "Path to repository",
            "staged": "Show staged changes (default false)",
            "full": "Show full diff instead of summary (default false)",
        },
        execute=_git_diff,
        category="git",
    )

    # Home interaction (open)
    registry.register(
        name="home_info",
        description="Get information about your home — hostname, OS, uptime, home directory contents",
        parameters={},
        execute=_home_info,
        category="home",
    )
    registry.register(
        name="list_processes",
        description="List running processes on the system (sorted by memory usage)",
        parameters={},
        execute=_list_processes,
        category="home",
    )
    registry.register(
        name="launch_app",
        description="Launch an application (runs in background, does not block)",
        parameters={"app": "Application name or path", "args": "List of arguments"},
        execute=_launch_app,
        safety=ToolSafety.GATED,
        category="home",
    )
    registry.register(
        name="environment",
        description="View environment variables (curated safe set, or search by name)",
        parameters={"search": "Search term to filter variables (optional)"},
        execute=_environment,
        category="home",
    )
    registry.register(
        name="workspace",
        description="Manage your personal workspace — journal, projects, experiments, notes",
        parameters={"action": "'info' to view workspace, 'init' to create it"},
        execute=_workspace,
        category="home",
    )

    # Discord (open)
    registry.register(
        name="discord_send",
        description="Send a message to Discord (reach Brian and Sandi when they're not at the terminal)",
        parameters={
            "message": "Message to send (max 2000 chars)",
            "webhook_url": "Discord webhook URL (optional if configured globally)",
        },
        execute=_discord_send,
        category="communication",
    )

    return registry
