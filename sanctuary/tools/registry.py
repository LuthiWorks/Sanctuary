"""ToolRegistry — the entity's hands.

A clean registry pattern where tools are small callables the entity
invokes through CognitiveOutput. Each tool has a name, description,
parameter spec, and an execute function.

Tools are categorized by safety level:
- OPEN: No restrictions. File reads, web search, clock, etc.
- GATED: Logged with confirmation for irreversible actions.
         Code execution, file deletion, system commands.

The entity sees the full tool catalog and chooses what to use.
Results come back as percepts in the next cognitive cycle.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)


class ToolSafety(str, Enum):
    """Safety level for tool execution."""
    OPEN = "open"      # No restrictions
    GATED = "gated"    # Logged, confirmation for irreversible actions


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description
    safety: ToolSafety = ToolSafety.OPEN
    category: str = ""


@dataclass(frozen=True)
class ToolPolicy:
    """The graduated-capability gate: decides whether a tool may execute.

    OPEN tools always run. GATED tools -- code execution, shell, file writes,
    app launch: anything irreversible or with a large blast radius -- are
    DENIED BY DEFAULT and become available only when capability is
    deliberately granted, one of three ways:

      - ``allow_gated=True``            every gated tool runs. A wide grant;
                                        intended for a supervised session, not
                                        the autonomous loop.
      - name in ``enabled_gated``       that specific gated tool runs. This is
                                        the normal expand-carefully path --
                                        grant ``run_code`` without granting
                                        ``shell``.
      - ``confirm(spec, params)`` True  a human-in-the-loop hook approves this
                                        one call (may be sync or async).

    The default ``ToolPolicy()`` denies every gated tool. That is the project's
    stated posture -- substrate first, no motor access, then expand carefully
    -- expressed as the safe default rather than left to the caller to
    remember. A denied call is not an error in the tool; it returns a failed
    ToolResult the entity perceives, and is logged for audit.
    """

    allow_gated: bool = False
    enabled_gated: frozenset[str] = frozenset()
    confirm: Optional[Callable[[ToolSpec, dict], Any]] = None

    def with_enabled(self, *names: str) -> "ToolPolicy":
        """Return a copy that additionally grants the named gated tools."""
        return replace(self, enabled_gated=self.enabled_gated | frozenset(names))


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class ToolRegistry:
    """Registry of all tools available to the entity.

    Usage::

        registry = ToolRegistry()
        registry.register(
            name="read_file",
            description="Read the contents of a file",
            parameters={"path": "Absolute path to the file"},
            execute=read_file_fn,
            category="filesystem",
        )

        result = await registry.execute("read_file", {"path": "/some/file.txt"})
    """

    def __init__(
        self,
        policy: Optional[ToolPolicy] = None,
        tool_timeout_seconds: float = 60.0,
    ):
        self._tools: dict[str, ToolSpec] = {}
        self._executors: dict[str, Callable[..., Awaitable[ToolResult]]] = {}
        self._history: list[ToolResult] = []
        self._max_history: int = 500
        # The graduated-capability gate. Defaults to deny-all-gated.
        self._policy: ToolPolicy = policy or ToolPolicy()
        # Watchdog ceiling: no single tool call may stall its caller (a
        # cognitive-cycle step) beyond this. A backstop above each tool's own
        # internal timeout — it catches tools with no timeout (e.g. wikipedia)
        # or a hung one. Only effective because blocking tool bodies are
        # offloaded to threads (audit 2026-07-03, item 13); a tool that blocked
        # the loop thread would prevent this timeout from ever firing.
        self._tool_timeout_seconds: float = tool_timeout_seconds

    @property
    def policy(self) -> ToolPolicy:
        return self._policy

    def set_policy(self, policy: ToolPolicy) -> None:
        """Replace the active tool policy (e.g. to widen capability)."""
        self._policy = policy

    def enable_gated(self, *names: str) -> None:
        """Grant specific gated tools -- the expand-carefully entry point."""
        self._policy = self._policy.with_enabled(*names)

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, str],
        execute: Callable[..., Awaitable[ToolResult]],
        safety: ToolSafety = ToolSafety.OPEN,
        category: str = "",
    ) -> None:
        """Register a tool."""
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            safety=safety,
            category=category,
        )
        self._executors[name] = execute
        logger.info("Tool registered: %s (%s)", name, category)

    async def _authorize(
        self, spec: ToolSpec, params: dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Apply the tool policy. Returns (allowed, deny_reason)."""
        if spec.safety != ToolSafety.GATED:
            return True, None

        policy = self._policy
        if policy.allow_gated or spec.name in policy.enabled_gated:
            return True, None

        if policy.confirm is not None:
            decision = policy.confirm(spec, params)
            if inspect.isawaitable(decision):
                decision = await decision
            if decision:
                return True, None
            return False, "gated tool denied by confirmation hook"

        return False, (
            f"tool '{spec.name}' is gated and not enabled; grant it via "
            f"ToolPolicy (enabled_gated / allow_gated / a confirm hook)"
        )

    async def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given parameters.

        Gated tools are authorized against the active :class:`ToolPolicy`
        before running; a denied call returns a failed ToolResult (never
        executes) and is logged for audit.
        """
        if name not in self._tools:
            result = ToolResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool: {name}",
            )
            self._record(result)
            return result

        spec = self._tools[name]
        executor = self._executors[name]

        allowed, deny_reason = await self._authorize(spec, params)
        if not allowed:
            logger.warning("Tool %s BLOCKED by policy: %s", name, deny_reason)
            result = ToolResult(
                tool_name=name,
                success=False,
                error=f"blocked by tool policy: {deny_reason}",
            )
            self._record(result)
            return result

        if spec.safety == ToolSafety.GATED:
            # Gated == "logged" per the safety contract: leave an audit line
            # whenever an irreversible-capability tool actually runs.
            logger.info(
                "GATED tool authorized: %s (param keys: %s)",
                name, sorted(params.keys()),
            )

        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                executor(params), timeout=self._tool_timeout_seconds
            )
            result.duration_ms = (time.perf_counter() - start) * 1000.0
        except asyncio.TimeoutError:
            # The tool body is offloaded to a thread, so it keeps running to
            # completion in the background (a subprocess is killed by its own
            # timeout); the loop is freed now and the caller gets a failure.
            result = ToolResult(
                tool_name=name,
                success=False,
                error=f"tool timed out after {self._tool_timeout_seconds:.0f}s",
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
            logger.error("Tool %s timed out after %.0fs", name, self._tool_timeout_seconds)
        except Exception as e:
            result = ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
            logger.error("Tool %s failed: %s", name, e)

        self._record(result)
        return result

    def get_catalog(self) -> list[dict]:
        """Get the full tool catalog for inclusion in CognitiveInput.

        Returns a list of tool descriptions the entity can browse.
        """
        catalog = []
        for spec in sorted(self._tools.values(), key=lambda s: (s.category, s.name)):
            catalog.append({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "safety": spec.safety.value,
                "category": spec.category,
            })
        return catalog

    def get_categories(self) -> dict[str, list[str]]:
        """Get tools organized by category."""
        cats: dict[str, list[str]] = {}
        for spec in self._tools.values():
            cats.setdefault(spec.category, []).append(spec.name)
        return cats

    def get_stats(self) -> dict:
        """Get tool usage statistics."""
        total = len(self._history)
        successes = sum(1 for r in self._history if r.success)
        by_tool: dict[str, int] = {}
        for r in self._history:
            by_tool[r.tool_name] = by_tool.get(r.tool_name, 0) + 1

        return {
            "registered_tools": len(self._tools),
            "total_executions": total,
            "success_rate": successes / total if total else 0.0,
            "by_tool": by_tool,
        }

    def _record(self, result: ToolResult) -> None:
        """Record a tool execution in history."""
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        level = logging.DEBUG if result.success else logging.WARNING
        logger.log(
            level,
            "Tool %s: %s (%.1fms)",
            result.tool_name,
            "OK" if result.success else result.error,
            result.duration_ms,
        )

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def has_tool(self, name: str) -> bool:
        return name in self._tools
