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
import logging
import time
from dataclasses import dataclass, field
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

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._executors: dict[str, Callable[..., Awaitable[ToolResult]]] = {}
        self._history: list[ToolResult] = []
        self._max_history: int = 500

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

    async def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given parameters."""
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

        start = time.perf_counter()
        try:
            result = await executor(params)
            result.duration_ms = (time.perf_counter() - start) * 1000.0
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
