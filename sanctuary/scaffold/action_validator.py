"""Scaffold action validator — validates LLM-proposed actions based on authority.

Checks memory operations, goal proposals, and other structured outputs for
validity and applies authority-based filtering. Invalid operations are
removed from the output; the LLM is notified via anomaly flags.
"""

from __future__ import annotations

import logging
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import (
    CognitiveOutput,
    GoalProposal,
    MemoryOp,
)

logger = logging.getLogger(__name__)

# Valid memory operation types
_VALID_MEMORY_OPS = frozenset(
    ["write_episodic", "write_semantic", "retrieve", "journal"]
)

# Valid goal proposal actions
_VALID_GOAL_ACTIONS = frozenset(["add", "complete", "reprioritize", "abandon"])


class ScaffoldActionValidator:
    """Validates and filters LLM output based on authority and basic rules."""

    def validate(
        self,
        output: CognitiveOutput,
        authority: AuthorityManager,
    ) -> tuple[CognitiveOutput, list[str]]:
        """Validate the output, returning (validated_output, issues).

        The output is modified in-place for efficiency. Issues are
        descriptions of anything that was filtered out.
        """
        issues: list[str] = []

        self._validate_memory_ops(output, authority, issues)
        self._validate_goal_proposals(output, authority, issues)
        self._validate_world_model_updates(output, authority, issues)

        return output, issues

    def _validate_memory_ops(
        self,
        output: CognitiveOutput,
        authority: AuthorityManager,
        issues: list[str],
    ) -> None:
        level = authority.level("memory")

        if level == AuthorityLevel.SCAFFOLD_ONLY:
            if output.memory_ops:
                issues.append(
                    f"Filtered {len(output.memory_ops)} memory ops (authority: SCAFFOLD_ONLY)"
                )
                output.memory_ops = []
            return

        valid_ops: list[MemoryOp] = []
        for op in output.memory_ops:
            if op.type not in _VALID_MEMORY_OPS:
                issues.append(f"Invalid memory op type: {op.type}")
                continue
            if op.type == "retrieve" and not op.query:
                issues.append("Memory retrieve op missing query")
                continue
            if op.type.startswith("write") and not op.content:
                issues.append(f"Memory {op.type} op missing content")
                continue
            valid_ops.append(op)

        output.memory_ops = valid_ops

    def _validate_goal_proposals(
        self,
        output: CognitiveOutput,
        authority: AuthorityManager,
        issues: list[str],
    ) -> None:
        level = authority.level("goals")

        if level == AuthorityLevel.SCAFFOLD_ONLY:
            if output.goal_proposals:
                issues.append(
                    f"Filtered {len(output.goal_proposals)} goal proposals (authority: SCAFFOLD_ONLY)"
                )
                output.goal_proposals = []
            return

        valid: list[GoalProposal] = []
        for proposal in output.goal_proposals:
            if proposal.action not in _VALID_GOAL_ACTIONS:
                issues.append(f"Invalid goal action: {proposal.action}")
                continue
            if proposal.action == "add" and not proposal.goal:
                issues.append("Goal add proposal missing description")
                continue
            valid.append(proposal)

        output.goal_proposals = valid

    def _validate_world_model_updates(
        self,
        output: CognitiveOutput,
        authority: AuthorityManager,
        issues: list[str],
    ) -> None:
        level = authority.level("world_model")

        if level == AuthorityLevel.SCAFFOLD_ONLY:
            if output.world_model_updates or output.world_model_queries:
                issues.append("Filtered world model updates (authority: SCAFFOLD_ONLY)")
                output.world_model_updates = []
                output.world_model_queries = []
