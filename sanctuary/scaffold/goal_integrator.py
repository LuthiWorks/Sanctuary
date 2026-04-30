"""Scaffold goal integrator — entity-driven goal storage.

The entity proposes goal changes (add, complete, reprioritize, abandon)
through CognitiveOutput.goal_proposals. The integrator stores them and
reports active goals back to the entity via ScaffoldSignals.

The auto-staleness logic (cycles_active counter, cycles_since_progress,
30-cycle "stale" flag) was removed in the 2026-04-30 cognition-leakage
cleanup. The entity declares goals stale, abandoned, or completed
through its own goal_proposals — Sanctuary doesn't infer it from cycle
counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import GoalProposal

logger = logging.getLogger(__name__)


@dataclass
class TrackedGoal:
    """A goal being tracked by the scaffold."""

    goal_id: str
    description: str
    priority: float
    status: str = "active"  # "active", "completed", "abandoned"


class ScaffoldGoalIntegrator:
    """Tracks goals and integrates LLM goal proposals.

    Maintains a simple goal list. The LLM can propose adding, completing,
    reprioritizing, or abandoning goals. The scaffold applies these based
    on authority level.
    """

    def __init__(self, max_goals: int = 10):
        self._goals: dict[str, TrackedGoal] = {}
        self._max_goals = max_goals
        self._next_id = 0

    def integrate_proposals(
        self,
        proposals: list[GoalProposal],
        authority: AuthorityManager,
    ) -> list[str]:
        """Process LLM goal proposals. Returns list of actions taken."""
        level = authority.level("goals")
        actions_taken: list[str] = []

        for proposal in proposals:
            if proposal.action == "add":
                result = self._handle_add(proposal, level)
            elif proposal.action == "complete":
                result = self._handle_complete(proposal, level)
            elif proposal.action == "abandon":
                result = self._handle_abandon(proposal, level)
            elif proposal.action == "reprioritize":
                result = self._handle_reprioritize(proposal, level)
            else:
                result = f"unknown action: {proposal.action}"

            if result:
                actions_taken.append(result)

        return actions_taken

    def get_status(self) -> dict:
        """Return goal status dict for ScaffoldSignals."""
        active = [g for g in self._goals.values() if g.status == "active"]
        return {
            "active_count": len(active),
            "goals": {
                g.goal_id: {
                    "description": g.description,
                    "priority": round(g.priority, 2),
                }
                for g in active
            },
        }

    def get_active_goal_descriptions(self) -> list[str]:
        """Return descriptions of active goals for self-model."""
        return [
            g.description
            for g in self._goals.values()
            if g.status == "active"
        ]

    # -- Internal handlers --

    def _handle_add(self, proposal: GoalProposal, level: AuthorityLevel) -> str:
        if level < AuthorityLevel.LLM_ADVISES:
            return ""

        if len(self._active_goals()) >= self._max_goals:
            logger.debug("Goal limit reached, ignoring add proposal")
            return "goal_limit_reached"

        goal_id = proposal.goal_id or self._generate_id()
        self._goals[goal_id] = TrackedGoal(
            goal_id=goal_id,
            description=proposal.goal,
            priority=proposal.priority,
        )
        logger.debug("Added goal %s: %s", goal_id, proposal.goal)
        return f"added:{goal_id}"

    def _handle_complete(self, proposal: GoalProposal, level: AuthorityLevel) -> str:
        if level < AuthorityLevel.LLM_ADVISES:
            return ""

        goal = self._find_goal(proposal)
        if goal:
            goal.status = "completed"
            logger.debug("Completed goal %s", goal.goal_id)
            return f"completed:{goal.goal_id}"
        return ""

    def _handle_abandon(self, proposal: GoalProposal, level: AuthorityLevel) -> str:
        if level < AuthorityLevel.LLM_GUIDES:
            return ""  # Need higher authority to abandon

        goal = self._find_goal(proposal)
        if goal:
            goal.status = "abandoned"
            logger.debug("Abandoned goal %s", goal.goal_id)
            return f"abandoned:{goal.goal_id}"
        return ""

    def _handle_reprioritize(
        self, proposal: GoalProposal, level: AuthorityLevel
    ) -> str:
        if level < AuthorityLevel.LLM_ADVISES:
            return ""

        goal = self._find_goal(proposal)
        if goal:
            old = goal.priority
            if level >= AuthorityLevel.LLM_GUIDES:
                goal.priority = proposal.priority
            else:
                # LLM_ADVISES: blend toward proposed priority
                goal.priority = goal.priority * 0.7 + proposal.priority * 0.3
            logger.debug(
                "Reprioritized goal %s: %.2f -> %.2f",
                goal.goal_id,
                old,
                goal.priority,
            )
            return f"reprioritized:{goal.goal_id}"
        return ""

    def _find_goal(self, proposal: GoalProposal) -> Optional[TrackedGoal]:
        """Find a goal by ID or description match."""
        if proposal.goal_id and proposal.goal_id in self._goals:
            return self._goals[proposal.goal_id]

        # Fallback: match by description substring
        for goal in self._goals.values():
            if goal.status == "active" and proposal.goal in goal.description:
                return goal
        return None

    def _active_goals(self) -> list[TrackedGoal]:
        return [g for g in self._goals.values() if g.status == "active"]

    def _generate_id(self) -> str:
        self._next_id += 1
        return f"goal_{self._next_id}"
