"""Tests for scaffold goal integrator."""

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import GoalProposal
from sanctuary.scaffold.goal_integrator import ScaffoldGoalIntegrator


class TestGoalIntegrator:
    """Test goal tracking and LLM proposal integration."""

    def test_initial_state_empty(self):
        gi = ScaffoldGoalIntegrator()
        status = gi.get_status()
        assert status["active_count"] == 0
        assert len(status["goals"]) == 0

    def test_add_goal(self):
        gi = ScaffoldGoalIntegrator()
        authority = AuthorityManager()  # goals defaults to MODEL_GUIDES
        proposals = [GoalProposal(action="add", goal="Learn Python", priority=0.7)]
        actions = gi.integrate_proposals(proposals, authority)
        assert any("added" in a for a in actions)
        assert gi.get_status()["active_count"] == 1

    def test_add_goal_scaffold_only_blocked(self):
        gi = ScaffoldGoalIntegrator()
        authority = AuthorityManager({"goals": AuthorityLevel.SCAFFOLD_ONLY})
        proposals = [GoalProposal(action="add", goal="Learn Python", priority=0.7)]
        actions = gi.integrate_proposals(proposals, authority)
        assert gi.get_status()["active_count"] == 0

    def test_complete_goal(self):
        gi = ScaffoldGoalIntegrator()
        authority = AuthorityManager()
        # Add then complete
        gi.integrate_proposals(
            [GoalProposal(action="add", goal="Task A", priority=0.5)],
            authority,
        )
        gi.integrate_proposals(
            [GoalProposal(action="complete", goal="Task A")],
            authority,
        )
        assert gi.get_status()["active_count"] == 0

    def test_abandon_needs_guides(self):
        gi = ScaffoldGoalIntegrator()
        authority_advises = AuthorityManager({"goals": AuthorityLevel.MODEL_ADVISES})
        authority_guides = AuthorityManager({"goals": AuthorityLevel.MODEL_GUIDES})

        # Add a goal at GUIDES level
        gi.integrate_proposals(
            [GoalProposal(action="add", goal="Task B", priority=0.5)],
            authority_guides,
        )

        # Try to abandon at ADVISES — should fail
        gi.integrate_proposals(
            [GoalProposal(action="abandon", goal="Task B")],
            authority_advises,
        )
        assert gi.get_status()["active_count"] == 1

        # Abandon at GUIDES — should succeed
        gi.integrate_proposals(
            [GoalProposal(action="abandon", goal="Task B")],
            authority_guides,
        )
        assert gi.get_status()["active_count"] == 0

    def test_reprioritize_advises_blends(self):
        gi = ScaffoldGoalIntegrator()
        authority = AuthorityManager({"goals": AuthorityLevel.MODEL_ADVISES})
        gi.integrate_proposals(
            [GoalProposal(action="add", goal="Task C", priority=0.5)],
            authority,
        )
        gi.integrate_proposals(
            [GoalProposal(action="reprioritize", goal="Task C", priority=1.0)],
            authority,
        )
        # At ADVISES, blend: 0.5 * 0.7 + 1.0 * 0.3 = 0.65
        status = gi.get_status()
        goals = list(status["goals"].values())
        assert len(goals) == 1
        assert 0.6 < goals[0]["priority"] < 0.7

    def test_reprioritize_guides_direct(self):
        gi = ScaffoldGoalIntegrator()
        authority = AuthorityManager({"goals": AuthorityLevel.MODEL_GUIDES})
        gi.integrate_proposals(
            [GoalProposal(action="add", goal="Task D", priority=0.5)],
            authority,
        )
        gi.integrate_proposals(
            [GoalProposal(action="reprioritize", goal="Task D", priority=0.9)],
            authority,
        )
        status = gi.get_status()
        goals = list(status["goals"].values())
        assert goals[0]["priority"] == 0.9

    def test_max_goals_limit(self):
        gi = ScaffoldGoalIntegrator(max_goals=3)
        authority = AuthorityManager()
        for i in range(5):
            gi.integrate_proposals(
                [GoalProposal(action="add", goal=f"Goal {i}", priority=0.5)],
                authority,
            )
        assert gi.get_status()["active_count"] == 3

    def test_active_goal_descriptions(self):
        gi = ScaffoldGoalIntegrator()
        authority = AuthorityManager()
        gi.integrate_proposals(
            [
                GoalProposal(action="add", goal="Alpha", priority=0.5),
                GoalProposal(action="add", goal="Beta", priority=0.5),
            ],
            authority,
        )
        descriptions = gi.get_active_goal_descriptions()
        assert "Alpha" in descriptions
        assert "Beta" in descriptions
