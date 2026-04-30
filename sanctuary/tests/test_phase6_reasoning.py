"""Tests for reasoning storage subsystems.

After the 2026-04-30 cognition-leakage cleanup, only the counterfactual
storage subsystem remains in sanctuary/reasoning/. The BeliefRevisionTracker,
UncertaintyQuantifier, and MentalSimulator (and the reflection-prompting
methods on CounterfactualReasoner) were removed because they had Sanctuary
doing what should be the entity's cognition. The deleted modules and tests
are preserved under _deprecated/cognition-leakage-2026-04-30/.
"""

from sanctuary.reasoning.counterfactual import (
    CounterfactualConfig,
    CounterfactualReasoner,
)


class TestCounterfactualReasoner:
    """Tests for counterfactual storage."""

    def test_record_decision(self):
        r = CounterfactualReasoner()
        r.record_decision(
            cycle=1,
            chosen_action="respond with empathy",
            alternatives=["ask question", "stay silent"],
            context_summary="User expressed frustration",
        )
        assert len(r._decisions) == 1
        assert r._decisions[0].chosen_action == "respond with empathy"
        assert len(r._decisions[0].alternatives) == 2

    def test_record_outcome(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=5, chosen_action="help", alternatives=["wait"])
        r.record_outcome(cycle=5, outcome="User was grateful", valence=0.8)
        assert r._decisions[0].outcome == "User was grateful"
        assert r._decisions[0].outcome_valence == 0.8

    def test_outcome_valence_clamped(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="extreme", valence=5.0)
        assert r._decisions[0].outcome_valence == 1.0

    def test_record_counterfactual(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="bad", valence=-0.5)
        r.record_counterfactual(
            decision_cycle=1,
            alternative_action="wait",
            imagined_outcome="nothing bad would have happened",
            confidence=0.6,
            lesson="Patience is sometimes better",
        )
        assert len(r._counterfactuals) == 1
        assert r._decisions[0].counterfactual_generated is True
        assert r._total_reflections == 1

    def test_recent_lessons(self):
        r = CounterfactualReasoner()
        for i in range(3):
            r.record_counterfactual(
                decision_cycle=i,
                alternative_action="alt",
                imagined_outcome="outcome",
                lesson=f"Lesson {i}",
            )
        lessons = r.get_recent_lessons(n=2)
        assert len(lessons) == 2
        assert lessons[0] == "Lesson 2"

    def test_stats(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="ok", valence=0.5)
        stats = r.get_stats()
        assert stats["total_decisions"] == 1
        assert stats["decisions_with_outcomes"] == 1

    def test_max_decision_history(self):
        config = CounterfactualConfig(max_decision_history=5)
        r = CounterfactualReasoner(config=config)
        for i in range(10):
            r.record_decision(cycle=i, chosen_action=f"act{i}", alternatives=["x"])
        assert len(r._decisions) == 5
