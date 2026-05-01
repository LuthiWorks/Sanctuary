"""Tests for the authority level management system."""

import pytest
from sanctuary.core.authority import (
    AuthorityLevel,
    AuthorityManager,
    DEFAULT_AUTHORITY,
)


class TestAuthorityLevel:
    def test_level_ordering(self):
        assert AuthorityLevel.SCAFFOLD_ONLY < AuthorityLevel.MODEL_ADVISES
        assert AuthorityLevel.MODEL_ADVISES < AuthorityLevel.MODEL_GUIDES
        assert AuthorityLevel.MODEL_GUIDES < AuthorityLevel.MODEL_CONTROLS

    def test_level_values(self):
        assert AuthorityLevel.SCAFFOLD_ONLY == 0
        assert AuthorityLevel.MODEL_ADVISES == 1
        assert AuthorityLevel.MODEL_GUIDES == 2
        assert AuthorityLevel.MODEL_CONTROLS == 3


class TestAuthorityManager:
    def test_default_initialization(self):
        """Default levels match the plan's initial authority table."""
        mgr = AuthorityManager()
        assert mgr.level("inner_speech") == AuthorityLevel.MODEL_CONTROLS
        assert mgr.level("self_model") == AuthorityLevel.MODEL_GUIDES
        assert mgr.level("attention") == AuthorityLevel.MODEL_ADVISES
        assert mgr.level("emotional_state") == AuthorityLevel.MODEL_GUIDES
        assert mgr.level("action") == AuthorityLevel.MODEL_ADVISES
        assert mgr.level("communication") == AuthorityLevel.MODEL_ADVISES
        assert mgr.level("goals") == AuthorityLevel.MODEL_GUIDES
        assert mgr.level("world_model") == AuthorityLevel.MODEL_GUIDES
        assert mgr.level("memory") == AuthorityLevel.MODEL_GUIDES
        assert mgr.level("growth") == AuthorityLevel.MODEL_CONTROLS

    def test_custom_initialization(self):
        mgr = AuthorityManager(initial_levels={"attention": 3, "memory": 0})
        assert mgr.level("attention") == AuthorityLevel.MODEL_CONTROLS
        assert mgr.level("memory") == AuthorityLevel.SCAFFOLD_ONLY

    def test_unknown_function_returns_scaffold_only(self):
        """Safe default: unknown functions get no the model authority."""
        mgr = AuthorityManager()
        assert mgr.level("nonexistent") == AuthorityLevel.SCAFFOLD_ONLY

    def test_promote(self):
        mgr = AuthorityManager(initial_levels={"attention": 1})
        new = mgr.promote("attention", reason="reliable performance")
        assert new == AuthorityLevel.MODEL_GUIDES
        assert mgr.level("attention") == AuthorityLevel.MODEL_GUIDES

    def test_promote_at_max_is_noop(self):
        mgr = AuthorityManager(initial_levels={"inner_speech": 3})
        new = mgr.promote("inner_speech")
        assert new == AuthorityLevel.MODEL_CONTROLS

    def test_demote(self):
        mgr = AuthorityManager(initial_levels={"goals": 2})
        new = mgr.demote("goals", reason="inconsistent behavior")
        assert new == AuthorityLevel.MODEL_ADVISES
        assert mgr.level("goals") == AuthorityLevel.MODEL_ADVISES

    def test_demote_at_min_is_noop(self):
        mgr = AuthorityManager(initial_levels={"test": 0})
        new = mgr.demote("test")
        assert new == AuthorityLevel.SCAFFOLD_ONLY

    def test_set_level(self):
        mgr = AuthorityManager(initial_levels={"action": 1})
        new = mgr.set_level("action", 3, reason="manual override")
        assert new == AuthorityLevel.MODEL_CONTROLS

    def test_audit_trail(self):
        """All authority changes must be logged."""
        mgr = AuthorityManager(initial_levels={"attention": 1})
        mgr.promote("attention", reason="test promotion")
        mgr.demote("attention", reason="test demotion")

        history = mgr.get_history()
        assert len(history) == 2
        assert history[0]["action"] == "promote"
        assert history[0]["old_level"] == 1
        assert history[0]["new_level"] == 2
        assert history[0]["reason"] == "test promotion"
        assert history[1]["action"] == "demote"
        assert history[1]["new_level"] == 1

    def test_model_has_authority(self):
        mgr = AuthorityManager()
        assert mgr.model_has_authority("inner_speech", minimum=3)
        assert mgr.model_has_authority("attention", minimum=1)
        assert not mgr.model_has_authority("attention", minimum=2)

    def test_get_all_levels(self):
        mgr = AuthorityManager()
        levels = mgr.get_all_levels()
        assert "inner_speech" in levels
        assert levels["inner_speech"] == AuthorityLevel.MODEL_CONTROLS
        # Ensure it's a copy
        levels["inner_speech"] = AuthorityLevel.SCAFFOLD_ONLY
        assert mgr.level("inner_speech") == AuthorityLevel.MODEL_CONTROLS

    def test_repr(self):
        mgr = AuthorityManager(initial_levels={"test": 2})
        r = repr(mgr)
        assert "AuthorityManager" in r
        assert "test=MODEL_GUIDES" in r
