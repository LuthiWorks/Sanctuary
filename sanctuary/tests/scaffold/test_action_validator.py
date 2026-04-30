"""Tests for scaffold action validator."""

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import CognitiveOutput, GoalProposal, MemoryOp
from sanctuary.scaffold.action_validator import ScaffoldActionValidator


class TestActionValidator:
    """Test authority-based action validation."""

    def test_valid_output_unchanged(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager()
        output = CognitiveOutput(
            inner_speech="Valid output",
            memory_ops=[MemoryOp(type="write_episodic", content="event happened")],
            goal_proposals=[GoalProposal(action="add", goal="Learn more")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(issues) == 0
        assert len(validated.memory_ops) == 1
        assert len(validated.goal_proposals) == 1

    def test_invalid_memory_op_type_filtered(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager()
        output = CognitiveOutput(
            inner_speech="test",
            memory_ops=[MemoryOp(type="delete_everything", content="oops")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.memory_ops) == 0
        assert any("invalid memory op" in i.lower() for i in issues)

    def test_retrieve_without_query_filtered(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager()
        output = CognitiveOutput(
            inner_speech="test",
            memory_ops=[MemoryOp(type="retrieve", query="")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.memory_ops) == 0
        assert any("query" in i.lower() for i in issues)

    def test_write_without_content_filtered(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager()
        output = CognitiveOutput(
            inner_speech="test",
            memory_ops=[MemoryOp(type="write_episodic", content="")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.memory_ops) == 0

    def test_scaffold_only_blocks_all_memory_ops(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager({"memory": AuthorityLevel.SCAFFOLD_ONLY})
        output = CognitiveOutput(
            inner_speech="test",
            memory_ops=[MemoryOp(type="write_episodic", content="event")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.memory_ops) == 0
        assert any("scaffold_only" in i.lower() for i in issues)

    def test_scaffold_only_blocks_all_goal_proposals(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager({"goals": AuthorityLevel.SCAFFOLD_ONLY})
        output = CognitiveOutput(
            inner_speech="test",
            goal_proposals=[GoalProposal(action="add", goal="New goal")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.goal_proposals) == 0

    def test_invalid_goal_action_filtered(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager()
        output = CognitiveOutput(
            inner_speech="test",
            goal_proposals=[GoalProposal(action="destroy", goal="everything")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.goal_proposals) == 0

    def test_goal_add_without_description_filtered(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager()
        output = CognitiveOutput(
            inner_speech="test",
            goal_proposals=[GoalProposal(action="add", goal="")],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.goal_proposals) == 0

    def test_scaffold_only_blocks_world_model(self):
        from sanctuary.core.schema import AddEntity, EntityQuery

        validator = ScaffoldActionValidator()
        authority = AuthorityManager({"world_model": AuthorityLevel.SCAFFOLD_ONLY})
        output = CognitiveOutput(
            inner_speech="test",
            world_model_updates=[
                AddEntity(name="alice", properties={"mood": "happy"}),
            ],
            world_model_queries=[EntityQuery(name="alice")],
        )
        validated, issues = validator.validate(output, authority)
        assert validated.world_model_updates == []
        assert validated.world_model_queries == []

    def test_mixed_valid_and_invalid(self):
        validator = ScaffoldActionValidator()
        authority = AuthorityManager()
        output = CognitiveOutput(
            inner_speech="test",
            memory_ops=[
                MemoryOp(type="write_episodic", content="valid event"),
                MemoryOp(type="bad_type", content="invalid"),
                MemoryOp(type="retrieve", query="find something"),
            ],
        )
        validated, issues = validator.validate(output, authority)
        assert len(validated.memory_ops) == 2
        assert len(issues) == 1
