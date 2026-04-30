"""Tests for the typed-relation world graph."""

import json
from pathlib import Path

import pytest

from sanctuary.core.schema import (
    AddEntity,
    AddRelation,
    EntityQuery,
    NeighborhoodQuery,
    Relation,
    RelationQuery,
    RemoveEntity,
    RetractRelation,
    UpdateProperty,
    WorldGraphEntity,
)
from sanctuary.memory.world_graph import (
    DEFAULT_SIZE_WARNING_THRESHOLD,
    SizeWarning,
    WorldGraph,
)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class TestApplyUpdate:
    def test_add_entity_creates_record(self):
        g = WorldGraph()
        g.apply_update(AddEntity(name="Alice", properties={"role": "visitor"}), cycle=1)
        assert "Alice" in g.entities
        assert g.entities["Alice"].properties["role"] == "visitor"
        assert g.entities["Alice"].first_seen_cycle == 1
        assert g.entities["Alice"].last_referenced_cycle == 1

    def test_add_entity_merges_properties(self):
        g = WorldGraph()
        g.apply_update(AddEntity(name="Alice", properties={"role": "visitor"}), cycle=1)
        g.apply_update(AddEntity(name="Alice", properties={"mood": "curious"}), cycle=5)
        # Both properties survive; first_seen_cycle preserved.
        assert g.entities["Alice"].properties == {"role": "visitor", "mood": "curious"}
        assert g.entities["Alice"].first_seen_cycle == 1
        assert g.entities["Alice"].last_referenced_cycle == 5

    def test_add_relation_auto_creates_endpoints(self):
        g = WorldGraph()
        g.apply_update(
            AddRelation(source="Alice", type="knows", target="Bob", confidence=0.8),
            cycle=2,
        )
        assert "Alice" in g.entities
        assert "Bob" in g.entities
        rels = g.entities["Alice"].relations
        assert len(rels) == 1
        assert rels[0].target == "Bob"
        assert rels[0].confidence == 0.8
        assert rels[0].asserted_at_cycle == 2

    def test_add_relation_dedup_updates_existing(self):
        g = WorldGraph()
        g.apply_update(AddRelation(source="A", type="knows", target="B", confidence=0.5), cycle=1)
        g.apply_update(
            AddRelation(
                source="A", type="knows", target="B",
                confidence=0.9, source_observation="confirmed via second meeting",
            ),
            cycle=3,
        )
        rels = [r for r in g.entities["A"].relations if r.retracted_at_cycle is None]
        assert len(rels) == 1
        assert rels[0].confidence == 0.9
        assert rels[0].source == "confirmed via second meeting"

    def test_retract_relation_preserves_history(self):
        g = WorldGraph()
        g.apply_update(AddRelation(source="A", type="knows", target="B"), cycle=1)
        g.apply_update(RetractRelation(source="A", type="knows", target="B"), cycle=5)
        rels = g.entities["A"].relations
        assert len(rels) == 1  # still in storage
        assert rels[0].retracted_at_cycle == 5

    def test_retract_missing_source_is_silent(self):
        g = WorldGraph()
        # Should not raise even if the source doesn't exist.
        g.apply_update(RetractRelation(source="ghost", type="knows", target="B"), cycle=5)

    def test_remove_entity_drops_record(self):
        g = WorldGraph()
        g.apply_update(AddEntity(name="X"), cycle=1)
        g.apply_update(RemoveEntity(name="X"), cycle=2)
        assert "X" not in g.entities

    def test_update_property_modifies(self):
        g = WorldGraph()
        g.apply_update(AddEntity(name="A", properties={"k": 1}), cycle=1)
        g.apply_update(UpdateProperty(name="A", key="k", value=2), cycle=3)
        assert g.entities["A"].properties["k"] == 2
        assert g.entities["A"].last_referenced_cycle == 3

    def test_update_property_missing_entity_is_silent(self):
        g = WorldGraph()
        g.apply_update(UpdateProperty(name="ghost", key="k", value=1), cycle=1)
        assert "ghost" not in g.entities

    def test_relation_types_seen_tracks_open_vocabulary(self):
        g = WorldGraph()
        g.apply_update(AddRelation(source="A", type="knows", target="B"), cycle=1)
        g.apply_update(AddRelation(source="B", type="recovering_from", target="C"), cycle=2)
        assert g.relation_types_seen == {"knows", "recovering_from"}


# ---------------------------------------------------------------------------
# Query resolution
# ---------------------------------------------------------------------------


class TestResolveQuery:
    def _seed(self, g: WorldGraph) -> None:
        g.apply_update(AddEntity(name="Alice", properties={"role": "visitor"}), cycle=1)
        g.apply_update(AddEntity(name="Bob"), cycle=1)
        g.apply_update(AddEntity(name="Carol"), cycle=1)
        g.apply_update(AddRelation(source="Alice", type="knows", target="Bob"), cycle=2)
        g.apply_update(AddRelation(source="Bob", type="knows", target="Carol"), cycle=2)
        g.apply_update(AddRelation(source="Alice", type="trusts", target="Bob"), cycle=2)

    def test_entity_query_returns_active_relations(self):
        g = WorldGraph()
        self._seed(g)
        result = g.resolve_query(EntityQuery(name="Alice"), cycle=10)
        assert result.found is True
        assert "Alice" in result.entities
        types_found = {rel.type for _, rel in result.relations}
        assert types_found == {"knows", "trusts"}

    def test_entity_query_missing_returns_not_found(self):
        g = WorldGraph()
        result = g.resolve_query(EntityQuery(name="ghost"), cycle=1)
        assert result.found is False
        assert result.entities == {}

    def test_relation_query_finds_all_pairs(self):
        g = WorldGraph()
        self._seed(g)
        result = g.resolve_query(RelationQuery(type="knows"), cycle=10)
        assert result.found is True
        pairs = {(src, rel.target) for src, rel in result.relations}
        assert pairs == {("Alice", "Bob"), ("Bob", "Carol")}

    def test_relation_query_no_matches_returns_not_found(self):
        g = WorldGraph()
        self._seed(g)
        result = g.resolve_query(RelationQuery(type="never_used"), cycle=10)
        assert result.found is False
        assert result.relations == []

    def test_neighborhood_depth_one(self):
        g = WorldGraph()
        self._seed(g)
        result = g.resolve_query(NeighborhoodQuery(name="Alice", depth=1), cycle=10)
        # depth=1 = seed + immediate neighbors.
        assert sorted(result.entities.keys()) == ["Alice", "Bob"]

    def test_neighborhood_depth_two_reaches_two_hops(self):
        g = WorldGraph()
        self._seed(g)
        result = g.resolve_query(NeighborhoodQuery(name="Alice", depth=2), cycle=10)
        assert sorted(result.entities.keys()) == ["Alice", "Bob", "Carol"]

    def test_neighborhood_missing_returns_not_found(self):
        g = WorldGraph()
        result = g.resolve_query(NeighborhoodQuery(name="ghost", depth=1), cycle=1)
        assert result.found is False

    def test_query_updates_last_referenced(self):
        g = WorldGraph()
        g.apply_update(AddEntity(name="A"), cycle=1)
        g.resolve_query(EntityQuery(name="A"), cycle=42)
        assert g.entities["A"].last_referenced_cycle == 42


# ---------------------------------------------------------------------------
# Retracted relations excluded from queries but preserved in storage
# ---------------------------------------------------------------------------


class TestRetractionVisibility:
    def test_retracted_relation_not_in_entity_query(self):
        g = WorldGraph()
        g.apply_update(AddRelation(source="A", type="knows", target="B"), cycle=1)
        g.apply_update(RetractRelation(source="A", type="knows", target="B"), cycle=2)
        result = g.resolve_query(EntityQuery(name="A"), cycle=3)
        assert result.relations == []

    def test_retracted_relation_not_in_relation_query(self):
        g = WorldGraph()
        g.apply_update(AddRelation(source="A", type="knows", target="B"), cycle=1)
        g.apply_update(RetractRelation(source="A", type="knows", target="B"), cycle=2)
        result = g.resolve_query(RelationQuery(type="knows"), cycle=3)
        assert result.found is False

    def test_retracted_relation_preserved_in_storage(self):
        g = WorldGraph()
        g.apply_update(AddRelation(source="A", type="knows", target="B"), cycle=1)
        g.apply_update(RetractRelation(source="A", type="knows", target="B"), cycle=2)
        # Storage keeps the record so the entity's prior beliefs aren't lost.
        rel = g.entities["A"].relations[0]
        assert rel.retracted_at_cycle == 2


# ---------------------------------------------------------------------------
# Persistence + migration
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "graph.json"
        g = WorldGraph()
        g.apply_update(AddEntity(name="Alice", properties={"role": "visitor"}), cycle=1)
        g.apply_update(AddRelation(source="Alice", type="knows", target="Bob"), cycle=2)
        g.save(path)

        g2 = WorldGraph()
        g2.load(path)
        assert g2.entity_count == 2  # Alice + auto-created Bob
        assert "Alice" in g2.entities
        assert g2.relation_types_seen == {"knows"}
        rel = g2.entities["Alice"].relations[0]
        assert rel.target == "Bob"

    def test_load_missing_file_no_error(self, tmp_path: Path):
        g = WorldGraph()
        g.load(tmp_path / "nonexistent.json")
        assert g.entity_count == 0

    def test_save_atomic(self, tmp_path: Path):
        """Save writes through a .tmp file then renames — partial state never visible."""
        path = tmp_path / "graph.json"
        g = WorldGraph()
        g.apply_update(AddEntity(name="A"), cycle=1)
        g.save(path)
        assert path.exists()
        # No leftover .tmp file after a clean save.
        assert not path.with_suffix(path.suffix + ".tmp").exists()

    def test_migration_from_old_world_model_format(self, tmp_path: Path):
        """Legacy WorldModel JSON: entities keyed by name with bare properties.

        WorldGraphEntity defaults handle missing relations, first_seen_cycle,
        and last_referenced_cycle through Pydantic — no migration script needed.
        """
        path = tmp_path / "graph.json"
        legacy = {
            "entities": {
                "Alice": {"name": "Alice", "properties": {"role": "visitor"}},
                "Bob": {"name": "Bob", "properties": {}},
            },
            "environment": {"setting": "atrium"},  # ignored by new loader
        }
        path.write_text(json.dumps(legacy), encoding="utf-8")

        g = WorldGraph()
        g.load(path)
        assert g.entity_count == 2
        assert g.entities["Alice"].properties == {"role": "visitor"}
        assert g.entities["Alice"].relations == []
        assert g.entities["Alice"].first_seen_cycle == 0


# ---------------------------------------------------------------------------
# Memory-footprint warning
# ---------------------------------------------------------------------------


class TestSizeWarning:
    def test_warning_below_threshold_is_none(self):
        g = WorldGraph(size_warning_threshold=5)
        g.apply_update(AddEntity(name="A"), cycle=1)
        assert g.emit_size_warning() is None

    def test_warning_fires_when_threshold_crossed(self):
        g = WorldGraph(size_warning_threshold=2)
        g.apply_update(AddEntity(name="A"), cycle=1)
        g.apply_update(AddEntity(name="B"), cycle=1)
        warning = g.emit_size_warning()
        assert warning is not None
        assert warning.entity_count == 2
        assert warning.threshold == 2
        assert "2 entities" in warning.message

    def test_warning_latches_until_count_drops(self):
        g = WorldGraph(size_warning_threshold=2)
        g.apply_update(AddEntity(name="A"), cycle=1)
        g.apply_update(AddEntity(name="B"), cycle=1)
        assert g.emit_size_warning() is not None
        # Subsequent calls without dropping below threshold return None.
        assert g.emit_size_warning() is None
        # Drop below threshold.
        g.apply_update(RemoveEntity(name="A"), cycle=2)
        assert g.emit_size_warning() is None
        # Cross again — fires once more.
        g.apply_update(AddEntity(name="C"), cycle=3)
        warning = g.emit_size_warning()
        assert warning is not None

    def test_default_threshold_constant(self):
        g = WorldGraph()
        assert g.size_warning_threshold == DEFAULT_SIZE_WARNING_THRESHOLD


# ---------------------------------------------------------------------------
# Open vocabulary — relation types are not validated
# ---------------------------------------------------------------------------


class TestOpenVocabulary:
    def test_novel_type_accepted_without_registration(self):
        g = WorldGraph()
        g.apply_update(
            AddRelation(source="A", type="never_seen_before", target="B"),
            cycle=1,
        )
        assert "never_seen_before" in g.relation_types_seen

    def test_arbitrary_relation_type_strings_work(self):
        g = WorldGraph()
        weird_types = ["snake_case", "with-dashes", "with spaces", "UPPERCASE", "🤝"]
        for t in weird_types:
            g.apply_update(AddRelation(source="A", type=t, target="B"), cycle=1)
        assert g.relation_types_seen >= set(weird_types)


# ---------------------------------------------------------------------------
# Apply-then-resolve semantics
# ---------------------------------------------------------------------------


class TestApplyThenResolve:
    def test_query_sees_just_added_entity(self):
        """The cycle applies updates before resolving queries from the same
        output, so the entity can ask about something it just added."""
        g = WorldGraph()
        cycle = 1
        # Simulate the cycle: apply ops first, then resolve queries.
        g.apply_update(AddEntity(name="Fresh"), cycle=cycle)
        result = g.resolve_query(EntityQuery(name="Fresh"), cycle=cycle)
        assert result.found is True
        assert "Fresh" in result.entities
