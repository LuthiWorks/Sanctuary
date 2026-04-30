"""Typed-relation world graph — structured cues for entity recall.

The graph is *not* primary storage. The entity's deep understanding lives
in the living weights. The graph helps it access specific facts quickly
and precisely — like a photo album, not a notebook. The entity writes
through ``CognitiveOutput.world_model_updates`` and asks via
``CognitiveOutput.world_model_queries``; results return one cycle later.

Open vocabulary: relation types are not enumerated. The entity introduces
new types as it needs them. ``relation_types_seen`` tracks what has been
introduced for query support, not for validation.

Retraction preserves history: ``retract_relation`` sets
``retracted_at_cycle`` rather than deleting. Queries return only active
relations (``retracted_at_cycle is None``); the storage keeps all of
them so the entity's prior beliefs are not lost. ``remove_entity`` is the
hard-delete escape hatch when an entity should not exist at all.

Persistence: a single JSON file, saved after every mutation, loaded at
boot. No change log, no snapshots, no replay infrastructure — the journal
records changes if the entity wants to. That's the entity's choice.

Memory-footprint guard: the graph emits a percept message when entity
count crosses a threshold. No auto-eviction. The entity decides what
to retract or remove.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    WorldModelOperation,
    WorldModelQuery,
    WorldQueryResult,
)

logger = logging.getLogger(__name__)


DEFAULT_SIZE_WARNING_THRESHOLD = 10_000


@dataclass
class SizeWarning:
    """Emitted by ``emit_size_warning`` when entity count crosses threshold.

    The cycle turns these into percepts the entity can see; the entity
    decides what to retract or remove. No auto-eviction.
    """

    entity_count: int
    threshold: int

    @property
    def message(self) -> str:
        return (
            f"World graph has grown to {self.entity_count} entities "
            f"(threshold {self.threshold}). You may want to retract "
            f"relations or entities you no longer need."
        )


class WorldGraph:
    """In-memory typed-relation graph with JSON persistence.

    Mutations apply immediately. Persistence is via ``save(path)`` (the
    cycle calls this after each mutation) and ``load(path)`` at boot.
    """

    def __init__(self, size_warning_threshold: int = DEFAULT_SIZE_WARNING_THRESHOLD):
        self.entities: dict[str, WorldGraphEntity] = {}
        self.relation_types_seen: set[str] = set()
        self.size_warning_threshold = size_warning_threshold
        # Latch so the warning fires once when first crossing the threshold,
        # not every cycle thereafter.
        self._size_warning_fired = False

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply_update(self, op: WorldModelOperation, cycle: int) -> None:
        """Apply one operation to the graph at ``cycle``.

        Operations are dispatched by type. Unknown shapes raise — Pydantic
        validation upstream should make that unreachable, but the
        explicit ``TypeError`` keeps the dispatch closed.
        """
        if isinstance(op, AddEntity):
            self._apply_add_entity(op, cycle)
        elif isinstance(op, AddRelation):
            self._apply_add_relation(op, cycle)
        elif isinstance(op, RetractRelation):
            self._apply_retract_relation(op, cycle)
        elif isinstance(op, RemoveEntity):
            self._apply_remove_entity(op)
        elif isinstance(op, UpdateProperty):
            self._apply_update_property(op, cycle)
        else:
            raise TypeError(f"Unknown world-graph operation: {type(op).__name__}")

    def _apply_add_entity(self, op: AddEntity, cycle: int) -> None:
        if op.name in self.entities:
            existing = self.entities[op.name]
            # Merge properties; don't overwrite first_seen_cycle.
            existing.properties.update(op.properties)
            existing.last_referenced_cycle = cycle
        else:
            self.entities[op.name] = WorldGraphEntity(
                name=op.name,
                properties=dict(op.properties),
                first_seen_cycle=cycle,
                last_referenced_cycle=cycle,
            )

    def _apply_add_relation(self, op: AddRelation, cycle: int) -> None:
        # Source entity must exist. If the entity asks to relate something
        # that hasn't been introduced, auto-create a placeholder rather
        # than failing — relations can describe partially-known things.
        if op.source not in self.entities:
            self.entities[op.source] = WorldGraphEntity(
                name=op.source,
                first_seen_cycle=cycle,
                last_referenced_cycle=cycle,
            )
        if op.target not in self.entities:
            self.entities[op.target] = WorldGraphEntity(
                name=op.target,
                first_seen_cycle=cycle,
                last_referenced_cycle=cycle,
            )

        source = self.entities[op.source]
        # If an active relation of the same shape already exists, update
        # confidence and source rather than appending a duplicate.
        for r in source.relations:
            if r.target == op.target and r.type == op.type and r.retracted_at_cycle is None:
                r.confidence = op.confidence
                if op.source_observation:
                    r.source = op.source_observation
                source.last_referenced_cycle = cycle
                self.relation_types_seen.add(op.type)
                return

        source.relations.append(
            Relation(
                type=op.type,
                target=op.target,
                asserted_at_cycle=cycle,
                confidence=op.confidence,
                source=op.source_observation,
            )
        )
        source.last_referenced_cycle = cycle
        self.relation_types_seen.add(op.type)

    def _apply_retract_relation(self, op: RetractRelation, cycle: int) -> None:
        """Mark active relations matching (source, type, target) as retracted.

        Multiple matches (shouldn't exist after dedup in add_relation, but
        defensive) all get retracted at the same cycle.
        """
        if op.source not in self.entities:
            return
        source = self.entities[op.source]
        retracted_any = False
        for r in source.relations:
            if (
                r.target == op.target
                and r.type == op.type
                and r.retracted_at_cycle is None
            ):
                r.retracted_at_cycle = cycle
                retracted_any = True
        if retracted_any:
            source.last_referenced_cycle = cycle

    def _apply_remove_entity(self, op: RemoveEntity) -> None:
        """Drop the entity entirely. Inbound relations from other entities
        are preserved as dangling references — querying the target by
        name will report not-found, but the source entity still has its
        record. The entity may want to retract those as a follow-up.
        """
        self.entities.pop(op.name, None)

    def _apply_update_property(self, op: UpdateProperty, cycle: int) -> None:
        if op.name not in self.entities:
            return
        entity = self.entities[op.name]
        entity.properties[op.key] = op.value
        entity.last_referenced_cycle = cycle

    # ------------------------------------------------------------------
    # Query resolution
    # ------------------------------------------------------------------

    def resolve_query(self, query: WorldModelQuery, cycle: int) -> WorldQueryResult:
        """Resolve one query against the current graph state.

        Updates ``last_referenced_cycle`` on entities the query touches —
        the act of asking is a form of attention. Queried-but-missing
        entities don't fault; the result has ``found=False``.
        """
        if isinstance(query, EntityQuery):
            return self._resolve_entity_query(query, cycle)
        if isinstance(query, RelationQuery):
            return self._resolve_relation_query(query)
        if isinstance(query, NeighborhoodQuery):
            return self._resolve_neighborhood_query(query, cycle)
        raise TypeError(f"Unknown world-graph query: {type(query).__name__}")

    def _resolve_entity_query(self, query: EntityQuery, cycle: int) -> WorldQueryResult:
        entity = self.entities.get(query.name)
        if entity is None:
            return WorldQueryResult(query=query, found=False)
        entity.last_referenced_cycle = cycle
        active_relations = [
            (entity.name, r) for r in entity.relations if r.retracted_at_cycle is None
        ]
        return WorldQueryResult(
            query=query,
            found=True,
            entities={entity.name: entity},
            relations=active_relations,
        )

    def _resolve_relation_query(self, query: RelationQuery) -> WorldQueryResult:
        pairs: list[tuple[str, Relation]] = []
        for entity in self.entities.values():
            for r in entity.relations:
                if r.type == query.type and r.retracted_at_cycle is None:
                    pairs.append((entity.name, r))
        # ``found`` is True iff at least one active relation of this type
        # exists. Empty result with ``found=False`` lets the entity
        # distinguish "no such relation in the graph" from "I asked
        # something else".
        return WorldQueryResult(
            query=query,
            found=bool(pairs),
            relations=pairs,
        )

    def _resolve_neighborhood_query(
        self, query: NeighborhoodQuery, cycle: int
    ) -> WorldQueryResult:
        if query.name not in self.entities:
            return WorldQueryResult(query=query, found=False)

        # BFS up to N hops. ``depth`` counts hops out from the seed:
        # depth=1 means seed + immediate neighbors (loop runs depth+1
        # times — once for the seed at hop 0, once per hop outward).
        visited: set[str] = set()
        frontier: list[str] = [query.name]
        relations: list[tuple[str, Relation]] = []

        for _ in range(query.depth + 1):
            next_frontier: list[str] = []
            for name in frontier:
                if name in visited:
                    continue
                visited.add(name)
                entity = self.entities.get(name)
                if entity is None:
                    continue
                entity.last_referenced_cycle = cycle
                for r in entity.relations:
                    if r.retracted_at_cycle is None:
                        relations.append((name, r))
                        if r.target not in visited:
                            next_frontier.append(r.target)
            frontier = next_frontier

        entities = {
            name: self.entities[name]
            for name in visited
            if name in self.entities
        }
        return WorldQueryResult(
            query=query,
            found=True,
            entities=entities,
            relations=relations,
        )

    # ------------------------------------------------------------------
    # Status / size
    # ------------------------------------------------------------------

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    def emit_size_warning(self) -> Optional[SizeWarning]:
        """Return a SizeWarning the first time entity_count crosses the
        threshold. Subsequent calls return None until the count drops
        below threshold and crosses again.
        """
        if self.entity_count >= self.size_warning_threshold:
            if not self._size_warning_fired:
                self._size_warning_fired = True
                return SizeWarning(
                    entity_count=self.entity_count,
                    threshold=self.size_warning_threshold,
                )
        else:
            self._size_warning_fired = False
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomically write the graph to ``path`` as JSON.

        Writes to ``path.tmp`` then renames, so a crash mid-write doesn't
        corrupt the existing file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entities": {
                name: entity.model_dump(mode="json")
                for name, entity in self.entities.items()
            },
            "relation_types_seen": sorted(self.relation_types_seen),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        """Load the graph from ``path``. Missing file is not an error —
        the graph stays empty (boot-from-scratch).

        Migration: the previous ``WorldModel`` format stored entities
        as ``{"name": {"properties": {...}}}`` with no relations, no
        cycle timestamps. ``WorldGraphEntity``'s defaults handle that
        format directly through Pydantic ``model_validate``.
        """
        path = Path(path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        # Forward-compat-ish: tolerate old format that's a bare
        # ``{"entities": {...}, "environment": {...}}`` from the old
        # WorldModel without a version field.
        entities_data = payload.get("entities", {})
        self.entities = {
            name: WorldGraphEntity.model_validate(
                {"name": name, **(data if isinstance(data, dict) else {})}
            )
            for name, data in entities_data.items()
        }

        # Rebuild relation_types_seen from current relations if it
        # wasn't persisted (older data).
        persisted_types = payload.get("relation_types_seen")
        if persisted_types is None:
            self.relation_types_seen = {
                r.type
                for entity in self.entities.values()
                for r in entity.relations
            }
        else:
            self.relation_types_seen = set(persisted_types)

        self._size_warning_fired = False  # Re-evaluate against fresh state.
