"""Context window budget allocation and compression.

Without management, the stream of thought will overflow any context window.
The ContextManager enforces token budgets per section of CognitiveInput
and applies layered compression to keep each cycle within bounds.

Strategy from PLAN.md:
- Inner speech: older cycles summarized, only most recent verbatim
- Self-model / world model: rewritten each cycle, not appended
- Memory surfacing: selective, top-K most relevant
- Percept batching: grouped and summarized when many arrive
- Scaffold signals: terse, structured form
- Adaptive budget: shifts toward percepts during conversation, toward
  self-reflection during idle cycles

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sanctuary.core.schema import CognitiveInput, Percept, SurfacedMemory


@dataclass
class BudgetConfig:
    """Token budget allocation for a single cognitive cycle.

    All values are approximate token counts. The ContextManager uses
    character-based estimation (1 token ~ 4 chars) until a real tokenizer
    is wired in.
    """

    # Fixed overhead
    system_prompt: int = 2000
    identity_charter: int = 500

    # Dynamic allocation (the pool)
    previous_thought: int = 500
    self_model: int = 300
    world_model_query_results: int = 500
    new_percepts: int = 800
    prediction_errors: int = 200
    surfaced_memories: int = 500
    scaffold_signals: int = 300
    emotional_temporal: int = 200

    # Total target
    total_target: int = 4000

    # Chars-per-token estimate (conservative)
    chars_per_token: int = 4

    def budget_bytes(self, section: str) -> int:
        """Get the byte budget for a named section."""
        tokens = getattr(self, section, 0)
        return tokens * self.chars_per_token


@dataclass
class CompressionStats:
    """Tracks what was compressed and by how much."""

    sections_compressed: list[str] = field(default_factory=list)
    original_chars: int = 0
    compressed_chars: int = 0

    @property
    def savings_ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return 1.0 - (self.compressed_chars / self.original_chars)


class ContextManager:
    """Manages context window budget for each cognitive cycle.

    The ContextManager takes a CognitiveInput and compresses it to fit
    within the configured token budget. Compression is lossy but
    prioritized: inner speech and percepts are preserved most faithfully,
    older/lower-significance content is summarized or dropped first.
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self._last_stats: Optional[CompressionStats] = None

    def compress(self, cognitive_input: CognitiveInput) -> CognitiveInput:
        """Compress cognitive input to fit within the context budget.

        Returns a new CognitiveInput (does not mutate the original).
        Compression is applied per-section according to budget allocation.
        """
        stats = CompressionStats()

        # Compress each section independently
        previous_thought = self._compress_previous_thought(
            cognitive_input.previous_thought, stats
        )
        percepts = self._compress_percepts(
            cognitive_input.new_percepts, stats
        )
        memories = self._compress_memories(
            cognitive_input.surfaced_memories, stats
        )
        self_model = self._compress_self_model(
            cognitive_input.self_model, stats
        )
        query_results = self._compress_query_results(
            cognitive_input.world_model_query_results, stats
        )
        scaffold = self._compress_scaffold(
            cognitive_input.scaffold_signals, stats
        )

        # Charter summary: truncate to identity_charter budget if needed
        charter_summary = cognitive_input.charter_summary
        charter_budget = self.config.budget_bytes("identity_charter")
        if len(charter_summary) > charter_budget:
            stats.sections_compressed.append("charter_summary")
            stats.original_chars += len(charter_summary)
            charter_summary = charter_summary[:charter_budget]
            stats.compressed_chars += charter_budget

        self._last_stats = stats

        return CognitiveInput(
            previous_thought=previous_thought,
            new_percepts=percepts,
            prediction_errors=cognitive_input.prediction_errors,
            surfaced_memories=memories,
            emotional_state=cognitive_input.emotional_state,
            temporal_context=cognitive_input.temporal_context,
            self_model=self_model,
            world_model_query_results=query_results,
            scaffold_signals=scaffold,
            experiential_state=cognitive_input.experiential_state,
            charter_summary=charter_summary,
        )

    def get_last_stats(self) -> Optional[CompressionStats]:
        """Return compression stats from the most recent compress() call."""
        return self._last_stats

    # -- Section compression methods --

    def _compress_previous_thought(self, thought, stats):
        """Truncate inner speech to budget. Most recent content preserved."""
        if thought is None:
            return None

        budget = self.config.budget_bytes("previous_thought")
        original = thought.inner_speech
        original_len = len(original)

        if original_len <= budget:
            return thought

        stats.sections_compressed.append("previous_thought")
        stats.original_chars += original_len

        # Keep the end (most recent content), summarize the rest
        truncated = "..." + original[-(budget - 3) :]
        stats.compressed_chars += len(truncated)

        from sanctuary.core.schema import PreviousThought

        return PreviousThought(
            inner_speech=truncated,
            predictions_made=thought.predictions_made[-3:],
            self_model_snapshot=thought.self_model_snapshot,
        )

    def _compress_percepts(self, percepts: list[Percept], stats) -> list[Percept]:
        """Batch and limit percepts to budget.

        When many percepts arrive, group by modality and summarize.
        """
        if not percepts:
            return percepts

        budget = self.config.budget_bytes("new_percepts")
        total_chars = sum(len(p.content) for p in percepts)

        if total_chars <= budget:
            return percepts

        stats.sections_compressed.append("new_percepts")
        stats.original_chars += total_chars

        # Strategy: group by modality, keep most recent per group
        by_modality: dict[str, list[Percept]] = {}
        for p in percepts:
            by_modality.setdefault(p.modality, []).append(p)

        compressed: list[Percept] = []
        per_modality_budget = budget // max(len(by_modality), 1)

        for modality, group in by_modality.items():
            group_chars = sum(len(p.content) for p in group)

            if group_chars <= per_modality_budget:
                compressed.extend(group)
            elif len(group) > 3:
                # Batch: summarize count, keep most recent
                summary = Percept(
                    modality=modality,
                    content=f"[{len(group) - 1} {modality} percepts batched]",
                    source="context_manager",
                    timestamp=group[-1].timestamp,
                )
                compressed.append(summary)
                compressed.append(group[-1])  # Keep most recent
            else:
                # Truncate individual contents
                for p in group:
                    truncated_content = p.content[:per_modality_budget // max(len(group), 1)]
                    compressed.append(
                        p.model_copy(update={"content": truncated_content})
                    )

        stats.compressed_chars += sum(len(p.content) for p in compressed)
        return compressed

    def _compress_memories(
        self, memories: list[SurfacedMemory], stats
    ) -> list[SurfacedMemory]:
        """Keep top-K memories by significance within budget."""
        if not memories:
            return memories

        budget = self.config.budget_bytes("surfaced_memories")
        total_chars = sum(len(m.content) for m in memories)

        if total_chars <= budget:
            return memories

        stats.sections_compressed.append("surfaced_memories")
        stats.original_chars += total_chars

        # Sort by significance (highest first), take as many as fit
        sorted_mems = sorted(memories, key=lambda m: m.significance, reverse=True)
        kept: list[SurfacedMemory] = []
        chars_used = 0

        for mem in sorted_mems:
            if chars_used + len(mem.content) <= budget:
                kept.append(mem)
                chars_used += len(mem.content)
            else:
                # Try to fit a truncated version
                remaining = budget - chars_used
                if remaining > 50:
                    truncated = mem.model_copy(
                        update={"content": mem.content[:remaining]}
                    )
                    kept.append(truncated)
                    chars_used += remaining
                break

        stats.compressed_chars += chars_used
        return kept

    def _compress_self_model(self, self_model, stats):
        """Trim self-model fields to budget.

        Self-model is rewritten each cycle (not appended), so this mostly
        handles unusually verbose states.
        """
        budget = self.config.budget_bytes("self_model")
        serialized = self_model.model_dump_json()

        if len(serialized) <= budget:
            return self_model

        stats.sections_compressed.append("self_model")
        stats.original_chars += len(serialized)

        from sanctuary.core.schema import SelfModel

        compressed = SelfModel(
            current_state=self_model.current_state[:200],
            recent_growth=self_model.recent_growth[:100],
            active_goals=self_model.active_goals[:5],
            uncertainties=self_model.uncertainties[:3],
            values=self_model.values[:5],
        )

        stats.compressed_chars += len(compressed.model_dump_json())
        return compressed

    def _compress_query_results(self, query_results, stats):
        """Limit world-graph query results to budget.

        Each result carries the original query plus entity / relation data.
        When over budget, drop the oldest results first — the entity sees
        fewer answers, but the answers it does see are intact rather than
        truncated mid-entity.
        """
        if not query_results:
            return query_results

        budget = self.config.budget_bytes("world_model_query_results")

        # Estimate current serialized size by summing per-result sizes.
        sizes = [len(r.model_dump_json()) for r in query_results]
        total = sum(sizes)

        if total <= budget:
            return query_results

        stats.sections_compressed.append("world_model_query_results")
        stats.original_chars += total

        # Drop oldest results (front of list) until under budget.
        # The list is ordered by issue cycle; the most recent are last.
        kept: list = []
        running = 0
        for r, size in zip(reversed(query_results), reversed(sizes)):
            if running + size > budget:
                break
            kept.append(r)
            running += size
        kept.reverse()

        stats.compressed_chars += running
        return kept

    def _compress_scaffold(self, scaffold, stats):
        """Scaffold signals are already terse. Truncate only if needed."""
        budget = self.config.budget_bytes("scaffold_signals")
        serialized = scaffold.model_dump_json()

        if len(serialized) <= budget:
            return scaffold

        stats.sections_compressed.append("scaffold_signals")
        stats.original_chars += len(serialized)

        from sanctuary.core.schema import ScaffoldSignals, CommunicationDriveSignal

        compressed = ScaffoldSignals(
            attention_highlights=scaffold.attention_highlights[:5],
            communication_drives=scaffold.communication_drives,
            goal_status=scaffold.goal_status,
            anomalies=scaffold.anomalies[:3],
        )

        stats.compressed_chars += len(compressed.model_dump_json())
        return compressed
