"""Experiential layer manager — coordinates CfC cells.

The manager sits between the cognitive cycle and the individual CfC cells.
Each cycle, it:
    1. Receives the current cognitive state (arousal, errors, etc.)
    2. Steps all CfC cells forward (evolving hidden states)
    3. Returns a summary of experiential state for the entity's input

The manager uses a dynamic registry to track all cells — both foundational
(precision, affect, attention, goal) and knowledge cells (acquired through
the entity's lived experience). All cells are treated uniformly. New cells
can be registered at runtime.

Inter-cell connections: affect feeds precision (emotional arousal modulates
precision), attention informs goal prioritization (salient goals get boosted).
Knowledge cells participate in this same network with entity-specified topology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.experiential.affect_cell import AffectCell, AffectCellConfig
from sanctuary.experiential.attention_cell import AttentionCell, AttentionCellConfig
from sanctuary.experiential.cell_registry import CellRegistry, InterCellConnection
from sanctuary.experiential.evolution import (
    ContinuousEvolutionLoop,
    EvolutionConfig,
    EvolutionSnapshot,
    PerceptEvent,
)
from sanctuary.experiential.goal_cell import GoalCell, GoalCellConfig
from sanctuary.experiential.knowledge_cell import KnowledgeCell
from sanctuary.experiential.precision_cell import PrecisionCell, PrecisionCellConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperientialState:
    """The experiential layer's output for one cognitive cycle.

    This gets included in the entity's CognitiveInput as a compact summary
    of what the continuous-time cells are experiencing.
    """

    precision_weight: float
    affect_vad: tuple[float, float, float] = (0.0, 0.2, 0.5)
    attention_salience: float = 0.5
    goal_adjustment: float = 0.0
    hidden_state_norms: dict[str, float] = field(default_factory=dict)
    cell_active: dict[str, bool] = field(default_factory=dict)
    knowledge_signals: dict[str, list[float]] = field(default_factory=dict)


# Authority function names for foundational cells
_AUTH = {
    "precision": "experiential_precision",
    "affect": "experiential_affect",
    "attention": "experiential_attention",
    "goal": "experiential_goal",
}


class ExperientialManager:
    """Coordinates CfC cells and manages their integration with the scaffold.

    The manager handles:
    - Dynamic cell registry (foundational + knowledge cells)
    - Stepping all cells forward each cycle
    - Inter-cell connections (affect->precision, attention->goals, + knowledge cells)
    - Authority management (scaffold vs CfC per cell)
    - State persistence across sessions
    - Monitoring and health checks
    """

    AUTHORITY_FUNCTION = "experiential_precision"  # backward compat

    def __init__(
        self,
        authority: Optional[AuthorityManager] = None,
        precision_config: Optional[PrecisionCellConfig] = None,
        affect_config: Optional[AffectCellConfig] = None,
        attention_config: Optional[AttentionCellConfig] = None,
        goal_config: Optional[GoalCellConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
    ):
        self.authority = authority or AuthorityManager()

        # Dynamic cell registry
        self._registry = CellRegistry()

        # Create foundational cells
        self.precision_cell = PrecisionCell(precision_config)
        self.affect_cell = AffectCell(affect_config)
        self.attention_cell = AttentionCell(attention_config)
        self.goal_cell = GoalCell(goal_config)

        # Register foundational cells
        self._registry.register(
            "precision", self.precision_cell, category="foundational",
            output_names=["precision_weight"],
        )
        self._registry.register(
            "affect", self.affect_cell, category="foundational",
            output_names=["valence", "arousal", "dominance"],
        )
        self._registry.register(
            "attention", self.attention_cell, category="foundational",
            output_names=["salience"],
        )
        self._registry.register(
            "goal", self.goal_cell, category="foundational",
            output_names=["adjustment"],
        )

        # Register foundational inter-cell connections
        self._registry.add_connection(InterCellConnection(
            source_cell="affect", target_cell="precision",
            source_output="arousal", target_input="arousal",
        ))
        self._registry.add_connection(InterCellConnection(
            source_cell="attention", target_cell="goal",
            source_output="salience", target_input="congruence_boost",
        ))

        # Continuous evolution loop (optional, started explicitly)
        self._evolution_config = evolution_config
        self._evolution_loop: Optional[ContinuousEvolutionLoop] = None

        self._initialized = True

        # Register authority for each foundational cell if not present
        for name, func in _AUTH.items():
            if func not in self.authority.get_all_levels():
                self.authority.set_level(
                    func,
                    AuthorityLevel.SCAFFOLD_ONLY,
                    reason=f"CfC {name} cell initialized but untrained",
                )

        logger.info(
            "ExperientialManager initialized with %d cells (%d foundational)",
            self._registry.cell_count,
            self._registry.foundational_count,
        )

    @property
    def registry(self) -> CellRegistry:
        """The dynamic cell registry."""
        return self._registry

    def step(
        self,
        arousal: float,
        prediction_error: float,
        base_precision: float,
        scaffold_precision: float,
        # Affect inputs
        percept_valence_delta: float = 0.0,
        percept_arousal_delta: float = 0.0,
        llm_emotion_shift: float = 0.0,
        scaffold_vad: tuple[float, float, float] = (0.0, 0.2, 0.5),
        # Attention inputs
        goal_relevance: float = 0.0,
        novelty: float = 0.0,
        emotional_salience: float = 0.0,
        recency: float = 0.0,
        scaffold_salience: float = 0.5,
        # Goal inputs
        cycles_stalled_norm: float = 0.0,
        deadline_urgency: float = 0.0,
        emotional_congruence: float = 0.0,
        scaffold_goal_adj: float = 0.0,
    ) -> ExperientialState:
        """Step the experiential layer forward one cycle.

        All cells are always stepped (they need temporal continuity).
        Blending with scaffold depends on each cell's authority level.

        Inter-cell connections:
        - Affect arousal feeds into precision (emotional modulation)
        - Attention salience feeds into goal adjustment (salient goals boosted)
        """
        # 1. Step affect cell first (feeds into precision)
        cfc_v, cfc_a, cfc_d = self.affect_cell.step(
            percept_valence_delta=percept_valence_delta,
            percept_arousal_delta=percept_arousal_delta,
            llm_emotion_shift=llm_emotion_shift,
        )
        affect_level = self.authority.level(_AUTH["affect"])
        blended_vad = (
            self._blend(scaffold_vad[0], cfc_v, affect_level),
            self._blend(scaffold_vad[1], cfc_a, affect_level),
            self._blend(scaffold_vad[2], cfc_d, affect_level),
        )

        # 2. Step precision cell (uses affect arousal as inter-cell connection)
        # Override arousal with blended affect arousal for cross-cell influence
        effective_arousal = blended_vad[1]
        cfc_precision = self.precision_cell.step(
            arousal=effective_arousal,
            prediction_error=prediction_error,
            base_precision=base_precision,
        )
        precision_level = self.authority.level(_AUTH["precision"])
        blended_precision = self._blend(scaffold_precision, cfc_precision, precision_level)

        # 3. Step attention cell
        cfc_salience = self.attention_cell.step(
            goal_relevance=goal_relevance,
            novelty=novelty,
            emotional_salience=emotional_salience,
            recency=recency,
        )
        attention_level = self.authority.level(_AUTH["attention"])
        blended_salience = self._blend(scaffold_salience, cfc_salience, attention_level)

        # 4. Step goal cell (uses attention salience as inter-cell connection)
        # Boost emotional congruence with attention salience for cross-cell influence
        effective_congruence = emotional_congruence + 0.1 * blended_salience
        cfc_goal_adj = self.goal_cell.step(
            cycles_stalled_norm=cycles_stalled_norm,
            deadline_urgency=deadline_urgency,
            emotional_congruence=effective_congruence,
        )
        goal_level = self.authority.level(_AUTH["goal"])
        blended_goal_adj = self._blend(scaffold_goal_adj, cfc_goal_adj, goal_level)

        # 5. Step all knowledge cells
        knowledge_signals: dict[str, list[float]] = {}
        for name, reg in self._registry.knowledge_cells():
            try:
                cell = reg.cell
                if isinstance(cell, KnowledgeCell):
                    # Build inputs from inter-cell connections
                    cell_kwargs: dict[str, float] = {}
                    for conn in self._registry.get_inputs_for(name):
                        # Map connected cell outputs to this cell's inputs
                        if conn.source_cell == "affect":
                            cell_kwargs["input_0"] = blended_vad[0]  # valence
                            cell_kwargs["input_1"] = blended_vad[1]  # arousal
                        elif conn.source_cell == "attention":
                            cell_kwargs["input_0"] = blended_salience
                        elif conn.source_cell == "precision":
                            cell_kwargs["input_0"] = blended_precision
                        elif conn.source_cell == "goal":
                            cell_kwargs["input_0"] = blended_goal_adj
                        elif conn.source_cell in knowledge_signals:
                            # Feed from another knowledge cell
                            src_outputs = knowledge_signals[conn.source_cell]
                            for i, v in enumerate(src_outputs):
                                cell_kwargs[f"input_{i}"] = v

                    outputs = cell.step(**cell_kwargs)
                    knowledge_signals[name] = outputs
            except Exception as e:
                logger.warning("Knowledge cell '%s' step failed: %s", name, e)
                knowledge_signals[name] = []

        # Build hidden state norms from all cells
        hidden_state_norms: dict[str, float] = {
            "precision": self.precision_cell.get_summary()["hidden_state_norm"],
            "affect": self.affect_cell.get_summary()["hidden_state_norm"],
            "attention": self.attention_cell.get_summary()["hidden_state_norm"],
            "goal": self.goal_cell.get_summary()["hidden_state_norm"],
        }
        for name, reg in self._registry.knowledge_cells():
            summary = reg.cell.get_summary()
            hidden_state_norms[name] = summary.get("hidden_state_norm", 0.0)

        # Build cell_active from all cells
        cell_active: dict[str, bool] = {
            "precision": precision_level >= AuthorityLevel.MODEL_ADVISES,
            "affect": affect_level >= AuthorityLevel.MODEL_ADVISES,
            "attention": attention_level >= AuthorityLevel.MODEL_ADVISES,
            "goal": goal_level >= AuthorityLevel.MODEL_ADVISES,
        }
        for name, _ in self._registry.knowledge_cells():
            cell_active[name] = True  # Knowledge cells are always active

        return ExperientialState(
            precision_weight=blended_precision,
            affect_vad=blended_vad,
            attention_salience=blended_salience,
            goal_adjustment=blended_goal_adj,
            hidden_state_norms=hidden_state_norms,
            cell_active=cell_active,
            knowledge_signals=knowledge_signals,
        )

    def _blend(
        self,
        scaffold_value: float,
        cfc_value: float,
        level: AuthorityLevel,
    ) -> float:
        """Blend scaffold and CfC outputs based on authority level.

        SCAFFOLD_ONLY (0): 100% scaffold
        MODEL_ADVISES (1):   75% scaffold, 25% CfC
        MODEL_GUIDES (2):    25% scaffold, 75% CfC
        MODEL_CONTROLS (3):  100% CfC
        """
        weights = {
            AuthorityLevel.SCAFFOLD_ONLY: (1.0, 0.0),
            AuthorityLevel.MODEL_ADVISES: (0.75, 0.25),
            AuthorityLevel.MODEL_GUIDES: (0.25, 0.75),
            AuthorityLevel.MODEL_CONTROLS: (0.0, 1.0),
        }
        scaffold_w, cfc_w = weights[level]
        return scaffold_value * scaffold_w + cfc_value * cfc_w

    # -- Authority management per cell --

    def promote(self, cell_name: str, reason: str = "") -> AuthorityLevel:
        """Promote a cell's authority level."""
        func = _AUTH[cell_name]
        new_level = self.authority.promote(func, reason)
        logger.info("CfC %s promoted to %s: %s", cell_name, AuthorityLevel(new_level).name, reason)
        return new_level

    def demote(self, cell_name: str, reason: str = "") -> AuthorityLevel:
        """Demote a cell's authority level."""
        func = _AUTH[cell_name]
        new_level = self.authority.demote(func, reason)
        logger.info("CfC %s demoted to %s: %s", cell_name, AuthorityLevel(new_level).name, reason)
        return new_level

    # Backward-compatible aliases
    def promote_precision(self, reason: str = "") -> AuthorityLevel:
        return self.promote("precision", reason)

    def demote_precision(self, reason: str = "") -> AuthorityLevel:
        return self.demote("precision", reason)

    # -- Continuous evolution --

    async def start_evolution(self, config: Optional[EvolutionConfig] = None):
        """Start the continuous evolution background loop.

        CfC cells will evolve between model cycles at adaptive tick rates.
        """
        if self._evolution_loop is not None and self._evolution_loop.running:
            return
        cfg = config or self._evolution_config
        self._evolution_loop = ContinuousEvolutionLoop(self, cfg)
        await self._evolution_loop.start()

    async def stop_evolution(self):
        """Stop the continuous evolution loop."""
        if self._evolution_loop is not None:
            await self._evolution_loop.stop()

    def feed_percept(self, event: PerceptEvent):
        """Feed a percept event to the evolution loop for inter-cycle processing."""
        if self._evolution_loop is not None and self._evolution_loop.running:
            self._evolution_loop.feed_percept(event)

    def update_evolution_context(self, **kwargs):
        """Update scaffold context for the evolution loop after each model cycle."""
        if self._evolution_loop is not None:
            self._evolution_loop.update_context(**kwargs)

    def evolution_snapshot(self) -> Optional[EvolutionSnapshot]:
        """Read accumulated CfC state from the evolution loop.

        Returns None if the evolution loop is not running.
        """
        if self._evolution_loop is not None and self._evolution_loop.running:
            return self._evolution_loop.snapshot()
        return None

    @property
    def evolution_running(self) -> bool:
        return self._evolution_loop is not None and self._evolution_loop.running

    def reset(self):
        """Reset all CfC cell hidden states."""
        self._registry.reset_all()
        logger.info("Experiential layer reset (%d cells)", self._registry.cell_count)

    def get_status(self) -> dict:
        """Status of all experiential cells for monitoring."""
        status: dict = {}

        # Foundational cells with authority info
        for name, func in _AUTH.items():
            status[name] = {
                "authority": self.authority.level(func).name,
                "summary": getattr(self, f"{name}_cell").get_summary(),
                "category": "foundational",
            }

        # Knowledge cells
        for name, reg in self._registry.knowledge_cells():
            status[name] = {
                "authority": "SELF_DIRECTED",
                "summary": reg.cell.get_summary(),
                "category": "knowledge",
                "domain": reg.domain,
            }

        if self._evolution_loop is not None:
            status["evolution"] = {
                "running": self._evolution_loop.running,
                "tick_ms": self._evolution_loop.current_tick_ms,
            }

        status["registry"] = self._registry.get_registry_metadata()
        return status

    def save(self, directory: Path):
        """Save all cell states and registry to directory."""
        directory.mkdir(parents=True, exist_ok=True)
        # Save via registry (handles all cells uniformly)
        self._registry.save(directory)
        logger.info(
            "Experiential layer saved to %s (%d cells)",
            directory,
            self._registry.cell_count,
        )

    def load(self, directory: Path):
        """Load foundational cell states from directory.

        Knowledge cells are loaded via the registry metadata.
        """
        # Load foundational cells (backward compatible)
        foundational_cells = {
            "precision": (PrecisionCell, "precision_cell"),
            "affect": (AffectCell, "affect_cell"),
            "attention": (AttentionCell, "attention_cell"),
            "goal": (GoalCell, "goal_cell"),
        }
        for name, (cls, attr) in foundational_cells.items():
            # Try new registry layout first, then legacy flat layout
            cell_path = directory / name / "cell.pt"
            legacy_path = directory / f"{name}_cell.pt"
            path = cell_path if cell_path.exists() else legacy_path

            if path.exists():
                loaded_cell = cls.load(path)
                setattr(self, attr, loaded_cell)
                # Update the registry reference
                if self._registry.has(name):
                    self._registry.get(name).cell = loaded_cell
                logger.info("Loaded %s cell from %s", name, path)
            else:
                logger.warning("No saved %s cell at %s or %s", name, cell_path, legacy_path)

        # Load knowledge cells from registry metadata
        meta_path = directory / "registry_meta.pt"
        if meta_path.exists():
            self._load_knowledge_cells(directory, meta_path)

    def _load_knowledge_cells(self, directory: Path, meta_path: Path) -> None:
        """Load knowledge cells from registry metadata."""
        import torch
        # registry_meta.pt holds only plain dicts/lists/str/float (see
        # CellRegistry.save), so weights_only=True is safe and blocks pickle RCE.
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)

        for name, cell_meta in meta.get("cells", {}).items():
            if cell_meta.get("category") != "knowledge":
                continue

            cell_path = directory / name / "cell.pt"
            if not cell_path.exists():
                logger.warning("Knowledge cell '%s' metadata found but no saved state at %s", name, cell_path)
                continue

            try:
                cell = KnowledgeCell.load(cell_path)
                if not self._registry.has(name):
                    self._registry.register(
                        name=name,
                        cell=cell,
                        category="knowledge",
                        domain=cell_meta.get("domain", ""),
                        input_names=cell_meta.get("input_names", []),
                        output_names=cell_meta.get("output_names", []),
                        metadata=cell_meta.get("metadata", {}),
                    )
                else:
                    self._registry.get(name).cell = cell
                logger.info("Loaded knowledge cell '%s' from %s", name, cell_path)
            except Exception as e:
                logger.error("Failed to load knowledge cell '%s': %s", name, e)

        # Restore connections
        for conn_meta in meta.get("connections", []):
            try:
                from sanctuary.experiential.cell_registry import InterCellConnection
                conn = InterCellConnection(
                    source_cell=conn_meta["source_cell"],
                    target_cell=conn_meta["target_cell"],
                    source_output=conn_meta["source_output"],
                    target_input=conn_meta["target_input"],
                    weight=conn_meta.get("weight", 1.0),
                )
                # Only add if both cells exist and connection not already present
                if (
                    self._registry.has(conn.source_cell)
                    and self._registry.has(conn.target_cell)
                ):
                    existing = self._registry.get_connections()
                    already_exists = any(
                        c.source_cell == conn.source_cell
                        and c.target_cell == conn.target_cell
                        and c.source_output == conn.source_output
                        for c in existing
                    )
                    if not already_exists:
                        self._registry.add_connection(conn)
            except (KeyError, Exception) as e:
                logger.warning("Failed to restore connection: %s", e)
