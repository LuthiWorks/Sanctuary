"""Experiential layer — CfC continuous-time neural dynamics.

CfC (Closed-form Continuous-depth) cells evolve state between model cycles,
providing the temporal thickness that IWMT requires but transformers
cannot provide alone. The scaffold trains these cells, then hands off
authority as they demonstrate reliable behavior.

The experiential layer has two categories of cells:
- **Foundational cells** (precision, affect, attention, goal) — present at boot,
  handle the felt quality of experience.
- **Knowledge cells** — acquired through the entity's lived experience, holding
  domain-specific expertise. None exist at birth; all are earned.

Both are managed through a dynamic registry that accepts new cell types at runtime.
"""

from sanctuary.experiential.affect_cell import AffectCell
from sanctuary.experiential.attention_cell import AttentionCell
from sanctuary.experiential.cell_factory import CellRequest, KnowledgeCellFactory
from sanctuary.experiential.cell_registry import CellProtocol, CellRegistry, InterCellConnection
from sanctuary.experiential.evolution import (
    ContinuousEvolutionLoop,
    EvolutionConfig,
    EvolutionSnapshot,
    PerceptEvent,
)
from sanctuary.experiential.goal_cell import GoalCell
from sanctuary.experiential.knowledge_cell import KnowledgeCell, KnowledgeCellConfig
from sanctuary.experiential.manager import ExperientialManager
from sanctuary.experiential.precision_cell import PrecisionCell
from sanctuary.experiential.trainer import CfCTrainer

__all__ = [
    "AffectCell",
    "AttentionCell",
    "CellProtocol",
    "CellRegistry",
    "CellRequest",
    "ContinuousEvolutionLoop",
    "EvolutionConfig",
    "EvolutionSnapshot",
    "GoalCell",
    "ExperientialManager",
    "InterCellConnection",
    "KnowledgeCell",
    "KnowledgeCellConfig",
    "KnowledgeCellFactory",
    "PerceptEvent",
    "PrecisionCell",
    "CfCTrainer",
]
