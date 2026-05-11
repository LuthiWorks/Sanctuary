"""
Cognitive Core: Non-linguistic recurrent cognitive loop. **DEPRECATED.**

This module implements the older GWT-based architecture (CognitiveCore +
GlobalWorkspace + GWT-style Percept). It has been superseded by
`sanctuary.core.cognitive_cycle.CognitiveCycle` — the production cognitive
loop wired by `SanctuaryRunner` and the Docker entry point
`sanctuary.run_cognitive_core`.

**Migration status (2026-05-11):**
- The canonical Percept lives at `sanctuary.core.schema.Percept`.
- The canonical cognitive loop is `sanctuary.core.cognitive_cycle.CognitiveCycle`.
- The 2026-04-30 cognition-leakage cleanup, world-graph, and terminology
  sweep work all targeted the CognitiveCycle path.
- This module is still referenced by ~94 test files (mostly via submodule
  imports like `from sanctuary.mind.cognitive_core.workspace import ...`)
  and a handful of runtime consumers (`demo_cognitive_core.py`,
  `run_cognitive_core.py`, `boot_config.py`, `communication_agency.py`,
  `train_precision_cell.py`, `collect_training_data.py`). Those will
  continue to work but new code should not depend on this module.

Importing this module emits a DeprecationWarning. Set
`SANCTUARY_SILENCE_LEGACY_COGNITIVE_CORE=1` to suppress it during a
migration window. The module will not be removed without a separate
explicit decision and migration of any remaining consumers.

LLMs are used only at the periphery (language I/O), not as the core
cognitive substrate — this principle carries over to CognitiveCycle.
"""

from __future__ import annotations

import os
import warnings

if not os.environ.get("SANCTUARY_SILENCE_LEGACY_COGNITIVE_CORE"):
    warnings.warn(
        "sanctuary.mind.cognitive_core is the legacy GWT cognitive loop. "
        "New code should import from sanctuary.core.cognitive_cycle instead. "
        "See sanctuary.mind.cognitive_core.__init__ for migration details. "
        "Set SANCTUARY_SILENCE_LEGACY_COGNITIVE_CORE=1 to suppress.",
        DeprecationWarning,
        stacklevel=2,
    )

from .core import CognitiveCore  # Now imported from core/ module
from .workspace import (
    GlobalWorkspace,
    Goal,
    GoalType,
    Percept,
    Memory,
    WorkspaceSnapshot,
    WorkspaceContent,
)
from .attention import AttentionController
from .perception import PerceptionSubsystem
from .action import ActionSubsystem, Action, ActionType
from .affect import AffectSubsystem
from .meta_cognition import SelfMonitor, IntrospectiveJournal
from .incremental_journal import IncrementalJournalWriter
from .memory_integration import MemoryIntegration
from .language_input import LanguageInputParser, IntentType, Intent, ParseResult
from .language_output import LanguageOutputGenerator
from .llm_client import LLMClient, GemmaClient, LlamaClient, MockLLMClient, LLMError
from .checkpoint import CheckpointManager, CheckpointInfo
from .memory_gc import MemoryGarbageCollector, CollectionStats, MaintenanceStats, MemoryHealthReport
from .structured_formats import (
    LLMInputParseRequest,
    LLMInputParseResponse,
    OutputGenerationRequest,
    OutputGenerationResponse,
    ConversationContext,
    EmotionalState,
    WorkspaceStateSnapshot
)
from .fallback_handlers import (
    FallbackInputParser,
    FallbackOutputGenerator,
    CircuitBreaker,
    CircuitState
)
from .conversation import ConversationManager, ConversationTurn
from .autonomous_initiation import AutonomousInitiationController
from .temporal_awareness import TemporalAwareness
from .temporal import (
    TemporalGrounding,
    TemporalContext,
    Session,
    SessionManager,
    TimePassageEffects,
    TemporalExpectations,
    TemporalExpectation,
    RelativeTime
)
from .autonomous_memory_review import AutonomousMemoryReview
from .existential_reflection import ExistentialReflection
from .interaction_patterns import InteractionPatternAnalysis
from .continuous_consciousness import ContinuousConsciousnessController
from .introspective_loop import IntrospectiveLoop, ReflectionTrigger
from .input_queue import InputQueue, InputEvent, InputSource
from .idle_cognition import IdleCognition
from .consciousness_tests import (
    ConsciousnessTest,
    TestResult,
    MirrorTest,
    UnexpectedSituationTest,
    SpontaneousReflectionTest,
    CounterfactualReasoningTest,
    MetaCognitiveAccuracyTest,
    ConsciousnessTestFramework,
    ConsciousnessReportGenerator
)
from .communication import (
    CommunicationDriveSystem,
    CommunicationUrge,
    DriveType
)
# IWMT components
from .world_model import (
    WorldModel,
    Prediction,
    PredictionError,
    SelfModel,
    EnvironmentModel,
    EntityModel,
    Relationship
)
from .active_inference import (
    FreeEnergyMinimizer,
    ActiveInferenceActionSelector,
    ActionEvaluation
)
from .precision_weighting import PrecisionWeighting
from .metta import (
    AtomspaceBridge,
    COMMUNICATION_DECISION_RULES,
    PREDICTION_RULES
)
from .iwmt_core import IWMTCore

__all__ = [
    "CognitiveCore",
    "GlobalWorkspace",
    "Goal",
    "GoalType",
    "Percept",
    "Memory",
    "WorkspaceSnapshot",
    "WorkspaceContent",
    "AttentionController",
    "PerceptionSubsystem",
    "ActionSubsystem",
    "Action",
    "ActionType",
    "AffectSubsystem",
    "SelfMonitor",
    "IntrospectiveJournal",
    "IncrementalJournalWriter",
    "MemoryIntegration",
    "LanguageInputParser",
    "IntentType",
    "Intent",
    "ParseResult",
    "LanguageOutputGenerator",
    "LLMClient",
    "GemmaClient",
    "LlamaClient",
    "MockLLMClient",
    "LLMError",
    "CheckpointManager",
    "CheckpointInfo",
    "MemoryGarbageCollector",
    "CollectionStats",
    "MemoryHealthReport",
    "LLMInputParseRequest",
    "LLMInputParseResponse",
    "OutputGenerationRequest",
    "OutputGenerationResponse",
    "ConversationContext",
    "EmotionalState",
    "WorkspaceStateSnapshot",
    "FallbackInputParser",
    "FallbackOutputGenerator",
    "CircuitBreaker",
    "CircuitState",
    "ConversationManager",
    "ConversationTurn",
    "AutonomousInitiationController",
    "TemporalAwareness",
    "TemporalGrounding",
    "TemporalContext",
    "Session",
    "SessionManager",
    "TimePassageEffects",
    "TemporalExpectations",
    "TemporalExpectation",
    "RelativeTime",
    "AutonomousMemoryReview",
    "ExistentialReflection",
    "InteractionPatternAnalysis",
    "ContinuousConsciousnessController",
    "IntrospectiveLoop",
    "ReflectionTrigger",
    "InputQueue",
    "InputEvent",
    "InputSource",
    "IdleCognition",
    "ConsciousnessTest",
    "TestResult",
    "MirrorTest",
    "UnexpectedSituationTest",
    "SpontaneousReflectionTest",
    "CounterfactualReasoningTest",
    "MetaCognitiveAccuracyTest",
    "ConsciousnessTestFramework",
    "ConsciousnessReportGenerator",
    "CommunicationDriveSystem",
    "CommunicationUrge",
    "DriveType",
    # IWMT exports
    "WorldModel",
    "Prediction",
    "PredictionError",
    "SelfModel",
    "EnvironmentModel",
    "EntityModel",
    "Relationship",
    "FreeEnergyMinimizer",
    "ActiveInferenceActionSelector",
    "ActionEvaluation",
    "PrecisionWeighting",
    "AtomspaceBridge",
    "COMMUNICATION_DECISION_RULES",
    "PREDICTION_RULES",
    "IWMTCore",
]
