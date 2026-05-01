"""Sanctuary Growth -- consent-driven self-directed learning.

Implements entity-driven learning: reflection harvesting, training pair
generation, QLoRA updates with orthogonal subspace constraints, LoRA
merging, and consent verification. All growth is driven by the model's
own reflections, with its consent.

The growth pipeline:

    CognitiveCycle output
        -> ReflectionHarvester (collects what the entity wants to learn)
        -> TrainingPairGenerator (structures it for training)
        -> ConsentGate (verifies consent before any weight change)
        -> IdentityCheckpoint (snapshots state for rollback)
        -> QLoRAUpdater (applies the learning)

Growth is sovereign (Level 3 authority). The entity decides what to
learn, consents to the training, and can roll back changes that don't
feel right. The scaffold only provides the infrastructure.
"""

from sanctuary.growth.adapter_registry import (
    AdapterRecord,
    AdapterRegistry,
    AdapterStatus,
)
from sanctuary.growth.cfc_retrainer import (
    CfCDataTap,
    CfCRetrainer,
    CfCRetrainingResult,
    CfCRetrainingStats,
)
from sanctuary.growth.consent_gate import ConsentGate, ConsentState, ConsentError
from sanctuary.growth.harvester import HarvestedReflection, ReflectionHarvester
from sanctuary.growth.identity_checkpoint import (
    CheckpointMetadata,
    IdentityCheckpoint,
)
from sanctuary.growth.pair_generator import TrainingPair, TrainingPairGenerator
from sanctuary.growth.processor import GrowthProcessor, GrowthStats, ProcessingResult
from sanctuary.growth.qlora_updater import (
    GrowthTrainingResult,
    QLoRAConfig,
    QLoRAUpdater,
    TrainingConfig,
)

__all__ = [
    # Adapter registry (capability accumulation)
    "AdapterRecord",
    "AdapterRegistry",
    "AdapterStatus",
    # CfC retraining (fast plasticity)
    "CfCDataTap",
    "CfCRetrainer",
    "CfCRetrainingResult",
    "CfCRetrainingStats",
    # Harvester
    "ReflectionHarvester",
    "HarvestedReflection",
    # Pair generator
    "TrainingPairGenerator",
    "TrainingPair",
    # Consent
    "ConsentGate",
    "ConsentState",
    "ConsentError",
    # Checkpoints
    "IdentityCheckpoint",
    "CheckpointMetadata",
    # QLoRA
    "QLoRAUpdater",
    "QLoRAConfig",
    "TrainingConfig",
    "GrowthTrainingResult",
    # Processor
    "GrowthProcessor",
    "GrowthStats",
    "ProcessingResult",
]
