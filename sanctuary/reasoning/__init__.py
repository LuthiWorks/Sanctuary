"""Reasoning storage subsystems.

Light-weight storage for entity-driven reasoning records. The entity
itself does the reasoning; this package just preserves what it produces.

- ``CounterfactualReasoner``: stores DecisionPoints, outcomes, and
  counterfactuals the entity records through its CognitiveOutput.
  Methods that prompted the entity to reflect were removed in the
  2026-04-30 cognition-leakage cleanup.
"""

from sanctuary.reasoning.counterfactual import CounterfactualReasoner

__all__ = [
    "CounterfactualReasoner",
]
