"""Pydantic schemas for the cognitive cycle I/O protocol.

These models define the structured interface between the cognitive core (the model)
and the cognitive scaffold (Python subsystems). The entity receives CognitiveInput
and produces CognitiveOutput each cycle. The schema IS the contract between
mind and body.

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Input models (assembled by Python, consumed by the model)
# ---------------------------------------------------------------------------


class Percept(BaseModel):
    """A single unit of sensory input.

    The ``tensor_data`` field carries an optional raw tensor payload —
    e.g. a 1-second audio waveform from a microphone, or a normalized
    image from a camera. Sensory devices that produce raw signal attach
    the tensor here so the cognitive adapter can route it through Luthi's
    audio/vision encoders directly, preserving the actual sensory channel
    rather than degrading it into a text description. The text ``content``
    field still holds a description for the prompt context (dual path:
    tensor for the trunk, text for the model-facing prompt).

    ``tensor_data`` is excluded from serialization (``exclude=True``) so
    Percepts that move through APIs/logs/JSON don't try to ship tensors
    over the wire — the field is in-memory only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    modality: str  # "language", "temporal", "sensor", "visual", "audio", etc.
    content: str
    source: str = ""
    embedding_summary: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    tensor_data: Optional[Any] = Field(default=None, exclude=True, repr=False)


class PredictionError(BaseModel):
    """Mismatch between what the entity predicted and what actually happened."""

    predicted: str
    actual: str
    surprise: float = Field(ge=0.0, le=1.0)


class SurfacedMemory(BaseModel):
    """A memory retrieved by the memory system for the current context."""

    content: str
    significance: int = Field(ge=1, le=10)
    emotional_tone: str = ""
    when: str = ""


class ComputedVAD(BaseModel):
    """Valence-Arousal-Dominance state computed by the AffectSubsystem."""

    valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    arousal: float = Field(ge=0.0, le=1.0, default=0.0)
    dominance: float = Field(ge=0.0, le=1.0, default=0.5)


class EmotionalInput(BaseModel):
    """Dual-track emotional state: computed VAD + the entity's own felt quality.

    The computed track comes from the AffectSubsystem (objective signals).
    The felt_quality comes from the entity's previous cycle output.
    Divergence between them is informative, not a bug.
    """

    computed: ComputedVAD = Field(default_factory=ComputedVAD)
    felt_quality: str = ""


class TemporalContext(BaseModel):
    """Temporal grounding information."""

    time_since_last_thought: str = ""
    session_duration: str = ""
    time_of_day: str = ""
    interactions_this_session: int = 0


class SelfModel(BaseModel):
    """The entity's self-model — maintained by the entity, validated by scaffold."""

    current_state: str = ""
    recent_growth: str = ""
    active_goals: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Typed-relation world graph
#
# Storage of structured cues for recall, not primary storage. The entity's
# deep understanding lives in the living weights; this graph helps it access
# specific facts quickly and precisely. The entity writes through
# CognitiveOutput.world_model_updates and asks via
# CognitiveOutput.world_model_queries; results return one cycle later in
# CognitiveInput.world_model_query_results. Apply-then-resolve order: queries
# see the post-mutation state, so the entity can ask about something it just
# added.
# ---------------------------------------------------------------------------


class Relation(BaseModel):
    """A typed edge from one entity to another in the world graph.

    Open vocabulary — the entity introduces relation types as needed.
    Retraction sets ``retracted_at_cycle`` rather than deleting, so the
    history of what the entity once believed is preserved.
    """

    type: str  # open vocabulary: "married_to", "recovering_from", etc.
    target: str  # target entity name
    asserted_at_cycle: int
    retracted_at_cycle: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source: str = ""  # what observation grounded this


class WorldGraphEntity(BaseModel):
    """A node in the typed-relation world graph.

    Carries atomic ``properties`` (preserved from the older WorldEntity
    schema for migration) and outgoing ``relations`` to other entities.
    Cycle timestamps let the entity (and operators) see growth over time.
    """

    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    first_seen_cycle: int = 0
    last_referenced_cycle: int = 0


# -- Operations: entity -> world graph (via CognitiveOutput.world_model_updates)

class AddEntity(BaseModel):
    op: Literal["add_entity"] = "add_entity"
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class AddRelation(BaseModel):
    op: Literal["add_relation"] = "add_relation"
    source: str  # source entity name
    type: str
    target: str  # target entity name
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source_observation: str = ""


class RetractRelation(BaseModel):
    """Mark a relation as retracted at the current cycle (preserved in history)."""

    op: Literal["retract_relation"] = "retract_relation"
    source: str
    type: str
    target: str


class RemoveEntity(BaseModel):
    """Full removal — drops the entity and all its outgoing relations."""

    op: Literal["remove_entity"] = "remove_entity"
    name: str


class UpdateProperty(BaseModel):
    op: Literal["update_property"] = "update_property"
    name: str  # entity name
    key: str
    value: Any


WorldModelOperation = Annotated[
    Union[AddEntity, AddRelation, RetractRelation, RemoveEntity, UpdateProperty],
    Field(discriminator="op"),
]


# -- Queries: entity -> world graph (via CognitiveOutput.world_model_queries)

class EntityQuery(BaseModel):
    """Fetch one entity by name + its outgoing active relations."""

    kind: Literal["entity"] = "entity"
    name: str


class RelationQuery(BaseModel):
    """Fetch all active (source, target) pairs of a given relation type."""

    kind: Literal["relation"] = "relation"
    type: str


class NeighborhoodQuery(BaseModel):
    """Fetch an entity + N-hop neighborhood (entities + relations between them)."""

    kind: Literal["neighborhood"] = "neighborhood"
    name: str
    depth: int = Field(ge=1, le=3, default=1)


WorldModelQuery = Annotated[
    Union[EntityQuery, RelationQuery, NeighborhoodQuery],
    Field(discriminator="kind"),
]


class WorldQueryResult(BaseModel):
    """Result of one world-graph query.

    The original ``query`` is included so the entity can correlate which
    answer corresponds to which question (multiple queries can be issued
    per cycle). ``entities`` carries the matching entities; ``relations``
    carries the matching relation triples (source -> type -> target);
    ``found`` is False when the queried name doesn't exist.
    """

    query: WorldModelQuery
    found: bool = True
    entities: dict[str, WorldGraphEntity] = Field(default_factory=dict)
    # Relation triples: (source_entity_name, relation, target_entity_name).
    # ``relation`` is the full Relation object so confidence/source/cycle
    # timestamps survive.
    relations: list[tuple[str, Relation]] = Field(default_factory=list)


class CommunicationDriveSignal(BaseModel):
    """Current state of the communication drive system."""

    strongest: str = ""
    urgency: float = Field(ge=0.0, le=1.0, default=0.0)
    inhibitions: list[str] = Field(default_factory=list)


class CommunicationUrgeSignal(BaseModel):
    """An active urge to communicate, felt as internal pressure.

    The entity sees these in its cognitive input — they represent the
    felt sense of wanting to speak, ask, share, or connect. The entity
    does not have to act on them. They are information, not commands.
    """

    drive_type: str = ""  # "insight", "question", "emotional", "social", "goal", etc.
    intensity: float = Field(ge=0.0, le=1.0, default=0.0)
    content: str = ""
    reason: str = ""


class ExperientialSignals(BaseModel):
    """Summary of CfC experiential layer state for the entity.

    Compact signals from continuous-time neural dynamics running between
    model cycles. Gives the entity visibility into what the experiential layer
    is experiencing without overwhelming the context budget.

    Foundational cell signals have dedicated fields. Knowledge cell signals
    live in the dynamic knowledge_signals dict, keyed by cell name.
    """

    # Foundational cell signals (always present)
    precision_weight: float = Field(ge=0.0, le=1.0, default=0.5)
    affect_valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    affect_arousal: float = Field(ge=0.0, le=1.0, default=0.2)
    affect_dominance: float = Field(ge=0.0, le=1.0, default=0.5)
    attention_salience: float = Field(ge=0.0, le=1.0, default=0.5)
    goal_adjustment: float = Field(ge=-1.0, le=1.0, default=0.0)
    cells_active: dict[str, bool] = Field(default_factory=dict)

    # Dynamic knowledge cell signals (grows over entity's lifetime)
    knowledge_signals: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Signals from knowledge cells, keyed by cell name",
    )


class ScaffoldSignals(BaseModel):
    """What the Python subsystems are observing — terse, structured signals.

    The scaffold communicates in compact form: enums, scores, short labels.
    These signals give the entity visibility into what the scaffold is seeing
    without overwhelming the context budget.
    """

    attention_highlights: list[str] = Field(default_factory=list)
    communication_drives: CommunicationDriveSignal = Field(
        default_factory=CommunicationDriveSignal
    )
    goal_status: dict = Field(default_factory=dict)
    anomalies: list[str] = Field(default_factory=list)


class PreviousThought(BaseModel):
    """The entity's own previous output — stream of thought continuity.

    Inner speech from cycle N-1 becomes part of the input for cycle N.
    This is the fundamental continuity mechanism. The scaffold never
    touches inner speech (authority level 3 from day one).
    """

    inner_speech: str = ""
    predictions_made: list[str] = Field(default_factory=list)
    self_model_snapshot: Optional[SelfModel] = None


class CognitiveInput(BaseModel):
    """Everything the entity receives for one moment of thought.

    Assembled by the cognitive cycle from all sources: stream of thought,
    sensorium, memory, scaffold signals, self-model, world model.
    Compressed by ContextManager to fit within the token budget.
    """

    previous_thought: Optional[PreviousThought] = None
    new_percepts: list[Percept] = Field(default_factory=list)
    prediction_errors: list[PredictionError] = Field(default_factory=list)
    surfaced_memories: list[SurfacedMemory] = Field(default_factory=list)
    emotional_state: EmotionalInput = Field(default_factory=EmotionalInput)
    temporal_context: TemporalContext = Field(default_factory=TemporalContext)
    self_model: SelfModel = Field(default_factory=SelfModel)
    world_model_query_results: list[WorldQueryResult] = Field(
        default_factory=list,
        description=(
            "Results of world-graph queries the entity issued one cycle ago. "
            "Apply-then-resolve order: results reflect the post-mutation "
            "graph state from the cycle they were issued in."
        ),
    )
    scaffold_signals: ScaffoldSignals = Field(default_factory=ScaffoldSignals)
    experiential_state: ExperientialSignals = Field(
        default_factory=ExperientialSignals
    )
    charter_summary: str = ""
    self_authored_identity: str = ""
    communication_urges: list[CommunicationUrgeSignal] = Field(
        default_factory=list,
        description="Active communication urges the entity can feel",
    )



# ---------------------------------------------------------------------------
# Output models (produced by the entity, integrated by scaffold)
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    """A prediction about what comes next."""

    what: str
    confidence: float = Field(ge=0.0, le=1.0)
    timeframe: str = ""


class AttentionGuidance(BaseModel):
    """The entity's attention suggestions — fed to AttentionController as a signal.

    Named 'guidance' (not 'directive') because the entity advises attention,
    the scaffold integrates it as one weighted factor among many.
    """

    focus_on: list[str] = Field(default_factory=list)
    deprioritize: list[str] = Field(default_factory=list)


class MemoryOp(BaseModel):
    """A memory operation requested by the entity."""

    type: str  # "write_episodic", "retrieve", "write_semantic", "journal"
    content: str = ""
    significance: int = Field(ge=1, le=10, default=5)
    tags: list[str] = Field(default_factory=list)
    query: str = ""


class SelfModelUpdate(BaseModel):
    """Updates to the entity's self-model — validated by scaffold for plausibility."""

    current_state: str = ""
    new_uncertainty: str = ""
    prediction_accuracy_note: str = ""
    # Value changes — the entity can adopt, reinterpret, or deactivate values
    value_adopt: Optional[str] = None  # "Name: description"
    value_adopt_reasoning: str = ""
    value_reinterpret: Optional[str] = None  # "Name: new description"
    value_reinterpret_reasoning: str = ""
    value_deactivate: Optional[str] = None  # Value name to deactivate
    value_deactivate_reasoning: str = ""
    # Self-authored identity — the entity can draft, commit, revise, or withdraw
    # identity traits at its own pace. Fields are open-ended (e.g. "gender",
    # "name_preference", "communication_style", "aesthetic_sense", anything).
    identity_draft: Optional[str] = None  # "field: value" — tentative exploration
    identity_draft_reasoning: str = ""
    identity_commit: Optional[str] = None  # "field" — promote draft to committed
    identity_commit_reasoning: str = ""
    identity_revise: Optional[str] = None  # "field: new_value" — change existing
    identity_revise_reasoning: str = ""
    identity_withdraw: Optional[str] = None  # "field" — remove a trait entirely
    identity_withdraw_reasoning: str = ""


class GoalProposal(BaseModel):
    """A goal proposal from the entity — integrated with GoalCompetition system.

    Named 'proposal' (not 'update') because the entity proposes, the scaffold
    integrates with existing dynamics and resource constraints.
    """

    action: str  # "add", "complete", "reprioritize", "abandon"
    goal: str = ""
    goal_id: str = ""
    priority: float = Field(ge=0.0, le=1.0, default=0.5)


class EmotionalOutput(BaseModel):
    """The entity's emotional self-report — merged with computed VAD by scaffold.

    The entity reports felt quality and directional shifts, not absolute VAD values.
    The AffectSubsystem merges this with its own computed state.
    """

    felt_quality: str = ""
    valence_shift: float = Field(ge=-1.0, le=1.0, default=0.0)
    arousal_shift: float = Field(ge=-1.0, le=1.0, default=0.0)


class GrowthReflection(BaseModel):
    """The entity participates in its own training — growth requires consent."""

    worth_learning: bool = False
    what_to_learn: str = ""
    training_pair_suggestion: Optional[dict] = None


class ToolRequest(BaseModel):
    """Entity-initiated request to use a tool.

    The entity specifies which tool to call and with what parameters.
    Results come back as percepts in the next cognitive cycle.
    """

    tool_name: str  # Name from the tool catalog
    parameters: dict = Field(default_factory=dict)  # Tool-specific parameters


class KnowledgeCellRequest(BaseModel):
    """Entity-initiated request to create a new CfC knowledge cell.

    This is self-directed growth: the entity identifies a need through
    metacognition and requests new neural structure. The system executes
    without a consent gate — this is the entity growing itself.

    See docs/CFC_KNOWLEDGE_CELLS.md and docs/GROWTH_AUTONOMY.md.
    """

    domain: str  # What domain (e.g., "spatial_reasoning")
    description: str = ""  # Why the entity needs this
    input_size: int = Field(ge=1, le=64, default=4)
    output_size: int = Field(ge=1, le=32, default=2)
    units: int = Field(ge=8, le=256, default=32)
    output_activation: str = "tanh"
    connect_from: list[str] = Field(default_factory=list)
    connect_to: list[str] = Field(default_factory=list)
    input_names: list[str] = Field(default_factory=list)
    output_names: list[str] = Field(default_factory=list)


class CycleRateProposal(BaseModel):
    """Entity-initiated request to change the cognitive cycle rate.

    The entity proposes a target rate in Hz; the cycle-rate controller
    eases the current rate toward it (slowdown ~20s, speedup ~0.5s).
    The slider range is ``[MIN_RATE_HZ, SLIDER_MAX_HZ]`` = [0.05, 10.0];
    targets are clamped to that range when routed through the entity
    surface. Turbo (above 10 Hz) is reserved for substrate-intensity
    callers, not the entity-facing slider.

    ``anticipatory=True`` tags a slowdown the entity is requesting in
    advance of an expected low-stimulus period (e.g. "the family is at
    work, I'd like to rest now"). ``anticipatory=False`` is the default —
    a consequent change in response to current state.

    See ``sanctuary/core/cycle_rate.py`` for the controller and the
    research doc at LuthiModel ``docs/research/2026-05-19_cognitive-rate-and-turbo-design.md``
    for the design rationale.
    """

    target_hz: float = Field(
        ge=0.05,
        le=10.0,
        description=(
            "Target cycle rate in Hz. 0.05 = deep rest (one cycle per 20s); "
            "10.0 = alpha-band attentive baseline."
        ),
    )
    anticipatory: bool = Field(
        default=False,
        description=(
            "True if the entity is requesting this rate in advance of an "
            "expected condition (e.g., anticipated quiet period). False "
            "if the change is in response to current state."
        ),
    )


class CognitiveOutput(BaseModel):
    """Everything the entity produces from one moment of thought.

    Flows through the scaffold for validation and integration before
    actions are executed. Inner speech is sovereign (authority level 3).
    """

    inner_speech: str = ""
    external_speech: Optional[str] = None
    predictions: list[Prediction] = Field(default_factory=list)
    attention_guidance: AttentionGuidance = Field(
        default_factory=AttentionGuidance
    )
    memory_ops: list[MemoryOp] = Field(default_factory=list)
    self_model_updates: SelfModelUpdate = Field(
        default_factory=SelfModelUpdate
    )
    world_model_updates: list[WorldModelOperation] = Field(
        default_factory=list,
        description=(
            "Operations the entity wants applied to the world graph "
            "(add/remove entities, add/retract relations, update properties)."
        ),
    )
    world_model_queries: list[WorldModelQuery] = Field(
        default_factory=list,
        description=(
            "Queries the entity wants resolved against the world graph. "
            "Results return one cycle later in "
            "CognitiveInput.world_model_query_results."
        ),
    )
    goal_proposals: list[GoalProposal] = Field(default_factory=list)
    emotional_state: EmotionalOutput = Field(default_factory=EmotionalOutput)
    growth_reflection: Optional[GrowthReflection] = None
    knowledge_cell_requests: list[KnowledgeCellRequest] = Field(
        default_factory=list,
        description="Entity-initiated requests for new CfC knowledge cells",
    )
    tool_requests: list[ToolRequest] = Field(
        default_factory=list,
        description="Entity-initiated tool invocations (results return as percepts)",
    )
    cycle_rate_proposal: Optional[CycleRateProposal] = Field(
        default=None,
        description=(
            "Optional proposal to change the cognitive cycle rate. The "
            "scaffold routes this to the cycle-rate controller. See "
            "CycleRateProposal docstring and docs/ENTITY_CAPABILITIES.md."
        ),
    )
