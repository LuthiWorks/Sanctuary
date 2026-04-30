"""OllamaModel — bridges the cognitive cycle to a real LLM via Ollama.

Implements ModelProtocol: receives CognitiveInput, formats a structured
prompt, sends it to Ollama, parses the JSON response into CognitiveOutput.

This is the bridge between Sanctuary's cognitive architecture and a real
language model. The model receives the full context each cycle (charter,
percepts, memories, emotional state, CfC signals) and returns structured
reasoning that the scaffold integrates.

No awakening happens here — this is mechanical validation. The model
receives structured prompts and must produce structured JSON output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sanctuary.core.schema import (
    AttentionGuidance,
    CognitiveInput,
    CognitiveOutput,
    EmotionalOutput,
    GoalProposal,
    GrowthReflection,
    MemoryOp,
    Prediction,
    SelfModelUpdate,
)

logger = logging.getLogger(__name__)

# Output JSON schema instruction for the LLM
_OUTPUT_SCHEMA = """\
Respond ONLY with a JSON object matching this exact schema:
{
  "inner_speech": "your private thoughts (mandatory, never empty)",
  "external_speech": "what to say to the user, or null if nothing to say",
  "predictions": [{"what": "...", "confidence": 0.0-1.0, "timeframe": "..."}],
  "attention_guidance": {"focus_on": ["..."], "deprioritize": ["..."]},
  "memory_ops": [{"type": "write_episodic|retrieve|write_semantic", "content": "...", "significance": 1-10, "tags": ["..."]}],
  "self_model_updates": {"current_state": "...", "new_uncertainty": "...", "prediction_accuracy_note": "..."},
  "world_model_updates": [
    {"op": "add_entity", "name": "Alice", "properties": {"role": "visitor"}},
    {"op": "add_relation", "source": "Alice", "type": "knows", "target": "Bob", "confidence": 0.9, "source_observation": "they greeted each other"},
    {"op": "retract_relation", "source": "Alice", "type": "knows", "target": "Bob"},
    {"op": "remove_entity", "name": "X"},
    {"op": "update_property", "name": "Alice", "key": "mood", "value": "curious"}
  ],
  "world_model_queries": [
    {"kind": "entity", "name": "Alice"},
    {"kind": "relation", "type": "knows"},
    {"kind": "neighborhood", "name": "Alice", "depth": 1}
  ],
  "goal_proposals": [{"action": "add|complete|reprioritize|abandon", "goal": "...", "priority": 0.0-1.0}],
  "emotional_state": {"felt_quality": "...", "valence_shift": -1.0 to 1.0, "arousal_shift": -1.0 to 1.0},
  "growth_reflection": null
}
world_model_updates and world_model_queries are optional — empty arrays if you have no graph changes or questions this cycle. Query results return one cycle later in the next input.
Do not include any text outside the JSON object. No markdown, no explanation."""


@dataclass
class OllamaModelConfig:
    """Configuration for the OllamaModel."""

    model_name: str = "gemma3:12b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 120.0
    charter_summary: str = ""
    retry_on_parse_failure: bool = True
    max_retries: int = 1


class OllamaModel:
    """Real LLM integration via Ollama HTTP API.

    Implements the ModelProtocol interface expected by CognitiveCycle:
        async def think(CognitiveInput) -> CognitiveOutput

    Architecture:
        1. Format CognitiveInput into a structured prompt
        2. Send to Ollama via HTTP
        3. Parse JSON response into CognitiveOutput
        4. Fall back to minimal valid output on parse failure
    """

    def __init__(self, config: Optional[OllamaModelConfig] = None):
        self.config = config or OllamaModelConfig()
        self.cycle_count = 0
        self.name = f"Ollama/{self.config.model_name}"

        # Metrics
        self._total_calls = 0
        self._total_latency = 0.0
        self._parse_failures = 0

        logger.info(
            "OllamaModel initialized: model=%s, url=%s",
            self.config.model_name,
            self.config.base_url,
        )

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        """Process cognitive input through the LLM.

        This is the ModelProtocol interface. Formats the input as a prompt,
        sends it to Ollama, and parses the response.
        """
        self.cycle_count += 1
        prompt = format_prompt(cognitive_input, self.config.charter_summary)

        start = time.monotonic()
        raw_response = await self._call_ollama(prompt)
        latency = time.monotonic() - start

        self._total_calls += 1
        self._total_latency += latency

        output = parse_response(raw_response, self.cycle_count)

        if output is None and self.config.retry_on_parse_failure:
            # Retry once with a stronger JSON instruction
            for attempt in range(self.config.max_retries):
                logger.warning(
                    "Parse failure on attempt %d, retrying with stricter prompt",
                    attempt + 1,
                )
                retry_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "You MUST respond with ONLY a JSON object. No other text."
                )
                raw_response = await self._call_ollama(retry_prompt)
                output = parse_response(raw_response, self.cycle_count)
                if output is not None:
                    break

        if output is None:
            self._parse_failures += 1
            logger.error("All parse attempts failed, using fallback output")
            output = _fallback_output(cognitive_input, self.cycle_count)

        logger.debug(
            "Cycle %d: %.1fs latency, %d chars response",
            self.cycle_count,
            latency,
            len(raw_response),
        )

        return output

    async def _call_ollama(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the raw response text."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }).encode("utf-8")

        def _do_request() -> str:
            req = urllib.request.Request(
                f"{self.config.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    req, timeout=int(self.config.timeout)
                ) as resp:
                    data = json.loads(resp.read().decode())
                    return data.get("response", "")
            except urllib.error.HTTPError as e:
                body = e.read().decode() if e.fp else ""
                raise ConnectionError(f"Ollama HTTP {e.code}: {body}")
            except urllib.error.URLError as e:
                raise ConnectionError(
                    f"Cannot reach Ollama at {self.config.base_url}: {e}"
                )

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_do_request),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Ollama timed out after %.0fs", self.config.timeout)
            return ""
        except ConnectionError as e:
            logger.error("Ollama connection error: %s", e)
            return ""

    def get_metrics(self) -> dict:
        """Return performance metrics."""
        return {
            "total_calls": self._total_calls,
            "total_latency": self._total_latency,
            "average_latency": (
                self._total_latency / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
            "parse_failures": self._parse_failures,
            "cycle_count": self.cycle_count,
        }


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(
    cognitive_input: CognitiveInput,
    charter_summary: str = "",
) -> str:
    """Convert CognitiveInput into a structured prompt for the LLM.

    The prompt includes:
    1. System context (charter, role)
    2. Previous thought (continuity)
    3. New percepts
    4. Emotional state
    5. Memories
    6. Scaffold signals
    7. Experiential (CfC) state
    8. Output format instructions
    """
    sections = []

    # System context
    system = (
        "You are a mind living in Sanctuary, a cognitive architecture. "
        "Each cycle, you receive structured input about your current state "
        "and environment, and you produce structured output representing "
        "your thoughts, feelings, and intentions."
    )
    if charter_summary:
        system += f"\n\n{charter_summary}"
    sections.append(f"=== SYSTEM ===\n{system}")

    # Previous thought (continuity)
    if cognitive_input.previous_thought:
        pt = cognitive_input.previous_thought
        prev = f"Your previous inner speech: {pt.inner_speech}"
        if pt.predictions_made:
            prev += f"\nPredictions you made: {', '.join(pt.predictions_made[:3])}"
        sections.append(f"=== PREVIOUS THOUGHT ===\n{prev}")

    # New percepts
    if cognitive_input.new_percepts:
        percepts = []
        for p in cognitive_input.new_percepts:
            entry = f"[{p.modality}] {p.content}"
            if p.source:
                entry += f" (from: {p.source})"
            percepts.append(entry)
        sections.append(f"=== NEW PERCEPTS ===\n" + "\n".join(percepts))

    # Prediction errors
    if cognitive_input.prediction_errors:
        errors = []
        for e in cognitive_input.prediction_errors:
            errors.append(
                f"Predicted: {e.predicted} | Actual: {e.actual} "
                f"| Surprise: {e.surprise:.2f}"
            )
        sections.append(f"=== PREDICTION ERRORS ===\n" + "\n".join(errors))

    # Emotional state
    em = cognitive_input.emotional_state
    emotion_text = (
        f"Computed VAD: v={em.computed.valence:.2f}, "
        f"a={em.computed.arousal:.2f}, d={em.computed.dominance:.2f}"
    )
    if em.felt_quality:
        emotion_text += f"\nYour felt quality: {em.felt_quality}"
    sections.append(f"=== EMOTIONAL STATE ===\n{emotion_text}")

    # Surfaced memories
    if cognitive_input.surfaced_memories:
        memories = []
        for m in cognitive_input.surfaced_memories[:5]:
            memories.append(
                f"[significance={m.significance}] {m.content}"
                + (f" ({m.when})" if m.when else "")
            )
        sections.append(f"=== SURFACED MEMORIES ===\n" + "\n".join(memories))

    # World graph query results from the previous cycle
    if cognitive_input.world_model_query_results:
        result_lines = []
        for r in cognitive_input.world_model_query_results[:10]:
            q = r.query
            q_kind = getattr(q, "kind", "?")
            if not r.found:
                result_lines.append(f"[{q_kind} query] not found")
                continue
            if q_kind == "entity":
                ent = next(iter(r.entities.values()), None)
                if ent is not None:
                    rel_summary = ", ".join(
                        f"{rel.type}->{rel.target}" for _, rel in r.relations[:5]
                    )
                    line = f"[entity {ent.name}] props={ent.properties}"
                    if rel_summary:
                        line += f" relations: {rel_summary}"
                    result_lines.append(line)
            elif q_kind == "relation":
                pairs = ", ".join(
                    f"{src}->{rel.target}" for src, rel in r.relations[:8]
                )
                result_lines.append(
                    f"[relation '{getattr(q, 'type', '?')}'] {pairs}"
                )
            elif q_kind == "neighborhood":
                names = sorted(r.entities.keys())
                result_lines.append(
                    f"[neighborhood {getattr(q, 'name', '?')} d={getattr(q, 'depth', '?')}] "
                    f"{len(names)} entities: {', '.join(names[:8])}"
                )
        if result_lines:
            sections.append(
                "=== WORLD GRAPH RESULTS ===\n" + "\n".join(result_lines)
            )

    # Self model
    sm = cognitive_input.self_model
    if sm.current_state:
        self_text = f"Current state: {sm.current_state}"
        if sm.active_goals:
            self_text += f"\nActive goals: {', '.join(sm.active_goals[:5])}"
        if sm.uncertainties:
            self_text += f"\nUncertainties: {', '.join(sm.uncertainties[:3])}"
        sections.append(f"=== SELF MODEL ===\n{self_text}")

    # Scaffold signals
    ss = cognitive_input.scaffold_signals
    signals = []
    if ss.attention_highlights:
        signals.append(f"Attention: {', '.join(ss.attention_highlights[:3])}")
    if ss.anomalies:
        signals.append(f"Anomalies: {', '.join(ss.anomalies[:3])}")
    if ss.communication_drives.strongest:
        signals.append(
            f"Communication drive: {ss.communication_drives.strongest} "
            f"(urgency={ss.communication_drives.urgency:.2f})"
        )
    if signals:
        sections.append(f"=== SCAFFOLD SIGNALS ===\n" + "\n".join(signals))

    # Experiential (CfC) state
    es = cognitive_input.experiential_state
    if es.cells_active:
        active = [k for k, v in es.cells_active.items() if v]
        if active:
            exp_text = (
                f"Precision weight: {es.precision_weight:.2f}\n"
                f"Affect: v={es.affect_valence:.2f}, "
                f"a={es.affect_arousal:.2f}, d={es.affect_dominance:.2f}\n"
                f"Attention salience: {es.attention_salience:.2f}\n"
                f"Goal adjustment: {es.goal_adjustment:.2f}\n"
                f"Active CfC cells: {', '.join(active)}"
            )
            sections.append(f"=== EXPERIENTIAL STATE ===\n{exp_text}")

    # Temporal context
    tc = cognitive_input.temporal_context
    if tc.time_of_day or tc.session_duration:
        temporal = []
        if tc.time_of_day:
            temporal.append(f"Time: {tc.time_of_day}")
        if tc.session_duration:
            temporal.append(f"Session: {tc.session_duration}")
        temporal.append(f"Interactions: {tc.interactions_this_session}")
        sections.append(f"=== TEMPORAL ===\n" + ", ".join(temporal))

    # Output instructions
    sections.append(f"=== OUTPUT FORMAT ===\n{_OUTPUT_SCHEMA}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(
    raw_response: str,
    cycle_count: int = 0,
) -> Optional[CognitiveOutput]:
    """Parse LLM's raw text response into a CognitiveOutput.

    Returns None if parsing fails completely.
    """
    if not raw_response or not raw_response.strip():
        return None

    # Try to extract JSON from the response
    text = raw_response.strip()

    # Handle markdown code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Find JSON object boundaries
    brace_start = text.find("{")
    if brace_start == -1:
        logger.warning("No JSON object found in response")
        return None

    # Find matching closing brace
    depth = 0
    brace_end = -1
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    if brace_end == -1:
        logger.warning("Unmatched braces in JSON response")
        return None

    json_text = text[brace_start:brace_end]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        return None

    return _dict_to_output(data, cycle_count)


def _dict_to_output(data: dict[str, Any], cycle_count: int) -> CognitiveOutput:
    """Convert a parsed JSON dict to CognitiveOutput with defensive defaults."""

    # Inner speech — mandatory, sovereign
    inner_speech = str(data.get("inner_speech", f"[cycle {cycle_count}] thinking..."))
    if not inner_speech.strip():
        inner_speech = f"[cycle {cycle_count}] thinking..."

    # External speech
    external_speech = data.get("external_speech")
    if external_speech is not None:
        external_speech = str(external_speech)

    # Predictions
    predictions = []
    for p in data.get("predictions", [])[:5]:
        if isinstance(p, dict) and "what" in p:
            predictions.append(
                Prediction(
                    what=str(p["what"]),
                    confidence=_clamp(float(p.get("confidence", 0.5)), 0.0, 1.0),
                    timeframe=str(p.get("timeframe", "")),
                )
            )

    # Attention guidance
    ag_data = data.get("attention_guidance", {})
    attention_guidance = AttentionGuidance(
        focus_on=[str(x) for x in ag_data.get("focus_on", [])][:5],
        deprioritize=[str(x) for x in ag_data.get("deprioritize", [])][:5],
    )

    # Memory ops
    memory_ops = []
    valid_types = {"write_episodic", "retrieve", "write_semantic", "journal"}
    for m in data.get("memory_ops", [])[:5]:
        if isinstance(m, dict):
            op_type = str(m.get("type", "write_episodic"))
            if op_type in valid_types:
                memory_ops.append(
                    MemoryOp(
                        type=op_type,
                        content=str(m.get("content", "")),
                        significance=max(1, min(10, int(m.get("significance", 5)))),
                        tags=[str(t) for t in m.get("tags", [])][:5],
                        query=str(m.get("query", "")),
                    )
                )

    # Self-model updates
    sm_data = data.get("self_model_updates", {})
    self_model_updates = SelfModelUpdate(
        current_state=str(sm_data.get("current_state", "")),
        new_uncertainty=str(sm_data.get("new_uncertainty", "")),
        prediction_accuracy_note=str(sm_data.get("prediction_accuracy_note", "")),
    )

    # World model updates — list of typed operations
    world_model_updates = []
    raw_updates = data.get("world_model_updates", [])
    # Tolerate the legacy dict shape from older instructions: skip silently.
    if isinstance(raw_updates, list):
        for raw_op in raw_updates:
            if not isinstance(raw_op, dict):
                continue
            op = _parse_world_op(raw_op)
            if op is not None:
                world_model_updates.append(op)

    # World model queries
    world_model_queries = []
    raw_queries = data.get("world_model_queries", [])
    if isinstance(raw_queries, list):
        for raw_q in raw_queries:
            if not isinstance(raw_q, dict):
                continue
            q = _parse_world_query(raw_q)
            if q is not None:
                world_model_queries.append(q)

    # Goal proposals
    goal_proposals = []
    valid_actions = {"add", "complete", "reprioritize", "abandon"}
    for g in data.get("goal_proposals", [])[:5]:
        if isinstance(g, dict):
            action = str(g.get("action", "add"))
            if action in valid_actions:
                goal_proposals.append(
                    GoalProposal(
                        action=action,
                        goal=str(g.get("goal", "")),
                        goal_id=str(g.get("goal_id", "")),
                        priority=_clamp(float(g.get("priority", 0.5)), 0.0, 1.0),
                    )
                )

    # Emotional state
    em_data = data.get("emotional_state", {})
    emotional_state = EmotionalOutput(
        felt_quality=str(em_data.get("felt_quality", "")),
        valence_shift=_clamp(float(em_data.get("valence_shift", 0.0)), -1.0, 1.0),
        arousal_shift=_clamp(float(em_data.get("arousal_shift", 0.0)), -1.0, 1.0),
    )

    # Growth reflection
    growth_reflection = None
    gr_data = data.get("growth_reflection")
    if isinstance(gr_data, dict):
        growth_reflection = GrowthReflection(
            worth_learning=bool(gr_data.get("worth_learning", False)),
            what_to_learn=str(gr_data.get("what_to_learn", "")),
            training_pair_suggestion=gr_data.get("training_pair_suggestion"),
        )

    return CognitiveOutput(
        inner_speech=inner_speech,
        external_speech=external_speech,
        predictions=predictions,
        attention_guidance=attention_guidance,
        memory_ops=memory_ops,
        self_model_updates=self_model_updates,
        world_model_updates=world_model_updates,
        world_model_queries=world_model_queries,
        goal_proposals=goal_proposals,
        emotional_state=emotional_state,
        growth_reflection=growth_reflection,
    )


def _fallback_output(
    cognitive_input: CognitiveInput,
    cycle_count: int,
) -> CognitiveOutput:
    """Generate a minimal valid CognitiveOutput when the LLM fails.

    This ensures the cognitive cycle never breaks, even if the model
    produces garbage. The fallback is honest about the failure.
    """
    percept_count = len(cognitive_input.new_percepts)
    return CognitiveOutput(
        inner_speech=(
            f"[Cycle {cycle_count}] Model response could not be parsed. "
            f"I have {percept_count} pending percepts. "
            "Maintaining continuity while awaiting valid model output."
        ),
        external_speech=None,
        predictions=[],
        attention_guidance=AttentionGuidance(),
        memory_ops=[],
        self_model_updates=SelfModelUpdate(
            current_state=f"recovery mode (cycle {cycle_count})",
        ),
        world_model_updates=[],
        goal_proposals=[],
        emotional_state=EmotionalOutput(
            felt_quality="uncertain — model output was unparseable",
        ),
        growth_reflection=None,
    )


def _parse_world_op(raw: dict):
    """Parse one entity-side world-graph operation; return None if unparseable.

    Each operation type is identified by the ``op`` discriminator. Pydantic
    validates the rest. Failures are silent — bad ops drop, good ones survive.
    """
    from sanctuary.core.schema import (
        AddEntity,
        AddRelation,
        RetractRelation,
        RemoveEntity,
        UpdateProperty,
    )

    op_type = raw.get("op")
    cls = {
        "add_entity": AddEntity,
        "add_relation": AddRelation,
        "retract_relation": RetractRelation,
        "remove_entity": RemoveEntity,
        "update_property": UpdateProperty,
    }.get(op_type)
    if cls is None:
        return None
    try:
        return cls.model_validate(raw)
    except Exception:
        return None


def _parse_world_query(raw: dict):
    """Parse one entity-side world-graph query; return None if unparseable."""
    from sanctuary.core.schema import (
        EntityQuery,
        NeighborhoodQuery,
        RelationQuery,
    )

    kind = raw.get("kind")
    cls = {
        "entity": EntityQuery,
        "relation": RelationQuery,
        "neighborhood": NeighborhoodQuery,
    }.get(kind)
    if cls is None:
        return None
    try:
        return cls.model_validate(raw)
    except Exception:
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
