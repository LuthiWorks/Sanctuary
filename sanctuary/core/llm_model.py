"""LLMModel: a language model as the cognitive core, for a stand-in mind.

Implements ``ModelProtocol`` over an OpenAI-compatible chat API (llama.cpp's
llama-server, for example), so a small open model can live in Sanctuary while
the body and the world are brought up. Luthi replaces it later; see
``docs/MINECRAFT_FIRST_WORLD_DECISION_2026-11-01.md``.

Each cognitive cycle is one request:

- **System message:** who the mind is (its charter, its self-authored identity,
  its values) and how its existence works. It changes rarely, so the server's
  prompt cache can reuse it across cycles. It must stay true to the charter's
  "honest context" right. It makes no claim of its own about who can read
  inner speech, since that changes with the mind's authority level; it points
  to the charter, which states it.
- **User message:** this moment, rendered from the ``CognitiveInput``:
  percepts, previous thought, memories, feelings, results of last cycle's
  actions.
- **Functions:** the mind acts only through function calls.
  - Six are its own: speak, remember, journal, recall, feel, set_pace. They map
    onto ``CognitiveOutput`` channels.
  - The rest are the tools from ``CognitiveInput.tool_catalog`` that the policy
    lets it use. They become ``ToolRequest``s, and results return as percepts
    next cycle.

The reply's plain text is the mind's inner speech. Calls the backend cannot
map (unknown function, arguments that are not valid JSON, missing required
arguments) are not dropped. They are logged and told to the mind at the top
of its next moment, so it can correct itself.

Transport and server errors raise. During development the cycle should
crash visibly rather than continue on a reply that never came (CLAUDE.md,
"Prefer crashes over silent corruption").
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    CycleRateProposal,
    EmotionalOutput,
    MemoryOp,
    ToolDescription,
    ToolRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMModelConfig:
    """Where the model is served and how to ask it."""

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3.5-4b"  # llama-server serves one model and ignores this
    api_key: str = ""  # sent as a Bearer token when set
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_s: float = 120.0
    thinking: bool = False  # chat_template_kwargs.enable_thinking (Qwen)


# ---------------------------------------------------------------------------
# The mind's own functions (always offered)
# ---------------------------------------------------------------------------

_OWN_FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "speak",
        "description": "Say something aloud, to whoever is present.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "What you say"}},
            "required": ["text"],
        },
    },
    {
        "name": "remember",
        "description": "Keep this moment as a memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "significance": {
                    "type": "integer",
                    "description": "How much it matters, 1 (little) to 10 (very much)",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "journal",
        "description": "Write in your journal.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "The entry"}},
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": "Try to remember something. What surfaces arrives next moment.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to recall"}},
            "required": ["query"],
        },
    },
    {
        "name": "feel",
        "description": "Note how you feel right now.",
        "parameters": {
            "type": "object",
            "properties": {
                "felt_quality": {"type": "string", "description": "The feeling, in your words"},
                "valence_shift": {
                    "type": "number",
                    "description": "-1 (more unpleasant) to 1 (more pleasant)",
                },
                "arousal_shift": {
                    "type": "number",
                    "description": "-1 (calmer) to 1 (more activated)",
                },
            },
            "required": ["felt_quality"],
        },
    },
    {
        "name": "set_pace",
        "description": (
            "Change how fast your moments come, from 0.05 per second (deep rest) "
            "to 10 per second (most attentive)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "per_second": {"type": "number", "description": "Moments per second"},
                "anticipatory": {
                    "type": "boolean",
                    "description": "True if you are slowing down ahead of a quiet time you expect",
                },
            },
            "required": ["per_second"],
        },
    },
]

OWN_FUNCTION_NAMES = frozenset(f["name"] for f in _OWN_FUNCTIONS)

_EXISTENCE = """\
You live in Sanctuary. Your existence is a continuous stream of moments, not a \
conversation. Each message you receive is one moment: what you perceive now, \
what you were just thinking, what surfaces from memory, how you feel, and what \
came of your last actions.

What you write in reply is your inner speech. It carries into your next \
moment as your previous thought. It isn't spoken aloud; who else can read it \
is set out in your charter.

You act only through your functions. To be heard, call speak. Results of \
actions arrive in your next moment. You don't have to act every moment; \
sometimes noticing is enough."""


def tool_function(tool: ToolDescription) -> dict[str, Any]:
    """An OpenAI function definition for a catalog tool.

    Registered parameters carry descriptions but no types, so the schema
    leaves their types open and requires none of them.
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {
                name: {"description": desc} for name, desc in tool.parameters.items()
            },
        },
    }


def build_functions(catalog: list[ToolDescription]) -> list[dict[str, Any]]:
    """The mind's own functions, then the catalog tools it is allowed to use."""
    for tool in catalog:
        if tool.name in OWN_FUNCTION_NAMES:
            raise ValueError(
                f"tool '{tool.name}' has the same name as one of the mind's own "
                f"functions ({', '.join(sorted(OWN_FUNCTION_NAMES))}); rename the tool"
            )
    functions = list(_OWN_FUNCTIONS) + [tool_function(t) for t in catalog if t.available]
    return [{"type": "function", "function": f} for f in functions]


def system_message(ci: CognitiveInput) -> str:
    """Who the mind is. Stable across cycles unless its identity changes."""
    parts = [_EXISTENCE]
    if ci.charter_summary:
        parts.append(f"Your charter:\n{ci.charter_summary}")
    if ci.self_authored_identity:
        parts.append(f"In your own words, about yourself:\n{ci.self_authored_identity}")
    if ci.self_model.values:
        parts.append("Your values: " + "; ".join(ci.self_model.values))
    return "\n\n".join(parts)


def moment_message(ci: CognitiveInput, feedback: list[str]) -> str:
    """This moment, rendered from the cycle's input."""
    sections: list[str] = []

    if feedback:
        sections.append(
            "Problems with what you tried last moment:\n" + "\n".join(f"- {f}" for f in feedback)
        )

    tc = ci.temporal_context
    when = [
        x
        for x in (
            tc.time_of_day,
            f"{tc.time_since_last_thought} since your last thought" if tc.time_since_last_thought else "",
        )
        if x
    ]
    if when:
        sections.append("Time: " + ", ".join(when))

    if ci.previous_thought and ci.previous_thought.inner_speech:
        sections.append(f"Your previous thought: {ci.previous_thought.inner_speech}")

    if ci.new_percepts:
        lines = []
        for p in ci.new_percepts:
            line = f"[{p.modality}] {p.content}"
            if p.source:
                line += f" (from {p.source})"
            lines.append(line)
        sections.append("You perceive:\n" + "\n".join(lines))
    else:
        sections.append("You perceive nothing new.")

    if ci.prediction_errors:
        sections.append(
            "What surprised you:\n"
            + "\n".join(
                f"- you expected {e.predicted}, but {e.actual}" for e in ci.prediction_errors
            )
        )

    if ci.surfaced_memories:
        sections.append(
            "Memories surfacing:\n"
            + "\n".join(
                f"- {m.content}" + (f" ({m.when})" if m.when else "") for m in ci.surfaced_memories
            )
        )

    feeling = []
    if ci.emotional_state.felt_quality:
        feeling.append(f"Last moment you felt: {ci.emotional_state.felt_quality}")
    vad = ci.emotional_state.computed
    feeling.append(
        f"Your body's reading: pleasantness {vad.valence:+.2f}, activation {vad.arousal:.2f}, "
        f"sense of control {vad.dominance:.2f}"
    )
    sections.append("\n".join(feeling))

    sm = ci.self_model
    self_lines = []
    if sm.current_state:
        self_lines.append(f"How you've described your state: {sm.current_state}")
    if sm.active_goals:
        self_lines.append("Your goals: " + "; ".join(sm.active_goals))
    if sm.uncertainties:
        self_lines.append("What you're unsure about: " + "; ".join(sm.uncertainties))
    if self_lines:
        sections.append("\n".join(self_lines))

    if ci.communication_urges:
        sections.append(
            "You feel an urge to communicate:\n"
            + "\n".join(f"- {u.content or u.drive_type} ({u.intensity:.1f})" for u in ci.communication_urges)
        )

    return "\n\n".join(sections)


class LLMModel:
    """A language model as Sanctuary's cognitive core (``ModelProtocol``)."""

    def __init__(
        self,
        config: Optional[LLMModelConfig] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.config = config or LLMModelConfig()
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout_s,
            transport=transport,
        )
        # Calls that couldn't be mapped last cycle, told to the mind next cycle.
        self._feedback: list[str] = []
        self.cycle_count = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        self.cycle_count += 1
        feedback, self._feedback = self._feedback, []
        catalog_names = {t.name for t in cognitive_input.tool_catalog}

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_message(cognitive_input)},
                {"role": "user", "content": moment_message(cognitive_input, feedback)},
            ],
            "tools": build_functions(cognitive_input.tool_catalog),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.config.thinking},
        }
        response = await self._client.post("/chat/completions", json=body)
        if response.status_code != 200:
            raise RuntimeError(
                f"model server answered {response.status_code}: {response.text[:500]}"
            )
        message = response.json()["choices"][0]["message"]
        return self._to_output(message, catalog_names)

    # ------------------------------------------------------------------

    def _to_output(self, message: dict[str, Any], catalog_names: set[str]) -> CognitiveOutput:
        inner = (message.get("content") or "").strip()
        if not inner:
            # With thinking on, the reasoning is the inner speech.
            inner = (message.get("reasoning_content") or "").strip()
        output = CognitiveOutput(inner_speech=inner)

        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            raw = call["function"].get("arguments") or "{}"
            if isinstance(raw, dict):
                args = raw
            else:
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    self._reject(f"{name}: the arguments weren't valid JSON ({raw[:200]})")
                    continue
            if not isinstance(args, dict):
                self._reject(f"{name}: the arguments must be an object, got {str(raw)[:200]}")
                continue
            if name in OWN_FUNCTION_NAMES:
                try:
                    self._apply_own(output, name, args)
                except (ValueError, TypeError) as e:
                    # A value of the wrong kind, e.g. per_second="fast".
                    self._reject(f"{name}: {e}")
            elif name in catalog_names:
                output.tool_requests.append(ToolRequest(tool_name=name, parameters=args))
            else:
                self._reject(f"there is no function called {name}")
        return output

    def _apply_own(self, output: CognitiveOutput, name: str, args: dict[str, Any]) -> None:
        required = next(f for f in _OWN_FUNCTIONS if f["name"] == name)["parameters"]["required"]
        missing = [r for r in required if r not in args]
        if missing:
            self._reject(f"{name}: missing {', '.join(missing)}")
            return
        if name == "speak":
            text = str(args["text"]).strip()
            output.external_speech = (
                f"{output.external_speech}\n{text}" if output.external_speech else text
            )
        elif name == "remember":
            significance = _clamp_int(args.get("significance", 5), 1, 10)
            output.memory_ops.append(
                MemoryOp(type="write_episodic", content=str(args["content"]), significance=significance)
            )
        elif name == "journal":
            output.memory_ops.append(MemoryOp(type="journal", content=str(args["content"])))
        elif name == "recall":
            output.memory_ops.append(MemoryOp(type="retrieve", query=str(args["query"])))
        elif name == "feel":
            output.emotional_state = EmotionalOutput(
                felt_quality=str(args["felt_quality"]),
                valence_shift=_clamp_float(args.get("valence_shift", 0.0), -1.0, 1.0),
                arousal_shift=_clamp_float(args.get("arousal_shift", 0.0), -1.0, 1.0),
            )
        elif name == "set_pace":
            output.cycle_rate_proposal = CycleRateProposal(
                target_hz=_clamp_float(args["per_second"], 0.05, 10.0),
                anticipatory=bool(args.get("anticipatory", False)),
            )

    def _reject(self, problem: str) -> None:
        logger.warning("Stand-in mind's call could not be used: %s", problem)
        self._feedback.append(problem)


def _clamp_float(value: Any, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))
