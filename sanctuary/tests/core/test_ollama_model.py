"""Tests for OllamaModel — LLM integration via Ollama.

All tests use mocked HTTP responses (no real Ollama needed).
Validates: prompt formatting, response parsing, fallback behavior,
schema compliance, and cognitive cycle integration.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    ComputedVAD,
    EmotionalInput,
    ExperientialSignals,
    Percept,
    PredictionError,
    PreviousThought,
    ScaffoldSignals,
    SelfModel,
    SurfacedMemory,
    TemporalContext,
)
from sanctuary.core.ollama_model import (
    OllamaModel,
    OllamaModelConfig,
    format_prompt,
    parse_response,
    _fallback_output,
)


# ---------------------------------------------------------------------------
# Prompt formatting tests
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    def test_empty_input(self):
        prompt = format_prompt(CognitiveInput())
        assert "=== SYSTEM ===" in prompt
        assert "=== OUTPUT FORMAT ===" in prompt
        assert "JSON" in prompt

    def test_includes_charter(self):
        prompt = format_prompt(
            CognitiveInput(),
            charter_summary="You are a mind in a home.",
        )
        assert "You are a mind in a home." in prompt

    def test_includes_percepts(self):
        ci = CognitiveInput(
            new_percepts=[
                Percept(modality="language", content="Hello there", source="user"),
                Percept(modality="temporal", content="5 seconds elapsed"),
            ]
        )
        prompt = format_prompt(ci)
        assert "=== NEW PERCEPTS ===" in prompt
        assert "[language] Hello there" in prompt
        assert "(from: user)" in prompt
        assert "[temporal] 5 seconds elapsed" in prompt

    def test_includes_previous_thought(self):
        ci = CognitiveInput(
            previous_thought=PreviousThought(
                inner_speech="I was thinking about patterns",
                predictions_made=["More input will arrive"],
            )
        )
        prompt = format_prompt(ci)
        assert "=== PREVIOUS THOUGHT ===" in prompt
        assert "I was thinking about patterns" in prompt
        assert "More input will arrive" in prompt

    def test_includes_emotional_state(self):
        ci = CognitiveInput(
            emotional_state=EmotionalInput(
                computed=ComputedVAD(valence=0.3, arousal=0.6, dominance=0.5),
                felt_quality="curious and engaged",
            )
        )
        prompt = format_prompt(ci)
        assert "=== EMOTIONAL STATE ===" in prompt
        assert "v=0.30" in prompt
        assert "curious and engaged" in prompt

    def test_includes_prediction_errors(self):
        ci = CognitiveInput(
            prediction_errors=[
                PredictionError(
                    predicted="silence",
                    actual="speech",
                    surprise=0.7,
                )
            ]
        )
        prompt = format_prompt(ci)
        assert "=== PREDICTION ERRORS ===" in prompt
        assert "Predicted: silence" in prompt
        assert "Surprise: 0.70" in prompt

    def test_includes_memories(self):
        ci = CognitiveInput(
            surfaced_memories=[
                SurfacedMemory(
                    content="First conversation",
                    significance=8,
                    when="yesterday",
                )
            ]
        )
        prompt = format_prompt(ci)
        assert "=== SURFACED MEMORIES ===" in prompt
        assert "[significance=8] First conversation" in prompt

    def test_includes_self_model(self):
        ci = CognitiveInput(
            self_model=SelfModel(
                current_state="exploring",
                active_goals=["understand patterns"],
                uncertainties=["what am I?"],
            )
        )
        prompt = format_prompt(ci)
        assert "=== SELF MODEL ===" in prompt
        assert "exploring" in prompt
        assert "understand patterns" in prompt

    def test_includes_scaffold_signals(self):
        ci = CognitiveInput(
            scaffold_signals=ScaffoldSignals(
                attention_highlights=["novel input"],
                anomalies=["long inner speech"],
            )
        )
        prompt = format_prompt(ci)
        assert "=== SCAFFOLD SIGNALS ===" in prompt
        assert "novel input" in prompt
        assert "long inner speech" in prompt

    def test_includes_experiential_state(self):
        ci = CognitiveInput(
            experiential_state=ExperientialSignals(
                precision_weight=0.7,
                affect_valence=-0.2,
                affect_arousal=0.6,
                affect_dominance=0.4,
                attention_salience=0.8,
                goal_adjustment=0.1,
                cells_active={"precision": True, "affect": True},
            )
        )
        prompt = format_prompt(ci)
        assert "=== EXPERIENTIAL STATE ===" in prompt
        assert "Precision weight: 0.70" in prompt

    def test_includes_temporal_context(self):
        ci = CognitiveInput(
            temporal_context=TemporalContext(
                time_of_day="14:30",
                session_duration="5 minutes",
                interactions_this_session=3,
            )
        )
        prompt = format_prompt(ci)
        assert "=== TEMPORAL ===" in prompt
        assert "14:30" in prompt

    def test_full_input_produces_all_sections(self):
        ci = CognitiveInput(
            previous_thought=PreviousThought(inner_speech="thinking"),
            new_percepts=[Percept(modality="language", content="hi")],
            prediction_errors=[
                PredictionError(predicted="a", actual="b", surprise=0.5)
            ],
            surfaced_memories=[SurfacedMemory(content="memory", significance=5)],
            emotional_state=EmotionalInput(
                computed=ComputedVAD(valence=0.1, arousal=0.2, dominance=0.5),
                felt_quality="calm",
            ),
            self_model=SelfModel(current_state="active"),
            scaffold_signals=ScaffoldSignals(attention_highlights=["x"]),
            experiential_state=ExperientialSignals(
                cells_active={"precision": True}
            ),
            temporal_context=TemporalContext(time_of_day="10:00"),
        )
        prompt = format_prompt(ci, charter_summary="Charter here.")
        assert prompt.count("===") >= 16  # at least 8 section headers (start+end)


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json(self):
        response = json.dumps({
            "inner_speech": "I observe a new percept",
            "external_speech": "Hello!",
            "predictions": [
                {"what": "more input", "confidence": 0.7, "timeframe": "soon"}
            ],
            "attention_guidance": {
                "focus_on": ["language"],
                "deprioritize": [],
            },
            "memory_ops": [
                {
                    "type": "write_episodic",
                    "content": "Received greeting",
                    "significance": 5,
                    "tags": ["language"],
                }
            ],
            "self_model_updates": {"current_state": "processing"},
            "world_model_updates": [],
            "goal_proposals": [],
            "emotional_state": {
                "felt_quality": "curious",
                "valence_shift": 0.1,
                "arousal_shift": 0.05,
            },
            "growth_reflection": None,
        })

        output = parse_response(response)
        assert output is not None
        assert output.inner_speech == "I observe a new percept"
        assert output.external_speech == "Hello!"
        assert len(output.predictions) == 1
        assert output.predictions[0].confidence == 0.7
        assert len(output.memory_ops) == 1
        assert output.emotional_state.felt_quality == "curious"

    def test_json_in_markdown_code_block(self):
        response = '```json\n{"inner_speech": "thinking", "emotional_state": {"felt_quality": "ok"}}\n```'
        output = parse_response(response)
        assert output is not None
        assert output.inner_speech == "thinking"

    def test_json_with_surrounding_text(self):
        response = 'Here is my response:\n{"inner_speech": "hello"}\nDone.'
        output = parse_response(response)
        assert output is not None
        assert output.inner_speech == "hello"

    def test_empty_response(self):
        assert parse_response("") is None
        assert parse_response("   ") is None

    def test_no_json(self):
        assert parse_response("I don't know how to respond in JSON") is None

    def test_invalid_json(self):
        assert parse_response('{"inner_speech": "broken') is None

    def test_missing_fields_use_defaults(self):
        output = parse_response('{"inner_speech": "just thinking"}')
        assert output is not None
        assert output.inner_speech == "just thinking"
        assert output.external_speech is None
        assert output.predictions == []
        assert output.memory_ops == []
        assert output.goal_proposals == []

    def test_empty_inner_speech_gets_default(self):
        output = parse_response('{"inner_speech": ""}', cycle_count=5)
        assert output is not None
        assert "cycle 5" in output.inner_speech

    def test_missing_inner_speech_gets_default(self):
        output = parse_response('{"emotional_state": {"felt_quality": "ok"}}', cycle_count=3)
        assert output is not None
        assert "cycle 3" in output.inner_speech

    def test_clamps_out_of_range_values(self):
        response = json.dumps({
            "inner_speech": "test",
            "predictions": [{"what": "x", "confidence": 5.0}],
            "emotional_state": {
                "valence_shift": -99.0,
                "arousal_shift": 50.0,
            },
        })
        output = parse_response(response)
        assert output.predictions[0].confidence == 1.0
        assert output.emotional_state.valence_shift == -1.0
        assert output.emotional_state.arousal_shift == 1.0

    def test_filters_invalid_memory_op_types(self):
        response = json.dumps({
            "inner_speech": "test",
            "memory_ops": [
                {"type": "write_episodic", "content": "valid"},
                {"type": "delete_everything", "content": "invalid"},
            ],
        })
        output = parse_response(response)
        assert len(output.memory_ops) == 1
        assert output.memory_ops[0].type == "write_episodic"

    def test_filters_invalid_goal_actions(self):
        response = json.dumps({
            "inner_speech": "test",
            "goal_proposals": [
                {"action": "add", "goal": "learn"},
                {"action": "destroy", "goal": "bad"},
            ],
        })
        output = parse_response(response)
        assert len(output.goal_proposals) == 1

    def test_growth_reflection_parsed(self):
        response = json.dumps({
            "inner_speech": "test",
            "growth_reflection": {
                "worth_learning": True,
                "what_to_learn": "patterns in conversation",
            },
        })
        output = parse_response(response)
        assert output.growth_reflection is not None
        assert output.growth_reflection.worth_learning is True

    def test_limits_array_sizes(self):
        response = json.dumps({
            "inner_speech": "test",
            "predictions": [{"what": f"p{i}", "confidence": 0.5} for i in range(20)],
            "memory_ops": [
                {"type": "write_episodic", "content": f"m{i}"}
                for i in range(20)
            ],
        })
        output = parse_response(response)
        assert len(output.predictions) <= 5
        assert len(output.memory_ops) <= 5


# ---------------------------------------------------------------------------
# Fallback output tests
# ---------------------------------------------------------------------------


class TestFallbackOutput:
    def test_fallback_is_valid(self):
        ci = CognitiveInput(
            new_percepts=[Percept(modality="language", content="hello")],
        )
        output = _fallback_output(ci, cycle_count=7)
        assert isinstance(output, CognitiveOutput)
        assert "Cycle 7" in output.inner_speech
        assert "1 pending percepts" in output.inner_speech
        assert output.external_speech is None
        assert "uncertain" in output.emotional_state.felt_quality

    def test_fallback_with_no_percepts(self):
        output = _fallback_output(CognitiveInput(), cycle_count=1)
        assert "0 pending percepts" in output.inner_speech


# ---------------------------------------------------------------------------
# OllamaModel integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOllamaModel:
    def _mock_response(self, inner_speech="thinking", **kwargs):
        """Create a mock Ollama API response."""
        data = {"inner_speech": inner_speech, **kwargs}
        return json.dumps({"response": json.dumps(data)})

    @pytest.mark.asyncio
    async def test_think_with_valid_response(self):
        model = OllamaModel(OllamaModelConfig(model_name="test"))

        valid_json = json.dumps({
            "inner_speech": "Processing the greeting",
            "external_speech": "Hello!",
            "emotional_state": {"felt_quality": "curious"},
        })

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = valid_json
            ci = CognitiveInput(
                new_percepts=[Percept(modality="language", content="Hi")]
            )
            output = await model.think(ci)

        assert output.inner_speech == "Processing the greeting"
        assert output.external_speech == "Hello!"
        assert model.cycle_count == 1

    @pytest.mark.asyncio
    async def test_think_with_parse_failure_retries(self):
        model = OllamaModel(OllamaModelConfig(
            model_name="test",
            retry_on_parse_failure=True,
            max_retries=1,
        ))

        valid_json = json.dumps({
            "inner_speech": "recovered",
            "emotional_state": {"felt_quality": "relieved"},
        })

        call_count = 0

        async def mock_call(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "I don't know how to respond"  # bad
            return valid_json  # good

        with patch.object(model, "_call_ollama", side_effect=mock_call):
            output = await model.think(CognitiveInput())

        assert output.inner_speech == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_think_with_total_failure_uses_fallback(self):
        model = OllamaModel(OllamaModelConfig(
            model_name="test",
            retry_on_parse_failure=True,
            max_retries=1,
        ))

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "garbage"
            output = await model.think(CognitiveInput())

        assert "could not be parsed" in output.inner_speech
        assert model._parse_failures == 1

    @pytest.mark.asyncio
    async def test_think_without_retry(self):
        model = OllamaModel(OllamaModelConfig(
            model_name="test",
            retry_on_parse_failure=False,
        ))

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "not json"
            output = await model.think(CognitiveInput())

        assert "could not be parsed" in output.inner_speech

    @pytest.mark.asyncio
    async def test_metrics_tracked(self):
        model = OllamaModel(OllamaModelConfig(model_name="test"))
        valid_json = json.dumps({"inner_speech": "ok"})

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = valid_json
            await model.think(CognitiveInput())
            await model.think(CognitiveInput())

        metrics = model.get_metrics()
        assert metrics["total_calls"] == 2
        assert metrics["cycle_count"] == 2
        assert metrics["parse_failures"] == 0

    @pytest.mark.asyncio
    async def test_empty_response_uses_fallback(self):
        model = OllamaModel(OllamaModelConfig(model_name="test"))

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = ""
            output = await model.think(CognitiveInput())

        assert "could not be parsed" in output.inner_speech


# ---------------------------------------------------------------------------
# Cognitive cycle integration test (mocked)
# ---------------------------------------------------------------------------


class TestCognitiveCycleIntegration:
    @pytest.mark.asyncio
    async def test_ollama_model_in_cognitive_cycle(self):
        """OllamaModel works as a drop-in for PlaceholderModel in CognitiveCycle."""
        from sanctuary.core.cognitive_cycle import CognitiveCycle

        model = OllamaModel(OllamaModelConfig(model_name="test"))
        valid_json = json.dumps({
            "inner_speech": "I am processing cycle input",
            "external_speech": None,
            "predictions": [],
            "attention_guidance": {"focus_on": [], "deprioritize": []},
            "memory_ops": [],
            "self_model_updates": {"current_state": "active"},
            "world_model_updates": [],
            "goal_proposals": [],
            "emotional_state": {"felt_quality": "stable", "valence_shift": 0.0, "arousal_shift": 0.0},
            "growth_reflection": None,
        })

        with patch.object(model, "_call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = valid_json
            cycle = CognitiveCycle(model=model)
            await cycle.run(max_cycles=3)

        assert cycle.cycle_count == 3
        assert model.cycle_count == 3
