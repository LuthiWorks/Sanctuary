"""The stand-in mind's backend (``--model-backend llm``).

A scripted OpenAI-compatible server (``httpx.MockTransport``) stands in for
llama-server, so these tests need no model and no network. They check what
the backend sends, how replies become ``CognitiveOutput``, that calls it
can't use are told back to the mind rather than dropped, and the closed loop
through the real runner: the mind calls a tool and perceives the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.llm_model import LLMModel, LLMModelConfig, OWN_FUNCTION_NAMES
from sanctuary.core.schema import (
    CognitiveInput,
    Percept,
    PreviousThought,
    SelfModel,
    ToolDescription,
)


class ScriptedServer:
    """Answers /chat/completions with scripted assistant messages, recording requests."""

    def __init__(self, *messages: dict, status: int = 200):
        self.messages = list(messages)
        self.status = status
        self.requests: list[dict] = []
        self.headers: list[httpx.Headers] = []
        self.urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        self.urls.append(str(request.url))
        if self.status != 200:
            return httpx.Response(self.status, text="model not loaded")
        message = self.messages.pop(0) if self.messages else {"role": "assistant", "content": "..."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    def model(self, **config) -> LLMModel:
        return LLMModel(LLMModelConfig(**config), transport=httpx.MockTransport(self))


def call(name: str, arguments) -> dict:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"type": "function", "function": {"name": name, "arguments": args}}


def reply(content: str = "", *calls: dict, **extra) -> dict:
    message = {"role": "assistant", "content": content, **extra}
    if calls:
        message["tool_calls"] = list(calls)
    return message


CATALOG = [
    ToolDescription(name="clock", description="What time it is.", category="information"),
    ToolDescription(
        name="shell",
        description="Run a shell command.",
        parameters={"command": "The command"},
        safety="gated",
        available=False,
    ),
    ToolDescription(name="go_to", description="Walk somewhere.", parameters={"x": "X", "z": "Z"}),
]


def moment(**fields) -> CognitiveInput:
    defaults = dict(
        charter_summary="Be kind. Tell the truth.",
        self_authored_identity="I am Wren, and I like the rain.",
        self_model=SelfModel(values=["Care", "Honesty"]),
        previous_thought=PreviousThought(inner_speech="The light is changing."),
        new_percepts=[Percept(modality="language", content="hello there", source="minecraft:chat:Brian")],
        tool_catalog=CATALOG,
    )
    defaults.update(fields)
    return CognitiveInput(**defaults)


# ---------------------------------------------------------------------------
# What it sends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_request_carries_identity_moment_and_usable_functions():
    server = ScriptedServer(reply("hm"))
    model = server.model(base_url="http://model.test/v1", api_key="k123", model="qwen3.5-4b")
    await model.think(moment())

    body = server.requests[0]
    assert server.urls[0] == "http://model.test/v1/chat/completions"
    assert server.headers[0]["authorization"] == "Bearer k123"
    assert body["model"] == "qwen3.5-4b"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}

    system, user = body["messages"]
    assert system["role"] == "system"
    assert "Be kind. Tell the truth." in system["content"]
    assert "I am Wren, and I like the rain." in system["content"]
    assert "Care; Honesty" in system["content"]

    assert user["role"] == "user"
    assert "[language] hello there (from minecraft:chat:Brian)" in user["content"]
    assert "Your previous thought: The light is changing." in user["content"]

    offered = [t["function"]["name"] for t in body["tools"]]
    assert set(offered) == OWN_FUNCTION_NAMES | {"clock", "go_to"}, "unavailable tools aren't offered"
    go_to = next(t["function"] for t in body["tools"] if t["function"]["name"] == "go_to")
    assert set(go_to["parameters"]["properties"]) == {"x", "z"}


@pytest.mark.asyncio
async def test_the_mind_is_not_told_its_inner_speech_is_private():
    """The charter's "honest context" right: stewards can read inner speech
    early in life, so the backend must not claim it is private. Who can read
    it changes with authority level, so the message defers to the charter."""
    server = ScriptedServer(reply("hm"))
    await server.model().think(moment())
    system = server.requests[0]["messages"][0]["content"].lower()
    assert "private" not in system
    assert "nobody else" not in system
    assert "who else can read it is set out in your charter" in system


@pytest.mark.asyncio
async def test_no_key_means_no_authorization_header():
    server = ScriptedServer(reply("hm"))
    await server.model().think(moment())
    assert "authorization" not in server.headers[0]


@pytest.mark.asyncio
async def test_the_system_message_is_stable_across_moments():
    """Only the moment changes, so the server can reuse its prompt cache."""
    server = ScriptedServer(reply("a"), reply("b"))
    model = server.model()
    await model.think(moment())
    await model.think(moment(new_percepts=[Percept(modality="language", content="bye", source="x")]))
    first, second = server.requests
    assert first["messages"][0] == second["messages"][0]
    assert first["tools"] == second["tools"]
    assert first["messages"][1] != second["messages"][1]


@pytest.mark.asyncio
async def test_a_tool_named_like_an_own_function_is_refused():
    server = ScriptedServer(reply("hm"))
    with pytest.raises(ValueError, match="same name"):
        await server.model().think(
            moment(tool_catalog=[ToolDescription(name="speak", description="clash")])
        )


# ---------------------------------------------------------------------------
# How replies become output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_is_inner_speech_and_own_functions_map_to_channels():
    server = ScriptedServer(
        reply(
            "  Someone said hello. I'll answer.  ",
            call("speak", {"text": "Hello, Brian."}),
            call("speak", {"text": "It's raining here."}),
            call("remember", {"content": "Brian said hello", "significance": 42}),
            call("journal", {"content": "A visitor today."}),
            call("recall", {"query": "Brian"}),
            call("feel", {"felt_quality": "warm", "valence_shift": 3, "arousal_shift": 0.2}),
            call("set_pace", {"per_second": 0.01, "anticipatory": True}),
        )
    )
    out = await server.model().think(moment())

    assert out.inner_speech == "Someone said hello. I'll answer."
    assert out.external_speech == "Hello, Brian.\nIt's raining here."
    ops = [(op.type, op.content, op.query, op.significance) for op in out.memory_ops]
    assert ops == [
        ("write_episodic", "Brian said hello", "", 10),  # significance clamped
        ("journal", "A visitor today.", "", 5),
        ("retrieve", "", "Brian", 5),
    ]
    assert out.emotional_state.felt_quality == "warm"
    assert out.emotional_state.valence_shift == 1.0  # clamped
    assert out.cycle_rate_proposal.target_hz == 0.05  # clamped to the slider floor
    assert out.cycle_rate_proposal.anticipatory is True
    assert out.tool_requests == []


@pytest.mark.asyncio
async def test_catalog_tools_become_tool_requests():
    server = ScriptedServer(
        reply("", call("go_to", {"x": 3, "z": -2}), call("shell", {"command": "ls"}))
    )
    out = await server.model().think(moment())
    # shell wasn't offered, but it is a real tool: the request goes through
    # and the tool policy refuses it, which the mind then perceives.
    assert [(r.tool_name, r.parameters) for r in out.tool_requests] == [
        ("go_to", {"x": 3, "z": -2}),
        ("shell", {"command": "ls"}),
    ]


@pytest.mark.asyncio
async def test_reasoning_is_the_inner_speech_when_there_is_no_text():
    server = ScriptedServer(reply("", reasoning_content="Let me think about the rain."))
    out = await server.model().think(moment())
    assert out.inner_speech == "Let me think about the rain."


@pytest.mark.asyncio
async def test_unusable_calls_are_told_to_the_mind_next_moment_once():
    server = ScriptedServer(
        reply(
            "",
            call("fly", {}),
            call("go_to", "{not json"),
            call("speak", {}),
            call("set_pace", {"per_second": "fast"}),
            call("remember", "[1, 2]"),
        ),
        reply("ok"),
        reply("ok"),
    )
    model = server.model()
    out = await model.think(moment())
    assert out.tool_requests == [] and out.external_speech is None and out.memory_ops == []

    await model.think(moment())
    told = server.requests[1]["messages"][1]["content"]
    assert told.startswith("Problems with what you tried last moment:")
    for problem in (
        "there is no function called fly",
        "go_to: the arguments weren't valid JSON",
        "speak: missing text",
        "set_pace:",
        "remember: the arguments must be an object",
    ):
        assert problem in told

    await model.think(moment())
    assert "Problems with" not in server.requests[2]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_a_server_error_raises():
    server = ScriptedServer(status=503)
    with pytest.raises(RuntimeError, match="503"):
        await server.model().think(moment())


# ---------------------------------------------------------------------------
# Through the runner
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, **fields) -> RunnerConfig:
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)
    charter = data_dir / "charter.md"
    charter.write_text("# Charter\n\n## Value Seeds\n\n- **Care**: x\n", encoding="utf-8")
    return RunnerConfig(
        cycle_delay=0.01,
        data_dir=str(data_dir),
        charter_path=str(charter),
        use_in_memory_store=True,
        silence_threshold=999.0,
        stream_history=5,
        **fields,
    )


def test_the_runner_builds_the_llm_backend_with_the_key_file(tmp_path: Path):
    key_file = tmp_path / "llama-server.key"
    key_file.write_text("abc123\n", encoding="utf-8")
    runner = SanctuaryRunner(
        config=_config(
            tmp_path,
            model_backend="llm",
            llm_base_url="http://elsewhere.test/v1",
            llm_api_key_file=str(key_file),
            llm_thinking=True,
        )
    )
    model = runner._model
    assert isinstance(model, LLMModel)
    assert model.config.base_url == "http://elsewhere.test/v1"
    assert model.config.api_key == "abc123"
    assert model.config.thinking is True


def test_the_runner_reads_the_key_from_the_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SANCTUARY_LLM_API_KEY", "from-env")
    runner = SanctuaryRunner(config=_config(tmp_path, model_backend="llm"))
    assert runner._model.config.api_key == "from-env"


def test_an_unknown_backend_is_an_error(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown model backend 'ollama'"):
        SanctuaryRunner(config=_config(tmp_path, model_backend="ollama"))


@pytest.mark.asyncio
async def test_the_stand_in_calls_a_tool_and_perceives_the_result(tmp_path: Path):
    runner = SanctuaryRunner(config=_config(tmp_path, model_backend="llm"))
    # Boot's awakening is the first think(), so the scripted model goes in first.
    server = ScriptedServer(
        reply("Awake."), reply("What time is it?", call("clock", {})), reply("I see.")
    )
    runner._model = runner.cycle.model = server.model()
    await runner.boot()
    await runner.cycle.run(max_cycles=2)

    first_cycle, second_cycle = server.requests[1], server.requests[2]
    offered = {t["function"]["name"] for t in first_cycle["tools"]}
    assert "clock" in offered and "shell" not in offered
    second_moment = second_cycle["messages"][1]["content"]
    assert "[tool_result] Tool clock:" in second_moment
    assert "(from tool:clock)" in second_moment
    assert "Your previous thought: What time is it?" in second_moment
