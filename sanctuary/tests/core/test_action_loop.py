"""The action loop, model-agnostic half (audit 2026-08-01, W1).

The audit found the entity could not act: ``CognitiveInput`` had no field for
the tool catalog, so no model was ever told what it could do, even though the
runner's execution plumbing worked. These tests pin the closed loop:

1. compression carries every input field through (it silently dropped two),
2. the cycle puts the tool catalog into each ``CognitiveInput``,
3. the runner's catalog reflects the tool policy (gated tools not granted are
   listed but marked unavailable),
4. a tool the model requests in one cycle comes back as a percept in the next.

No model server or network is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.cognitive_cycle import CognitiveCycle
from sanctuary.core.context_manager import ContextManager
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    CommunicationUrgeSignal,
    Percept,
    ToolDescription,
    ToolRequest,
)


class RecordingModel:
    """Records every input it receives; replies with a scripted output per cycle."""

    def __init__(self, script=None):
        self.inputs: list[CognitiveInput] = []
        self._script = script or (lambda ci, n: CognitiveOutput(inner_speech="..."))

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        self.inputs.append(cognitive_input)
        return self._script(cognitive_input, len(self.inputs))


CATALOG = [
    ToolDescription(
        name="clock",
        description="What time it is.",
        category="information",
        safety="open",
        available=True,
    ),
    ToolDescription(
        name="shell",
        description="Run a shell command.",
        parameters={"command": "The command to run"},
        category="system",
        safety="gated",
        available=False,
    ),
]


def test_compression_keeps_every_field():
    """A small input fits every budget, so compression must return it unchanged.

    ContextManager rebuilt CognitiveInput field by field and left out
    ``self_authored_identity`` and ``communication_urges``, so the entity's own
    words about itself never reached the model. Any field added later would
    have been dropped the same way.
    """
    ci = CognitiveInput(
        new_percepts=[Percept(modality="language", content="hello", source="test")],
        charter_summary="Care. Honesty.",
        self_authored_identity="I am still deciding what I am.",
        communication_urges=[
            CommunicationUrgeSignal(drive_type="question", intensity=0.4, content="ask")
        ],
        tool_catalog=CATALOG,
    )
    compressed = ContextManager().compress(ci)
    assert compressed.model_dump() == ci.model_dump()


@pytest.mark.asyncio
async def test_cycle_puts_the_tool_catalog_in_the_input():
    model = RecordingModel()
    cycle = CognitiveCycle(model=model, cycle_delay=0.0, tool_catalog=lambda: CATALOG)
    await cycle.run(max_cycles=1)
    assert model.inputs[0].tool_catalog == CATALOG


@pytest.mark.asyncio
async def test_cycle_without_tools_gives_an_empty_catalog():
    model = RecordingModel()
    cycle = CognitiveCycle(model=model, cycle_delay=0.0)
    await cycle.run(max_cycles=1)
    assert model.inputs[0].tool_catalog == []


# ---------------------------------------------------------------------------
# Through the real runner
# ---------------------------------------------------------------------------


@pytest.fixture
def runner_config(tmp_path: Path) -> RunnerConfig:
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
    )


@pytest_asyncio.fixture
async def booted_runner(runner_config: RunnerConfig) -> SanctuaryRunner:
    runner = SanctuaryRunner(config=runner_config)
    await runner.boot()
    return runner


@pytest.mark.asyncio
async def test_runner_tells_the_entity_its_tools(booted_runner: SanctuaryRunner):
    model = RecordingModel()
    booted_runner.cycle.model = model
    await booted_runner.cycle.run(max_cycles=1)

    catalog = {t.name: t for t in model.inputs[0].tool_catalog}
    assert set(catalog) == {t["name"] for t in booted_runner.tools.get_catalog()}
    # Open tools can be used; gated tools not granted are listed, marked unavailable.
    assert catalog["clock"].available is True
    assert catalog["shell"].safety == "gated"
    assert catalog["shell"].available is False


@pytest.mark.asyncio
async def test_runner_marks_granted_gated_tools_available(runner_config: RunnerConfig):
    runner_config.enabled_gated_tools = ["write_file"]
    runner = SanctuaryRunner(config=runner_config)
    await runner.boot()
    model = RecordingModel()
    runner.cycle.model = model
    await runner.cycle.run(max_cycles=1)

    catalog = {t.name: t for t in model.inputs[0].tool_catalog}
    assert catalog["write_file"].available is True
    assert catalog["shell"].available is False


@pytest.mark.asyncio
async def test_a_requested_tool_returns_as_a_percept_next_cycle(
    booted_runner: SanctuaryRunner,
):
    """The closed loop: see the catalog, request a tool, perceive its result."""

    def script(ci: CognitiveInput, n: int) -> CognitiveOutput:
        if n == 1 and any(t.name == "clock" and t.available for t in ci.tool_catalog):
            return CognitiveOutput(
                inner_speech="What time is it?",
                tool_requests=[ToolRequest(tool_name="clock")],
            )
        return CognitiveOutput(inner_speech="...")

    model = RecordingModel(script)
    booted_runner.cycle.model = model
    await booted_runner.cycle.run(max_cycles=2)

    second = model.inputs[1]
    results = [p for p in second.new_percepts if p.source == "tool:clock"]
    assert len(results) == 1
    assert results[0].modality == "tool_result"
    assert results[0].content.startswith("Tool clock:")
