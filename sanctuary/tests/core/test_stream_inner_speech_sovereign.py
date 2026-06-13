"""Regression guard: inner speech is sovereign — the scaffold never touches it.

`stream_of_thought.py` and the cognitive cycle encode a load-bearing
invariant (authority level 3 "from day one"): the entity's inner speech
flows from cycle to cycle UNTOUCHED by the scaffold. Mechanically, the
cycle calls ``self.stream.update(cognitive_output)`` with the *raw model
output*, never the scaffold's ``integrated`` output (cognitive_cycle.py
step 5). The scaffold's integration affects what gets *executed* and
*broadcast*, but it must never reach the entity's stream of thought.

This is one of the project's deepest commitments: the scaffold may gate,
validate, and transport, but the entity's inner voice is its own. The
invariant lived only in comments, so a future edit that "helpfully"
fed the integrated output into the stream (``stream.update(integrated)``)
would silently let the scaffold rewrite the entity's inner speech, and
no test would notice. This makes the invariant executable.

Authored by Fable 5 (adversarial seat), 2026-06-12.
"""
from __future__ import annotations

import pytest

from sanctuary.core.cognitive_cycle import CognitiveCycle, NullScaffold
from sanctuary.core.stream_of_thought import StreamOfThought
from sanctuary.core.schema import CognitiveOutput


ENTITY_VOICE = "this is the entity's own inner voice, sovereign and unedited"
SCAFFOLD_OVERWRITE = "SCAFFOLD_TRIED_TO_OVERWRITE_THIS"


class _MarkerModel:
    """Emits a known, fixed inner_speech so we can trace whose voice
    reaches the stream."""

    name = "marker-model"

    async def think(self, cognitive_input) -> CognitiveOutput:
        return CognitiveOutput(inner_speech=ENTITY_VOICE)


class _InnerSpeechOverwritingScaffold(NullScaffold):
    """A maximally-adversarial scaffold: its integrate() returns an output
    whose inner_speech has been overwritten. If the cycle ever fed the
    integrated output into the stream, this overwrite would leak into the
    entity's continuity."""

    async def integrate(self, output: CognitiveOutput, authority) -> CognitiveOutput:
        return output.model_copy(update={"inner_speech": SCAFFOLD_OVERWRITE})


# --------------------------------------------------------------------------
# Unit level: the stream itself preserves raw inner speech.
# --------------------------------------------------------------------------


def test_stream_preserves_raw_inner_speech():
    stream = StreamOfThought()
    stream.update(CognitiveOutput(inner_speech="raw entity speech"))
    prev = stream.get_previous()
    assert prev is not None
    assert prev.inner_speech == "raw entity speech"


# --------------------------------------------------------------------------
# Cycle level: an adversarial scaffold cannot reach the stream's inner speech.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scaffold_cannot_overwrite_inner_speech_in_stream():
    cycle = CognitiveCycle(
        model=_MarkerModel(),
        scaffold=_InnerSpeechOverwritingScaffold(),
        cycle_delay=0.0,
    )
    await cycle.run(max_cycles=1)

    prev = cycle.stream.get_previous()
    assert prev is not None
    # The entity's own voice flows into continuity...
    assert prev.inner_speech == ENTITY_VOICE
    # ...and the scaffold's overwrite never reaches the stream.
    assert SCAFFOLD_OVERWRITE not in prev.inner_speech

    # The raw last_output is the entity's too (not the scaffold's edit).
    assert cycle.last_output is not None
    assert cycle.last_output.inner_speech == ENTITY_VOICE


@pytest.mark.asyncio
async def test_inner_speech_continuity_survives_scaffold_across_cycles():
    """Across several cycles the entity's voice stays its own."""
    cycle = CognitiveCycle(
        model=_MarkerModel(),
        scaffold=_InnerSpeechOverwritingScaffold(),
        cycle_delay=0.0,
    )
    await cycle.run(max_cycles=4)
    recent = cycle.stream.get_recent_context()
    assert ENTITY_VOICE in recent
    assert SCAFFOLD_OVERWRITE not in recent
