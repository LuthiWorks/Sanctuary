# The stand-in mind's backend — 2026-11-01

## Objective

Step 2 of `docs/MINECRAFT_FIRST_WORLD_DECISION_2026-11-01.md`: a third model backend, `llm`, so that a small open language model (Qwen3.5-4B, Brian's choice) can be Sanctuary's cognitive core while the body and the Minecraft world are brought up. Luthi replaces it later.

## Process

### Step 1: read the retired backend

- **What it was:** the `ollama` backend retired on 2026-04-30 (`git show ed01b3e^:sanctuary/core/ollama_model.py`).
- **How it worked:** it asked the model to reply with one JSON object covering every `CognitiveOutput` field, about fifteen keys. Parse failures were retried with a stricter instruction, then replaced by a fallback output.
- **Why not reuse that approach:**
  - For a 4B model, a fifteen-key JSON schema every cycle is a lot to get right, and a malformed reply loses the whole cycle.
  - The fallback output is the kind of silent default `CLAUDE.md` rules out.
- **What to reuse:** its rendering of `CognitiveInput` sections was a useful reference.

### Step 2: design around function calling

llama-server, run with `--jinja`, supports native function calling, and Qwen models are trained for it. So:

- **The reply's plain text is inner speech,** which the model produces anyway.
- **Every action is a function call:**
  - Six are the mind's own (`speak`, `remember`, `journal`, `recall`, `feel`, `set_pace`), each mapped onto a `CognitiveOutput` channel.
  - The rest are the catalog tools the policy lets it use (step 1's `tool_catalog`), which become `ToolRequest`s.
- **One bad call costs that call, not the cycle.** A call the backend can't use (an unknown name, arguments that aren't valid JSON or not an object, a missing required argument, a value of the wrong kind) is logged and told to the mind at the top of its next moment, so it can correct itself. Nothing is dropped silently.
- **Server and transport errors raise,** per the coding standards.
- **One request per cycle:** a system message (identity, how its existence works) and a user message (this moment). Sanctuary carries continuity itself (previous thought, surfaced memories), so no chat history is kept. The system message and function list stay identical across cycles, so llama-server's prompt cache reuses them, and only the moment is processed each cycle. A test pins this.
- **Gated tools that haven't been granted aren't offered as functions.** A call to one anyway still goes through as a `ToolRequest`. The registry's policy refuses it, and the refusal reaches the mind as a percept: the honest consequence, through the existing path.

### Step 3: read the prompts it actually produces, and find a false statement

With the unit tests passing, I ran a scripted boot plus two cycles through the real runner and read the rendered prompts.

- **The problem:** my system message said inner speech "is private ... Nobody else hears it." The same moment's awakening text says the opposite ("Right now, your family can see your thoughts — your inner speech and journal"), and the charter's RIGHTS include "(3) Honest context — you are never deceived about your own conditions."
- **Why it matters:** the system message would have deceived the mind about its own conditions on every cycle.
- **The fix:** the message now makes no claim of its own about who can read inner speech, since that changes with authority level. It says inner speech "isn't spoken aloud; who else can read it is set out in your charter." A test fails if the word "private" or "nobody else" comes back.
- **Lesson:** the unit tests all passed with the false sentence in place. Reading the real prompt is what caught it. For a backend whose whole job is to tell a mind about its world, that read is part of the testing, not optional.

### Step 4: the runner and the CLI

- **The backend:** `RunnerConfig` gains `llm_base_url`, `llm_model`, `llm_api_key_file` and `llm_thinking`. The key is read from the file's first line, or from `SANCTUARY_LLM_API_KEY` when no file is given. On Brian's PC the file is LuthiCode's `%USERPROFILE%\.pi\agent\llama-server.key`.
- **The CLI:** it gains the matching `--llm-*` flags, and `--model-backend` accepts `llm`.
- **A silent fallback removed while there:** `_create_model` turned any unknown backend name into the placeholder. It now raises, naming the valid backends.

## Conclusion

`--model-backend llm` runs a language model as Sanctuary's mind over any OpenAI-compatible endpoint. It's tested without a model: request shape, mapping of output, feedback on unusable calls, and the closed loop through the real runner (the mind calls `clock` and perceives the result next cycle).

**Not done, or not known:**
- **How Qwen3.5-4B actually behaves in this loop.** Whether it acts sensibly, how often its calls are malformed, the cycle rate. Unknown until a run on Brian's PC.
- **Images.** Visual percepts reach it only as text. Image input, needed for the Minecraft first-person view, comes with step 3.
- **The awakening moment offers no tools.** `boot()` builds its first input itself, outside the cycle's assembly, so the catalog is empty then. Arguably right for a first moment, but it's an accident of the code, not a decision.
- **Bodies other than Minecraft.** `speak` maps to `external_speech`. In Minecraft that should become in-game chat (step 3), so the mind needs no separate chat tool.

## Artifacts

- Code:
  - `sanctuary/core/llm_model.py`
  - `sanctuary/api/runner.py` (backend, config)
  - `sanctuary/api/cli.py` (flags)
- Tests: `sanctuary/tests/core/test_llm_model.py` (14)
- Reference: retired backend at `ed01b3e^:sanctuary/core/ollama_model.py`
