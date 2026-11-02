# Closing the action loop, model-agnostic half — 2026-11-01

## Objective

Step 1 of `docs/MINECRAFT_FIRST_WORLD_DECISION_2026-11-01.md`, which is the model-agnostic half of the 2026-08-01 audit's W1.

The audit found that the entity could not act. `CognitiveInput` had no field for the tool catalog, so no model was ever told which tools exist. The runner's execution path, from `tool_requests` to a `tool_result` percept on the next cycle, was correct but unreachable.

The stand-in mind for Minecraft (a ~4B language model) needs that path before anything else. The Luthi half of W1, routing M9 decoder selections into `CognitiveOutput`, is deliberately left for later ("then we'll worry about Luthi").

## Process

### Step 1: confirm the audit's finding still holds

- **No history to account for:** `git log --since=2026-08-01` on `schema.py`, `runner.py`, `luthi_model.py`, `cognitive_cycle.py` and `placeholder.py` shows no commits since the audit.
- **The gap is still there:** `CognitiveInput` has no catalog field, `ToolRegistry.get_catalog()` has no caller outside its own tests, and `SanctuaryRunner._execute_tool_requests` works. `test_ws_server.py::test_cycle_driven_tool_request_actuates_world` already drives it with a hand-built output.
- **A booted runner registers 26 tools.** Six are gated and, under the default deny-all `ToolPolicy`, cannot run: `run_code`, `write_file`, `launch_app`, `network_reach`, `network_scan` and `shell`.

### Step 2: decide what "the catalog" shows

- **The choice:** list only the tools the entity can use, or list everything and mark what is usable.
- **Listing everything, marked, is chosen.** The graduated-capability design means the entity's reach grows over time, and hiding gated tools would make that growth invisible to it. A new `ToolDescription.available` flag carries the policy's answer, and each model adapter decides how to present it.
  - A function-calling language model should be offered available tools only, or it will call tools that are certain to be refused.
  - Luthi may be shown the whole surface.
- **The confirmation-hook case:** "available" is True there too, because the call may run; the hook can still decline. This is the one judgment call in the definition, so it is documented on `ToolDescription` and `ToolRegistry.is_available`.

### Step 3: write the tests first, and find a second bug

`sanctuary/tests/core/test_action_loop.py` was written before the change. One test checks that context compression carries a small input through unchanged, because the new field had to survive compression to reach the model.

- **It failed for an unexpected reason.** Comparing the dumps showed `ContextManager.compress()` drops two existing fields, not just the new one: `self_authored_identity` and `communication_urges`.
- **How:** it rebuilt `CognitiveInput` field by field, and those two fields, added later, were never added to the rebuild.
- **The consequence:** the entity's self-authored identity, assembled every cycle in `CognitiveCycle._assemble_input`, never reached the model. `LuthiModel._format_input` reads `ci.self_authored_identity` (`luthi_model.py:849`) and always found it empty. Communication urges were dropped too, though nothing currently fills them in (`_assemble_input` never sets the field), so that loss had no effect yet.
- **This is the audit's pattern again: correct code, not connected,** found here only because a test checked the whole object rather than the sections it compresses.
- **The fix:** `compress()` now returns `cognitive_input.model_copy(update=...)`, replacing only the sections it compresses. Every other field, current or future, is carried through unchanged. The test compares the full `model_dump()`, so a field dropped later fails it.

### Step 4: wire the catalog

- **The cycle:** `CognitiveCycle(..., tool_catalog=provider)`. It's a provider called every cycle rather than a fixed list, because tools are registered after the cycle is constructed and gated tools can be granted while running.
- **The registry:** `ToolRegistry.is_available(name)`, and each `get_catalog()` entry now includes `available`.
- **The runner:** passes `self._describe_tools`, which turns registry entries into `ToolDescription`s.
- **Unchanged:** `LuthiModel._format_input` still tells Luthi nothing about tools. That belongs to the Luthi half of W1.

## Conclusion

- **The model-agnostic action loop is closed.** Any model that reads `CognitiveInput.tool_catalog` can request a tool and perceive its result on the next cycle. This is tested end to end through the real runner with a scripted model.
- **A dropped-field bug is fixed:** the entity's self-authored identity and communication urges now reach the model. That changes what Luthi sees each cycle: it will now receive its self-authored identity text whenever there is any. A reviewer should treat that as a behavior change, even though it is the documented intent.

**Not done, and still open:**
- **Tool requests bypass the scaffold.** The tool hook receives the model's raw output, not the scaffold-integrated one, so requests skip scaffold validation. `ToolPolicy` is the only gate. That matches the code's current intent, but step 6 of the cycle's docstring says only validated actions are executed. Worth a reviewer's eye.
- **Broad exception handlers remain:** `_execute_tool_requests` still wraps its body in `except Exception`, contrary to CLAUDE.md's coding standards. It is untouched here, and narrowing it is a separate change.
- **The Luthi half of W1.**

## Artifacts

- Tests: `sanctuary/tests/core/test_action_loop.py` (6 tests; 5 failed before the change)
- Code:
  - `sanctuary/core/schema.py` (`ToolDescription`, `CognitiveInput.tool_catalog`)
  - `sanctuary/core/context_manager.py`
  - `sanctuary/core/cognitive_cycle.py`
  - `sanctuary/tools/registry.py`
  - `sanctuary/api/runner.py`
- Plan: `docs/MINECRAFT_FIRST_WORLD_DECISION_2026-11-01.md`
