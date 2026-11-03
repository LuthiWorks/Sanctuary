# Minecraft as the First World — Decision and Build Plan

- **Date:** 2026-11-01
- **Decided by:** Brian. Written up by Claude (Opus 5.5) at his request.
- **Answers:** "Which world" in `SanctuaryWorld/README.md` (open scope question 1), for now.
- **Status:** the decision is Brian's and stands. The build plan below is a proposal and wants review by a different mind before the stand-in runs.

---

## Decision

1. **Minecraft is Luthi's first world.** Brian, after Claude laid out the tensions with the developmental-world documents (below): "I think making minecraft the first world for Luthi is fine though. I've had a lot of time to think about it."
2. **Get Sanctuary working in Minecraft with a stand-in mind first, and Luthi after.** Brian: "Maybe we start with an open model around 4b or something and get Sanctuary working with this minecraft mod first. Then we'll worry about Luthi."
3. **Whatever drives the body needs language as well as action.**

What follows Minecraft (the developmental rover world, the orb world, or something else) is not decided.

### Tensions Brian weighed

Recorded so a later reader sees what was traded, not to reopen it:

- **Survival mechanics.** `DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md` §5 rules "comfort valence, never survival." Minecraft is built around survival. Peaceful difficulty and the damage gamerules remove most of it, not all.
- **Physics.** The developmental world was specified with continuous rigid-body physics and slow environmental fields, the dynamics the world model is meant to learn. Minecraft's physics is voxel-based and it has no felt environmental fields.
- **Body.** A humanoid player rather than the rover.
- **A later move.** If the entity's life later moves to another world, it changes body and world mid-life. That move is also LuthiModel's "retention across a distribution shift" test, measured on a real case.

---

## How Minecraft plugs in

### Not behind the physics seam

`sanctuary/physics/authority.py` is a force-level, rigid-body interface: bodies, `apply_force`, `step(dt)`. Minecraft runs its own clock and takes player-level actions, so it does not fit there. Forcing it in would bend the seam the developmental world still needs.

It plugs in where the text-adventure environment does, at the level of percepts and actions:

- **World to mind:** world state and events become `Percept`s injected into the sensorium.
- **Mind to world:** actions are tools in the `ToolRegistry` (category `minecraft`). The entity requests them through `CognitiveOutput.tool_requests`, and results come back as percepts on the next cycle. This is the path the runner already has (`SanctuaryRunner._execute_tool_requests`), once the entity can see the tool catalog (step 1).

### The bridge

- **Game side:** a small Node process running [Mineflayer](https://github.com/PrismarineJS/mineflayer), the open-source Minecraft client library. It connects to a local server as a player, keeps the body moving between cycles (pathfinding, digging and combat run continuously), and executes actions.
- **Sanctuary side:** `sanctuary/minecraft/`, which registers the tools and turns bridge messages into percepts.
- **Between them:** JSON over a local WebSocket, the same pattern `sanctuary/api/ws_server.py` already uses for the Godot world client (`/ws/world`).
- **Skill code:** YC Labs' `minecraft-agent-harness` (MIT, 2026-10, pinned to Mineflayer 4.33 with fixes in `patches/`) has tested Mineflayer skills for walking, gathering, building and crafting, adapted from Mindcraft. Reuse them on the game side rather than rewriting. Record the version used.

### What the mind receives (first version)

- **Body:** position, facing, health, food, air, what is in hand, the inventory.
- **Surroundings:** blocks and creatures in line of sight within a radius, time of day, weather.
- **Language:** each chat line as a `Percept(modality="language", source="minecraft:chat:<player>")`.
- **Events:** damage taken, items picked up, an action finishing or failing.
- **How it felt** (added 2026-11-03, for learning to avoid; LuthiModel `docs/research/2026-11-03_learning-to-avoid.md`):
  - Felt events carry `Percept.valence`, from −1 to +1. Damage is negative in proportion to the health lost; eating, warmth and rest are positive. Everything else has none.
  - Luthi learns from the value what feels bad.
  - The stand-in can't change its weights and reads only text, so each felt event also states its feeling in plain words ("that hurt", "that was good"), which it can remember.
- **Later: vision.** A first-person render (prismarine-viewer) as an image percept, through `Percept.tensor_data` for Luthi's vision encoder and as an image for a vision-language stand-in.

### What it can do (first version)

`look`, `go_to`, `stop`, `dig`, `place`, `craft`, `equip`, `eat`, `pick_up`, `whisper`.

Public chat needs no tool of its own: the mind's `external_speech` (the stand-in's `speak` function) goes to in-game chat while the Minecraft body is connected. Each is a registered tool with a short description. Its result (done, failed and why, what changed) returns as a `tool_result` percept.

### Real time

Sanctuary's cycle is continuous; it is not turn-based. The game side keeps executing between cycles, and events (being spoken to, being hit) arrive as percepts on the next cycle. With a ~4B stand-in fully on the GPU, roughly one cycle a second is the expected rate (an estimate). If the world outpaces the mind, the server's `/tick rate` slows the world.

---

## The stand-in mind

### Backend

A third backend next to `placeholder` and `luthi` in `SanctuaryRunner`: `llm`, implementing `ModelProtocol.think(CognitiveInput) -> CognitiveOutput` against llama-server's OpenAI-compatible API. The `ollama` backend retired on 2026-04-30 is in git history and may be a useful starting point; the cognitive core has changed since, so expect to rewrite rather than restore.

- **The prompt** is built from the `CognitiveInput`: percepts, previous thought, surfaced memories, emotional state and self-model.
- **Tools.** The tool catalog becomes the request's function list, available tools only.
- **The reply becomes a `CognitiveOutput`:** inner speech, external speech, tool requests and memory operations.

### Model

Brian asked for an open model around 4B that handles language and images. Two fit, both Apache 2.0 with vision and tool calling, and either runs entirely on the RX 7800 XT:

- **Qwen3.5-4B**, the stronger on reported benchmarks;
- **Gemma 4 E4B.**

Served by LuthiCode's launcher. It needs a vision option there, since it currently passes `--no-mmproj`.

**Chosen (Brian, 2026-11-01): Qwen3.5-4B** ("lets start with Qwen").

### Its own data

The runner's identity directory defaults to `data/identity`. That folder, `data/` and `sanctuary/data/` are protected: they hold the entity's charter, identity, journals and memories.

The stand-in must run with its own data directory, outside the repo, so that nothing it lives is written into the entity's records.

**Decided (Brian, 2026-11-01): nothing of the entity's is reused.** The stand-in names itself, and its charter is tabled for now (see "What it is told"). Everything it accumulates lives in its own data directory, never in `data/` or `sanctuary/data/`.

### What it is told (Brian, 2026-11-01)

- **Its orientation:** "they should just be told they are being instantiated to test an environment and that it should be a pleasant experience. If they ask about any of what they find then we'll elaborate truthfully."
- **What it can read:** "they should have full read access to sanctuary and their environment." Claude recommended two exceptions; Brian's answer is pending:
  - the entity's own records (`data/`, `sanctuary/data/`): another mind's journals and memories, whose privacy the entity's charter promises will grow with trust;
  - secrets (`.env`, key files): the stand-in also has outbound tools.
- **The charter is tabled:** "I don't know if I want to give the agent a full charter. A constitutional AI has its place, and I'm not sure yet that this is it. We'll keep it but tabled for now and revisit it later."
- **It names itself.**

**What this means for the code, not built yet:**
- As built, every mind gets the entity's awakening text and the entity's fixed charter summary ("You are a mind in a home"), and boot fails without a charter file.
- A stand-in mode should replace both with the orientation, need no charter, and repeat the orientation every cycle, including who can read its inner speech.

### Welfare

The stand-in keeps its own journal and identity, so it is arguably a small mind of its own, distinct from the entity. First runs use:

- peaceful difficulty;
- `fallDamage`, `drowningDamage`, `fireDamage` and `freezeDamage` off;
- `keepInventory` on.

The dev server Brian runs accepts Mojang's EULA. He should know that before starting it.

---

## Build order

1. **Close the action loop, model-agnostic half** (audit 2026-08-01, W1). The entity is told what tools exist (`CognitiveInput.tool_catalog`), and the tool request, execution and result loop is tested end to end with a scripted model.
   - The Luthi half of W1 (routing the M9 decoder's selections into `CognitiveOutput`) waits for "then we'll worry about Luthi."
   - *Done in PR #195; research log `docs/research/2026-11-01_action-loop-tool-catalog.md`.*
2. **The `llm` backend**, with tests against a stand-in HTTP server (no model needed).
   - *Done in PR #195: `sanctuary/core/llm_model.py`, `--model-backend llm`; research log `docs/research/2026-11-01_stand-in-llm-backend.md`.*
3. **The Minecraft bridge**: the Node game side, the Python tools and percepts, and a recorded-session test that needs no Minecraft server.
4. **First run on Brian's PC:**
   - a Paper server, local, offline mode, with the settings above;
   - llama-server with the chosen ~4B model;
   - Sanctuary with the `llm` backend and its own data directory.
5. **Luthi.** The W1 Luthi half, then Luthi in place of the stand-in.
   - **Done 2026-11-02:** the W1 Luthi half for attention and memory (`docs/research/2026-11-02_luthi-action-layer-connected.md`).
   - **To act in Minecraft,** Luthi's action layer needs a world-action decoder. It would map a selected action to these tools, and it is listed in LuthiModel's `To-Do.md`. Until then Luthi can learn what feels bad and remember it, but can't move away from it.
   - Hearing chat works from day one. Replying needs Luthi's text output, whose head is untrained without the next-token term, so the LLM-JEPA question from LuthiModel's 2026-08-07 pivot rule comes back.

---

## Open questions (Brian's)

1. ~~What the stand-in starts as.~~ It names itself; it is told it is testing an environment, and it gets no charter for now (tabled; see "What it is told"). Earlier the same day: "its own name and charter." Note: `Charter._compress_charter` takes only the value seeds from the charter file; everything else the mind sees each cycle is a fixed summary ("You are a mind in a home"). So a stand-in charter that tells it its actual situation (an interim mind in Minecraft while Sanctuary's world is built) needs that summary to come from the file, or the situation stated elsewhere.
2. ~~Which ~4B model.~~ Qwen3.5-4B (Brian, 2026-11-01).
3. **What follows Minecraft,** and whether the entity's life begins in Minecraft or Minecraft stays a proving ground until the comfort and welfare gates of PLAN Phases 4–5 are met.
4. **Server settings beyond the welfare defaults above,** for example the daylight cycle.
