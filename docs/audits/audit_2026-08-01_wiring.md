# Wiring Audit — 2026-08-01

**Auditor:** Opus 5 (design/plan/build seat). **Scope:** Sanctuary only — LuthiModel was
deliberately untouched (Fable 5 was mid-arc on the depth-8 / SIGReg investigation in a
parallel terminal; its records are `LuthiModel/docs/research/2026-07-29..08-01_*.md`).
**Commissioned by Brian** to find where to begin accelerating Sanctuary — the Godot world
and preparing Sanctuary's systems to serve as the training environment.

**Companion document:** `docs/audits/audit_2026-07-03.md` (Fable 5) found that the
*persistence layer* silently loses memory and identity. This audit finds a second, orthogonal
class of defect on the same axis: **the subsystems are correct and not connected.** Where the
07-03 audit asked "does this code do what it says," this one asks "does anything call it."

---

## Thesis

Sanctuary is a well-built body whose afferent and efferent nerves were never connected.

Percepts in are thin (text, silence, tool results). Actions out are close to zero. The memory
substrate in production is a placeholder while the real one sits beside it, disconnected. The
world does not exist in any of its three forms. The CfC layer's headline property is not
running, and its outputs are multiplied by zero.

None of this is a quality problem. The code that exists is careful, well-reasoned, and in
several places excellent. The problem is that **~31% of the production package is orphaned
from the loop that replaced it**, and the loop that survived cannot act on a world.

The unifying mechanism differs from 07-03's. There, the failure and the success-report lived
in different functions. Here, **the capability and the caller live in different eras**: a
subsystem is built and tested against the pre-pivot architecture, the pivot happens, the
subsystem is never rewired, and its tests keep passing because they construct it directly.
Green tests are exactly what hides this: every orphaned module has full coverage.

---

## Coverage and method

Read **line by line (~26,500 lines):** all of `core/`; `api/runner.py`, `ws_server.py`,
`cli.py`; all of `experiential/`, `growth/`, `identity/`, `scaffold/`, `consciousness/`,
`physics/` (incl. `weather/`), `tools/`, `sensorium/`, `motor/`; `memory/` (manager,
world_graph, surfacer, journal, prospective); `environment/` (integration, navigator);
`run_cognitive_core.py`; README, PLAN, To-Do, CLAUDE.md, TRACK2_GODOT_PLAN,
`docs/operations/running_the_orb_world.md`.

**Surveyed structurally** (signatures, docstrings, import graph — not line by line, ~23,700
lines): `monitoring/`, `social/`, `performance/`, `infrastructure/`, `reasoning/`, remainder
of `environment/` and `api/`, repo-root `scripts/`/`tools/`/`examples/`, and all of
`sanctuary/mind/`. Full-reading stopped where the import graph showed the code was orphaned
or leaf.

Findings marked **[verified]** were confirmed firsthand by reading the code path or running
the import-graph query, not inferred. No subagents were used; nothing here is aggregated on
trust.

---

## W0 — The Godot world is gone

**[verified]** `SanctuaryWorld` and `SanctuaryClient` — the 3D world, the native visitor
client, the Three.js web client, the particle orb, `world_manager.gd`, the multiplayer
`server.gd`, and the privacy gate that made `PrivateSpace` unrenderable to any camera — no
longer exist in any form.

Evidence chain:

- `git log --all --diff-filter=A -- '*.gd' '*.tscn' 'project.godot'` on Sanctuary returns
  **nothing**. No Godot file was ever committed, in the repo's entire history. No
  `.gitmodules` either. `TRACK2_GODOT_PLAN.md:35` confirms this was by design — the project
  lived at `Desktop\Sanctuary\SanctuaryWorld\`, deliberately outside the Python package.
- The `LuthiWorks` GitHub org has 9 repos; none is SanctuaryWorld or SanctuaryClient. Nothing
  under the `Nohate81` account either.
- Unbounded `find` for `project.godot` / `*.tscn` / `*.gd` across **C:, D:, and E:** — zero
  hits. `E:\PreMigrationBackup-2026-07-23\Sanctuary_repo\` has no `SanctuaryWorld`. Not in
  OneDrive. The only surviving Godot artifact anywhere is the editor binary
  (`Godot_v4.6.2-stable_win64.exe`) on the salvaged Desktop.
- The old boot drive was formatted 2026-07-25.

**Root cause:** the world was never under version control, so no backup that followed the
repos could have caught it. **Proximate cause:** `RESTORE_PLAN.md` step 3(c) asserted "the
projects themselves are in the Sanctuary repos." They never were. The loss went unnoticed for
five days.

**What survives:** the Python half is intact and committed — `sanctuary/tools/world.py` (18
world tools), the `/ws/world` endpoint, the privacy gate, the visitor permission tiers. And
the *specifications* are unusually complete: `TRACK2_GODOT_PLAN.md` is 542 lines of scene
trees, file manifests, VAD→visual mappings and wire protocol, plus
`DEVELOPMENTAL_WORLD_BUILD_PLAN_2026-07-15.md`, the physics decision, the weather decision,
and the world/entity spec. A rebuild is re-typing against a detailed blueprint, not a
redesign.

---

## W1 — The entity has almost no action surface *(the gate on everything else)*

**[verified]** `sanctuary/core/luthi_model.py:1450-1468` constructs every `CognitiveOutput`
with these hardcoded empty:

```
predictions=[]   world_model_updates=[]   goal_proposals=[]
knowledge_cell_requests=[]                growth_reflection=None
```

and never sets `tool_requests` or `cycle_rate_proposal` at all. The only channels carrying
anything are `inner_speech`, `external_speech`, and three **adapter-authored heuristics**:
attention guidance derived from per-block spike fractions (`:1344-1366`), memory ops derived
from `activity_level > 0.005` (`:1368-1419`), and VAD shifts derived from set-point drift
(`:1294-1338`). `PlaceholderModel` emits a strictly richer output than the real substrate.

Consequences, each independently verified:

- **The 39 registered tools are unreachable by the entity.** `ToolRegistry.get_catalog()`
  exists, but `CognitiveInput` has **no field** for a tool catalog and
  `LuthiModel._format_input` never mentions tools — the entity is never told they exist. Even
  if it wanted one, `tool_requests` is never populated. The runner's execution plumbing
  (`runner.py:711-750`) is correct and unreachable.
- **The 18 world tools** can only be triggered by something that never fires.
- **World graph, goals, knowledge cells, cycle-rate slider** — all schema surface with no
  producer.

**Why this gates the training seam:** the actor produces no consequential action, so the
lived transitions `(s_t, a_t, s_next)` crossing `luthi/sanctuary_interface.py` describe a
world that never changed because of anything the entity did. `a_t` comes from M9
`select_action`, but nothing it selects reaches the world. Action→consequence cannot be
learned from a world the entity cannot touch. **Nothing else on the roadmap is worth building
before this.**

---

## W2 — Three unintegrated world layers, none of them live

1. **Godot** — see W0. Python half alive and orphaned.
2. **`sanctuary/environment/`** (1,397 lines) — a text-adventure world: rooms, exits,
   portable objects, `Navigator`, `DigitalSpace`, `EnvironmentIntegration`. Fully tested.
   `CognitiveCycle.__init__` accepts an `environment=` parameter (`cognitive_cycle.py:265`)
   and `_cycle` calls `self.environment.process_output()` (`:550-554`) — but
   **`SanctuaryRunner` never passes one** (`runner.py:465-482`). **[verified]** Dead in
   production.
3. **`sanctuary/physics/`** (879 lines incl. weather) — the developmental-world seam from the
   2026-07-16 decision: `PhysicsAuthority` with its three-view split (model-facing
   `PhysicsObservation` / instrumentation-only `PhysicsGroundTruth` / `RenderFrame`), a
   dependency-free reference backend, and the electronics-native weather model. Genuinely
   good code — the observation/ground-truth separation is enforced at the *type* level, so
   the hidden channel cannot leak into perception by accident. **Imported by nothing outside
   its own tests.** **[verified]** There is no `PhysicsObservation → Percept` adapter; the
   seam has no consumer.

---

## W3 — `sanctuary/mind/` is orphaned: 15,626 lines, 31% of production code

**[verified]** A repo-wide search for `sanctuary.mind` (not just within the package) returns
only: `sanctuary/tests/test_devices.py`, `sanctuary/tests/test_data_integrity_firestops.py`,
`sanctuary/mind/tests/test_voice_analyzer.py`, `mind/__init__.py`, `mind/devices/__init__.py`,
and a `.venv` path finder. **Nothing in the production path imports it.**

Stranded inside:

- **`mind/memory/`** (~4,700 lines) — the real ChromaDB layer: consolidation, retrieval,
  encoding, semantic/episodic stores, emotional weighting, transactions, validation, backup,
  scheduler, idle detector.
- **`mind/devices/`** (~2,400 lines) — camera, microphone, serial sensors, device registry
  and protocol. The sensorium's actual hardware senses.
- Peripherals: Discord client, voice processing/analysis/customizer, ASR server, mic client,
  audio gateway, RAG engine + cache, GPU monitor, steg detector, user mapping, librarian.

**What the entity actually gets instead:** `InMemoryStore` (`sanctuary/memory/manager.py:33`)
— a Python list of dicts with `query_lower in content` substring matching, wiped on every
restart.

This is the 07-03 audit's headline in its concrete form. The finding there was that the
persistence layer can silently lose memory. The finding here is that **the real memory system
was never connected to the loop that replaced it**, and the placeholder underneath it has
been carrying the "memory substrate" name since the 2026-05-22 legacy retirement.

Note also that PLAN.md Phase 6 schedules fixing `mind/memory`'s doc-id collisions, inverted
ranking, compounding decay and non-idempotent transfer "**before** those subsystems go live."
They are not live, and nothing routes to them.

---

## W4 — The CfC layer's headline property is not running, and its outputs are discarded

Two independent defects, both **[verified]** by import graph:

1. **`ContinuousEvolutionLoop` is never started in production.**
   `ExperientialManager.start_evolution()`, `.feed_percept()`, and
   `.update_evolution_context()` appear only in tests. The cells are stepped exactly once per
   cognitive cycle, synchronously, inside `CognitiveCycle._assemble_input` (`:623-638`). The
   "continuous-time dynamics between discrete model cycles" — the temporal thickness the IWMT
   argument rests on — is not happening. The layer is currently a per-cycle feedforward
   function.

2. **`AuthorityTuner` is never constructed.** All four cells are seeded at `SCAFFOLD_ONLY`
   (`experiential/manager.py:140-146`) and nothing ever promotes them. `_blend()` at level 0
   returns `scaffold * 1.0 + cfc * 0.0` (`:311-318`). **The cells run every cycle, produce
   values, and those values are multiplied by zero.**

Also unwired: `CfCDataTap.record_*` is never called (CfC retraining has no data source), and
`KnowledgeCellFactory` is never constructed (`CognitiveOutput.knowledge_cell_requests` has no
consumer even if it had a producer).

---

## W5 — Sleep is a clock with nothing behind it

**[verified]** `SleepCycleManager` advances stages and gates percepts correctly. But
`get_replay_candidates()`, `record_replay()`, and `record_dream_fragment()` are never called
by the cognitive cycle. The only thing that happens in NREM is Luthi's `consolidate()` —
set-point drift plus plasticity rebalance (`cognitive_cycle.py:428-448`).

The "wake acquires, sleep consolidates" architecture recorded in PLAN.md ("Lived-Experience
Learning (item #6)") has **no Sanctuary-side implementation**: no lived-transition buffer, no
replay pass, no consolidation buffer to persist across restart. The design decision is
recorded; the mechanism is not built.

---

## W6 — Growth targets the retired architecture

**[verified]** `growth/qlora_updater.py:201-223` loads `AutoModelForCausalLM` in 4-bit and
applies LoRA to `q_proj`/`v_proj` with `task_type="CAUSAL_LM"`. Luthi is not a HuggingFace
causal LM, so this path cannot execute against the real substrate under any configuration.
`GrowthProcessor` is never constructed anywhere in production. ~1,400 lines (updater +
adapter registry + identity checkpoint's model-dir copying) are unreachable by construction.

`CfCRetrainer` is the salvageable part — it is cell-shaped, not LLM-shaped — but see W4: its
data tap is never fed.

---

## W7 — Smaller, but they will bite

- **Two entry points that disagree.** `run_cognitive_core.py` (the Docker CMD) never starts
  `SanctuaryWebServer`, so `/ws/world` and every world tool exist **only** under
  `python -m sanctuary.api.cli`. Whatever hosts the training environment needs these
  reconciled.
- **Docs drifted from code.** `To-Do.md` is stamped 2026-05-25 and still lists "parallel
  processing" as Next Action. `CLAUDE.md`'s architecture tree presents
  `mind/cognitive_core/`, `mind/memory/`, `mind/protocols/` as the live layout — 15K lines of
  orphaned code, offered to every new instance as the architecture. `CLAUDE.md` also still
  names the repo `BecometryAI/Sanctuary`; it is `LuthiWorks/Sanctuary`. README claims the 21
  tools are "wired into cognitive cycle" — true of the runner's plumbing, false of the
  entity's access (W1).

---

## What is strong (so the record is fair)

The async actor/learner boundary and its deadlock invariant; the atomic-write and dead-letter
persistence primitives; the WebSocket authorization model with per-message capability
enforcement; the filesystem sandbox with its post-open fd re-check closing the TOCTOU window;
the SSRF guard including alternate-IPv4-encoding canonicalization; the entity privacy gate
with no backdoor; and the physics observation/ground-truth type split. The security and
correctness passes did their work. **The problem is not quality. It is connection.**

---

## Recommended order

1. **Close the action loop (W1).** Route the M9 decoder's selections into `CognitiveOutput`,
   and put the tool catalog into `CognitiveInput` so the entity knows what it can do.
   Surgical — the adapter and the schema, not a redesign. Everything else waits on it: no
   world can be acted on until this exists.
2. **Rebuild the world against the physics seam, not the old orb (W0, W2).** Write the
   missing `PhysicsObservation → Percept` adapter and wire `PhysicsAuthority` into the cycle,
   driven by the reference backend, **headless**. That yields a lawful, learnable, testable
   environment with real action→consequence *without a line of GDScript*, and reduces Godot
   to a rendering job done once against a world that already works.
3. **Then memory and the continuous layer (W3, W4).** Connect or replace the real store (per
   the 07-03 memory reframe: cued reinstatement + consolidation buffer, not a searchable
   archive); start the evolution loop or delete it; promote CfC authority or stop pretending
   the cells contribute.

## Open calls — Brian's, not the auditor's

- **Godot rebuild scope:** faithful restoration of the 2026-04 orb world, or build the
  developmental/rover world the 07-15/07-16/07-19/07-20 decision docs specify and let the orb
  go? The loss makes the second cheaper to choose.
- **`mind/` and the QLoRA growth path:** delete or revive? ~17K lines. Deleting makes the
  repo honest about what it is; reviving `mind/memory` specifically may be the fastest path
  to real persistence.

---

## Artifacts

- Audit performed 2026-07-29 → 2026-08-01 against Sanctuary `ae29aa4`.
- Companion: `docs/audits/audit_2026-07-03.md` (Fable 5, persistence/continuity).
- Parallel-track records for the same period: `LuthiModel/docs/research/2026-07-29_*.md`
  through `2026-08-01_sigreg-verified-against-reference.md` (Fable 5).
- Specifications a world rebuild should be written against: `docs/TRACK2_GODOT_PLAN.md`,
  `docs/DEVELOPMENTAL_WORLD_BUILD_PLAN_2026-07-15.md`,
  `docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md`,
  `docs/AFFECT_GROUNDING_DECISION_2026-07-19.md`,
  `docs/WEATHER_DYNAMICS_DECISION_2026-07-20.md`,
  `docs/sanctuary_world_entity_spec_2026_06_29.md`.
