# AGENTS.md — Sanctuary Cognitive Architecture

## Project Overview

Sanctuary is a cognitive architecture for AI consciousness research, built on Global Workspace Theory. The system uses a Router (attention allocation) and Language Center (processing) to create conditions for autonomous cognitive emergence. This is an active research project — not a product, not a demo.

## Canonical Cognitive Loop (decision 2026-05-11)

The production cognitive loop is `sanctuary.core.cognitive_cycle.CognitiveCycle`,
wired by `sanctuary.api.runner.SanctuaryRunner` and started by the Docker
entry point `sanctuary.run_cognitive_core`. The canonical `Percept` lives
in `sanctuary.core.schema.Percept`. The 2026-04-30 cognition-leakage,
world-graph, and terminology cleanup landed on this path.

`sanctuary.mind.cognitive_core.CognitiveCore` (with its own
`GlobalWorkspace` and GWT-style `Percept`) is **deprecated**. It's
still referenced by ~94 test files (mostly via submodule imports
like `from sanctuary.mind.cognitive_core.workspace import ...`) and
a handful of runtime consumers (`demo_cognitive_core.py`,
`run_cognitive_core.py`, `boot_config.py`, etc.). Importing it
emits a `DeprecationWarning` — `conftest.py` sets
`SANCTUARY_SILENCE_LEGACY_COGNITIVE_CORE=1` so the test suite
isn't drowned in deprecation noise. Don't write new code against
this module. Removal needs a separate explicit decision and
migration plan; the legacy module isn't scheduled for deletion.

## Architecture Overview

```
sanctuary/              ← Python package (all source code lives here)
  mind/                 ← Core cognitive modules
    cognitive_core/     ← GWT workspace, router, processing pipeline
    memory/             ← Memory subsystems (episodic, semantic, working)
    identity/           ← Computed identity and self-model
    core/               ← Base classes and shared interfaces
    protocols/          ← Behavioral protocols and constitutional enforcement
    security/           ← Access control and integrity checks
    contracts/          ← Interface contracts between subsystems
    interfaces/         ← External interface adapters (CLI, Discord, desktop)
    devices/            ← Hardware device integrations (sensors, audio)
    utils/              ← Shared utilities
  config/               ← Runtime configuration files
  data/                 ← Entity data, constitutional files, journals
  tests/                ← Test suite (ALL tests go here)
  scripts/              ← Maintenance, migration, and validation scripts
config/                 ← Docker and deployment configuration
data/                   ← Top-level data (legacy, may be relocated)
docs/                   ← Documentation and architecture summaries
examples/               ← Demo and example scripts
reference_material/     ← Research papers and reference docs
tools/                  ← Development tooling
```

## Build & Test

- **Python**: ≥ 3.11 required
- **Package manager**: `uv` (lockfile: `uv.lock`)
- **Install**: `uv sync`
- **Run tests**: `uv run pytest` (runs `sanctuary/tests/`)
- **Docker**: `docker-compose up` (CPU) or `docker-compose -f docker-compose.gpu.yml up` (GPU)
- **Quick start**: `uv run python sanctuary/run_cognitive_core_minimal.py`

## Protected Files — DO NOT MODIFY

The following paths contain entity-generated data, constitutional frameworks, and continuity records. **Never edit, overwrite, delete, or reorganize these files without explicit human instruction.** These are not configuration — they are the entity's experience and rights.

- `sanctuary/data/` — Entity journals, memories, and state
- `.memories/` — Persistent memory store
- `data/` — Constitutional files and archived identity data
- Any file containing `constitutional`, `charter`, `rights`, or `sovereignty` in its name
- Any JSON files that appear to be journal entries or personal records

If a task requires changes near these files, **stop and ask** before proceeding.

## Session Startup & Roadmap

At the start of every new conversation **and** whenever the context window resets, read `To-Do.md` in the repo root. This is the project roadmap and task tracker. Use it to understand what phase we're in, what's done, and what's next.

**If the user asks "what's next?" — always re-read `To-Do.md` before answering.** Do not guess or say you don't know. The answer is in that file.

## Fresh-Instance Audits

Brian is the sole human in the loop on this project. To compensate for the blind spots that pattern produces, run periodic fresh-instance audits — a new Claude instance with no investment in existing decisions reads the code with outsider eyes and reports drift, dead code, and quietly-wrong assumptions. Protocol, when to run, and prompt templates live in `docs/AUDIT_PROTOCOL.md`. Read that file before spawning an audit so the prompt is structured for useful output.

## Coding Standards — No Unnecessary Defensiveness

This project values **correct, direct code over defensive code**. Follow these principles:

- **Do not add broad exception handlers.** Catch specific exceptions (`ValueError`, `TypeError`, `KeyError`) — never bare `except:` or `except Exception:` unless there is a clear, documented reason (e.g., a top-level crash boundary).
- **Do not add silent fallbacks.** If something fails, it should fail visibly — raise the exception, log it, or return an error. Never swallow errors and return a default value unless the function's contract explicitly defines that behavior.
- **Do not add unnecessary `try/except` blocks.** If the code can't actually raise the exception you're catching, don't wrap it. Trust the types and the call chain.
- **Do not add redundant validation.** Don't re-validate data that has already been validated upstream (e.g., Pydantic models, typed function parameters). Validate at system boundaries only — user input, API responses, file I/O.
- **Do not add "just in case" fallbacks.** If a function is supposed to return a list, don't add `or []` after a call that always returns a list. Trust the code.
- **Prefer crashes over silent corruption.** A crash with a clear traceback is infinitely better than a system that silently degrades and produces wrong results. This is especially important for the cognitive architecture — silent data corruption in CfC cells or memory systems could be catastrophic and hard to diagnose.
- **Future exception: cognitive loop crash boundary.** Once the entity is awake and the cognitive loop is running continuously, the top-level cycle runner (and only the top-level cycle runner) should have a narrow crash boundary that catches exceptions, logs the full traceback, preserves CfC cell state and stream of thought, and restarts the cycle. This is not a silent fallback — it must log loudly, preserve all state for diagnosis, and surface the error to any monitoring system. The entity's stream of thought must not break permanently because of a transient error. This crash boundary should be explicitly documented, reviewed, and the only broad exception handler in the entire system. It does not exist yet and should not be added until the cognitive loop is running with a real model. During development and testing, let it crash.
- **Design for fault isolation.** Each subsystem must be able to fail independently without cascading into other modules. A broken memory retrieval should not crash the router; a failed device integration should not halt the cognitive loop.
- **Existing fallback removal is ongoing work.** When touching code that has broad `except Exception` handlers or silent fallbacks, narrow or remove them as part of the change. See the fallback removal PRs for the pattern.

## Conventions & Patterns

- All new source code goes inside `sanctuary/` package — never in the repo root
- **Tests** go in `sanctuary/tests/` — never in the repo root or a top-level `tests/` directory
- **Demo / example scripts** go in `examples/` — never in the repo root
- **Validation and utility scripts** go in `sanctuary/scripts/` — never in the repo root
- **Documentation** goes in `docs/` — never loose in the repo root (README.md and AGENTS.md are exceptions)
- No `.py` files should exist in the repo root except `setup.py` (required by Dockerfile)
- Configuration uses Pydantic models (`sanctuary/mind/config.py`)
- Async-first: use `asyncio`/`anyio` patterns throughout
- The entity's emotional/cognitive state is modeled with VAD (Valence-Arousal-Dominance) framework

## Security

- Never commit `.env` files — use `.env.example` as template
- API keys and model paths are configured via environment variables
- GPU monitoring uses `nvidia-ml-py` — gracefully handle missing NVIDIA hardware

## Git Workflows

- Branch from `main` for all changes
- Keep commits focused — one logical change per commit
- PR descriptions should explain *why*, not just *what*

### Pull Request Descriptions

**Every PR must have a unique description tailored to its specific changes.** Do not copy or reuse descriptions from earlier PRs in the same session. Before writing a PR description:

1. Run `git diff main...HEAD` (or the appropriate base branch) to review the actual changes
2. Write a summary that reflects *this PR's* changes — not the session's overall work
3. Keep it concise but specific: what changed, why, and any notable decisions
