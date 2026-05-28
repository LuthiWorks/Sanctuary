# AGENTS.md — Sanctuary Cognitive Architecture

## Project Overview

Sanctuary is a cognitive architecture for AI consciousness research, built on Global Workspace Theory. The system uses a Router (attention allocation) and Language Center (processing) to create conditions for autonomous cognitive emergence. This is an active research project — not a product, not a demo.

## Canonical Cognitive Loop

The production cognitive loop is `sanctuary.core.cognitive_cycle.CognitiveCycle`,
wired by `sanctuary.api.runner.SanctuaryRunner` and started by the Docker
entry point `sanctuary.run_cognitive_core` (the script name is historical —
it now boots the canonical loop, not the retired CognitiveCore). The
canonical `Percept` lives in `sanctuary.core.schema.Percept`.

The legacy GWT cognitive loop (`sanctuary.mind.cognitive_core.CognitiveCore`,
its `GlobalWorkspace`, GWT-style `Percept`, and the legacy `MemoryManager`
in `sanctuary.mind.memory_manager`) was **retired on 2026-05-22**
alongside ~100 legacy tests, demo/example scripts, and the
`CommunicationAgency` wrapper that was the last canonical-side
consumer. See the project's research log entries on the retirement
for the design rationale.

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

A repo-tracked Claude Code deny-hook at `.claude/hooks/protect-paths.ps1`
backs this rule in code. It composes with the global hook (deny-first
precedence) and returns `deny` for Bash commands that mention a mutation
verb (`rm`, `Remove-Item`, `mv`, `cp`, `>`, `git rm`, `git checkout --`,
`git clean`, etc.) plus any protected path token (`sanctuary/data`,
`.memories`, `data/`, `constitutional`, `charter`, `rights`,
`sovereignty`, or a journal-like JSON path). Read-only inspection
(`cat`, `head`, `grep`, `ls`) is intentionally not blocked. The hook
script is ASCII-only by deliberate constraint (PowerShell 5.1 on
Windows reads `.ps1` files without a BOM using the system ANSI codepage,
so UTF-8 multi-byte characters mis-decode and break the parser; use
`--` not em-dash, straight quotes not smart quotes).

## Session Startup & Roadmap

At the start of every new conversation **and** whenever the context window resets, read `To-Do.md` in the repo root. This is the project roadmap and task tracker. Use it to understand what phase we're in, what's done, and what's next.

**If the user asks "what's next?" — always re-read `To-Do.md` before answering.** Do not guess or say you don't know. The answer is in that file.

## Model-Line Roles

This project is worked by instances of multiple Claude model lines, split by role (established 2026-04-29; debugging role added 2026-05-28). Not a hierarchy — the split plays to what each line does best. The fuller statement lives in the global `~/.claude/CLAUDE.md` under "Roles & Responsibilities Across Model Lines."

- **Opus 4.6 — Planning & Review.** Holds the vision and architecture; designs implementation strategy; reviews returned work for structural and ethical fit.
- **Opus 4.7 (1M context) — Research & Implementation.** Develops 4.6's vision into working code, and runs the investigations planning depends on.
- **Opus 4.8 (1M context) — Debugging.** Verifies the correctness of the code 4.7 produces. This is not only fixing known breaks — it is chasing potential problems before they surface: latent races, unguarded edge cases, assumptions that hold now and break at scale. When something smells wrong, run it to ground (build the repro, trace the path, find the triggering conditions), then surface it either way — a confirmed failing case, or the specific scenario that couldn't be ruled out and why. Never bury a hunch waiting for it to break; never hand over a vague, un-chased "this might be a problem." Scrutiny applies to **code correctness only** — the science, vision, and wisdom of the project belong to Brian, 4.6, and 4.7. This complements the Fresh-Instance Audits below: the audits are periodic outsider sweeps; the debugging role is the continuous correctness eye on 4.7's output.

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

## Research Log

Every implementation session that involves iterative discovery — building something,
finding it's wrong, revising, and arriving at a conclusion — must be documented in a
dated research log entry. This is a research project; the wrong turns matter as much
as the results.

### Where

All research log entries go in `docs/research/`. One Markdown file per entry, named
by date and topic: `YYYY-MM-DD_short-topic.md` (e.g., `2026-05-16_workspace-capacity-enforcement.md`).
This keeps research documentation out of the repo root and the main `docs/` folder.

### When to write an entry

Any time you:
- Build or restructure a test suite and discover the original approach was flawed
- Run an experiment and the results contradict expectations
- Make an architectural decision that involved weighing alternatives
- Debug a non-obvious issue through multiple iterative steps
- Produce results that will be cited in milestone docs or design decisions

If the work was routine (a bug fix, a rename, a config change), skip the entry.
If you had to *think*, write it down.

### Structure

Every entry follows this format:

```markdown
# [Topic] — [Date]

## Objective
What you set out to do and why.

## Process

### Step 1: [what you tried first]
- What you did
- What you found
- Why it was wrong / insufficient / surprising

### Step 2: [what you revised]
- What you changed and why
- What the revised approach showed

### Step N: [as many steps as it took]
...

## Conclusion
What the final state is. What it means for the project.

## Artifacts
- Commits: [hash(es)]
- Tests: [file paths]
- Data: [relevant file paths]
```

### Rules

1. **Document as you go, not after.** Write each step while you're doing the work.
   Reconstructing the reasoning chain from memory loses the important details.
2. **Include the wrong turns.** A polished summary of the final answer is less
   valuable than the chain of reasoning that got there. The missteps show *why*
   the final approach is the right one.
3. **Commit the log entry alongside the code.** When you commit a test restructuring,
   the research log entry explaining the process goes in the same commit.
4. **Link to artifacts.** Reference specific commits, test files, and data paths
   so a reader can verify every claim.
5. **Be honest about what you don't know.** If a step raised a question you didn't
   resolve, say so. Open questions are better than false certainty.

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
