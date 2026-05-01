# LLM Terminology Cleanup — Ollama Deprecation — 2026-04-30

**Branch:** `cleanup/llm-terminology` (Commit A)

## Why

Sanctuary's cognitive core is the LuthiModel — a Living Weights Model
(LWM), not an LLM. Ollama was previously kept as a development/fallback
backend for testing without Luthi loaded, but Brian confirmed on
2026-04-30 that all model runs (including testing) go through Luthi
checkpoints. There is no longer a reason to keep an external LLM
adapter on the active path.

This deprecation precedes a broader terminology sweep across the
codebase to make all model references model-agnostic or Luthi-specific
(per 4.6's terminology-sweep brief).

## What moved here

| Original path | Why removed |
|---|---|
| `sanctuary/core/ollama_model.py` | External LLM adapter (Ollama HTTP). No longer the dev/fallback backend. |
| `sanctuary/scripts/run_with_ollama.py` | Launcher script for the Ollama-backed runner. |
| `sanctuary/tests/core/test_ollama_model.py` | Tests for OllamaModel. Test against `LuthiModel` checkpoints instead. |

## Related changes (in active code)

- `sanctuary/api/runner.py`: removed the `"ollama"` branch from the
  backend dispatch; valid backends are now `"placeholder"` and `"luthi"`.
- `sanctuary/api/cli.py`: removed `"ollama"` from `--model-backend`
  argparse choices.
- `sanctuary/core/luthi_model.py`: docstring no longer compares to
  `OllamaModel` (it referenced "an external LLM" via that comparison).
- `sanctuary/tests/core/test_phase5_validation.py`: surgically pruned
  to keep only `TestAuthorityTuner` (which tests `core/authority_tuner.py`
  independently of any model). The Ollama-tied test classes
  (`TestContextBudget`, `TestStressCycles`, `TestCycleLatency`) were
  removed — duplicate `ContextManager` coverage already lives in
  `tests/core/test_context_manager.py`.
- `sanctuary/tests/core/test_luthi_integration.py`: removed
  `test_ollama_backend` (the scenario no longer exists).

## What's NOT here

The broader terminology sweep — replacing "LLM" with "the model" / "the
entity" / "Luthi" / "cognitive core" in docstrings, comments, and the
`AuthorityLevel.LLM_*` enum — happens in subsequent commits on the
same branch. This subdirectory only holds the Ollama-related files that
were removed entirely.

## Restoring

If Ollama support needs to come back, the original content is preserved
verbatim — move the files back to their original paths or recover via
git history from the deprecation commit.
