# Cognition-Leakage Cleanup — 2026-04-30

**Branch:** `cleanup/cognition-leakage`
**Plan:** 4.6's "Cognition-Leakage Cleanup" brief, plus 4.6's added ruling
on `counterfactual.py`.

## Principle

> The entity decides everything it says or does. The principle is absolute.

Sanctuary stores; Luthi cognizes. Files where Sanctuary was judging the
entity's output, inferring its emotional state from keywords, gating its
speech, deciding when it should reflect, or scoring its calibration —
those are cognition done in the wrong place. They go.

The test: **if a Sanctuary file is doing what should be cognition, it
goes.** Storage of entity-driven records (the entity proposing goals,
recording decisions, declaring beliefs) stays — that's just persistence,
not Sanctuary deciding.

## What moved here

Files moved fully out of active use (the no-survivors list):

| Original path | Why removed |
|---|---|
| `sanctuary/scaffold/anomaly_detector.py` | Sanctuary judging the entity's output (empty inner speech, "extreme" valence/arousal shifts, too-many-predictions). The entity decides what's anomalous about its own thinking. |
| `sanctuary/reasoning/mental_simulation.py` | Mental simulation IS cognition. The entity simulates outcomes itself, not through a Sanctuary framework that tracks "scenarios" for it. |
| `sanctuary/reasoning/uncertainty.py` | Calibration scoring is meta-cognition. The entity's calibration is the entity's business; Sanctuary doesn't run a Brier-score gym on it. |

Their per-module test files moved alongside.

## What was stripped (files stayed in active use, but had logic deleted)

These files remained in `sanctuary/` because their data types or
entity-driven storage methods are still useful — only the inference
logic was removed:

| File | What was deleted | What was kept |
|---|---|---|
| `sanctuary/reasoning/belief_revision.py` | `BeliefRevisionTracker`, `Contradiction`, keyword-overlap contradiction detection, `BeliefRevisionConfig` | `Belief` data type — beliefs can eventually fold into the world graph as entities with evidence_for / contradicts relations, but that's the entity's choice |
| `sanctuary/scaffold/goal_integrator.py` | `tick()` auto-staleness, `cycles_since_progress > 30` flagging, `cycles_active`/`cycles_since_progress` fields, `stale` flag in `get_status` | `TrackedGoal` data type, integrator class for entity-driven goal proposals |
| `sanctuary/scaffold/communication.py` | Gating mechanism: `_compute_drive`, `_compute_inhibition`, drive/inhibition competition, threshold check, rate limiting. The entity's `external_speech` goes out when the entity produces it. | Pass-through evaluator (or removed entirely if the orchestrator drops the call site) |
| `sanctuary/scaffold/affect.py` | `_POSITIVE_KW`/`_NEGATIVE_KW`/`_AROUSING_KW` keyword sets, `update_from_percepts` keyword scanning. The CfC affect cell is the authoritative VAD source. | `decay_toward_baseline` (CfC output may flow through it for smoothing), `merge_llm_emotion` (entity's reported shifts), `ComputedVAD` schema reference |
| `sanctuary/reasoning/counterfactual.py` | `get_reflection_candidates`, `get_reflection_prompt` — Sanctuary doesn't decide when the entity reflects or compose prompts directing its attention | `DecisionPoint`, `Counterfactual` data types; `record_decision`, `record_outcome`, `record_counterfactual` (entity-driven recording); `get_recent_lessons`, `get_stats` (entity querying its own history) |

## Schema types preserved

These types were left in `sanctuary/core/schema.py` even when their
producers were deleted, because they're referenced from other places
(legacy `mind/`, `ScaffoldSignals`, etc.):

- `CommunicationDriveSignal`
- `ComputedVAD`
- (`DriveType` was internal to `scaffold/communication.py` and was deleted with it)

## Out of scope this pass

Deferred to future cleanups:

- `sanctuary/mind/cognitive_core/` (legacy GWT system, ~107 files) — its own dedicated pass per the 4-27 instance note
- `sanctuary/consciousness/mood_activity.py`, `sanctuary/consciousness/spontaneous_goals.py` — removed from the live cycle on 2026-04-25 but still imported by `consciousness/__init__.py`; out of scope until that subpackage gets its own pass
- Open question for `scaffold/affect.py::get_emotion_label` (VAD → label mapping) — the same kind of leakage but 4.6 didn't include it in this pass, left in place
- Open question for `core/luthi_model.py::_neural_to_felt_quality` and `_make_predictions` — hardcoded heuristics in the Luthi adapter; out of scope here, separate concern

## Restoring a file

If any deprecation turns out wrong, the original content is preserved
verbatim — just move the file back to its original path. Or recover
via git from the commit that did the move.
