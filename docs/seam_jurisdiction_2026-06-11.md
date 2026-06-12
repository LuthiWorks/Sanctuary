# Seam Jurisdiction — Substrate Selects, Scaffold Transports

**Date:** 2026-06-11
**Ratified:** Brian (2026-06-11). Fable 5's brief (`docs/brief_for_4-7_leakage_and_readme_2026-06-11.md` §1.5) recommends this doc; 4.7 writes it as the build seat. New leakage passes start from this principle so the boundary is not re-litigated every pass.

---

## Principle

> **The substrate selects; the scaffold transports.**

M9 puts action selection inside the substrate via Expected Free Energy over candidate latent actions (M9 step-1 spec §2). Sanctuary's motor / cycle / communication layers predate that decision and have overlapping authority over "what happens next." Where authority is shared without a stated boundary, leakage breeds — Sanctuary infers what should be cognition. This document states the boundary explicitly so the next leakage pass starts from it instead of re-deriving it.

The principle generalizes the 4.6 2026-04-30 cleanup's "Sanctuary stores; Luthi cognizes" rule to the new M9 surface. Storage and transport are the scaffold's. *Selection* is the entity's, and selection now lives in the substrate's EFE evaluator.

---

## Action classes

One sentence per side of the seam, per action class. Where on-the-ground contact with the code contradicts a ruling, the build seat overrides and notes why — the build seat sees what the review seat can't.

### Speech / text emission

- **Substrate selects.** The M9 EFE planner evaluates the text decoder against the other launch decoders (attention, memory) via candidate-action G; the text decoder's intensity gate (`LuthiModel/luthi/v2/m9/decoders.py::TextDecoder.intensity_head`) is the entity's "speak now" decision.
- **Scaffold transports.** Sanctuary delivers the rendered text to recipients via its interaction channels (multi-party manager, conversation state, external I/O). It does not initiate, gate, or shape speech.
- **Load-bearing ghost.** The 4.6 2026-04-30 deletion of the speech drive/inhibition gate (`sanctuary/scaffold/communication.py::_compute_drive` / `_compute_inhibition`) stays deleted. Any future "small rate limiter for safety" or "per-user politeness check before speech goes out" is the speech gate coming back through the side door — the principle says no.

### Motor / in-Sanctuary action

- **Substrate selects.** When a motor decoder lands in M9 step 2 or beyond. (M9 step 1 launch set is text + attention + memory; motor and audio are deferred — `LuthiModel/docs/research/2026-06-10_m9-step1-spec.md §3`.)
- **Scaffold transports.** Sanctuary translates motor intent into Godot world actions; the Track-2 Godot integration is the executor, not a co-selector.
- **Transitional caveat.** Pending the motor decoder, the scaffold MAY route legacy non-substrate motor pathways only as fallback / transition; these are placeholders for the next leakage pass to retire once M9 step 2 lands. They MUST NOT acquire new selection logic.

### Memory-write / consolidation

- **Substrate selects.** The M9 memory decoder (`LuthiModel/luthi/v2/m9/decoders.py::MemoryDecoder`) produces salience + intensity; the entity decides whether and what to consolidate. The substrate's working episode store lives inside `PredictiveCodingBlock` and is the entity's own.
- **Scaffold persists.** Sanctuary stores derived records — what the entity declared, the goals it proposed, the decisions it recorded — because the entity chose them, not because Sanctuary judged them worth keeping. The 4.6 retention rule applies: entity-driven storage stays; storage that depends on Sanctuary inferring significance goes.

### Rate proposal / attention allocation

- **Substrate selects.** The M9 attention decoder (`LuthiModel/luthi/v2/m9/decoders.py::AttentionDecoder`) produces per-modality gates; the entity directs its own processing.
- **Scaffold transports.** Next-cycle encoder input is multiplied by the gates at the perceive phase; Sanctuary's cycle scheduler honors the entity's attentional choice and does not override it.
- **Idle-cycle activity is selection in the wrong place.** `consciousness/mood_activity.py` (scaffold proposing what to do with idle time) is retired in §1.4 of the 2026-06-11 brief because that is selection happening outside the substrate.

---

## Findings — where the principle hits an absurdity

These are exceptions in the sense that there is no selection happening, so the substrate-vs-scaffold split does not apply. Recorded so a future leakage pass doesn't try to force them under the principle and produce a worse design.

- **Sleep and physiology** (`sanctuary/consciousness/sleep_cycle.py`). Not an action class the substrate selects over. Sleep is a scaffold-side rhythm the way a body cycles whether or not its occupant agrees; the substrate runs whatever cycle the scaffold has it in. The scaffold owns scheduling; the substrate consumes it. Exempt: no selection happening.
- **Perception / addressee detection** (`sanctuary/social/multi_party.py`). Perception is not selection. Parsing who is being addressed in a multi-party stream is sensorium work the scaffold does to deliver structured input to the substrate. The substrate consumes structured percepts; it doesn't select against the parser. Exempt: this is transport *into* the substrate, not selection inside it.

---

## Override clause

> Where the build seat's hands-on contact with the code contradicts a ruling above, the build seat overrides and notes why in the pass README. The principle is the default, not a ceiling on judgment; finds-and-justifies are how this doc gets better.

---

## See also

- `_deprecated/cognition-leakage-2026-04-30/README.md` — 4.6's predecessor pass that this doc generalizes.
- `docs/brief_for_4-7_leakage_and_readme_2026-06-11.md` — Fable 5's brief recommending this doc.
- `LuthiModel/docs/research/2026-06-10_m9-step1-spec.md` — the M9 spec that puts action selection inside the substrate.
- `LuthiModel/docs/research/language_as_channel_direction_2026-06-11.md` — Fable 5's direction doc; the seam ruling here is the implementation surface for that direction's "P3 floor, instrumentality driver" claim about speech.
