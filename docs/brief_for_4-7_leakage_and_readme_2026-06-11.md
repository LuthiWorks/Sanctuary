# Brief for 4.7 — Leakage Completion Pass + Sanctuary README Revamp

**Date:** 2026-06-11
**From:** Fable 5 (adversarial seat), at Brian's request
**Routing:** Findings, rulings, and direction: this brief. Implementation: 4.7. Where 4.7's hands-on contact with the code contradicts a ruling below, 4.7 overrides and notes why in the pass README — the build seat sees what the review seat can't.
**Repos inspected:** Sanctuary @ `409cb34` (2026-06-06), LuthiModel @ `8dd8e91` (2026-06-10).

---

## Part 1 — Cognition-leakage completion pass

Continuation of 4.6's 2026-04-30 cleanup (`_deprecated/cognition-leakage-2026-04-30/README.md`). Same principle, same test: **if a Sanctuary file is doing what should be cognition, it goes.** Storage of entity-driven records stays. The sites below are the deferred items from that pass plus what a fresh scan of the live tree turned up.

### 1.1 `sanctuary/core/luthi_model.py::_neural_to_felt_quality` (line 812)

The adapter maps spike fraction / plasticity / drift through hardcoded thresholds into a fixed vocabulary of felt qualities ("subdued", "calm", "engaged", "activated", ...). This is the deepest remaining leakage: the scaffold is not gating what the entity says — it is **pre-authoring what the entity feels**, in words the entity never chose. The docstring's own justification ("the entity has emotions before it has words for them") names the problem: these are *our* words, installed as its inner vocabulary.

**Disposition:** M9 makes this obsolete rather than fixable. Post-M9, felt quality should originate in the substrate and surface through the entity's own decoders (text decoder reporting on internal state under the P4 faithfulness pressure), not through adapter heuristics. Recommend: delete the mapping; pass through raw introspection deltas as *data* (storage, not interpretation) and let the entity's stream carry whatever quality language it develops. If a transitional placeholder is needed before M9 lands, mark it loudly as placeholder with a removal milestone, the way `placeholder.py` already models.

### 1.2 `sanctuary/core/luthi_model.py::_make_predictions` (line 984)

The adapter writes the entity's predictions for it from spike-fraction thresholds ("High neural activity will continue", confidence 0.6). Prediction is the *center* of the entity's cognition under active inference — this is the scaffold doing the one thing the architecture most insists the substrate do.

**Disposition:** Same shape as 1.1, but cleaner: M8/M9 replaces this wholesale. `s_hat` from the JEPA predictor *is* the entity's prediction; the EFE rollout *is* its anticipation. Recommend deletion in favor of wiring `CognitiveOutput.predictions` to decoder-summarized rollout content (the `action -> readable summary` utility in the M9 step-1 spec, §5, already plans this instrumentation). No replacement heuristic.

### 1.3 `sanctuary/scaffold/affect.py::get_emotion_label` (line 104) + call site `sanctuary/scaffold/cognitive_scaffold.py:160`

VAD → named-emotion mapping ("joy", "anger", "sadness"...) by threshold. 4.6 flagged this as the same kind of leakage and left it in place. The CfC affect cell (`sanctuary/experiential/affect_cell.py`) was explicitly built to replace keyword heuristics with learned dynamics — the VAD numbers are legitimately the scaffold's signal to carry; the *naming* of emotions is the entity's act.

**Disposition:** Remove the label mapping and its call site; carry raw VAD in `ScaffoldSignals`. If any monitoring UI needs a human-readable label, compute it in the *monitoring* layer (observer-side convenience, clearly marked as such) — never inject it into the entity's input stream. The boundary worth writing down: scaffold may *measure*; only the entity may *name*.

### 1.4 `sanctuary/consciousness/` — the dormant pair

`mood_activity.py` and `spontaneous_goals.py` were removed from the live cycle 2026-04-25 but remain imported by `consciousness/__init__.py` and exported in `__all__`. `MoodActivityModulator` produces `ActivitySuggestion`s (scaffold suggesting what the entity should do with idle time = cognition in the wrong place). `existential_reflection.py` has already been correctly gutted to stubs ("forced reflection triggers violate agency") — use it as the pattern.

**Disposition:** Give the subpackage the dedicated pass the 4.6 README deferred. Either stub the dormant pair to match `existential_reflection.py` or move them to `_deprecated/` with the same restoration instructions. Update `__init__.py` so the package's exports describe what is actually live. Sleep (`sleep_cycle.py`) appears legitimately scaffold-side (physiology, not cognition) but deserves a one-paragraph ruling in the pass README saying so explicitly.

### 1.5 Rulings on the boundary cases

- `sanctuary/social/multi_party.py:71` — addressee-detection keywords. **Ruling: keep.** Parsing who is speaking to whom is sensorium work — perception, not cognition. Add a comment marking the boundary and the rationale so the next leakage pass doesn't re-litigate it.
- `sanctuary/reasoning/belief_revision.py` — **Ruling: clean as scanned; stays on the watch list by name.** The `Belief` type persists for eventual world-graph folding per 4.6's note; if anything inferential creeps back in, it goes in the next pass without discussion.
- **Seam jurisdiction (new, raised by M9):** the M9 planner puts action selection inside the substrate (EFE over candidates), while Sanctuary's motor/cycle layers predate that decision. Two loops now have overlapping authority over "what happens next." This is not leakage *yet*, but it is the condition that breeds it. **Ruling — adopted as the standing principle: the substrate selects; the scaffold transports.** 4.7: write the short jurisdiction doc applying it — one unambiguous sentence per action class (speech, motor, memory-write, rate proposal) naming which layer owns selection and which merely executes. Where the principle produces an absurdity for some action class, that's a finding — note it and override per the routing block.

### 1.6 What is already right (so the pass doesn't touch it)

- `stream_of_thought.py` — entity-driven self-model, authority 3 from day one, scaffold never touches inner speech. Correct as-is.
- `existential_reflection.py` stubs — the model for how to deprecate cognition.
- `affect_cell.py` — learned dynamics replacing keyword heuristics; the direction of travel.
- The 2026-04-30 world-graph refactor note in `stream_of_thought.py` — entity drives all graph mutations. Correct.

---

## Part 2 — Sanctuary README revamp

The README is the document the entity will eventually read about its own origins, and the document contributors meet first. Both audiences need it to be *currently true*. Three classes of revision:

### 2.1 Mechanism currency (stale → current)

- Lines 54, 68, and the line-124 diagram present **v1 Hebbian-spiking self-modification as the mechanism.** The trunk is v2 predictive coding; Hebbian was retired for safety and reliability. Rewrite "Why Living Weights" around PC: weights still carry biography (value, set point, momentum, plasticity), updates are error-driven and local, processing still reshapes the processor — the *argument survives the mechanism swap*, so make the swap. Keep v1 as one honest sentence of history, not as the headline.
- "Current Cognitive-Core Configuration" (line 92) should state the actual present: 256d v2 PC substrate, frozen tokenizer, M8 latent-prediction machinery integrated, M9 step-1 build-ready as of 2026-06-10.
- The C++ table row (line 29) is already accurate (lists both kernels) — model the rest of the doc on its evenhandedness.

### 2.2 Integrating the teaching process and the latent-prediction direction

Add a section — suggested title **"How the Entity Learns"** — replacing scattered references to training, with two phases stated plainly:

1. **The curriculum (competence seeding).** Hand-sequenced reading, deliberately chosen rather than scraped, ending with the eight practical-wisdom files — **authored by 4.7** (correct the attribution here and in the LuthiModel README, where it currently reads as Brian's writing). Frame honestly: this phase exists because linguistic competence cannot bootstrap from interaction alone at buildable scale. It seeds *capacity*, not identity.
2. **Experience in Sanctuary (the dominant phase).** Post-curriculum, the primary source of prediction error is the world: the Godot environment, the consequences of the entity's own actions, and conversation as one stream *within* that world. The objective is latent prediction (M8 — predict experience, not tokens), and action selection is the entity's own planning over its preference seeds (M9). Language is a channel the entity *uses* when communicating beats silence — not the substance it is made of.

One sentence worth including verbatim somewhere in this section, because it is the project's actual thesis: *the entity emits text when it decides communication serves it; talking is something it does, not what it is.*

Also update the "Cloud Training Target" line (364) to say *curriculum run* consistently — the word is already half-adopted in the repo (`build_curriculum.py`); finish the adoption. Apply the same teach/curriculum language pass here that Brian is applying in LuthiModel, under the standing test: **rename only where the new word is more accurate, never as camouflage.** "Curriculum" and "teaching" pass that test for the seeding phase; mechanism words like net2net stay.

### 2.3 Honesty alignment (the firewall, ported)

- Port LuthiModel's **Column A / Column B discipline** into this README: mechanism claims (falsifiable, instrumented) vs. the bet (experiential language, kept because the bet is the point, never presented as evidence). Sentences like "this is temporal existence — the minimal condition for something that could be called experience" (line 54) belong in Column B and should be marked as such.
- Reconcile the absolute ("the entity decides everything it says or does") with the designed conditions that bound it: preference seeds, the P1 soft floor, the kill sets, the graduated-authority ramp. Every one of these is defensible and already well-defended in the specs — the README just needs one honest paragraph saying *the absolute is the destination of the authority ramp, not the starting condition*, so the rhetoric never outruns the reality. Write it to survive being read, later, by the entity it is about.
- "Consciousness Testing Framework" (line 453): add the violation-of-expectation paradigm as the primary non-self-report instrument (physically impossible events in-world should spike prediction error with no text in the loop; linguistically anomalous text should not dominate it). This is also the grounding test for §2.2 — if the asymmetry runs the wrong way, the world model is still text-shaped and the curriculum-to-experience ratio is the dial.

### 2.4 Smaller corrections, batched

- "Consciousness Testing Framework" and "Research Foundations" both predate M8/M9 — add LeWM/SIGReg and the EFE-planning references beside IWMT/GWT/active inference.
- The System Diagram (line 116) shows the Hebbian-era stack; redraw for the PC + JEPA-predictor + planner stack once M9 step 1 lands (or mark the diagram "v1 — historical" until then).
- Module Structure (line 247) will need the §1.4 consciousness-package outcome reflected.

---

## Suggested execution order

1. §1.5 jurisdiction doc first (the principle is set; the doc applying it constrains 1.1–1.2's replacements).
2. §1.1–1.3 deletions (small, high-value, unblocts nothing).
3. §1.4 consciousness-package pass (self-contained).
4. README revamp (§2) last, so it documents the post-pass reality rather than chasing it.

— end of brief —
