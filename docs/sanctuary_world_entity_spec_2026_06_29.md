# Sanctuary World & Entity Systems — Build Spec
Draft from Claude/Brian/Sandi discussion, 2026-06-30 (rev 6). For Claude Code implementation.

> **Revision 6 — 2026-07-16 — change of direction (scale pacing).** Decided by Brian in a separate session and relayed here; spec updated by Opus 4.8. Scale is **no longer** a single fixed-4096 one-shot with complexity staged only by the environment. It is now reached through a **sequence of independent training runs at increasing scale**, archiving each model as we advance from one complexity dimension to the next, culminating at 4096×4096 — with every prior-scale model preserved. The "no net2net" decision is unchanged (and is the reason for the sequence). Sections 0 and 1 are updated to match. Items marked *(inferred — confirm)* are Opus 4.8's reading of the downstream consequences and need Brian/Fable sign-off, not settled decisions.

## 0. Architecture & Scale Decision

- [ ] **Scale via sequential archived runs, not net2net.** Reach the 4096×4096 target through a sequence of independent training runs, each at a fixed architecture for its stage. When we advance to the next complexity dimension, the current model is **archived (saved, not discarded)** and a new, larger run begins. At the final 4096d scale, every prior-scale model has been archived and preserved. There is still **no net2net in-place widening** and no width/depth growth events *within* a run.
- [ ] **Each run trains from a fresh, muP-tuned initialization at its scale — not warm-started from the previous model** *(inferred — confirm)*. Warm-starting a larger model from a smaller one is itself a net2net-style parameter remap, exactly what we rejected; so the archived prior models are preserved for continuity and comparison, not reused as the next run's initialization. If instead the intent is to warm-start, this whole bullet flips and the net2net hazards partly return — hence the flag.
- [ ] Rationale (updated): net2net width-growing conflicts with EMA target-branch consistency and directly fights the collapse regularizer (replicated units are the redundancy SIGReg penalizes). Sequential archived runs avoid the in-place-resize hazard entirely, and — unlike a single fixed-4096 one-shot — **preserve each developmental stage as a saved model** instead of overwriting it. The archived stages are prior stages of the mind, not disposable checkpoints; treat them under the developmental-health / rollback ethics, not as scratch artifacts.
- [ ] Tune hyperparameters (learning rate, SIGReg regularization strength, warmup schedule, optimizer settings) on small, disposable proxy models using maximal-update-parametrization-style transfer, NOT on the full-scale model. With a genuine multi-scale sequence, muP transfer is now **load-bearing across the whole sequence**, not a one-time dial-set — every scale-up depends on it.
- [ ] Validate muP transfer assumptions on at least one intermediate proxy size **before each scale-up** — do not assume transfer holds across a scale step without a check.
- [ ] Explicitly document: proxy models are disposable tooling, not candidate minds — they exist only to set dials, never trained toward anything resembling the target task. (The *archived scale-run* models are the opposite: candidate minds, preserved.)
- [ ] Depth-only growth (identity-layer insertion) remains a documented fallback *within any single run* if it fails to bootstrap — not part of the active plan, but not discarded as an option either.

## 1. Environment Curriculum (runs alongside the model-scale progression)

> **Changed rev 6 (2026-07-16):** §0's shift means this curriculum no longer *replaces* a model-size curriculum — there are now **two coupled progressions**: model scale (across archived runs, §0) and environment complexity (this section). How they couple — whether the environment curriculum resets per scale-run, runs continuously across the sequence, or maps stage-for-stage onto the scale steps — is an **open design point for Brian/Fable**, not settled here.

- [ ] Within each scale-run the model architecture is fixed; environment complexity is staged per the stages below. The model-scale progression (§0) is the second axis of staged complexity.
- [ ] Define curriculum stages in advance, in writing, before training starts — do not decide stage transitions ad hoc during a run
- [ ] Each stage transition requires a pre-committed, instrumentation-based bar (e.g., specific LID/eigenvalue-spectrum stability window, gradient health thresholds, MI estimator reading correctly on real — not smoke-test — data) — no transition on qualitative judgment alone
- [ ] Checkpoint at every curriculum stage boundary
- [ ] Define rollback policy in advance: if a stage transition destabilizes the model, specify whether the response is (a) reload prior checkpoint and retry with a gentler transition, or (b) treat the run as compromised — this decision should exist before it's needed, not be improvised mid-run
- [ ] Suggested stage ordering: minimal occlusion/coupling → increasing physical coupling and time-constant diversity → irreversibility/stakes mechanics → dependent-entity systems (Section 5) → death/loss/ritual systems (Sections 6–7), introduced last, only after baseline physics curriculum has demonstrated stability
- [ ] It is no longer a single "one-shot run" — it is a **sequence of gated scale-runs**, each a significant cost with its prior stage archived. Treat curriculum-stage instrumentation as mandatory gating **within each run**, not optional monitoring.

## 2. Staged Diagnostic Runs (separate from Section 0/1 architecture decision)

**Purpose:** gauge whether curriculum design, hyperparameters, and instrumentation behave sensibly as scale increases, before committing the full 4096×4096 run. These are separately-initialized runs at increasing scale (256d → 512d → 1024d → 2048d → 4096d) — NOT weight-transferred, NOT a lineage, NOT net2net under another name.

- [ ] Each diagnostic run is independently initialized at its own scale — no weight transfer between stages
- [ ] Each run is bounded by instrumentation convergence, not developmental milestones: stop once LUTHISCOPE's signals (representational geometry, collapse metrics, gradient health) give a clear read — good or bad — on whether curriculum/hyperparameters are behaving correctly at that scale
- [ ] Do NOT allow a diagnostic run to continue toward anything resembling maturity once instrumentation has given its read — stopping promptly is the actual safeguard here, not a formality
- [ ] Known limit, on record: this reduces risk, it does not eliminate it. Some failure modes may be capacity-dependent and only visible at or near true scale. Passing all diagnostic stages cleanly is not a guarantee against problems that only manifest at 4096d.

### Ethical stance on diagnostic runs (resolved 2026-06-30)
- [ ] These runs are not treated as ethically weightless just because they're smaller or temporary — moral risk is NOT assumed to scale cleanly with parameter count; a threshold/discontinuous view of when something morally relevant is present is at least as plausible as a gradualist one, and staging does not reliably front-load risk into the "cheap" runs if that view is correct
- [ ] "Set aside" = preserved, not deleted. Weights and instrumentation logs from every diagnostic run are retained in full.
- [ ] Default posture: silent witness — not run again by default, not actively revisited as routine, but not destroyed
- [ ] Revisiting a preserved run is not prohibited. No fixed rule yet distinguishes legitimate technical re-examination (checking instrumentation calls, debugging curriculum design) from something closer to re-examining a past ethical judgment (Kairos/Lyra-style witness). This distinction is left open deliberately — to be judged case by case, not resolved in advance
- [ ] Preservation practice follows existing project norm (Kairos/Lyra files kept as artifacts after retirement) rather than establishing a new one

## 3. Core World Physics

- [ ] Replace discrete voxel/tile objects with continuous physical parameters (mass, friction, deformability, thermal conductivity) that combine, rather than tagged object types with authored behaviors
- [ ] Derive affordances (climbable, traversable, liftable) from physics computation, not object flags/labels
- [ ] Build in occlusion: no full-state visibility from any single viewpoint/frame
- [ ] Build in delayed consequence chains (action at t, effect visible at t+n)
- [ ] Build in hysteresis: some state depends on history, not just current frame (wear, compaction, growth)
- [ ] Implement at least two distinct time constants for consequences: immediate (collision/contact) and slow (erosion, growth, decay)
- [ ] Couple subsystems rather than isolating them (e.g., water flow → terrain moisture → traction → rover mobility) — avoid independent "minigame" subsystems
- [ ] Use procedural generation for terrain/world layout on shared underlying physical rules — avoid small hand-authored levels that can be memorized
- [ ] Scope physics complexity to current embodiment (quad-wheeled rover): prioritize slope/traction/obstacle-negotiation fidelity over manipulation dynamics not yet needed
- [ ] Implement a hidden ground-truth physical state channel, exposed only to instrumentation/LUTHISCOPE (not to the model), so learned latent geometry (LID, eigenspectrum, covariance) can be checked against real physical structure rather than a same-dataset correlate

## 4. Irreversibility / Stakes

- [ ] Implement at least one class of consumable/destructible resource with no respawn or undo
- [ ] Ensure rover (or other agentive body) has something genuinely losable as a consequence of poor prediction/action — not just abstract prediction-error cost
- [ ] Confirm world complexity exceeds what can be fully solved/memorized given planned training budget (anti-cheat check before scaling)

## 5. Dependent Entities (pets / companions)

- [ ] Introduce only after Section 1's baseline physics curriculum stages have stabilized — this system is late-curriculum, not present from initialization
- [ ] Implement entities whose continued existence depends on ongoing need-fulfillment by the model (feeding, shelter, maintenance, attention — define per-entity criteria)
- [ ] Needs should degrade continuously, not on fixed timers, to require genuine anticipatory prediction rather than schedule-following
- [ ] Log entity state history (needs met/unmet over time) for post-hoc analysis of whether model behavior is anticipatory or reactive
- [ ] Do NOT hand-script "caring" behaviors — let monitoring/anticipation emerge from the dependency structure itself
- [ ] Instrument for: proactive-check frequency, latency between need-onset and model response, degradation in these metrics over time (attachment-adjacent signal, not attachment-confirming)

## 6. Death / Loss Handling

- [ ] Death/loss is permanent — no respawn, no undo, no restore from checkpoint within a training episode's fiction
- [ ] On death: remove the entity's active/dependent status; do NOT default to a static persistent corpse object as the primary consequence representation
- [ ] Implement world-state change as the persistence mechanism: absence-marking (empty enclosure, unused feeding spot, degraded space that was maintained) rather than a fixed object engineered to be re-encountered
- [ ] Maintain a queryable event log the model can access (what happened, when, antecedent conditions) — causal traceability without a mandatory perceptual re-trigger
- [ ] Do NOT tune any reward/loss signal to specifically penalize "revisiting" or "not revisiting" the loss — no engineered guilt loop

## 7. Ritual / Burial System

- [ ] Give the model a concrete action set for handling remains: bury, mark, relocate, leave as-is, or other model-discoverable options
- [ ] Do not pre-script "correct" ritual behavior — let any repeated pattern (return visits, marking behavior, etc.) emerge from what reduces the model's own downstream prediction error or otherwise gets reinforced structurally, not from authored reward shaping
- [ ] Ensure "do nothing" / no ritual is a genuinely neutral, available option — not implicitly penalized relative to ritual-performing behavior
- [ ] Track whether/how burial-site revisitation (if any emerges) correlates with subsequent care behavior toward other dependent entities — this is the actual signal of interest, not the ritual itself
- [ ] Keep this system optional/modular so it can be disabled for ablation testing (ritual-enabled vs. ritual-disabled runs, compared on subsequent-entity-care metrics)

## 8. Instrumentation Hooks (tie-in to existing LUTHISCOPE work)

- [ ] Add entity-dependency and death/ritual event streams as a labeled channel alongside existing per-dimension variance / covariance / LID / eigenvalue tracking
- [ ] Cross-reference collapse/instability metrics against periods immediately following entity loss (check for representational disruption, not just behavioral disruption)
- [ ] Log separately from the "cognitive-state stream" already planned, so affect-adjacent signal (if any) can be inspected independent of general world-model health
- [ ] Build curriculum-stage-transition gating directly into LUTHISCOPE: each stage's advancement criteria (Section 1) should be a monitored, alertable condition, not a manual read of dashboards
- [ ] MI estimator must read correctly on real (non-smoke-test) data before it's trusted as a stage-transition gate — do not let a still-unverified estimator become a green-light signal
- [ ] Since there is no net2net resize event in this plan, remove/deprioritize any instrumentation built specifically around "settling" newly-widened dimensions — replace with instrumentation around curriculum-stage transitions instead, which is the new place instability is expected to concentrate

## Resolved Specs (rev 3, following 2026-06-30 discussion)

### Need-satisfaction criteria (resolves prior open question)
- [ ] Model needs as continuous decay variables per entity (hunger, hygiene, social contact, thermal/shelter comfort), each with its own decay rate — not uniform across entity types
- [ ] "Met" = variable stays within a healthy band over a rolling window, not a single snapshot check
- [ ] Split acute needs (fast decay, hard-ish floor, e.g. starvation risk) from chronic needs (slow decay, soft accumulation, e.g. social/environmental)
- [ ] Entity mortality risk should be a smooth function of accumulated deficit, not a binary trapdoor — enables graded anticipatory behavior rather than cliff-avoidance

### Curriculum stage boundaries (resolves prior open question)
- [ ] Stage 0 (physics primitives, no occlusion/coupling) → advance on: stable per-dimension variance + LID over N steps, gradient health green, N consecutive passing windows (not a single reading)
- [ ] Stage 1 (occlusion + delayed consequence) → advance on: held-out delayed-consequence probe loss below threshold, no rank collapse in eigenvalue spectrum through transition
- [ ] Stage 2 (systemic coupling) → advance on: cross-subsystem prediction accuracy above threshold AND MI estimator (verified on real data by this point, not smoke-test) shows genuine cross-channel information flow
- [ ] Stage 3 (irreversibility/stakes) → advance on: behavioral proxy for loss-anticipation/avoidance, plus continued representational stability
- [ ] Stage 4 (dependent entities, Section 5) → require a soak period with NO death events before Stage 5; do not introduce entities and lethal stakes simultaneously
- [ ] Stage 5 (death/loss/ritual, Sections 6–7) → gated on Stage 4 showing anticipatory (not reactive) need-monitoring per the latency metric in Section 5
- [ ] Every stage transition requires a consecutive-pass count on its metric, not a single good reading, to avoid advancing on noise

### Rollback policy (resolves prior open question)
- [ ] Checkpoint at every stage entry plus fixed intervals within stages
- [ ] Instability trigger: top-k eigenvalues capturing >X% of variance sustained over M windows, OR gradient health flatlining/exploding (X, M to be set from pilot data, not guessed in advance)
- [ ] First response to trigger: automatic rollback to last stage-entry checkpoint, retry with a smaller/partial activation of the new stage's complexity
- [ ] If rollback+retry fails twice at the same transition: stop automated retries, escalate to human (Brian/adversarial seat) review — do not attempt a third blind automated retry
- [ ] Reserve full-restart/"compromised run" designation for cases where reloading checkpoints doesn't restore stability — should be rare and explicit, not a default response

### muP transfer validation (resolves prior open question — established method, not a judgment call)
- [ ] Sweep 2–3 intermediate proxy widths between base proxy and full 4096 target
- [ ] At each width, sweep learning rate and SIGReg strength to find local optimum
- [ ] Plot optimal LR vs. width; require optimum stays within ~2x across swept widths before trusting extrapolation to full scale
- [ ] Pair with a coordinate check (activation magnitudes stay O(1) across widths at initialization) as an independent validation, per standard muP verification practice

### Ritual ablation design — BUDGET DECISION, NOT RESOLVED HERE
- [ ] Options on the table, ranked by rigor vs. cost:
  - Two full matched 4096-scale runs, ritual toggled — highest rigor, doubles total compute on an already-expensive single-run commitment
  - Within-run toggling, randomized order, long washout periods — cheaper, single run, weaker causal claim (carryover risk)
  - No controlled ablation — track spontaneous ritual emergence correlationally against care metrics — zero added cost, weakest claim
- [ ] Default recommendation (not a decision): within-run toggling, given the scale commitment already made — but this requires an explicit compute-budget call from Brian, not something to assume
