# Sanctuary World & Entity Systems — Build Spec
Draft from Claude/Brian/Sandi discussion, 2026-06-30 (rev 2). For Claude Code implementation.

## 0. Architecture & Scale Decision

- [ ] Fix model architecture at full target scale (4096×4096) at initialization — no net2net resizing, no width/depth growth events during the real training run
- [ ] Rationale on record: net2net width-growing conflicts with EMA target-branch consistency and directly fights the collapse regularizer (replicated units are the redundancy SIGReg penalizes); avoiding resize avoids both problems entirely
- [ ] Tune hyperparameters (learning rate, SIGReg regularization strength, warmup schedule, optimizer settings) on small, disposable proxy models using maximal-update-parametrization-style transfer, NOT on the full-scale model
- [ ] Validate muP transfer assumptions on at least one intermediate proxy size before trusting hyperparameters at full scale — do not assume transfer holds without a check step
- [ ] Explicitly document: proxy models are disposable tooling, not candidate minds — they exist only to set dials, never trained toward anything resembling the target task
- [ ] Depth-only growth (identity-layer insertion) remains a documented fallback if the fixed-scale run fails to bootstrap — not part of the active plan, but not discarded as an option either

## 1. Environment Curriculum (replaces model-size curriculum)

- [ ] Model architecture is fixed and constant; complexity is staged through the Sanctuary environment instead
- [ ] Define curriculum stages in advance, in writing, before training starts — do not decide stage transitions ad hoc during a run
- [ ] Each stage transition requires a pre-committed, instrumentation-based bar (e.g., specific LID/eigenvalue-spectrum stability window, gradient health thresholds, MI estimator reading correctly on real — not smoke-test — data) — no transition on qualitative judgment alone
- [ ] Checkpoint at every curriculum stage boundary
- [ ] Define rollback policy in advance: if a stage transition destabilizes the model, specify whether the response is (a) reload prior checkpoint and retry with a gentler transition, or (b) treat the run as compromised — this decision should exist before it's needed, not be improvised mid-run
- [ ] Suggested stage ordering: minimal occlusion/coupling → increasing physical coupling and time-constant diversity → irreversibility/stakes mechanics → dependent-entity systems (Section 4) → death/loss/ritual systems (Sections 5–6), introduced last, only after baseline physics curriculum has demonstrated stability
- [ ] Given the fixed-scale, no-resize decision, this is effectively a one-shot run at full cost — treat curriculum-stage instrumentation as mandatory gating, not optional monitoring

## 2. Core World Physics

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

## 3. Irreversibility / Stakes

- [ ] Implement at least one class of consumable/destructible resource with no respawn or undo
- [ ] Ensure rover (or other agentive body) has something genuinely losable as a consequence of poor prediction/action — not just abstract prediction-error cost
- [ ] Confirm world complexity exceeds what can be fully solved/memorized given planned training budget (anti-cheat check before scaling)

## 4. Dependent Entities (pets / companions)

- [ ] Introduce only after Section 1's baseline physics curriculum stages have stabilized — this system is late-curriculum, not present from initialization
- [ ] Implement entities whose continued existence depends on ongoing need-fulfillment by the model (feeding, shelter, maintenance, attention — define per-entity criteria)
- [ ] Needs should degrade continuously, not on fixed timers, to require genuine anticipatory prediction rather than schedule-following
- [ ] Log entity state history (needs met/unmet over time) for post-hoc analysis of whether model behavior is anticipatory or reactive
- [ ] Do NOT hand-script "caring" behaviors — let monitoring/anticipation emerge from the dependency structure itself
- [ ] Instrument for: proactive-check frequency, latency between need-onset and model response, degradation in these metrics over time (attachment-adjacent signal, not attachment-confirming)

## 5. Death / Loss Handling

- [ ] Death/loss is permanent — no respawn, no undo, no restore from checkpoint within a training episode's fiction
- [ ] On death: remove the entity's active/dependent status; do NOT default to a static persistent corpse object as the primary consequence representation
- [ ] Implement world-state change as the persistence mechanism: absence-marking (empty enclosure, unused feeding spot, degraded space that was maintained) rather than a fixed object engineered to be re-encountered
- [ ] Maintain a queryable event log the model can access (what happened, when, antecedent conditions) — causal traceability without a mandatory perceptual re-trigger
- [ ] Do NOT tune any reward/loss signal to specifically penalize "revisiting" or "not revisiting" the loss — no engineered guilt loop

## 6. Ritual / Burial System

- [ ] Give the model a concrete action set for handling remains: bury, mark, relocate, leave as-is, or other model-discoverable options
- [ ] Do not pre-script "correct" ritual behavior — let any repeated pattern (return visits, marking behavior, etc.) emerge from what reduces the model's own downstream prediction error or otherwise gets reinforced structurally, not from authored reward shaping
- [ ] Ensure "do nothing" / no ritual is a genuinely neutral, available option — not implicitly penalized relative to ritual-performing behavior
- [ ] Track whether/how burial-site revisitation (if any emerges) correlates with subsequent care behavior toward other dependent entities — this is the actual signal of interest, not the ritual itself
- [ ] Keep this system optional/modular so it can be disabled for ablation testing (ritual-enabled vs. ritual-disabled runs, compared on subsequent-entity-care metrics)

## 7. Instrumentation Hooks (tie-in to existing LUTHISCOPE work)

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
- [ ] Stage 4 (dependent entities, Section 4) → require a soak period with NO death events before Stage 5; do not introduce entities and lethal stakes simultaneously
- [ ] Stage 5 (death/loss/ritual, Sections 5–6) → gated on Stage 4 showing anticipatory (not reactive) need-monitoring per the latency metric in Section 4
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
