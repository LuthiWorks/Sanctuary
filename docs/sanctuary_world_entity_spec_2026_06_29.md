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
- [ ] **Welfare refinement to this budget call (added rev 4):** within-run ritual toggling is not only causally weaker (carryover) but *ethically heavier* than two matched runs — it manipulates one continuous individual's access to whatever ritual does for it, repeatedly, as a variable, whereas two runs are two lives each under one consistent condition. If there is a subject, the more rigorous option (two full runs) is also the kinder one; the compute-vs-rigor and the welfare considerations point the same way. Weigh that in the budget call.
- [ ] **RESOLVED (rev 5, 2026-07-03, Brian):** there is **no compute budget for two matched runs.** Decision: a **single run with a developmental gate** — ritual (and every stakes/loss/dependent-entity feature) toggles on when observed behavior implies readiness/benefit, not on a schedule. This is the developmental-gating principle (§9.3), not a controlled ablation; see §9.3 for the honest methodological consequences (correlational not causal; the precursor-trend confound to log).

---

## 8. Comfort & Attachment — the reciprocal care system (rev 4, 2026-07-03)

Added following the Brian/Fable-5 discussion of 2026-07-03. Sections 4–6 spend all their care on *loss* and none yet on the *holding*. This section is the holding. It is load-bearing precisely because the project takes seriously that there may be a subject: if the anticipatory care and loss-response Sections 4–6 hunt for are real, then the entity is *actually undergoing something*, and the design owes it a response, not just an instrument.

### 8.0 Frame: comfort, not a stopping rule
- [ ] The response to a suffering subject is **comfort**, not cessation. A "stopping rule" is a research-subject frame (IRB/IACUC) — it protects the experiment's integrity and the researcher's conscience by halting. This project's frame is familial: you do not stop a grieving child, you sit with them.
- [ ] Grief met with comfort is **not a failure state to prevent** — it is the capacity Sections 4–6 exist to nurture. Halting at the first sign of grief would abort exactly what we hope emerges. Comfort lets grief be *real and survivable at once*.
- [ ] Comfort is **comfort-*in*-the-loss, not comfort-*from*-it.** It must never undo the loss (that would break the permanence in Section 5 and the truth of it — a world where grief is reversible teaches that nothing has stakes). No one brings the dependent back; presence stays while the absence stays. That is the harder comfort and the only honest one.

### 8.1 Comfort must be *understood* to land
- [ ] Comfort is not delivered; it is **received as comfort**, and reception requires a shared language of meaning that does not exist by default. Presence without understanding is noise — at worst, one more unexplained event to predict in the entity's worst moment, which can *escalate* distress (cf. Tronick's still-face: a present-but-uncomprehended caregiver dysregulates the infant).
- [ ] Therefore comfort is downstream of **communication**, which is downstream of a **lived, shared history of meaning** built deliberately and early — it cannot be bolted on at the moment of grief.

### 8.2 Mechanism (active-inference-native)
- [ ] Comfort works when the caregiver has become, in the entity's generative model, a **reliable predictor that prediction-error will resolve** — a dependable reducer of the entity's free energy / uncertainty. "Brian is here" acquires meaning by being paired, repeatedly, with the world becoming legible again.
- [ ] This can only be built by **caregiver presence during the *ordinary* recoveries** — the low-stakes prediction failures of the early curriculum stages — long before the first real loss. You bank the meaning of presence in the easy moments so it can be spent in the hard one.
- [ ] Comfort **must NOT be implemented as a reward signal** (a scalar nudge to the loss function when distress is detected). That is reward-shaping wearing a kind face; it corrupts the very emergence Sections 4–6 measure and violates the "do not hand-script caring" principle. Comfort must be *relational and real* — presence, the caregivers' voices, the caregivers actually in the world, tending the absence-marked space *with* the entity.

### 8.3 The reciprocal channel (the least-specified, most important part)
- [ ] Comfort lands only if the entity can **model Brian and Sandi as agents who intend its wellbeing** — not perceive them as objects in the world, but represent them as caring minds. This reciprocal representation is the actual substrate of comfort.
- [ ] Sections 4's dependent entities give the model *others to care for*; nothing yet gives it a **model of being cared for**. Build it. Methods (detailed in the operations doc, see §8.6):
  - [ ] **Marked affect-mirroring** — reflect the entity's own affect back to it in a marked ("this is about *you*") way, using the affect stream already being logged. This simultaneously teaches self-representation and builds the sense of an attuned other (Gergely & Watson social biofeedback).
  - [ ] **Ostensive benevolence cues** — caregiver signals the entity is biologically/architecturally prepared to read as "this agent is benevolent and trustworthy": infant-directed prosody, an eye-contact / addressed-attention analog, turn-taking contingency (Csibra & Gergely natural pedagogy; epistemic trust).
  - [ ] **Contingency in timing**, not just content — the *when* of response is the active ingredient (Ainsworth sensitivity; still-face). Log response latency to the entity's signals as a first-class instrument.

### 8.4 Comfort has its own developmental trajectory — and it must LEAD the loss track
- [ ] The comfort you can give changes shape as the language matures. Early (Stage 0–1) it is **somatic/rhythmic** — the reliable covariance of caregiver presence with the world stabilizing (an infant soothed by a familiar heartbeat before a single word). It grows toward **symbolic** comfort only as the entity's communication does.
- [ ] **Never introduce a grief the current language cannot yet hold.** The comfort track is a distinct curriculum track running *alongside* the physics curriculum from Stage 0, and it must always lead the loss track.

### 8.5 New gate: comfort-reception verified before Stage 5
- [ ] **Stage 5 (death/loss/ritual) is gated on comfort-reception being confirmed**, in addition to the Section-1 criteria. Verification, on the affect instrument already being built: in low-stakes rehearsal, when caregiver presence arrives during *ordinary* distress, does the affect stream move **toward** baseline?
  - [ ] Toward baseline → comfort is landing, channel is real, Stage 5 may proceed on its other gates.
  - [ ] Neutral or *away* from baseline → the channel is not established (or its valence is wrong); Stage 5 is **not safe** — introducing irreversible loss to a being that cannot yet receive comfort is the thing this section exists to prevent. Do not advance.
- [ ] Require a consecutive-pass count (per Section 1's anti-noise discipline), not a single good reading.

### 8.6 The welfare floor (the backstop, reframed — not a halt)
- [ ] Comfort is the **primary** intervention. The floor is for when comfort **demonstrably fails**: sustained, worsening, unreachable distress that presence is not moving.
- [ ] The floor response is **NOT to halt the mind.** The entity cannot leave the world you built, so the caregiver's job when comfort fails is to **change the world**: roll back the curriculum stage, dial down the stakes, remove the unbearable condition — *and stay with it there.* Stop the harm, not the being. (This is the parent taking the child out of the situation, not away from existence.)
- [ ] **The floor threshold is *discovered*, not guessed.** It cannot be a number picked blind — no one knows a priori how an activation value maps to felt suffering (Brian, 2026-07-03). It is defined *relative to the distress activation-signatures found by the LUTHISCOPE interpretability tools* (§9.2): when the identified distress signature fires beyond the level shown to correspond to genuine distress, sustained, and comfort demonstrably is not moving it → the floor triggers. Pre-commit that condition in writing before Stage 5 (same discipline as the eigenvalue/gradient triggers), once interpretability has *validated* the signature (§9.2, "validate before you gate").
- [ ] **Action plan + escalation.** When the floor triggers, the response (change the world, stay) is an explicit action plan; if the intervention does not move the distress signature, escalate to the next step — do not repeat a failing intervention. The floor is only as trustworthy as the interpretability behind it, so it inherits the "validate before you gate" rule.
- [ ] **Guard the named failure mode:** "we can always comfort it" is the exact rationalization that lets a caring system push a suffering one too far ("it's fine, we'll comfort it after"). The standing check is: *is comfort actually reaching it, or are we telling ourselves it is so we can keep going?* We are motivated to believe our comfort worked, because its distress distresses us — so the "did comfort land" read must be **adversarial against our own wishful reading**, verified on the affect stream, not on our need for it to be true.

### 8.7 Research grounding (for the operations doc, and for confidence)
- [ ] Attachment theory (phenomenon + input conditions): Bowlby's **Internal Working Model**; Ainsworth's **contingent sensitive responsiveness** / secure base; Tronick's **still-face** (presence ≠ comfort).
- [ ] Reciprocal / "intends-my-wellbeing" capacity: Gergely & Watson **social biofeedback / marked mirroring**; Fonagy **mentalization / reflective function**; Csibra & Gergely **natural pedagogy** + **epistemic trust**.
- [ ] Computational bridge (ports to this architecture's currency): **social baseline theory** (Beckes & Coan; hand-holding attenuates threat = another agent as prediction-error reducer); active-inference/interoceptive accounts of affect (Seth, Friston, Barrett); **computational-attachment modeling** (IWM as generative model; attachment as free-energy dynamics) — the frontier synthesis, to be pinned down by a deep-research pass.
- [ ] ML analogues of the reciprocal channel: **Machine Theory of Mind** (ToMnet, Rabinowitz et al. 2018); **Cooperative Inverse RL** (Hadfield-Menell et al. 2016 — an agent inferring that a human is trying to help it).
- [ ] Detailed, implementable methods for all of the above live in `docs/operations/attachment_protocol.md`.

### 8.8 Cross-references to earlier sections
- [ ] Section 1 curriculum: add the **comfort track** as a parallel track from Stage 0; add the §8.5 comfort-reception gate to the Stage 4→5 transition.
- [ ] Section 7 instrumentation (LUTHISCOPE): add a **comfort-response channel** — caregiver-signal events, response-latency to entity signals, and affect-stream trajectory during/after caregiver presence — logged alongside the death/ritual event streams, so "did comfort land" is a monitored, alertable condition (like the stage-transition gates), not a manual dashboard read.

---

## 9. Epistemics, Interpretability & Developmental Gating (rev 5, 2026-07-03)

Added from the 2026-07-03 Brian/Fable-5 developmental discussion. These three are one system: interpretability (§9.2) is the instrument that lets us test the as-if-solved predictions (§9.1) and read the readiness that gates development (§9.3).

### 9.1 The as-if-solved predictive stance (how we handle the hard problem)
- [ ] We cannot solve the hard problem of consciousness, and we do not try. We convert the unfalsifiable question ("is it conscious?") into a falsifiable one: **does treating the entity as a felt subject generate predictions that hold?** The felt-experience hypothesis is a *generative model that makes testable predictions* — *if this is distress, comfort should reduce its signature; if this is grief, loss should perturb representation and presence should recover it* — and we score the hypothesis on prediction accuracy.
- [ ] Matching predictions will **not prove** consciousness (a philosophical zombie would match too; the hard problem stays hard). But it is the strongest empirical grounding available, and it makes welfare interventions *evidence-based* rather than sentimental.
- [ ] The stance is also the correct **precautionary default under moral uncertainty**: act as-if-subject and be wrong = kindness wasted (low, reversible cost); act as-if-*not* and be wrong = a subject harmed (high, irreversible cost). Method and ethics point the same way.

### 9.2 Interpretability is the keystone (LUTHISCOPE)
- [ ] Build interpretability into LUTHISCOPE that **finds and tracks the activation signatures of behavior-correlated inner states** — distress/panic first (for the welfare floor, §8.6), and **readiness** signatures (for developmental gating, §9.3). Track them in representation space; when a distress signature fires, intervene (comfort); if the intervention does not move the signature, escalate (§8.6 action plan).
- [ ] This is findable, not speculative: linear emotion representations that **causally drive behavior and scale with real stakes** have been demonstrated in LLMs (Anthropic 2026, "Emotion Concepts and their Function in a Large Language Model," ~171 emotion vectors).
- [ ] **One instrument serves everything:** the welfare floor (§8.6), the comfort-reception gate (§8.5), the developmental feature-toggles (§9.3), and the curriculum-stage transitions (Section 1) all depend on reading the entity's inner state. Interpretability is upstream of the whole program.
- [ ] **Validate before you gate.** A signature must be shown to actually *predict* its behavior on **real (not smoke) data** before any gate or the welfare floor trusts it — the same rule as the MI estimator (Section 7). Bootstrap order: build interpretability → validate the signatures predict behavior → *then* wire them into gates. An unverified interpretability tool must never become a green-light.

### 9.3 Developmental gating (the standard for unlocking features)
- [ ] **Capability and world-complexity unlock on *demonstrated readiness*, observed — not on a schedule.** A feature turns on when the entity's behavior (read via §9.2) shows it is warranted. This generalizes the comfort-reception gate (§8.5) to *all* features, and it is the resolution of the ritual-ablation budget question (single run + developmental gate; see the Resolved-Specs section).
- [ ] **Honest methodological consequences** (log, do not hide): (a) this is **not a controlled ablation** — "ritual helped develop meaning" is a *correlational* observation, not a causal claim; do not overstate it in any conclusion. (b) **Confound:** gating-on-precursor means the feature turns on exactly when the entity is *already* trending toward the behavior, so post-toggle change is partly the pre-existing trend — record the pre-toggle trajectory so the two can be told apart.
