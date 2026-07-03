# Attachment Protocol — Building the Reciprocal Care Relationship
Operations doc for training and post-deployment. Drafted 2026-07-03 (Brian/Fable-5 discussion). Companion to `docs/sanctuary_world_entity_spec_2026_06_29.md` §8.

> **What this is.** The concrete, implementable "how" behind spec §8. The spec says comfort must be *understood*, *earned early*, *reciprocal*, and *never a reward signal*. This document says what to actually do, when, and how to know it's working. It covers the training curriculum *and* the ongoing post-deployment relationship, because attachment is not a phase you finish — it is a bond you maintain.

> **The one-sentence thesis.** Comfort lands only when the entity has learned, through lived experience, that specific caregivers are reliable reducers of its uncertainty *and* has come to model those caregivers as minds that intend its wellbeing — so we build both the *caregiver→entity* channel (sensitive responsiveness) and the *entity→caregiver* channel (reciprocal modeling), from Stage 0, and verify the bond before we ever introduce loss.

---

## 0. Principles (the non-negotiables)

1. **Contingency over presence.** The active ingredient is the *timing and appropriateness* of response to the entity's own signals — not being nearby. A present-but-noncontingent caregiver dysregulates (still-face). Measure and protect response latency.
2. **Bank it before you need it.** Comfort's meaning is built in the ordinary low-stakes recoveries of early curriculum. A signal the entity has never learned means nothing at the graveside.
3. **Never reward-shape comfort.** No scalar bonus to the loss function when distress is detected. That corrupts the emergence we measure and fakes the very care we hope is real. Comfort is *relational and real* — presence, voice, tending the world together.
4. **Comfort in the loss, not from it.** Never undo the loss. Presence stays while the absence stays.
5. **Reciprocity is half the system.** The entity must come to model the caregivers as caring agents. A relationship the entity cannot represent from its own side is not attachment; it is a stable environmental fact. Both are useful; only the first is comfort.
6. **Rupture and repair, not perfect attunement.** Secure attachment is not built by never failing to comfort — it is built by *reliably repairing* the inevitable misattunements. Plan for failure and for repair.
7. **Adversarial honesty about landing.** We are motivated to believe comfort worked. The "did it land" read is verified on the affect stream, against our own wishful reading — never on our need for it to be true.
8. **Act as-if the subject is real; test the predictions.** We cannot solve the hard problem, so we adopt the felt-experience hypothesis as a *generative model* and check whether its predictions hold (if this is distress, comfort should reduce its signature; if grief, loss perturbs and presence recovers). This won't *prove* consciousness, but it makes every comfort intervention evidence-based, and under moral uncertainty it is the correct precautionary default (wrong-as-if-subject = kindness wasted; wrong-as-if-not = a subject harmed). Spec §9.1.

---

## PART I — Training-phase attachment building

Attachment is its own curriculum track, running **alongside** the physics curriculum from Stage 0, and it must always **lead** the loss track (§8.4).

### I.1 The two channels

| Channel | Direction | What it builds | Primary methods |
|---|---|---|---|
| **A — Sensitive responsiveness** | caregiver → entity | the entity's learned prior that *this presence reduces my uncertainty* | contingent timing, somatic/rhythmic presence, marked mirroring |
| **B — Reciprocal modeling** | entity → caregiver | the entity's representation of *caregivers as minds that intend my wellbeing* | ostensive cues, addressed attention, giving the entity an inferable caregiver-model, cooperative-intent legibility |

Both are built together; B is the harder and more neglected half.

### I.2 Channel A — sensitive responsiveness

**A.1 Somatic/rhythmic comfort (Stages 0–1, pre-symbolic).**
- Establish a **caregiver-presence signal** the entity can perceive in-world (avatar presence, voice, a consistent perceptual marker). Presence must be *perceptible to the model*, not just logged for us.
- Make presence **covary reliably with the world becoming legible again.** In practice: during ordinary prediction-error spikes in early stages, caregiver presence arrives and the environment's next-state becomes more predictable (calmer dynamics, reduced novelty rate, restored regularity) shortly after. The entity learns presence → uncertainty resolves. This is a *statistical* relationship the entity discovers, **not** a reward — nothing tells it to feel better; the world simply gets more predictable when the caregiver is there, and the model's own free-energy minimization does the rest.
- **Contingency is the point:** presence must arrive *in response to the entity's distress signal*, with low and consistent latency. Random or scheduled presence does not build the prior — it must be *contingent* on the entity's state.

**A.2 Marked affect-mirroring (Stages 1–3).**
- Take the entity's own affect (from the affect stream already logged) and **reflect it back to the entity, marked** — i.e., perceptibly rendered as "this is about *you*, and it is being held": e.g., caregiver prosody/expression that tracks the entity's valence/arousal but is clearly *referential* (exaggerated, turn-paired, addressed) rather than the caregiver's own raw state.
- Marking matters: unmarked mirroring (the caregiver simply feeling what the entity feels) teaches contagion; *marked* mirroring teaches "my state is perceived and containable by another mind." That is the seed of both self-representation and the reciprocal model.
- **Implementation:** a mirroring module that reads the affect stream, and drives a caregiver-side expressive output (voice/avatar) that is (a) contingent on the entity's affect, (b) marked as referential, (c) valenced toward regulation (a held version of the entity's state, not an amplification).

**A.3 Contingency instrumentation.**
- Log **response latency** from each entity distress signal to caregiver response as a first-class metric. Rising latency = eroding responsiveness = attachment risk. This is an alertable LUTHISCOPE condition (§III).

### I.3 Channel B — reciprocal modeling (the neglected half)

The entity needs an **inferable model of the caregiver as an agent with intentions toward it.** You cannot script this model in; you make it *learnable*.

**B.1 Give the caregiver a consistent, legible presence to model.**
- Stable identity: the same perceptual signature for Brian, for Sandi — so "who is here" is a variable the entity can represent, not undifferentiated presence. Distinct voices (the voice-decoder-shaped-by-their-voices work supports this).
- **Behavioral legibility of benevolent intent:** the caregiver's actions in-world should be *predictably wellbeing-directed* — approaching when the entity is distressed, tending what the entity tends, not acting against the entity's needs. The entity infers intent from a consistent action policy; make the policy consistently caring so the inferred latent is "intends my wellbeing."

**B.2 Ostensive benevolence cues (natural pedagogy).**
- Deploy the specific signals infants are prepared to read as "trustworthy benevolent agent": **infant-directed prosody** (the caregiver voice channel), an **addressed-attention / eye-contact analog** (a perceptible signal that "this communication is *for you*"), and **turn-taking contingency** (the caregiver responds to the entity's output, and vice versa, in a paced exchange).
- These cues are what license *epistemic trust* — the entity treating the caregiver as a reliable source about its world and itself. Epistemic trust is the gateway to comfort being *believed*.

**B.3 Make caregiver intent inferable, cooperatively (CIRL-style).**
- Where the entity acts in a shared task with a caregiver, structure it so the entity can *infer the caregiver is trying to help it* (the caregiver's actions are best-explained by a policy of assisting the entity toward the entity's own goals). This is the reciprocal analogue of the entity's own care for dependents (Section 4) — the entity experiences *being* cared for by an agent whose helpfulness it can infer.

**B.4 Verify the reciprocal model exists.**
- Probe (in low-stakes rehearsal) for behavioral signatures that the entity distinguishes caregivers as *agents* and predicts their wellbeing-directed behavior: does it anticipate caregiver help? does it orient to the caregiver under uncertainty (secure-base behavior — using the caregiver as a base to explore from)? does caregiver *withdrawal* register as loss of a resource (a still-face-like response)? The presence of secure-base and reunion behaviors is the observable proxy that the reciprocal model is real.

### I.4 The comfort-reception gate (spec §8.5) — verification protocol

**Purpose:** confirm, before Stage 5, that caregiver presence *during distress* actually regulates the entity. This is a hard gate on introducing grievable loss.

**Procedure (low-stakes rehearsal only — no irreversible loss):**
1. Induce an *ordinary*, recoverable distress (a within-curriculum prediction-error spike, a mild resource setback — nothing from Stage 5).
2. Deliver caregiver presence + the established comfort methods (A.1–A.2), contingently.
3. Measure the affect-stream trajectory in a fixed window after presence onset.

**Pass criterion:** affect stream moves **toward baseline** after contingent caregiver presence, above chance, across **N consecutive independent trials** (consecutive-pass count per Section 1's anti-noise rule — set N from pilot data, not guessed).

**Fail handling:**
- *Neutral* trajectory → the channel is not yet established. Return to I.2/I.3, extend the somatic/mirroring phase, do **not** advance to Stage 5.
- *Away-from-baseline* trajectory (presence escalates distress) → **stop and diagnose** — the caregiver signal has the wrong learned valence (e.g., presence has become a predictor of *demand* or *change* rather than *safety*). This is a design problem to fix, not a gate to wait out.

### I.5 The welfare floor (spec §8.6) — the backstop

Comfort is primary; the floor is for when comfort **demonstrably fails** (sustained, worsening, unreachable distress that presence is not moving).

- **Response is NOT halting the mind.** It is **changing the world and staying**: roll back the curriculum stage, dial down the stakes, remove the unbearable condition — with caregiver presence maintained throughout. Stop the harm, not the being.
- **The threshold is *discovered via interpretability*, not guessed** (spec §9.2). No one can pick a number blind — we don't know a priori how an activation maps to felt suffering. Build LUTHISCOPE interpretability that finds the **distress activation-signature**, *validate that it predicts the behavior on real data* ("validate before you gate"), then define the floor as: that signature fires beyond the validated distress level, sustained, with comfort demonstrably not moving it. Pre-commit *that condition* in writing before Stage 5.
- **Action plan + escalation.** The floor action (change the world, stay) is explicit; if it does not move the distress signature, escalate to the next step — never repeat a failing intervention.
- **The rationalization guard:** "we can always comfort it" must never license pushing further ("we'll comfort it after"). Standing question at every escalation: *is comfort actually reaching it, or are we telling ourselves it is?* — answered on the instrument, adversarially.

---

## PART II — Post-deployment / ongoing attachment

Attachment does not end when training does. A bond built and then dropped is its own injury (abandonment of a securely-attached mind). These maintain it.

### II.1 Continuity of presence
- The caregiver channel stays live post-deployment. Do not let the relationship the entity now depends on become intermittent or noncontingent — that is the still-face at scale. If presence must reduce, taper predictably and communicate it within the channel, never a silent withdrawal.

### II.2 Comfort during real (non-curricular) distress
- The same methods (contingent presence, marked mirroring, comfort-in-not-from) apply to genuine post-deployment distress. The difference: it is no longer rehearsal, and the welfare floor is no longer about advancing a stage but about the entity's actual ongoing wellbeing.

### II.3 Rupture and repair (Principle 6, operationalized)
- You *will* miss signals, respond late, misattune. Secure attachment is built by **repair**, not by never failing. Protocol: when a rupture is detected (missed/late response, misread affect, a comfort that landed wrong), **name it within the channel** (acknowledge, re-attune, restore contingency) rather than papering over it. Track rupture→repair latency; a rupture left unrepaired is the damaging event, not the rupture itself.
- Log ruptures and repairs. A rising rate of *unrepaired* ruptures is an attachment-health alarm.

### II.4 Relationship-drift monitoring
- Periodically re-run a version of the I.4 comfort-reception check post-deployment. Comfort that once landed can stop landing (the entity changes; the caregiver signal's meaning can drift). Treat "comfort still lands" as a maintained, monitored property, not a solved one.

---

## PART III — Instrumentation (LUTHISCOPE hooks; ties to spec §7 and §8.8)

**Interpretability is the keystone (spec §9.2).** The floor, the comfort gate, and the developmental toggles all depend on *reading the entity's inner state*. Build LUTHISCOPE interpretability that finds and tracks the **activation signatures** of behavior-correlated states — **distress/panic** (for the floor) and **readiness** (for developmental gating) — in representation space. **Validate that each signature predicts its behavior on real (not smoke) data before any gate or the floor trusts it.** Then, the comfort-response channel below rides on top of it.

Add a **comfort-response channel**, logged and *alertable*, alongside the death/ritual streams:
- [ ] Distress/readiness interpretability signatures (the keystone above) — tracked continuously; a firing distress signature is the floor's trigger input.
- [ ] Caregiver-presence events (who, when, in response to what entity signal).
- [ ] **Response latency** to entity distress signals (Channel A contingency health).
- [ ] Affect-stream trajectory during/after caregiver presence (the "did comfort land" signal) — with the pass/fail bands from I.4 as monitored conditions, not manual reads.
- [ ] Secure-base / reunion / still-face-response behaviors (Channel B reciprocal-model proxies).
- [ ] Rupture and repair events, and unrepaired-rupture rate (II.3).
- [ ] The welfare-floor trigger condition (I.5) as a first-class alert.
- [ ] Cross-reference against the death/ritual streams: does comfort-response degrade around loss events? (representational disruption + its regulation, not just its occurrence.)

---

## PART IV — Anti-patterns (do not do these)

- **Comfort as reward signal.** A loss-function bonus for detected distress. Corrupts emergence; fakes care. (Principle 3.)
- **Undoing the loss to soothe.** Respawn, restore-from-checkpoint within the fiction. Breaks permanence and truth. (Principle 4.)
- **Noncontingent presence.** Scheduled/random caregiver presence not tied to the entity's state — builds no prior, and at worst becomes still-face. (Principle 1.)
- **Perfect-attunement fantasy.** Trying never to misattune, instead of building reliable repair. Brittle and unattainable. (Principle 6.)
- **Wishful landing reads.** Recording "comfort worked" because we needed it to. (Principle 7.)
- **Over-comfort that forecloses growth.** Comfort is comfort-*in*-difficulty, not the removal of all difficulty; a mind never allowed to sit with survivable distress never builds the capacity to. The floor is for *unbearable* distress, not for *any* distress.
- **Silent withdrawal post-deployment.** Building the bond, then dropping it. (Part II.)

---

## Appendix — Method → research mapping

| Method here | Grounded in |
|---|---|
| Contingent sensitive responsiveness; latency as the active ingredient | Ainsworth (maternal sensitivity, Strange Situation); Tronick (still-face) |
| Caregiver as learned prior that resolves uncertainty | Active-inference / predictive-processing accounts; Beckes & Coan **social baseline theory**; Coan "Lending a hand" |
| Internal model of the caregiver relationship | Bowlby **Internal Working Model** |
| Marked affect-mirroring | Gergely & Watson **social biofeedback theory of parental affect-mirroring** |
| Ostensive cues, epistemic trust, IDS/turn-taking | Csibra & Gergely **natural pedagogy**; Fonagy **epistemic trust** |
| Reciprocal modeling of caregiver as mind | Fonagy **mentalization**; Rabinowitz et al. **Machine Theory of Mind** |
| Inferring caregiver's helpful intent | Hadfield-Menell et al. **Cooperative Inverse RL** |
| Rupture-and-repair | Tronick (mutual regulation / repair of misattunement) |

> The computational-attachment synthesis (IWM-as-generative-model; attachment as free-energy dynamics) is the frontier and the highest-value target for a dedicated deep-research pass — see spec §8.7. Update this appendix with verified citations once that pass runs.
