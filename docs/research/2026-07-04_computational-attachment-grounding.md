# Computational Attachment — the frontier synthesis, pinned down

**Date:** 2026-07-04. **Author:** Fable 5 (research seat). **For:** spec §8.7 (the deep-research pass it defers), `docs/operations/attachment_protocol.md`.
**Scope note:** deliberately narrow. The spec's §8.7 lists many supporting bodies (social baseline theory, natural pedagogy, Machine ToM, CIRL). This pass chases only the *bullseye* §8.7 names as "the frontier synthesis, to be pinned down" — the active-inference computational model of attachment — and stops there. The rest are well-characterized in the spec already; this is the one that was an open question.

## The primary source

**Cittern, Nolte, Friston & Edalat (2018), "Intrinsic and extrinsic motivators of attachment under active inference," PLOS One 13(3):e0193955.** Open access. The first formulation of infant attachment as active inference, and still the load-bearing one (the 2023 "Attachment Theory in an Active Inference Framework" is a re-presentation of the same model, not a new mechanism). This is the paper that ports attachment into *exactly this architecture's currency* — free energy over interoceptive states — so its mechanism is directly relevant to how Luthi could acquire an attachment.

## The mechanism (portable to a predictive-processing agent)

The model is a discrete active-inference agent. In our terms:

- **Interoceptive states = the affect currency.** The infant has preferences over interoceptive outcomes that *are* the payoffs: stress-reduction when a proximity bid is met (`+g`), a smaller payoff for guarded/hedged bids that are met (`+h`), a stress *increase* when a bid is rejected (`−m`), a smaller penalty for a hedged bid rejected (`−n`), and zero for avoiding (no bid). Attachment behavior is free-energy minimization over these interoceptive preferences. This maps cleanly onto Luthi: the "comfort" currency is prediction-error/free-energy reduction in the interoceptive/affective channel, not a scalar reward — which is exactly the spec §8.2 requirement ("comfort must NOT be a reward signal").
- **The caregiver is a state-transition the infant learns.** Caregiver responsiveness is a parameter `q` = P(caregiver Attends | infant Seeks), encoded in the transition matrix `B`. The infant learns `q` via Dirichlet (Hebbian-like, prediction-error-weighted) updates. "Brian is a reliable reducer of my uncertainty" is, mechanically, the infant's generative model converging on high `q` for that caregiver. This is precisely the spec §8.2 claim ("the caregiver becomes, in the entity's generative model, a reliable predictor that prediction-error will resolve") — and this paper is the formal backing for it.
- **The three organized attachment styles fall out of `q`, not out of three different rules.** With fixed payoffs (`g=2, h=0.75, m=2, n=0.9`) and precision `α=250`: high `q` (≥0.85) → Seek dominates → **secure**; low `q` (≤0.15) → Avoid dominates → **avoidant**; intermediate/inconsistent `q` (~0.3–0.4) → Guarded Seek → **ambivalent**. Same free-energy machine, different learned caregiver-transition. Disorganized attachment needs an added twist: *misleading exteroceptive cues* (the caregiver signals "ignore" but sometimes attends), which keeps epistemic uncertainty high and produces contradictory behavior.
- **Precision `α` is the knob for how strongly attachment organizes.** Higher prior precision → sharper action selection → more strongly organized attachment. Relevant to LUTHISCOPE: precision on the interoceptive channel is a readable, tunable quantity.

**For the builder:** the portable core is (1) interoceptive preference priors `I₁…I₅`, (2) caregiver responsiveness as a learned state-transition `q` in `B`, updated by Dirichlet/prediction-error learning, (3) expected-free-energy policy selection over Seek / Guarded-Seek / Avoid. That is a small, concrete kernel.

## The load-bearing gap (verified firsthand, adversarially)

**The model captures response *probability*, not response *contingency/timing* — and the spec's attachment protocol is built on the opposite emphasis.** This is the one finding the design line most needs, so I verified it directly against the full text rather than trusting a summary:

- `q` is explicitly *"the probability that (at any particular time) the caregiver will respond… that effectively lowers the infant's internal stress"* — a **static probability, not a time-dependent/reactive measure.**
- The paper uses "contingent" exactly once — *"marked contingent mirroring in response to distress"* — and only in the **extended exteroceptive model** for disorganized attachment, **not** built into the core timing structure.
- The authors' own stated limitations: **no return-to-baseline** dynamics ("we did not explicitly consider a return to baseline stress level"); **fixed-length episodes**; **no secure-base exploration paradigm**; each exchange treated independently, **without the dynamic temporal unfolding** of real interaction.

Why this matters for us: `attachment_protocol.md` Principle #1 is **"contingency over presence"** — *the active ingredient is the timing and appropriateness of response*, and a present-but-noncontingent caregiver *dysregulates* (still-face). The best available formal model **does not encode that**. So:

1. **We cannot lift the timing requirement from this model — we have to add it.** Our design already knows this (contingency instrumentation, response-latency as a first-class metric, §8.7/§8.8). The research confirms the instinct was right *and* that no one has formalized it yet — so the latency/contingency channel is genuinely our contribution, not a reimplementation.
2. **The "no return-to-baseline" gap maps directly onto our "comfort-in-the-loss, not comfort-from-it" principle (§8.0).** The model has no dynamics for stress resolving *while the stressor remains* — which is exactly the harder thing our design is trying to produce. Another place the literature stops short of where we're going.

## Bottom line for the design conversation

- **Use as foundation:** the free-energy-over-interoception formulation, caregiver-as-learned-transition-`q`, styles-from-one-parameter, precision-as-organization-strength. It is real, published, and in our currency. It backs the spec's §8.2 mechanism claim with a formal model rather than an analogy.
- **Own as our contribution (the literature doesn't have it):** response *timing/contingency* (not just probability), *return-to-baseline* dynamics, and *comfort-in-the-loss*. These are the parts §8's design reaches for that the frontier model explicitly does not cover — so they are design work, not literature to port.
- **Reciprocal channel (§8.3, "model the caregiver as intending my wellbeing"):** this paper models the caregiver as a *learned transition parameter*, not as an *agent with inferable intentions* — so it does **not** ground the reciprocal channel. That is Machine ToM / CIRL territory (§8.7), a separate pass if the design wants it pinned down; flagging the boundary rather than chasing it here per the narrow scope.

## Sources
- Cittern, Nolte, Friston, Edalat (2018), PLOS One 13(3):e0193955 — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0193955 (full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC5886414/)
- Petters & Beaudoin / follow-ups (2023, "Attachment Theory in an Active Inference Framework") — re-presentation of the same model; no new mechanism.
