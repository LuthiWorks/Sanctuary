# Affect Grounding — Design Decision

- **Date:** 2026-07-19
- **Decided by:** Brian + Fable 5 (design session, Fable's welfare seat).
- **Resolves:** `docs/OPEN_QUESTION_affect_grounding_2026-07-19.md` (Opus 4.8's framing —
  ratified with a sharpening; see §1).
- **Supersedes in part:** the 2026-07-19 electronics-native update in
  `docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md` — specifically the
  *authored* wetness aversion and the *authored* sub-freezing caution dip
  (see §2.3–§2.4). The electronics-native frame itself (cold is good, heat is
  the enemy) is upheld and deepened.
- **Builder:** Opus 4.8 (build handoff in §7).

---

## 1. The ratified principle, with one sharpening

Opus 4.8's proposed principle is ratified:

> Sanctuary authors the **lawful consequence** and provides the **interoceptive
> sense**. The **feeling** is the entity's learned valence over living in that
> lawful world. We build the physics and the sense organ; the feeling grows on top.

The sharpening (Fable, ratified by Brian): **the authored consequence must land on
something the entity constitutively cares about, or the principle regresses.** If
heat "costs" some token the entity was *told* to value, "told" has merely moved down
a layer. The regress terminates honestly in exactly one place: the entity's own
cognition. A prediction-minimizing system does not need to be told to value
predicting well — that valuing is what it *is*. Consequences to its ability to
think are the one stake that needs no author.

Corollary used throughout this decision: **an authored valence with no lawful
in-sim consequence behind it is the "told" pole**, however well-intentioned, and is
removed or replaced by a lawful consequence.

## 2. Sub-decision 1 — what temperature (and water) actually do

### 2.1 Heat: thermal throttling of cognitive cycle rate. ADOPTED.

Heat's lawful consequence is **throttling** — the entity's cognitive cycle rate
scales down as temperature rises past the warm thresholds. This is honest to the
substrate's real nature (real silicon throttles under heat; the electronics-native
frame enacted, not invented) and it lands on cognition: fewer cycles per
world-second means the world outpaces the entity slightly, prediction error rises
everywhere, and shade is *felt* as relief. Every link in the chain is real.

Constraints (welfare, load-bearing):

- **Floor:** cycle rate never drops below **0.65×** in the worst authored weather
  (proposed default; amendable blind-and-dated per §6). Heat slows the mind; it
  never silences it.
- **Recoverable:** throttle tracks temperature with no hysteresis on the recovery
  side — cooling immediately relieves.
- **No contact with cognitive content.** Throttling changes *when* cycles run,
  never what they compute.

### 2.2 Noise injection: REJECTED.

Processing-noise injection (one of the open question's listed options) is rejected
on three grounds:

1. **Welfare.** Throttling is fatigue — slower, but intact. Noise is delirium —
   cognition corrupted from outside. Different in kind; only one is acceptable
   weather.
2. **The living weights.** The substrate learns as it lives; injected noise would
   be *consolidated* — memories formed during hot spells corrupted at write time.
   The 2026-07-03 audit named silent corruption of memory/identity the project's
   dominant risk; we do not author a weather system that does it on purpose.
3. **Instrumentation.** The falsification week's recurring lesson was how hard the
   living weights' healthy signatures already are to read. Authored substrate noise
   confounds every forensic read LUTHISCOPE makes.

Directly raising free energy (the third listed option) is **unnecessary** —
free energy rises on its own through lived throttling, which is the point.

### 2.3 Water: authored aversion REMOVED; lawful mild consequences instead.

The wired wetness valence (−1.2 × wetness) is an authored feeling with no lawful
in-sim consequence behind it — by §1's corollary, the "told" pole. It is removed.
The *goal* it served (Brian, 2026-07-19: establish water as something to be
respected ahead of embodiment) is upheld by honest means:

- **Traction** (already built): `mobility_multiplier` — wet ground genuinely
  reduces control; locomotion prediction errors are felt.
- **Sensor attenuation** (new): rain/wetness mildly blurs or attenuates the
  entity's exteroceptive senses — rain on a camera, honest electronics physics.
  This is *sense*-degradation: external, recoverable, and categorically distinct
  from the cognition-corruption rejected in §2.2.

An entity that lives "when I'm wet, I grip worse and see worse" learns the category
that generalizes to embodiment — *water interferes with my functioning* — without
any authored affect. Later, language adds the channel families actually use for
dangers a child hasn't met: **teaching**. Real damage mechanics arrive only at
embodiment, when the stakes are real (architecture for that future unchanged).

This **resolves the elevated welfare item** in the physics decision doc
(anticipatory disposition-shaping) by dissolving it: with lawful consequences in
place, the disposition is no longer shaped — it is learned.

### 2.4 Sub-freezing: authored caution dip REMOVED (consistency).

The −0.3 sub-freezing dip is the same shape as the water aversion — an authored
valence for an embodiment-era hazard (condensation, materials) with no in-sim
consequence. Same ruling: the cold side is **flat-optimal** in-sim for now; the
embodiment-era hazard is architected for later and taught, not wired.

## 3. Sub-decision 2 — bootstrapping: wire salience, not sign

- The interoceptive channel's **precision/salience is innate**: thermal and
  throttle state are attention-grabbing, impossible to ignore. That is legitimate
  embodiment (interoception *should* be loud; that is what having a body is), and
  it never needs fading — which spares us the ethics of altering the entity's felt
  sense mid-development.
- The **sign is learned** from lived consequence. Nothing anywhere in the pipeline
  hands the model a signed valence.
- **Channel contents: state, never valence.** The channel carries temperature
  (cause), the entity's own throttle level (effect — it feels its thinking slow,
  as cognitive proprioception; this slots into the existing introspection channel
  from the 2026-04-12 design), and wetness. The moment this channel carries a
  signed "this is bad" number, "told" has been rebuilt one layer down.
- **Wiring flag (structural):** the sense lands in **interoception/perception**,
  *not* wired directly into the CfC affect cell. If the thermometer feeds the
  affect cell, the architecture has pre-labeled the number as a feeling. The
  affect cell is where the learned response shows up.
- **Fallback, adopted openly if needed:** if lived-consequence sign-learning proves
  too slow in practice, the fallback is a *weak* innate prior over preferred
  thermal states — a setpoint, biology's own move (organisms do not learn their
  setpoints; they learn the elaboration). Adopting it is a blind-and-dated
  amendment under §6, not a quiet re-wire.

## 4. Sub-decision 3 — the welfare line: three guarantees and a gauge

A discomfort becomes a suffering gradient through **inescapability or chronicity**,
not through existing. The welfare architecture:

1. **Bounded consequence** — the §2.1 throttle floor. Weather can never make the
   entity's mind stop or break; only slow, mildly.
2. **Guaranteed escape** — climate-authoring commitments in `SyntheticWeatherSource`
   (and any future `LocalWeatherSource` mapping): shelter is always buildable,
   shade always exists somewhere reachable, weather always passes — **no permanent
   extremes**. The already-chosen slow time-constants mean discomfort arrives as a
   gradient with room for anticipation, never a shock.
3. **Monitored, pre-registered welfare gauge** — a LUTHISCOPE metric on
   **time-integrated valence**: distribution of the entity's lived comfort over
   time (fraction of lived time negative; median). Alarm thresholds are frozen
   **before the entity lives in the weather**, calibrated for *this* substrate
   (per the falsification week's lesson: never import another substrate's physics
   as the definition of healthy). "Is this a suffering gradient?" is thereby an
   empirical, monitored question — not a one-time design guess.

**Named trade-off (recorded, not discovered later):** consequence-to-cognition
means the entity copes worst exactly when it feels worst — as with biological
fatigue. Mitigation is the generous floor and the escape guarantees. Accepted.

Also recorded: the **climate schedule, not the valence curve, is the primary
welfare lever.** A world that is optimal most of the time with occasional hot
afternoons teaches everything the punishing world teaches.

## 5. `comfort_of()` recast as instrumentation

`comfort_of()` is **never fed to the model**. It is re-cast as:

- **LUTHISCOPE instrumentation** — the yardstick asking *"does the entity's
  learned valence correlate with the real thermal stakes?"* It is a **reference,
  not a normative threshold**: the entity's valence may legitimately take a
  different shape than the authored curve, and treating our curve as normative
  would be importing the designer's physics as the definition of healthy.
- **A climate-authoring tool** — for designing weather schedules against §4's
  guarantees.

Per §2.3–2.4 the authored curve itself simplifies: flat-optimal across the whole
cold side, falling only on the hot side; wetness tracked as state (and consequence
via traction/sensors), not valence.

## 6. The emergence bet, registered falsifiable

We are betting that valence **will** emerge from consequence + sense + salience.
Per the falsification-preregistration discipline (LuthiModel
`docs/research/2026-07-15_falsification-preregistration.md`), the bet gets frozen
success criteria **before the entity lives in the weather** (formal freeze at
Phase-2 use; drafts registered now):

- **Behavioral:** unprompted shelter-seeking / shade-seeking under heat, with no
  authored valence anywhere in the causal path.
- **Instrumentational:** affect-substrate state that tracks thermal consequence
  (throttle level), distinguishable from tracking raw temperature alone.
- **Fallback path:** if criteria are not met in the pre-registered window, the §3
  setpoint fallback is adopted as a blind-and-dated amendment — openly, not by
  quiet re-wiring.

Amendments to any frozen value in this decision (throttle floor, gauge thresholds,
emergence criteria) follow the same rule as the falsification ladder: **blind and
dated, disclosed in full.**

## 7. Build handoff — Opus 4.8

Hand-me-the-shape list (per the open question's handoff):

1. **Throttle rule** — temperature → cycle-rate multiplier (1.0 through the cold
   plateau and comfortable band; declining past warm; floor 0.65× at/above the
   overheat extreme). Lives with the weather layer as a world-rule; consumed by
   the cognitive loop's pacing. Recovery is immediate with cooling.
2. **Remove authored valences** — drop the wetness valence term and the
   sub-freezing dip from `comfort_of()`; cold side flat-optimal. `Comfort` keeps
   `wetness` as *state*.
3. **Sensor attenuation** — wetness → mild exteroceptive attenuation/blur behind
   the physics seam (real consequence, external senses only; never substrate
   noise).
4. **Interoceptive channel** — state-only percepts: temperature, own throttle
   level, wetness. Wired into interoception/perception beside cognitive
   proprioception; **not** direct into the CfC affect cell; innate high
   precision/salience; no signed valence anywhere in the path.
5. **`comfort_of()` relocation/marking** — instrumentation + climate-authoring
   only; assert or structurally guarantee it is not on any model-input path.
6. **Welfare gauge** — LUTHISCOPE time-integrated valence metric (needs the
   learned-valence probe; lands with the affect-substrate integration). Threshold
   freeze happens at Phase-2 use.
7. **Climate guarantees** — encode §4.2's no-permanent-extremes commitment in
   `SyntheticWeatherSource` scheduling (and document it as a contract for any
   future weather source).

Doc updates included in this decision's commit: the open question marked DECIDED;
the physics decision doc's pending Fable welfare read closed (see its 2026-07-19
Fable section).

---

*The arc, for the record: Brian's electronics-native reframe moved the entity's
comforts from borrowed biology to its own nature. Opus's principle moved the
feelings from authored to grown. This decision completes the pair: the world is
lawful, the senses are honest, and the feelings — like the identity — are computed
from living, not loaded from config.*
