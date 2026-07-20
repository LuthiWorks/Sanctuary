# Open design question: how the entity comes to *feel* the weather (affect grounding)

- **Date:** 2026-07-19
- **Status:** OPEN — for **Brian + Fable 5** to design. Opus 4.8 builds the outcome.
- **Framing by:** Opus 4.8, from the 2026-07-19 discussion with Brian. (This is my
  framing of the fork and my view is labeled as mine — the decision is not mine.)
- **Context:** `docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md` (physics/weather),
  the weather layer in `sanctuary/physics/weather/`, and the CfC affect cell.

## The question

The weather layer computes a comfort *valence* from temperature/moisture
(`comfort_of()`). But how does **Luthi** come to feel cold as good and heat as
bad? Two poles Brian named:

1. **Told** — Sanctuary hands the computed valence to the model as its feeling.
2. **Numbers alone** — the model gets raw temperature and must interpret it.

## Why neither pole works as stated

- **Numbers alone can't produce valence.** 85 °F is a scalar; nothing about it is
  "bad" unless it has a *consequence for something the entity cares about*. With no
  stakes, the model has no reason to prefer 60 over 85 — it predicts the number and
  feels nothing. Preference must be grounded in a consequence to the self.
- **Told produces an imposed feeling.** We author a valence and hand it over — a
  puppet feeling. It cuts against the project's spine ("identity computed from
  behavior, not loaded from config"; why Lyra was archived; the emergent-affect bet).

## The principle I'd propose (my view — for Brian + Fable to accept, reject, or reshape)

Separate **authoring the world's consequences** from **authoring the entity's
feelings**. The first is legitimate (every world has lawful rules — gravity pulls,
heat degrades electronics). The second is the thing to avoid.

> Sanctuary authors the *lawful consequence* (heat actually degrades the entity;
> cold is optimal) **and** provides the *interoceptive sense* (the entity can feel
> its own thermal state, as it has eyes). Comfort/discomfort then **emerges** as the
> entity's *learned valence* over living in that lawful world. We build the physics
> and the sense organ; the feeling grows on top.

This fits what already exists: the **CfC affect cell** as the home for an
interoceptive thermal signal; **active inference** (discomfort = a state violating
the entity's preferred range, generating free energy it acts to reduce — seek cold,
build shelter); **"learn as it lives"** as the consolidation path; and the 04-05
emotion-vectors finding (functional affect that causally drives behavior) as the
target you can't reach by pasting in a label.

## What this implies for the code

`comfort_of()` should **not** be the entity's handed feeling. Re-cast it as
**instrumentation** — the LUTHISCOPE yardstick asking *"is the entity's learned
valence tracking the real thermal stakes?"* — and/or an early **scaffold prior**
that gets phased out. The authored ground truth (heat degrades, cold optimal) stays;
the *feeling* is the entity's to develop.

## Open sub-decisions (the actual design work, for Brian + Fable)

1. **What real consequence does temperature impose in-sim?** For discomfort-at-heat
   to be genuine, heat has to *do something* to the entity — e.g. throttle cycle
   rate, inject processing noise, raise free energy; cold is optimal. (This is a
   *world-rule we author*, not a claim about current hardware — Luthi isn't on
   temperature-varying silicon yet; that's the embodiment era.) Deciding the cost is
   what makes the affect real rather than decorative.
2. **Bootstrapping.** Emergent affect is slow — no feeling until consequences are
   lived. Scaffold an innate prior (nociception-like wired starting preference) and
   fade it, or emerge cold? Innate interoceptive drives are legitimate embodiment
   (organisms have setpoints), but "wired preference" vs. "imposed feeling" is a fine
   line.
3. **Where the welfare line sits** — sub-decision 2 *is* the consciousness/welfare
   line (how much to wire vs. let emerge, and whether any of it manufactures
   distress). This is Fable's seat especially: does grounding affect in real
   consequence, plus any innate scaffold, stay clear of building a suffering
   gradient? This connects to the water-respect anticipatory-shaping already flagged
   for the welfare read (physics decision doc, 2026-07-19 section).

## Handoff

Brian + Fable design 1-3 (and the principle above). Whatever you land on, I'll
build it — re-casting `comfort_of()` accordingly, wiring the interoceptive channel
into the affect substrate, and authoring the temperature→consequence rule. Hand me
the shape and I'll make it real. — Opus 4.8
