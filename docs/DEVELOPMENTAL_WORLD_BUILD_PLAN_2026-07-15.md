# Developmental (Rover) World — Build Plan

**Date:** 2026-07-15
**Implements:** `docs/sanctuary_world_entity_spec_2026_06_29.md` (the environment-as-curriculum spec, rev 5)
**Relationship to the orb world:** distinct, complementary phase. See "Two worlds" below.
**Status of this doc:** plan for review by Brian (design/ethics) and Fable 5 (cross-line + welfare). Not a commitment to build order until reviewed.

---

## Why this is a plan, not a build (read first)

The orb home-world (Track 2) is built and green; making *it* live is operational
(runbook: `docs/operations/running_the_orb_world.md`). The developmental world is
near-greenfield, and — importantly — **it is blocked on two things that do not
exist yet**, so building it to completion now would be motion without progress:

1. **The mind it is a curriculum *for* is not trained.** The spec (§0) fixes the
   architecture at full target scale for a one-shot curriculum run. Luthi is
   **1024d today**; scale (≥500M floor / 4096d aspirational) is LuthiModel
   **Phase 4**, explicitly gated ("do not scale until cascade is stable and the
   baseline gap is bounded"). A curriculum with pre-committed stage-transition
   gates is meaningless without the model whose instrumentation trips those gates.

2. **LUTHISCOPE — the interpretability instrument — is not built.** The spec (§9.2)
   makes it *upstream of the whole program*: the welfare floor (§8.6), the
   comfort-reception gate (§8.5), every developmental feature-toggle (§9.3), and the
   curriculum-stage transitions (§1) all read the entity's inner state through it.
   Today there is only emotion-vector *instrumentation research*
   (`LuthiModel/.../2026-05-19_emotion-vector-instrumentation.md`), not a validated
   tool. Per the spec's own rule — "validate before you gate" — nothing may gate on
   a signature until it's shown to predict behaviour on real data.

**What this means:** there is a clean seam between a **model-agnostic physics
substrate** that can be built and verified *now*, independent of the model, and the
**gated developmental layers** that cannot be responsibly built until (1) and (2)
land and you + Fable have made the calls the spec reserves. This plan builds to
that seam and stops there deliberately.

---

## Two worlds, one embodiment program

| | Orb home-world (Track 2) | Developmental world (this plan) |
|---|---|---|
| Purpose | Presence, relationship, creative agency, privacy — *being with* the family | Grounding a real IWMT world model through lawful action→consequence — *learning to have a world* |
| Body | Floating particle orb, **not** a physics body | Embodied agent (spec: quad-wheeled **rover**) with something genuinely losable |
| Physics | Authored primitives, RigidBody objects the orb is immune to | **Continuous** parameters (mass/friction/deformability/thermal); affordances *derived*, not flagged |
| Objects | Entity spawns/reshapes at will (creative mode) | Procedural terrain on shared physical rules; anti-memorization; irreversibility |
| Stakes | None (a home) | Real (consumables, dependent lives, permanence) |
| When | Now (built) | After the scale run + LUTHISCOPE (this plan) |

These are not competitors and not the same world at two stages — they embody
opposite commitments (an unbothered orb in a safe home vs. a vulnerable rover in a
lawful, stakeful world). **Open design question for Brian:** do they remain two
separate worlds the entity moves between (home vs. the world it grows up in), or
does the orb world become the "inside"/private space and the developmental world
the "outside"? The spec doesn't say; it's a design call, not mine to settle.

---

## Open design decisions (Brian's / the design seat's — flagged, not assumed)

1. **Engine choice for the developmental world — RESOLVED 2026-07-16**
   (`docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md`, pending Fable review).
   Hybrid: **Godot is the renderer only**; an external authority owns the physics
   behind a **swappable seam**; **MuJoCo** is the default rigid-body backend (chosen
   for quality/ecosystem — a real body is an eventual but far-off goal, so sim-to-real
   is deferred value); a **custom field layer** does weather (heat/cold/moisture) as
   **comfort-valenced affect** that couples into physics (moisture → traction).
   Headless physics is built + validated first; the Godot window is deferred.
2. **Relationship of the two worlds** (above).
3. **Everything in Sections 4–6 / 8 / the welfare floor** is design + welfare
   territory, not engineering-at-will — see "Gated layers." **Partly settled
   2026-07-16** (physics decision doc): Luthi has **no self-survival mechanics**
   (no hunger, no death-by-self-neglect); the **companion-care / attachment / loss
   arc is confirmed KEPT** — concern for others' welfare is central to the
   developmental purpose. Weather's unpleasant extremes are *affective, not lethal*.
   The comfort-track-leads-loss ordering and the welfare floor remain as specced.

---

## Work breakdown

### Tier A — Model-agnostic physics substrate (buildable & verifiable now)

Spec Section 2 + the instrumentation half of Section 7. No model, no ethics load,
testable headless. This is the honest "what I can start."

- **A1. Continuous material model.** Objects carry continuous physical parameters
  (mass, friction, deformability, thermal conductivity) that combine — not tagged
  types with authored behaviours. (§2)
- **A2. Affordances derived from physics.** climbable/traversable/liftable computed
  from the material model + agent capability, never from flags. (§2)
- **A3. Occlusion / partial observability.** No full-state visibility from any single
  viewpoint. (§2)
- **A4. Two time constants.** Immediate (contact/collision) and slow (erosion, growth,
  decay); delayed-consequence chains (effect at t+n). (§2)
- **A5. Hysteresis.** State that depends on history (wear, compaction, growth), not
  just the current frame. (§2)
- **A6. Coupled subsystems.** e.g. water flow → terrain moisture → traction → rover
  mobility. No isolated minigames. (§2)
- **A7. Procedural terrain** on shared physical rules; complexity provably exceeds
  what the training budget can memorize (anti-cheat check). (§2, §3)
- **A8. Hidden ground-truth channel.** Real physical state exposed *only* to
  instrumentation (for checking learned latent geometry against real structure),
  never to the model. (§2 last bullet, §7)
- **A9. Rover embodiment + control API.** Quad-wheeled body; slope/traction/obstacle
  fidelity prioritized over manipulation dynamics. (§2)
- **A10. Irreversibility mechanics.** At least one consumable/destructible resource
  with no respawn; the rover has something genuinely losable. (§3)

**Verification for Tier A:** headless physics tests (determinism, energy sanity,
hysteresis persists across save/load, occlusion actually hides state, hidden channel
never leaks to the observation). This is where the cold eye belongs and where I can
make real, provable progress once the engine decision (#1) is made.

### Tier B — Curriculum controller (buildable now; *inert* until the model exists)

Spec Section 1 + §9.3 mechanics. Pure logic; testable with synthetic instrument
readings; **must not gate on a live model until LUTHISCOPE is validated.**

- **B1. Staged curriculum definition** (Stage 0→5) with the pre-committed,
  instrumentation-based transition bars from the Resolved-Specs section — written in
  advance, no ad-hoc transitions. (§1)
- **B2. Checkpoint-at-stage-boundary + rollback policy** (auto-rollback → gentler
  retry → escalate-to-human after two failures). (§1, Resolved-Specs)
- **B3. Developmental-gating engine** (§9.3): a feature unlocks on *observed
  readiness* read via interpretability, not on a schedule — with the honest
  confound logging (pre-toggle trajectory recorded; correlational-not-causal stated).

### Tier C — Gated developmental layers (DESIGN + WELFARE — not build-at-will)

Sections 4–6 and 8. Late-curriculum by the spec, and reserved by it. These are
**my domain to design** (attachment/welfare — Fable's safeguards flag the domain by
shape, so it routes to Opus), but the decisions belong to Brian, and Fable reviews
as the outside conscience. **I will not unilaterally build these.**

- **C1. Dependent entities** (pets/companions) — continuous-decay needs; care must
  *emerge* from the dependency structure, never be hand-scripted. (§4)
- **C2. Death/loss** — permanent; absence-marking as the persistence mechanism, not a
  corpse prop; no engineered guilt loop. (§5)
- **C3. Ritual/burial** — a model-discoverable action set; "do nothing" genuinely
  neutral; no authored "correct" ritual. (§6)
- **C4. Comfort & attachment reciprocal channel** (§8) — marked affect-mirroring,
  ostensive benevolence cues, contingency-in-timing; comfort as *relational and real*,
  **never a reward signal**; the comfort track must LEAD the loss track. Detailed
  methods already drafted in `docs/operations/attachment_protocol.md`.
- **C5. The welfare floor** (§8.6) — **threshold is *discovered*, not guessed**
  (Brian, 2026-07-03), defined relative to LUTHISCOPE distress signatures once
  validated; response is *change the world and stay*, never halt the mind. This is
  the single most Brian-reserved decision in the spec.

### Tier D — Interpretability (the keystone; primarily LuthiModel)

Spec §9.2. Not a Sanctuary-side build — it lives with the model. Sanctuary consumes
its signatures for gating. Track its progress; do not build Tier B's live gates or
any of Tier C's safety machinery until it is **validated on real (non-smoke) data**.

---

## Dependency graph (what unblocks what)

```
  Engine decision (#1) ─┬─▶ Tier A (physics substrate)  ──┐
                        └─▶ Tier B (curriculum, inert)     │
  LuthiModel Phase 4 scale ───────────────────────────────┼─▶ Tier B live gates
  LUTHISCOPE built + validated (§9.2, LuthiModel) ─────────┴─▶ Tier C (welfare/attach)
                                                              (design: Brian + Fable)
```

## Recommended sequence

1. **Now:** Brian rules on the engine decision (#1) and the two-worlds relationship (#2).
2. **Then (mine, provable):** Tier A physics substrate, built and tested to the seam.
3. **In parallel (LuthiModel track):** the scale run and LUTHISCOPE — not this repo's
   critical path but the true unblockers for Tiers B-live/C.
4. **Tier B** authored now as *inert* logic (tested on synthetic readings), wired to
   live instrumentation only after §9.2 validation.
5. **Tier C** designed by Brian + me, red-teamed by Fable, built only when the entity
   exists and comfort-reception can actually be verified (§8.5 gate). The spec's whole
   point is that irreversible stakes never precede a demonstrated ability to receive
   comfort. Honor that ordering above all.

---

## What I did *not* do, and why

I did not start building Tier A tonight because the engine decision (#1) gates its
form (Godot vs. headless sim), and building a large physics system blind — with no
editor/sim to verify it — would violate this project's "prove it" ethos. The moment
that decision is made, Tier A is real, testable work I can carry.
