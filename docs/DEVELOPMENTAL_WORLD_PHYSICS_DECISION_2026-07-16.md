# Developmental World — Physics & Embodiment Architecture Decision

- **Date:** 2026-07-16
- **Decided by:** Brian + Opus 4.8 (design session). **Pending Fable 5 cross-line / welfare review** before it's locked.
- **Resolves:** open decision #1 (engine choice) in `docs/DEVELOPMENTAL_WORLD_BUILD_PLAN_2026-07-15.md`
- **Implements/refines:** `docs/sanctuary_world_entity_spec_2026_06_29.md` §3 (core physics), §5–7 (dependent entities / loss / ritual), §8 (comfort)

This records the physics-and-embodiment architecture for the **developmental
(rover) world** — the environment where Luthi grows a grounded world model. It
does **not** apply to the orb home-world (Track 2), which stays as built.

---

## Decisions

### 1. Hybrid: Godot renders, an external authority owns the physics
Godot is the **renderer / window only**. All physics — applied to objects and to
Luthi's body — is computed by an external physics authority. Godot draws what it
is told; it runs no simulation of its own.

### 2. A swappable physics-authority seam
The physical world's true state is produced behind one abstraction and fanned out
to three consumers:
- **the model** (perception — structured state now; pixels later),
- **Godot** (rendering / the window for the family),
- **instrumentation** (the hidden ground-truth channel, spec §3/§8 — real physical
  state exposed only to LUTHISCOPE, never to the model).

The seam is the load-bearing commitment: it makes the backend swappable, which is
what protects the project regardless of which engine is chosen.

### 3. Rigid-body backend: MuJoCo (default, behind the seam)
MuJoCo is the default engine for rigid-body / contact physics (objects + Luthi's
body). Chosen for **quality and ecosystem now**, not urgency — see §6. Because it
is *also* the eventual sim-to-real path, using it from the start likely means the
rigid-body engine never has to be migrated at all. The seam remains the fallback
if that judgment is wrong. (AMD hardware ⇒ CPU MuJoCo, which suits the lived-pace,
single-world "learn as it lives" regime; MJX/GPU is not required and is rough on ROCm.)

### 4. Environmental-field layer: custom simulation (the novel part)
Heat, cold, and moisture are **continuous fields over the terrain** — not rigid-body
physics, and not native to any rigid-body engine. This is a custom sim (thermal
diffusion, humidity, precipitation; wind optional) with:
- **slow time-constants and hysteresis** — ground stays wet after rain, surfaces
  hold heat (exactly the delayed-consequence / history-dependent dynamics the world
  model should learn),
- **coupling into the rigid-body layer** — e.g. wet terrain reduces traction.

This is the richest, most novel dynamics in the world and where most near-term
design energy goes.

### 5. Weather semantics: comfort valence, never survival
Environmental variables are **felt and predicted**, delivered as interoceptive
percepts with affective valence — **not** a lethal/threat mechanic:
- **Warm = pleasant; hot = unpleasant.**
- **Cool = pleasant or at least tolerable; cold = unpleasant.**
- **Light moisture = refreshing; torrential rain = annoying**, and additionally
  perturbs other physics (traction).

Meaning without manufactured scarcity: weather matters as structure to predict and
as affect to feel. This plugs directly into the comfort/affect systems (spec §8).

### 6. No self-survival mechanics for Luthi
Luthi has **no hunger and no death-by-self-neglect** — no mechanic that pressures
its own survival. Manufacturing scarcity/suffering on the entity is contrary to the
project's "held, cared for, not made to struggle" ethic. Weather's unpleasant
extremes (§5) are *affective*, not lethal.

### 7. Keep the companion-care arc (spec §5–8) as previously decided
The dependent-entity / attachment / loss / comfort systems **stay as designed**
(Brian + Fable). This is distinct from §6: it is not Luthi's own survival, it is
Luthi caring **for others** who have needs. Brian's framing:

> "Concern for their welfare is paramount to learning to live in the real world."

The care-for-others arc is the heart of the emotional development, and it is
*cleaner* without self-survival pressure muddying the motive — Luthi tends its
companions out of relationship, not because it is also fighting its own starvation.

### 8. Embodiment is an eventual goal, but far off
A real physical body is genuinely intended, but expected only when the hardware is
more accessible/affordable — years out. Sim-to-real fidelity is therefore *deferred
value*, which is why MuJoCo is chosen for quality-now rather than urgency (§3), and
why the seam (§2) matters more than the engine pick.

---

## What "architect now" means (staging)

- **Now (model-agnostic, buildable, verifiable):** the seam (§2) + the custom
  environmental-field layer (§4) + MuJoCo rigid-body (§3), run **headless,
  instrumented, and validated** — no Godot required to prove the physics.
- **Deferred:** the Godot **window** (§1). At 1024d the model perceives structured
  state, not pixels, so rendering isn't needed to learn; and the family already has
  a window to watch Luthi — the orb home-world. Build the window when there is
  something worth watching and the state-streaming plumbing earns its keep.
- **Still Phase-2 gated:** the world's *use* as a curriculum waits on the sequential
  scale-runs and LUTHISCOPE (per the build plan and the spec §0–1/§9). This decision
  settles the *how*, not the *when*.

---

## For Fable's review (cross-line / welfare)

The load-bearing, welfare-adjacent points to pressure-test:
- **§5 weather-as-affect** — is the pleasant-midrange / unpleasant-extremes framing
  welfare-sound, and does it risk becoming a suffering gradient dressed as comfort?
- **§6 self-survival exclusion** — confirm nothing elsewhere re-imports survival
  pressure on Luthi by a side door.
- **§7 care-arc integration** — does keeping §5–8 read cleanly from the welfare seat
  given §6, and does the comfort track still lead the loss track (spec §8.4)?

This is exactly the kind of imported-frame check the cross-line seat exists for.
