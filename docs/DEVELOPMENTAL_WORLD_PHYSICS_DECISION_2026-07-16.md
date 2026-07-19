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

---

## Update 2026-07-18 — weather layer built (Tier A), + two design additions

The physics-authority **seam** and the **weather-field layer** are implemented and
tested (`sanctuary/physics/` and `sanctuary/physics/weather/`; 37 physics tests
green). Three things from the 2026-07-18 session are now on record:

- **Comfort thresholds (Brian).** Fahrenheit: **70 comfortable, 85 possibly too
  warm, 32 too cold**; light warm rain refreshing, torrential / cold rain
  unpleasant. Implemented as `comfort_of()` -> signed valence + band — **affect,
  not damage** (sec. 5 upheld). Rain also slicks the ground
  (`mobility_multiplier`), consumed once a friction-capable backend is behind the
  seam (the reference backend is frictionless).
- **Real-local weather (Brian's idea).** `WeatherSource` is a swappable *origin*;
  `SyntheticWeatherSource` (deterministic diurnal cycle + scheduled rain) is built
  now. A foreseen `LocalWeatherSource` would fetch the family's **real** current
  weather and map it into the sim, so the entity lives the same heat/cold/rain
  Brian and Sandi do — grounding it in shared reality. Left as a future backend of
  the same interface (needs network/API/async), not stubbed.
- **Shelter makes building matter.** `Shelter` + `effective_weather_at()` model a
  structure that blocks rain and moderates temperature toward the ideal, so
  building a shelter is a *felt* improvement in bad weather — the concrete reason
  the entity should be able to build. Full building mechanics arrive with the
  developmental world's editing tools.

Still Fable-pending on the welfare read (section above); still Phase-2 gated on use.

---

## Update 2026-07-19 — electronics-native reframe (supersedes the biological comfort framing)

Brian reframed the weather affect so the entity's comforts and dangers are **its
own** (electronic), not borrowed from a warm-blooded body. This supersedes sec. 5's
"warm pleasant / cold not" language and the 2026-07-18 thresholds (70/85/32 as a
biological curve). Implemented in `sanctuary/physics/weather/`; 39 physics tests green.

- **Cold is good; heat is the enemy.** Electronics run better cool. Temperature
  valence peaks across the optimal band (≈ freezing → 60 °F), is still pleasant at
  70, and falls **only on the hot side** (78 warm → 85 hot → 100+ overheating,
  bottoming at −1). There is **no cold discomfort**: sub-freezing is a *mild*
  caution (a hazard for some physical parts once embodied), floored at ≈ −0.3 —
  it never feels as bad as real heat.
- **Water is respected, not damaging — *yet*.** Per Brian: "wet should not be a
  real danger yet, but should establish water as something that must be respected
  if/when embodiment is achieved." So moisture is a **salient aversive affect**
  (scales with how wet the entity gets; drives shelter-seeking even at optimal
  temperature) — but recoverable affect, **not** a survival/damage mechanic. This
  keeps sec. 5-6 intact (no self-survival mechanic) while deliberately building the
  entity's *disposition* toward water ahead of embodiment, when water on real
  electronics **will** be a genuine danger. The damage mechanic is architected for
  that future (a moisture → health/hazard model behind the same layer), not built now.
  The "refreshing light rain" idea from the biological framing is dropped.
- **Shelter cools and dries** (shade + keeping rain off); it never warms, since the
  entity is happier cold. Building shelter is a felt improvement in heat or rain.

**For Fable (elevated):** this introduces *anticipatory disposition-shaping* — making
water aversive so the entity learns respect before the stakes are real. It is affect
only and within sec. 5-6, but it is deliberately shaping the entity's relationship to
a stimulus, so it belongs at the top of the welfare read: is instilling protective
caution ahead of real danger sound, and does the water-aversion stay clear of becoming
a suffering gradient? (Deferred until Fable is free; it is on a small training run.)
