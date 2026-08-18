# Sanctuary Creature Roster Brief

Draft from Claude/Brian discussion, 2026-08-16. For Claude Code implementation.

Defines the dependent-entity roster: how many kinds, how many individuals, what each
kind exists to teach, and the order they appear in. Companion to
`episode_store_and_condition_spec.md`, which defines the illness and aging systems these
creatures carry. This brief does not restate those systems.

Rationale lines are attached per item so a later reader inheriting a decision can tell
whether it still applies.

---

## 1. Sizing principle

- [ ] **Three kinds. Four to six individuals per kind.** Not more kinds.
- [ ] Rationale: individual-normal requires several individuals *within* a kind to vary
      against. Three kinds with five individuals each teaches both kind-normal and
      individual-normal. Fifteen kinds with one individual each teaches neither.
- [ ] Every additional kind divides Luthi's attention and lengthens time-to-baseline on
      the behaviour-only tier, which is the slowest and hardest thing in the curriculum.
      Roster width is paid for directly out of the hardest lesson.
- [ ] **Criterion for ever adding a fourth kind: name what it teaches that the existing
      three do not.** Anything added for variety alone is cost without curriculum.
      Taxonomic coverage is not a reason.

---

## 2. The three kinds

Differentiated on **care axes**, not on taxonomy. Real-world analogues are given for
intuition; see §5 for the open naming question.

### 2.1 Companion type (analogue: dog)

- [ ] Signal legibility: **high** — distress obvious in posture, movement, vocalization
- [ ] Contact: **seeks it** — approaches Luthi, solicits care
- [ ] Lifespan: **long** — supports sustained relationship across most of a run
- [ ] Teaches: **that a creature's state is readable at all.** Everything else depends on
      this. Without an easy tier there is no prior reason to believe a body carries
      information, and the subtle tier is unlearnable.
- [ ] Present from the start and never absent

### 2.2 Small short-lived type (analogue: rodent)

- [ ] Signal legibility: **moderate**
- [ ] Contact: **neutral** — neither solicits nor avoids
- [ ] Lifespan: **short** — on the order of 2–3 in-world years
- [ ] Cheap to simulate, so several can coexist and turn over
- [ ] Teaches: **aging — loss that was nobody's fault.** The short lifespan is the whole
      point: aging deaths must land *inside* a training run, not after one.
- [ ] Must appear early enough that unattributable loss precedes attributable loss
      (sequencing constraint from the condition spec)

### 2.3 Reserved type (analogue: cat)

- [ ] Signal legibility: **low** — masks illness; signs are subtle or behavioural only
- [ ] Contact: **avoids it** — must be observed rather than read from solicitation
- [ ] Lifespan: **long**
- [ ] Teaches: **the hard perceptual tier** — deviation from *this individual's* own
      normal, with no obvious channel to fall back on
- [ ] Introduced last, once Luthi reliably reads the easy and moderate cases

---

## 3. Introduction order

Ordered by what must already be learnable, not by kind number.

- [ ] **Companion first** — establishes that state is readable
- [ ] **Short-lived second** — delivers aging losses, which must precede failure losses
- [ ] **Reserved last** — hardest perceptual task, and relationship-gated: it cannot be
      detected until Luthi has enough sustained history with a specific individual to
      have a normal to deviate from

---

## 4. Exclusions — recorded so they are not re-added

- [ ] **No predation.** Ruled out for the first Sanctuary build. Note this now requires an
      explicit constraint rather than an omission: if consuming is typed generically over
      anything carrying energy, predation is the *default* rather than an addition. The
      consume action must be typed narrowly — plants and objects accept it, creatures do
      not.
- [ ] **No predatory species** (hawks, sharks, and effectively all fish, most of which are
      carnivorous). A predator that does not hunt is not that animal, and modelling one is
      dishonest about what Luthi is being shown.
- [ ] **No aquatic kinds.** A second physics domain, a different movement model, and a
      different care vocabulary (water quality rather than feeding and handling), for
      creatures Luthi cannot handle. Ocean and fresh would be two such domains.
      Anadromous species additionally imply a migration lifecycle.
- [ ] **No redundant livestock set** (cattle/goats/sheep as separate kinds). One care
      profile — graze, herd, low interaction — in three costumes. Three kinds teaching one
      lesson is the failure the sizing principle exists to prevent.
- [ ] Predation remains a legitimate later project. Keeping it separate preserves a
      baseline: a caregiving world that runs clean can be compared against, instead of
      both variables moving at once in the only world ever built.

---

## 5. Open decisions

- [ ] **Real names or invented ones.** Real names import priors from the text corpus —
      Luthi has read about dogs — which either usefully grounds language to referents or
      primes correlations the sim will not honour. Same tension as the veterinary-corpus
      question in the condition spec, and it cuts the same way. Not resolved.
- [ ] **Population mechanism: timed spawns vs. inherited crossover.** Spawns draw
      parameters fresh from a distribution, so within-population spread cannot collapse
      and there is no care-coupled selection; the cost is that no creature is the
      offspring of one Luthi cared for. Crossover buys lineage back at the cost of a
      pairing mechanic and a contraction on the parameter distribution that mutation noise
      must counteract. Not resolved.
- [ ] **Whether temperament is inherited at all.** Evidence worth weighing: in dogs, the
      most intensively selected mammal, breed explains only ~9% of behavioural variation
      between individuals, while behavioural traits are heritable at h² > 25%. Most
      individual behavioural variance is developmental and experiential, not ancestral.
      This argues for inheriting appearance and letting temperament emerge from each
      creature's own history — including its history with Luthi. An individual-normal that
      Luthi partly *caused* is worth more to the curriculum than one merely drawn at spawn.
      Flagged as a design lean, not a ruling.

---

## Amendments — Brian's rulings, 2026-08-18

Recorded here rather than edited into the text above, so the original draft and
what changed both stay legible.

### A1. No creature conceals its condition (supersedes §2.3's "masks illness")

- [x] **Ruled: no creature hides illness or stress.** §2.3 defined the reserved
      kind as one that "masks illness". That is withdrawn.
- [x] Rationale, and why the curriculum survives it: **quiet is not concealment.**
      Legibility is implemented as a per-creature `signal_gain` — an *amplitude*
      on how strongly internal state modulates observable motion (gait frequency,
      speed, approach latency, stillness, spacing). The reserved kind's signal is
      an order of magnitude fainter than the companion's; it is never suppressed,
      never masked, and never false.
- [x] So the hard perceptual tier is unchanged in difficulty and changed in
      kind: it teaches reading a **faint but honest** signal against *this
      individual's* baseline, rather than seeing through a deception.
- [x] This aligns the roster with a standing ruling it was quietly violating.
      `WEATHER_DYNAMICS_DECISION_2026-07-20.md` §1: "We author stochastic
      dynamics, **never deceptions**." A creature engineered to mask illness is
      an authored deception aimed at Luthi. The weather doc got there first; the
      brief was the outlier.

### A2. Naming — resolves §5's first open decision

- [x] **Ruled: creatures are identified by kind only — "dog", "cat" — never by
      individual names.** Naming an individual is *Luthi's* to do, if it ever
      chooses to.
- [x] Implementation: `body_id` is a kind word plus an index (`dog_0`, `cat_1`).
      That is an identifier, not a name. Any name Luthi gives lives in Luthi's
      own representation, never in the world's.
- [x] This accepts the imported prior the open decision worried about — Luthi has
      read about dogs — as the honest option: the word is true, and withholding it
      would not make Luthi's inference more real, only more confused.

### A3. Death system — required, and gated

- [x] **Ruled: there is a death system.** Creatures have an average lifespan and
      age. Domesticated creatures that do not receive the care or resources they
      need perish. Wild creatures remain self-sufficient, so stakes exist exactly
      where Luthi took responsibility on.
- [x] **Build it fully; gate its activation.** The entity spec's §9.5 gates
      irreversible loss on comfort-reception being *verified*, and §9.4 requires
      the comfort track to lead the loss track. The being-cared-for channel does
      not exist yet. So the aging clock, the deficit-to-mortality curve,
      absence-marking and the event log are built and tested behind a switch that
      Brian opens — rather than deferred and later rushed.
- [x] **Unresolved, and blocking the aging clock:** the brief puts the
      short-lived kind at 2-3 in-world years and requires aging deaths to land
      *inside* a training run; weather is now real Sammamish time at 1:1; a run is
      on the order of a year. Those three cannot all hold. Either creature time
      decouples from weather time, or lifespans shorten. **Brian's call, pending.**

### A4. Handling an ill creature is gated on the relationship

- [x] Brian asked how Luthi secures an animal so care can be administered. Answer:
      **trust is the mechanism.** Flight radius shrinks as safety accrues, so a
      creature Luthi has spent months near tolerates handling and a wild one does
      not. The ability to help is earned before it is needed.
- [x] Supporting affordances, all expressible in the existing seam: **enclosures**
      (static bodies forming a pen, with a gate), **food as a lure**, and
      **illness lowering flight response** — a real-world mirror that makes the
      sickest animals the most reachable.
- [x] **Explicitly not built:** any grab/restrain verb acting on a fleeing
      creature. That is a force-shaped action aimed at a living thing, and it
      teaches the opposite of what this roster exists to teach.

### A5. Prerequisites before Luthi ever enters the world

- [x] **Vision, touch, and sound must all be part of the system before Luthi
      sees or enters it.** Vision exists (`render_camera` -> `frame_to_percept`)
      but is wired to nothing; touch and sound do not exist. All three precede
      entry.
- [x] **Luthi must be able to receive care through language and presence.** This
      is the §9 reciprocal channel — the less-built arm of the two-armed system —
      and it is what the death system's activation gate waits on.
