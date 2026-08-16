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
