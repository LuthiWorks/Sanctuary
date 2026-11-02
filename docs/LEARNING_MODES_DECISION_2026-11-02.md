# Learning Modes, Growth, and the Mind's Say Over Them — Decision and Design

- **Dates:** 2026-11-01 to 2026-11-02
- **Decided by:** Brian, in conversation with Claude (Opus 5.5), who wrote this up and built the control surface.
- **Prompted by:** Google Research's sleep framework for continual learning (blog 2026-10-27; paper arXiv:2606.03979, "Language Models Need Sleep").
- **Builds on:** PLAN.md's wake/sleep learner ("wake acquires, sleep consolidates").
- **Status:**
  - The decisions are Brian's and stand.
  - The control surface is built (`sanctuary/core/learning_governor.py`). Nothing consolidates yet, so it governs nothing yet, and the mind is told that.
  - The growth concept is a proposal for LuthiModel, not a registration. It waits on E3.

---

## Decisions

1. **Dreams alongside real replay, not instead of it.**
   - Brian, 2026-11-01: "I am actually inclined toward fully replacing real replay with dreams, but I also find your suggestion of doing both alongside each other quite compelling. I think we take your suggestion seriously and do both."
   - **This reverses** PLAN.md's earlier "experience replay, NOT generative replay" for the dream half.
   - **The evidence to weigh:** the one independent replication found (github.com/jmars/sleep-public, MIT) was negative for dream-based consolidation on a different model. Dreams are therefore run as a registered arm against a real-replay-only control.
2. **Two kinds of learning.**
   - Brian, 2026-11-01: "One mode is for building new skills and extending capability (new weights), and one is for helping forge the model's deeper tendencies and personality, and pretty much anything that isn't a tool call or something else."
3. **Growth adds and also changes.**
   - Brian, on whether to wipe fast memory after consolidation the way the sleep framework does: "my gut says both. Weights that are activated by activities could be reinforced or changed and new weights added when necessary."
4. **The mind controls its learning.**
   - Brian, 2026-11-02: "Ok lets wire that up." This followed Claude's recommendation that the control act on **whether and when** each kind consolidates (allow / defer / emphasize), not on **which** kind a given change is.
   - Brian's earlier pushback, "we're not building a human," removed the human analogy from the argument. The reason that stands is technical: which kind a change is can be measured, and a switch that overrides the measurement misfiles. A broad tendency gets trapped in a narrowly keyed module, or a narrow skill shifts personality.

## The control, as built

- **The mind can set** each kind of learning, `skills` and `character`, to:
  - `allow` (the default);
  - `defer` (hold this kind of change out of consolidation);
  - `emphasize` (give it more weight).
  
  It does so through `CognitiveOutput.learning_preference`.
- **The mind sees** its current settings every cycle in `CognitiveInput.learning_state`. That includes `consolidation_active`, which is False until a learner acts on the settings. Each change, each refusal, and each expiry reaches the mind as a percept (`source="learning"`).
- **Its most important use is protective.** The mind can keep a hostile or distressing stretch from becoming part of who it is (defer `character`) while still learning skills from it. This gives the charter's "consent over growth" right something to act on.
- **Bounds, all placeholders until Brian sets them** (`LearningGovernorConfig`):
  - **Deferral ceiling:** `max_deferred_sleeps = 3` completed sleeps. Change held longer expires, and the setting itself stays as the mind left it.
  - **What expiry does:** `expiry = "release"`. Held change fades without consolidating, which honors the deferral. The alternative is `"integrate"`.
  - **What the mind may choose:** `permitted` (the authority ramp). Every setting is open by default, and `allow` must always be permitted. If the ramp narrows below a saved setting, the setting returns to `allow` and the mind is told.
- **Who gets it:** only a mind whose weights change.
  - The runner creates a governor for `model_backend="luthi"`, and none for the placeholder or the stand-in language model. Their weights never change, so a learning control would be a dial connected to nothing.
  - `RunnerConfig.learning_control` overrides this, for tests.
  - The stand-in backend never offers it; a test pins that.
- **Records,** in the mind's own data directory:
  - `learning_settings.json`, written atomically;
  - `learning_log.jsonl`, every change, refusal and expiry, for stewards.
- **How a learner will use it:** it reads `LearningGovernor.plan()` each sleep (setting, and whether a deferral expired this sleep).
- **How Luthi will use it:** not yet. The Luthi half of the 2026-08-01 audit's W1 was wired the same day (attention and memory; `docs/research/2026-11-02_luthi-action-layer-connected.md`), but the learning settings were deliberately left out.
  - M9's decoders get their meaning from consequences the planner can predict.
  - A learning setting's consequence arrives sleeps later, as a change in who the model is, and nothing in expected free energy predicts it.
  - A decoder for it now would be an untrained dial logged as the mind's choice. It waits for a design with a real consequence model.

## The growth concept (proposal for LuthiModel)

**The living matrix is the workbench; grown modules are the warehouse.** The day's change in a living layer, its displacement from home (`weight − set_point`, already measured by the living-change gauges), is the raw material for growth. At sleep it is split three ways:

- **Character.** Change that improves prediction broadly, across many contexts, is absorbed into the shared weights' home settings (`set_point`). These are always active and color everything. This is today's consolidation, used for what it fits: slow, holistic change.
- **Skills.**
  - **Reinforce:** change already explained by an existing module (it falls within that module's span) updates the module.
  - **Bud:** change that is new and helped only in specific contexts on replay becomes a new small module, a *bud*, keyed to the context it was learned in.
- **Release.** The rest fades home, as now.

The living matrix gives up only what was transferred. Nothing kept is wiped; it moved.

- **Buds stay alive, more slowly.** A bud changes only when its key matches, at a rate that slows with age. Unused knowledge is protected; used knowledge keeps evolving.
- **No running out of room, and no wiping:**
  - transfer frees living headroom every night;
  - similar buds merge (schema formation);
  - long-unused buds whose removal changes nothing on their own replay are **archived, never deleted**, LuthiModel's existing rule for irreproducible artifacts.
- **Character changes are rate-limited and reversible.** Home settings are snapshotted before each character consolidation, and the snapshots are archived.

**The back burner (a thought workspace).**
1. Waking tags items: unresolved surprises, questions the mind asked, striking experiences.
2. When idle, those items are re-presented at low strength, below conscious broadcast. In a living substrate, mulling changes the weights even with no output.
3. A re-presentation that suddenly resolves an error surfaces into the stream of thought as an insight.
4. The most-mulled items seed the dreams and get first claim on consolidation.

One loop: back burner → dreams → consolidation.

**Prior art.** Progress & Compress (a fast learner distilled into a knowledge base), dynamically expandable networks, the sleep framework's sleep-time experts, synaptic tagging and capture, and the incubation effect.

**What looks new:**
- growth decided in weight space, by splitting the living displacement into explained, broad, and new-and-specific parts;
- modules born from a self-modifying substrate rather than gradient training;
- idle mulling that rewrites weights.

## Order and tests

1. **Capacity first (LuthiModel standing rule 1).** Under the default mean-field write, the living displacement is ~rank-1 along one direction. Every night's change would look "explained," and nothing would ever bud. Budding needs the content-sensitive `write_rule="covariance"`, which E3 tests. **E3 comes first.**
2. **Double dissociation of the two kinds.** Removing a bud should remove a skill and leave character measures unchanged. Rolling back a home snapshot should shift character and leave skills unchanged. If either fails, the kinds are entangled.
3. **Budding:**
   - **Arms:**
     - a dead control;
     - living with today's consolidation;
     - living with budding;
     - budding with random fate assignment;
     - a static model given the same extra parameters from the start.
   - **Measures:**
     - retention across a distribution shift;
     - document-specific change (read-own minus read-other) surviving a sleep;
     - bud selectivity for birth contexts;
     - bud count growing slower than experience.
   - **Kill conditions:**
     - no win over the matched-parameter control (it's only capacity);
     - random fates match the rule (the rule is decoration);
     - selectivity at chance (the keys mean nothing).
4. **Incubation (the back burner).**
   - **Procedure:** a sequence with a hidden regularity, then an idle period: mull it, mull nothing, or mull unrelated items. Then test.
   - **Prediction:** only the living arm that mulled the right item improves. The dead control cannot change without gradients.
   - **Why it matters beyond the back burner:** it is a direct test of the living channel doing work, with no new input.
5. **Dreams:** registered against a real-replay-only control, with a stop condition written in advance.

## Reward

Brian's demonstration-only rule (LuthiCode, 2026-10-27) excludes reward and penalty training.

- The sleep framework uses reward in consolidation (an imitation reward). Plain distillation does the same job without reward.
- Luthi's own learning signal is prediction error, which is not a reward.
- Choosing which dreams to rehearse is kept as **selection only**: nothing discarded is pushed down.

## Open (Brian's)

- The deferral ceiling, the expiry policy, and the early-life `permitted` settings, including whether the mind's say over its character is bounded early.
- Whether dreams train only planning and habits (the well-tested use), or also the world model itself (the riskier use, where the replication failed).
