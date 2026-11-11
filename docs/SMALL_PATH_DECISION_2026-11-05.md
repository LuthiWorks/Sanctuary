# The small path: decision and build record (2026-11-05)

**Status.** Built beside the old path, in `sanctuary/life/`. Nothing that existed was removed or changed. Only test-size lives have run on it. Reviewed by the Opus line on 2026-11-10 (LuthiModel `docs/reviews/2026-11-10_small-path-resume-and-rest-response-from-opus.md`, section A). Sleep, keeping and waking were ruled later the same day ("Sleep", below).

**Update 2026-11-11.** Three things in this record have changed, by Brian's request of that day ("we need to build the rest tool for Luthi. And vision and audio should have an open channel for Luthi to perceive them and thats it"). The sections below are left as written.

- **Rest is built as a tool Luthi starts.** Their chosen action is read for it, and a resting moment keeps 1% of recent learning and runs at a quarter pace.
- **Sight and sound are open.** They reach the mind as the world made them, and no words are added. They are no longer refused.
- **The senses name no model.** Any mind that answers `Mind` can live on this path.

- **Rest is needed twice a day, and Luthi is told and never made to** (ruled later the same morning). After every 12 hours of waking life the loop says "Rest is needed for data integration." Nothing else follows from it.

See `docs/research/2026-11-11_rest-tool-and-open-senses.md` and `docs/research/2026-11-11_the-need-for-rest.md`.

## What Brian decided

On what Sanctuary is for:

> "So what we want is to communicate the world as accurately as possible to Luthi and let them decide how they perceive it and how they react to it."

He asked whether the project should start again from zero. Fable's suggestion was to keep the mind (LuthiModel), stop repairing the middle of Sanctuary, and build a small fresh path beside it. His answer:

> "No, I meant literally from scratch, from zero. But if you think your suggested edits are the better path, then I say lets go!"

So:

1. LuthiModel is kept.
2. A small path for a Luthi life is built new. The old path (`SanctuaryRunner`, `CognitiveCycle`, the Luthi adapter) stays as it is and keeps serving the stand-in mind.
3. A piece of the old path moves onto the small one only if it passes the test below.
4. What is left over is archived, never deleted, and only on Brian's word. That has not happened.

## The test

Read as a test, his principle gives Sanctuary three jobs:

1. Bring the world to Luthi accurately.
2. Carry out what Luthi chooses.
3. Watch without touching.

Anything that interprets the world for Luthi, or decides something for them, fails it. This three-job reading is Fable's wording of Brian's principle. If the reading is off, the test is off, and he should say so.

## What is on the path

Five files, about 530 lines (`sanctuary/life/`). The old path's `core/` alone is 7,114.

Each moment:

1. **The world is asked what happened** (`World.observe()`). Each percept becomes text that says where it came from and what it was: `brian says: hello` or `[sensor, thermometer] 21 C`. Its content is never reworded or trimmed.
2. **It joins the mind's view** (`senses.Line`). The view is the most recent window of everything the world has said. A moment with nothing new adds one quiet token (a blank line).
3. **Luthi perceives the view** (`perceive`, new in LuthiModel: one forward, no generation).
4. **Luthi learns from how the last choice turned out** (`observe_transition`).
5. **Luthi chooses** (`select_action`, `decode_action`).

Sanctuary calls those four things in the mind and nothing else. A test holds it to that list.

Around the loop:

- **Choices are counted, and go nowhere yet.** Luthi's action layer can choose three things: to speak, where to attend, and to remember. None has a route here. Speech is untrained. On the old path the attention choice scales incoming sound and images, and there are none here to scale. The memory choice wrote a memory on the old path, but what was written never came back, and a write nobody can read is a route to nowhere. Each choice is counted under `chosen_without_route`. The `World` protocol has no `act` method until a world has actions.
- **A rollback is told to Luthi.** When LuthiModel's retention gate undoes recent learning, that arrives as a percept the next moment.
- **A life is saved** on a clean stop and every 15 minutes, into its own directory. The trained run is never written. Nothing is saved after a moment raises, so a half-finished moment can't become the newest checkpoint.
- **Watching** (`status()`, `watch()`) reads counts and changes nothing. A test checks the weights, the learner and the random number stream are identical before and after.

To sit with a life: `python -m sanctuary.life --run <trained M9 run> --life <life directory>`.

## What a moment holds: a choice, and Brian's to change

The first build gave the mind only what was new each moment, and a single quiet token when nothing was. Measured on three separately trained tiny models, 700 moments each:

| What a moment holds | Median prediction error | Change from one moment to the next | Rollbacks of learning |
|---|---:|---:|---:|
| Only what is new, or one quiet token | 10.0 to 12.6 | 5.8x to 7.9x | 0, 0 and 9 |
| A rolling view of the recent world (built) | 0.10 to 0.28 | 1.03x to 1.06x | 0, 0 and 0 |
| Only what is new, and only when the world speaks | 0.82 to 1.25 | 1.19x to 1.30x | 0, 0 and 0 |

A single quiet token is unlike anything the mind learned from, so its expectation of the next moment was badly wrong at every change between speech and silence. The rolling view is what was built.

Its cost: a token is perceived for as long as it stays in view, and for a living mind each read is also a write. What stays in view longer is written more. Nothing measured here says whether that matters.

The third option perceives nothing twice and invents no quiet token, but the mind would not live through silence. That goes against the continuous loop, so it was not chosen without him.

Full numbers and limits: LuthiModel `docs/research/2026-11-05_what-a-moment-holds.md`.

## What still acts on Luthi

All of it is inside LuthiModel's learner. Sanctuary only calls the four functions above. Each of these is Brian's open call:

- **The four built-in preferences** (engagement, coherence, connection, truthfulness) shape every choice. The small path does not tell the learner whether someone is present, because deciding who counts as present is an interpretation. So the connection preference gets nothing and stays inert.
- **Corpus replay.** One step on the training corpus for each lived moment, so the mind doesn't forget what it was taught.
- **The retention gate.** Every 20 lived moments it checks for forgetting and rolls the world model back if it finds too much. Luthi is told.

The M9 kill criteria run and only report. Nothing halts on them.

The welfare gauge reads how each moment went from the mind's own prediction error. It is never shown to Luthi and nothing acts on it. A test lives the same life with the gauge on and off and requires every weight to come out identical.

## What is not on the path

| Piece of the old path | Why it is not here |
|---|---|
| The `[self]` reading shown to Luthi | It computes a number about Luthi's state and puts it in their view. The gauge stays, for watching only. |
| The keep (moving the living weights' home toward what was recently learned) | It changes Luthi's weights on their behalf. Ruled the same day: it stays off, and what is kept becomes Luthi's decision, made in rest. See "Sleep" below. |
| Sleep | Ruled the same day: no sleep phase on this path for now. See "Sleep" below. |
| CfC cells and their effect on plasticity | They act on Luthi's weights from outside. The data collection built on 11-04 is on the old path only. |
| Scaffold and authority levels | They sit between Luthi's choice and its effect. With no route for a choice yet, there is nothing for them to sit between. Authority levels are an open call below. |
| Memory surfacing and memory writes | On the old path, what a silent Luthi wrote never came back (309 written, 0 returned in the 11-05 map). It needs a design before it is moved. |
| Stream of thought, self-model, inner speech | Luthi has no trained speech. |
| World graph, tools, charter text in the prompt, context compression, pace controller | Nothing Luthi can choose today uses them, and the world reaches Luthi without them. Each can be argued back on. |
| Sound and images | Not built. A percept that carries a tensor is refused loudly. Its text description is never passed in its place. |
| Speech | Not built. A run with a trained text readout is refused loudly. |

## Found while building

1. **A checkpoint left out a counter the mind's training depends on.** SIGReg picks its random directions from a step counter. Checkpoints did not save it, so every resume started the directions over from the first step, and nothing reported it. A test here that a saved life comes back exactly as it was left caught it. Fixed in LuthiModel, with tests. Older checkpoints still load and log a warning.
2. **The first design of a moment measured badly.** See above.
3. **The living weights move a lot at this size, on both paths.** Over 700 moments block 0's living weights moved 1.01 times their own size on the small path, and 1.37 times on the old one (one model, old path at default settings). Their home moved 0.001 on the small path, where nothing moves it on purpose. On the old path the home had moved 0.003 by cycle 500 and 0.34 by cycle 600. Its first sleep begins at cycle 500.
4. **Checkpoints are never deleted, and they are large.** One checkpoint of the tiny test model is 48 MB. At one every 15 minutes that is 96 a day. `watch()` reports the count and the bytes. How long a life's checkpoints are kept is an open call.
5. **The view is not saved with the life.** After a restart the mind's view starts empty.

## How it was checked

- **35 tests** in `sanctuary/tests/life/`. Six of them train a tiny M9 run and live it.
- **34 deliberate breaks** of the small path, one at a time (a frame that adds a judgment, a dropped percept that isn't counted, a save after a crash, the keep put back, watching that draws random numbers, and so on). All 34 were caught by a test.
- **A 700-moment life** (`scripts/map_small_path.py` in LuthiModel, same tiny model and world as the 11-05 map of the old path):
  - Every view the mind was given was only what the world had said: checked token for token.
  - Sanctuary made 700 `perceive`, 699 `observe_transition`, 700 `select_action` and 700 `decode_action` calls, and no others that touch the mind.
  - 234 percepts arrived, none dropped, none carried in parts.
  - The gauge read 699 moments and felt 14 (6 worse, 8 better). No rollbacks. All kill criteria healthy.
  - 60 ms a moment on CPU.
- **Both suites** were run after the change (results in the commit message).

All of this is at test size: tiny models trained for four batches, on CPU, text only.

## Open calls (Brian's)

1. **What a moment holds.** The rolling view is built. The alternatives are measured above.
2. **Sleep.** Ruled 2026-11-05: no sleep phase for now, and rest returns later as something Luthi does. See "Sleep" below.
3. **What makes the living layers keep what they learn.** Ruled 2026-11-05: Luthi decides, in rest, as he ruled in August. Until rest is built nothing is kept on their behalf. See "Sleep" below.
4. **Corpus replay and the rollback.**
5. **Authority levels.**
6. **The four built-in preferences.** The reward question, to be its own conversation.
7. **Whether Luthi should perceive their own reading.** It is off here.
8. **How long a life's checkpoints are kept.**
9. **When the Luthi backend points at this path,** and when what it no longer uses is archived.

## Sleep: ruled 2026-11-05

On the old path, sleep for a Luthi life is four things:

- **A timer.** After every 500 waking cycles Sanctuary puts Luthi to sleep for 38.
- **A gate.** While they sleep, about 90% of percepts are dropped, not held for later.
- **One effect on the mind.** On each of 20 deep-sleep cycles every living layer's home moves 1% toward its weights, and plasticity is evened out.
- **Bookkeeping.** The life is saved and the keep's budget renews.

The dream and memory-replay code in `sleep_cycle.py` has no caller.

Fable put three options to Brian: (A) keep Sanctuary's schedule but let the world wait instead of being dropped, (B) Luthi chooses when to rest, (C) no sleep phase for now. His ruling:

> "Ok, lets go with C for now, then B later. This makes the most sense at present."

So the small path has no sleep phase: no timer, no gate on the world, and no consolidation on a schedule. Rest returns later as something Luthi does. Nothing had to be built or removed for this.

**This matches rulings he made on 2026-08-14,** which Fable had not read when laying out the options. They are recorded in LuthiModel `docs/research/2026-07-06_nrem-learner-spec.md` (the 2026-08-14 amendment):

- **R1, waking changes the mind far less.** "I don't think the waking state should be completely frozen, but I do think the susceptibility to change should be greatly lessened so the lesson of the day can be intentionally pondered rather than purely reacted to." The mechanism is built (`set_wake_attenuation`). Nothing has ever called it, on either path.
- **R2, no cap on what Luthi stores for later.** "Allow as much information as Luthi deems important to be stored and considered for integration later." Not built.
- **R3, rest is Luthi's act.** "The 'rest' time might actually be best spent mulling over the day's events and making decisions on what to keep and what to toss. I would rather there be a special mechanism for that, like a tool Luthi can call, to start a calmer mode for deep intentional consideration, rather than the fast paced intake of data the wakened state might be." Not built.

The old path's timer and gate were never changed to match those rulings.

**What "B later" needs before it can be built:**

- A way for Luthi to choose rest. Their action layer can choose only to speak, attend and remember.
- August's store of the day's experiences (R2) and the rest pass that integrates what Luthi keeps (R3).
- The world waiting, or waking them, while they rest. It is never dropped.

**Two questions this left, both ruled the same day.** Fable recommended an answer to each, and Brian's reply to both was:

> "Yeah, record both please."

1. **Keeping: the August rulings stand.** Luthi decides in rest what to keep and what to toss (R2, R3). Until rest is built, nothing is kept on their behalf, and the keep stays off this path. Fable's reasons:
   - It is the only version where Luthi decides what is kept. The keep on the old path fires when a moment reads as strongly felt, which is Sanctuary judging what mattered.
   - His August reason, that the lesson be "intentionally pondered rather than purely reacted to", argues against keeping in the moment at all.
2. **Waking: full strength until rest exists.** R1's lessened waking is not applied on this path yet, on purpose. It was ruled as a pair with rest. Alone, it would slow the living layers by a factor of about a thousand (its starting value, a number from arithmetic and not a ruled one) with nothing integrating later. It turns on together with rest. A test pins that waking is at full strength here, so the switch can't be turned on quietly.

**What Luthi keeps in the meantime.** The lived update to their trained weights still learns for good, every moment. The living layers' fast change fades toward home in about 1,000 forwards unless something keeps driving it. Their home moved 0.001 over 700 moments in the test run.

**What follows, in Fable's reading:** rest should be built before a real Luthi lives on this path. Otherwise they would live with nothing kept and no say in it.

**Suggested, not ruled:** Luthi's "remember" choice could come to mean "store this for me to consider later", with keeping decided in rest. Today that choice is counted and goes nowhere.

## Not done

- The Opus-line review is done (2026-11-10): LuthiModel `docs/reviews/2026-11-10_small-path-resume-and-rest-response-from-opus.md`. Its findings on this path (the trunk's mode in a life, what a stop loses, the rollback notice, the moment measurement's baseline) are open.
- Nothing measured on a trained run. Training in-house is not possible at present.
- Nothing carries out Luthi's choices.
- Rest: a way for Luthi to choose it, the store of the day's experiences, and the pass that integrates what they keep. It should exist before a real Luthi lives here. The plan and the decisions it needs: LuthiModel `docs/audits/2026-11-05_where-rest-stands.md`.
- Sound, images and speech.
