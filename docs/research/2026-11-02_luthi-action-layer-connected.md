# Luthi's action layer, connected — 2026-11-02

## Objective

Brian, 2026-11-02: "Ok awesome. Lets get this on the training path and
connect it to sanctuary." "This" is LuthiModel's M9 action layer, the part of
Luthi that chooses actions.

The LuthiModel half (training it, loading a trained run) is recorded in
`LuthiModel/docs/research/2026-11-02_m9-on-the-training-path.md`. This entry
is the Sanctuary half. It closes the Luthi side of the 2026-08-01 wiring
audit's W1 ("Route the M9 decoder's selections into `CognitiveOutput`").

Before today:

- The adapter called `select_action` every cycle and kept the result as a
  diagnostic only.
- Attention guidance and memory operations came from adapter-authored
  heuristics: spike fractions, and an activity threshold.
- The only M9 that had ever run here was a fresh, untrained one built by
  `examples/validate_seam_integration.py`.

## Process

### Step 1: how a trained run gets in

- `LuthiModelConfig` gains `m9_run_dir` and `life_dir`, and the runner and
  CLI gain `--luthi-m9-run` and `--luthi-life-dir`.
- `load()` hands both to `luthi.sanctuary_interface.load_m9_run` and
  attaches the returned trainer as actor and learner. Sanctuary still never
  reaches past the interface.
- The runner refuses an M9 run without a life directory. A life's
  checkpoints are never deleted, so where they go is a choice for Brian,
  not a default buried in `data_dir`.

### Step 2: carrying out the selected action

`decode_action` reports, per output channel, whether the action acts there
and with what content.

- **Attention.** When the attention channel acts, its gates become attention
  guidance. Modalities gated at or above one half go to `focus_on` and the
  rest to `deprioritize`, using Sanctuary's percept modality names.
- **Attention has a consequence.** I found that `attention_guidance` has no
  consumer anywhere in the cycle, so routing gates there alone would have
  been a dial connected to nothing. The M9 attention decoder specifies its
  own consequence: next cycle's sensory tokens are scaled by the gates. The
  adapter now does that (`_attend`). The text gate has nothing to scale,
  since text arrives as tokens, so it reaches the cycle only as guidance.
- **Memory.** When the memory channel acts, the adapter writes an episodic
  memory of the moment, with significance 1–10 from the chosen salience.
  The content is what was perceived. A model with no trained speech has no
  words of its own to store.
- **No decoder, no change.** An actor that cannot say what its action does
  (a bare `M9Actor`) leaves the old heuristics in place.

### Step 3: keeping an untrained readout silent

LuthiModel's KNOWN_INCOMPLETE #4: "Sanctuary must not present text from a
pure-JEPA checkpoint as Luthi's utterance." The adapter did not enforce it;
it generated inner and external speech from whatever loaded.

`load_m9_run` reports `speech_trained`. When it is False:

- no external speech is generated;
- inner generation runs only the step-0 forward the seam reads its state
  from, then returns no text.

Decoding further would feed the trunk its own noise through living weights,
token by token.

### Step 4: keeping the life

- **First try, wrong.** I put the save in `consolidate()`. The cycle calls
  that every NREM cycle, not once per sleep. That would have written a
  checkpoint of about 1.4 GB every NREM cycle, never deleted, with errors
  logged as "non-fatal".
- **Where it went instead.** The save moved into the cycle (step 0b). It
  runs once after each completed sleep, on the consolidation worker so the
  write cannot stall the loop, and it is not caught.
- **At shutdown.** `save_state()` saves the life first, also uncaught. I
  found the CLI never called `save_state()` at all, so it does now, after
  the cycle stops. A CLI session would otherwise have lost everything
  since the last sleep. It had also never saved the CfC layer's state, which
  it now does too.

### Step 5: running it

- **Tests.** `test_luthi_m9_life.py` ends with an end-to-end test. It
  trains a tiny M9 run with LuthiModel's own driver, then Sanctuary boots
  on it and runs three cycles. The test checks that:
  - the selection was decoded;
  - the learner took a lived step;
  - shutdown saved one life checkpoint;
  - the run directory is untouched.
- **A longer look.** On the 64d smoke run, by hand, for eight cycles:
  - a lived update every cycle after the first;
  - attention and memory came from the decoded action;
  - no speech;
  - about 0.25 s per cycle on CPU after an 18 s first cycle (loading and
    tokenizing).
- **What the run found.** It found the corpus-cache problem fixed on the
  LuthiModel side: the cache was being written into this repo.

### Step 6: the learning settings

`LEARNING_MODES_DECISION_2026-11-02.md` said the settings "join Luthi's
action vocabulary when the Luthi half of W1 is wired". W1's Luthi half is
wired, but the learning settings deliberately are not.

M9's decoders get their meaning from consequences the planner can predict.
A learning setting's consequence arrives sleeps later, as a change in who
the model is, and nothing in expected free energy predicts that. A decoder
for it would be an untrained dial logged as the mind's choice. It waits for
a design with a real consequence model. Until then nothing consolidates by
the settings for Luthi anyway (`consolidation_active=False`, and Luthi is
told).

## Conclusion

Sanctuary can run Luthi with its trained action layer:

- Luthi's chosen attention changes what it perceives next.
- Luthi's chosen memory writes are made.
- Its lived experience trains the action layer and the world model.
- Its life is kept across sleeps and restarts.
- An untrained text readout stays silent.

What remains open:

- **Nothing has run on Brian's PC.** The first real test is a training run
  at scale (M9 is on by default since Brian's later ruling the same day;
  `--no-m9` opts out), then Sanctuary on it.
- **The "active" calls are noise until the activity bands warm up** (their
  window is 32 cycles) and the decoders have been trained on something more than
  cycle-consistency. In the smoke run the text channel was flagged "active"
  on five of eight cycles, with nowhere to go.
- **Whether attention scaling is the right consequence** for a modality
  that arrives only some cycles (Minecraft gives no images yet) is
  untested.

## Artifacts

- Code:
  - `sanctuary/core/luthi_model.py`: `_load_m9_run`, `save_life`, `_decode`,
    `_attend`, `_chosen_attention`, `_chosen_memory`, the speech gate;
  - `sanctuary/core/cognitive_cycle.py`: step 0b;
  - `sanctuary/api/runner.py`;
  - `sanctuary/api/cli.py`.
- Tests:
  - `sanctuary/tests/core/test_luthi_m9_life.py` (7, including the end to
    end). These need `LUTHI_PATH`; CI skips them, like the other seam tests.
  - `sanctuary/tests/core/test_luthi_life_keeping.py` (3: save per sleep,
    save at shutdown, a life directory required). These need no luthi, so
    CI runs them.
- The LuthiModel side: commits `a397bf9`, `aaac761`.
