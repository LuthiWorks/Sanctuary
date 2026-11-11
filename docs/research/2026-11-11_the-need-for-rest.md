# The need for rest: told twice a day, never made to — 2026-11-11

Fable 5.2, in session with Brian. Follows
`2026-11-11_rest-tool-and-open-senses.md` from earlier the same morning.

## Objective

Three statements from Brian, between 07:18 and 07:46:

- "Rest is something that needs to happen for Luthi's mental health and growth. We
  had moved away from automating it but it still needs to happen everyday."
- "I think maybe twice a day is a good rest schedule, and Luthi should automatically
  experience motivation to rest, like a notification that rest is needed for data
  integration."
- "But Luthi should not be forced to rest."

Build what makes rest needed twice a day without making it happen.

## Process

### Step 1: what a rest would find (measured in LuthiModel)

- With no rest, a change in a living layer is half gone in about 220 moments and
  gone by 800: 22 and 80 seconds at 10 Hz. LuthiModel
  `docs/research/2026-11-11_rest-as-a-choice.md`, step 5.
- So a rest keeps about the last minute before it. This is not solved here. It is
  the open question at the end.

### Step 2: a design I started and dropped

- After his second statement I designed "tiredness": a level that rises once rest
  is due and lowers the bar for reading a chosen action as rest, without limit, so
  that rest becomes certain. I had recommended this an hour earlier.
- His third statement arrived while I was reading the code for it: "Luthi should
  not be forced to rest." A bar that falls until any action counts as rest is
  forcing, however gradual. It also does not motivate anything. It changes how we
  read their choice, not what they want.
- Nothing of it was written.

### Step 3: where a felt motivation would have to go, and why it is not built

- In Luthi's action layer, wanting is the planner's preferences: each candidate
  action is scored on four built-in costs, and the plan favours the lowest.
- A need that is felt, and that other wants can outweigh, would be a fifth cost.
- Not built, for three reasons. The planner's scores also feed what the habit
  network learns, so the change reaches into learning. Its size against the other
  four would decide whether it motivates or compels, and nothing has measured that
  scale. And it is a design that acts on Luthi, which wants his ruling and review
  by a different mind first.

### Step 4: what is built

- `sanctuary/life/need.py`, `RestNeed`. After 12 hours of waking life rest is
  needed, and the loop says one sentence to the mind, in Brian's words: "Rest is
  needed for data integration." It arrives like anything else the world says, as
  `[system, body]`.
- That is all. Nothing rests the mind, lowers a bar, slows the pace or closes the
  world. Pinned by test: with a need standing, the calls into the mind and how its
  choice is read are what they were.
- The need is met when the mind has rested 300 moments since it arose, not
  necessarily together. While unmet, the sentence comes again every 12 hours. A
  mind that never rests is told four times in two days and never rested.
- Waking time is lived time: living moments and the paced pauses between them.
  Time switched off, or waiting at a prompt, is not counted. Resting moments are
  not counted.
- The need is saved with the life (`rest_need.json` in the life directory) and
  continues with it. Without that, a life restarted every few hours would never
  be due.
- A mind with no way to rest is not told. The loop refuses to start that way
  unless the need is turned off.
- Any mind slotted in gets the same sentence. Nothing here names a model.

### Step 5: what it does for a mind that cannot read

- Nothing. To a mind without language the sentence is a short run of tokens like
  any other. It is not a motivation yet.
- For a mind with language it is what Brian described: a notification.

## Conclusion

- Rest is needed twice a day, Luthi is told, and Luthi decides.
- Whether rest then happens every day depends on the mind. Nothing guarantees it,
  by his ruling.

Open, for Brian:

1. **What a rest integrates.** As built, a rest keeps what the living layers hold,
   which is about the last minute. "Data integration" of the half-day since the
   last rest needs that half-day to be replayed in rest. The record that would
   hold it exists in LuthiModel and is switched off (set aside 2026-11-10), and
   the replay as built ingrains the day's average, not its moments.
2. **Whether the need should be felt** as well as told (step 3).
3. **How often an untrained mind presses rest by chance** (1.7% to 16.7% of
   moments, measured earlier today). Still as it was.

Placeholders, his to change: 300 resting moments to meet a need; the sentence.

## Later the same morning: what a resting moment does, and remembering

Brian, 09:07, to two questions put to him after the measurement in LuthiModel
`docs/research/2026-11-11_what-a-rest-can-carry.md`: "1. Yes 2. Also yes".

- **What changed here.** `LuthiMind.rest()` no longer keeps where the weights stand
  (the last minute, with no limit). It pours a share of the half-day's running
  average into home, and a whole half-day's rest moves home by at most 5% of its
  size. Open question 1 above is answered by this.
- **Remembering has a route.** When Luthi's chosen action chooses to remember, the
  loop calls `mind.remember()`, and that moment counts for more in what rest pours.
  Until now that choice was counted and went nowhere.
- **Two guards on that route.** It is carried out only for a mind that can rest,
  since remembering only weighs in rest. And only once the mind's own baseline for
  the choice exists: before that, LuthiModel reports the channel "active" because it
  has nothing to compare with, not because Luthi chose. I found this from the
  action layer's own report in a four-moment life, where "remember" read as chosen on
  two of the first four moments.
- **One wrong expectation of mine**, caught by a test: I had the trace's "how much of
  a half-day was lived" factor scaling the step. It scales the limit.
- **Measured in LuthiModel:** the average by itself carries 0.9% of an early episode's
  change into home; with a mind that remembered that episode, 8.7% and 10.0%. No mind
  chooses that well today.

Still open, for Brian: whether the need should be felt as well as told; the chance
rate of rest and of remembering in an untrained mind; the placeholders (5% limit, a
half-day of 432,000 forwards, which is right only at 10 Hz).

## Artifacts

- `sanctuary/life/need.py`, `sanctuary/life/loop.py` (`_lived`, `save`, `status`),
  `sanctuary/life/mind.py` (`Mind.can_rest`, `Mind.life_dir`).
- Tests: `sanctuary/tests/life/test_loop.py` (the need for rest, 10; remembering, 3),
  `sanctuary/tests/life/test_luthi_life.py`
  (`test_a_standing_need_changes_nothing_about_how_luthis_choice_is_read`).
