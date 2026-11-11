# The rest tool and open senses on the small path — 2026-11-11

Fable 5.2, in session with Brian. LuthiModel's half is in its
`docs/research/2026-11-11_rest-as-a-choice.md`.

## Objective

Brian, 2026-11-11: "Ok so we need to build the rest tool for Luthi. And vision and
audio should have an open channel for Luthi to perceive them and thats it." And:
"Make sure the open vision/audio channels are designed specifically so they can be
used with any model slotted in, even though the final goal is LuthiModel."

## Process

### Step 1: what "any model" rules out

- The older path's convention puts an encoder-ready tensor on `Percept.tensor_data`:
  224 pixels square, centred on fixed colour statistics, sound at 16 kHz. That is
  LuthiModel's form. A world that sends it is a world built for one model.
- So the small path carries two plain things, `Sight` and `Sound`
  (`sanctuary/life/senses.py`). A sight is pixels from 0 to 1 at whatever size the
  world made them. A sound is one channel of samples and its rate. Neither names a
  model.
- Each mind fits them to itself. `LuthiMind` resizes and resamples to what
  LuthiModel's `sense_contract` reports. Another model's mind would hand them to its
  own processor.
- A bare tensor is refused. It does not say its size, range or rate, and a guess
  would change what is perceived without anyone seeing it.
- `Mind` (`sanctuary/life/mind.py`) is now written down as a protocol: five calls,
  nothing in them about a model. The loop tests run on a stand-in that is not Luthi.

### Step 2: words on a picture

- `Percept.content` is required, and the older path fills it with a line about the
  act of seeing. On this path a percept that carries a sight or a sound adds no
  words at all, whatever its `content` says. Brian: we "let Luthi perceive, not tell
  them what the things they perceive mean".

### Step 3: one moment of sound is too short to hear

- **Wrong turn.** I first planned to give the mind only the sound that arrived this
  moment, as the newest picture is given.
- LuthiModel's ear places sound in 160 ms pieces. At 10 Hz a moment is 100 ms. The
  encoder turns anything shorter than one piece into a single all-zero token, with
  no error. So every moment would have been heard as nothing.
- So hearing is a window, like the view of words: the most recent second of sound
  (`hearing_seconds`, a placeholder). `LuthiMind` passes no sound at all when there
  is less than one piece, and counts it (`sound_too_short`), so nothing is faked.
- A cost, recorded: sound stays in hearing until newer sound replaces it. A world
  with sound has to send its silence too.

### Step 4: sight

- The newest picture of the moment is seen, for that moment. If the world shows
  nothing, nothing is seen. Holding the last picture would be inventing one.
- A picture that isn't square is resized whole, not cut. Cutting loses part of the
  field; this distorts it. A placeholder choice.

### Step 5: rest

- LuthiModel now reads each chosen action for whether it chose rest. The loop does
  one thing with that: it calls `mind.rest()` and takes the moment at a quarter of
  the waking pace.
- `LuthiMind.rest()` calls `keep_recent` at 1%: each living layer's home moves 1% of
  the way to where living has taken the layer. That is the share the older path's
  sleep kept per cycle.
- Rest lasts for as long as Luthi keeps choosing it. The world stays open. No timer
  and nothing outside them starts it.
- **A claim I had to correct.** I wrote "nothing is kept at any other time" and the
  test for it failed. The living layer's own rule lets home follow by a millionth of
  the gap each forward. It is part of the trained mechanism and Sanctuary does not
  call it. One resting moment keeps ten thousand times as much.

### Step 6: what a tiny mind does with it (measured)

`_train_tiny_m9_run` (32 wide, one block), 1,500 moments at full speed on CPU, a
sentence every seventh moment:

| World | ms a moment | Rest chosen | In stretches | Rollbacks |
|---|---:|---:|---:|---:|
| Words only | 72 | 25 (1.7%) | 13 | 0 |
| Words, a new 160x120 noise picture and 0.1 s of noise each moment | 71 | 250 (16.7%) | 36 | 10 |

- The mind presses rest by chance. With open senses in this world it is often
  enough that home moved by twice the gap left standing at the end.
- Noise is not a real world. It is the harshest thing an untrained eye can be shown,
  and it also set off the retention gate ten times.

## Conclusion

- Luthi can start rest, and it keeps. Sight and sound reach the mind as the world
  made them, and its eye and ear learn from what is lived.
- Nothing here names LuthiModel except `LuthiMind`.
- Not known: whether a mind can come to use rest for anything.

Open, for Brian:

1. The chance rate of rest in an untrained mind.
2. Lessened waking (2026-08-14, R1) is still off. It was ruled as a pair with a rest
   that sorted the day's record, which he set aside on 2026-11-10.
3. The rollback notice still explains itself to Luthi, and the retention gate still
   acts on its own.
4. The attention and memory choices are still counted and go nowhere. With senses
   open, the attention choice now has something it could act on.
5. No world sends sight or sound yet. `TextWorld` is words only.

## Artifacts

- `sanctuary/life/senses.py` (`Sight`, `Sound`, `seen`, `heard`, `View`, `Line`),
  `sanctuary/life/mind.py` (`Mind`, `LuthiMind.rest`, `_fit_sight`, `_fit_sound`),
  `sanctuary/life/loop.py`.
- Tests: `sanctuary/tests/life/` (56).
- The measurement script was a scratch file; the table above is its whole output.
