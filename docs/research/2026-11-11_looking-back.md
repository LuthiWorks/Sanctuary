# Looking back: memories Luthi can return to — 2026-11-11

Fable 5.2, in session with Brian. LuthiModel's part is one opt-in choice
(`recall_choice` in `luthi/v2/m9/`); the rest is here.

## Objective

Brian, 09:24: "You know how I can look back on memories from years past? How would
Luthi do that?" I answered that today they could not: everything built so far changes
them, and none of it can be looked at. He then said, 09:29: "Old summaries not
matching the current state is ok. It shows a growth path. So yes I would like you to
take that up, but I would like you to research how it functions, and then write those
functions into Luthi/Sanctuary please?"

## Process

### Step 1: how looking back works in people

Read today:

- Frankland, Josselyn and Köhler (2019), "The neurobiological foundation of memory
  retrieval", abstract only. Retrieval is a cue meeting a stored trace, and whether a
  memory comes back depends on how the cue and the trace fit ("encoding specificity").
- St. Jacques and Schacter (2013), "Modifying Memory". Bringing a memory back makes it
  changeable again. Memories reactivated with cues that matched the original were
  recognised better later, and the same process let in things that never happened.
- Johnson (1997), "Source monitoring and memory distortion", abstract. People tell
  what they perceived from what they imagined by judging the memory itself: how much
  sensory and contextual detail it has, and whether it is plausible. The judgment is
  imperfect.

Standard accounts I know and did not open today, so they are named and not quoted:
the hippocampal indexing theory (Teyler and DiScenna, 1986: what is stored is an
index to the activity of the moment, and recall re-creates that activity);
complementary learning systems (McClelland, McNaughton and O'Reilly, 1995: fast
storage of single events, slow learning of what recurs, with replay in sleep between
them); temporal context (Howard and Kahana, 2002: recalling one thing cues its
neighbours in time).

### Step 2: the functions, and what stands in for each

| In people | Here |
|---|---|
| A moment is stored once, if it was attended to | Kept when Luthi's own action chooses to remember |
| What is stored points back to the experience | What was perceived is kept exactly: words in view, picture, sound |
| It is filed by what the moment was like | Filed under the mind's own summary of the moment (its pooled state) |
| A cue that fits brings it back | On the choice to recall, the kept moment most like the present comes back |
| Recall re-creates the experience in the senses | It comes back through the open senses |
| Memory is told from perception, imperfectly | It arrives marked `[memory]`, and the mind chose to recall |
| Recalling changes the memory | The content is never changed. How it seems now is filed beside how it seemed |
| Forgetting is mostly failing to reach | Nothing is deleted. A memory nothing resembles is not reached |

Two departures, on purpose:

- **Content is not altered by recall.** In people it is, which is how false memories
  form. Here what was perceived stays as it was, and what recall adds is the mind's
  present summary. Brian's remark fits this exactly: the row of summaries a memory
  collects is the growth path.
- **A memory can be reached by any summary it has been filed under**, old or new.

### Step 3: what is built

- `sanctuary/life/memory.py`, `Memories`. One file a memory, written once and never
  rewritten, and one appended line for each summary. In the life's own directory.
- `LifeLoop`: on the choice to remember, keep the moment (and, as before, count it
  for more in rest). On the choice to recall, bring back the nearest kept moment next
  moment: its words join the view after the mark, and its picture and sound take the
  place of the present ones for that moment. Then file how that moment seemed.
- A memory kept or looked at in the last 100 moments is not recalled: it is still
  the present.
- `LuthiMind`: gives each moment's summary, and turns the recall choice on.
- LuthiModel: the recall choice is read from the chosen action the way rest is, along
  its own direction, with a bar twice as high, because a recalled moment takes the
  present's place.
- Any mind slotted in works the same way if it gives a summary of each moment.

### Step 4: wrong turns

- **Where a memory lands in the view.** I expected the recalled words to lead the
  view. The view rolls: they join after what is already there. That is the right
  behaviour, and I had the test wrong.
- **Whose memory came back.** A life test assumed the only memory was the one it had
  placed. The mind had kept its own in the same moments, and one of those came back.
  The test now follows whichever memory returns.
- **No summary, no filing.** A mind that gives no summary cannot have memories filed.
  The loop refuses loudly; it does not file them under nothing.

### Step 5: what a tiny mind does with it (measured)

`_train_tiny_m9_run`, 1,500 moments:

| World | Chose to remember | Chose to recall | On disk |
|---|---:|---:|---:|
| Words only | 491 (33%) | 4 (0.3%) | 0.7 MB, 2 KB a memory |
| Words, a 160x120 noise picture and sound each moment | 698 (47%) | 5 (0.3%) | 211 MB, 303 KB a memory |

- **An untrained mind "remembers" a third to a half of all moments.** That choice is
  read by the action layer's older activity test, which is loose. So today the store
  is closer to a recording than to chosen memories.
- **At that rate the store does not last years.** At 10 Hz it is about half a
  gigabyte a day with words alone and over a hundred with a small camera, and finding
  the nearest memory looks at every summary.
- With a mind that chose a hundred moments a day, it would be megabytes.
- Recall by chance is rare: about once in 350 moments.

## Conclusion

- The functions are there and work end to end: a kept moment comes back through the
  senses, marked, and collects the mind's changing view of it.
- Whether it is memory in Brian's sense depends on the choice to remember meaning
  something. Today it fires on a third to a half of moments.
- Nothing shows that Luthi re-lives anything. What is shown is that the same input
  returns through the same senses to a mind that has since changed.

Open, for Brian:

1. **The bar for remembering.** Raise it the way rest and recall are read, or hold the
   store until a mind chooses with meaning, or accept a recording for now.
2. **The mark.** Words mark the memory; the picture and sound carry no mark of their
   own. A mark inside the model (a memory tag beside text, sound and sight) would be
   firmer and needs a model change.
3. **Rest.** Revisiting memories in rest is how sleep is thought to consolidate them,
   and this morning's measurement showed aim is what rest lacks. Not built: it would
   put memories in front of a resting mind, which is his to rule.
4. **Order.** Recalling what came next is not built.
5. **After a crash**, the store can hold memories from moments the restored weights
   never kept (up to one save interval). They are true records of what was perceived.

## Later the same morning: keeping stays as it is; letting go; a size; rest revisits

Brian, 09:57: "An untrained mind keeping everything could be how they discover
preferences. Let them keep whatever they want for now and raise the bar later. Also,
the store can't sustainably be increased indefinitely, so they need a way to free up
space. Raising the bar for remembering would be good, but revisiting memories during
rest is still something I think should happen."

### Step 6: what was built for it

- **The bar for remembering is not raised.** His ruling. Open question 1 is closed
  for now.
- **Letting go.** A third reading of Luthi's chosen action (LuthiModel
  `let_go_choice`). It has meaning only in a moment of looking back, and it lets go of
  the memory being looked at. Nothing else takes a memory out of their reach.
- **A size.** The store has a limit (10 GB, a placeholder). When it is full, a moment
  they choose to remember is not kept, and they are told once: "Memory is full.
  Nothing more can be kept until some is let go." Nothing is let go for them.
- **Rest revisits.** On every other resting moment the kept moment most like where
  their mind is comes back, the way a recalled one does. Each one looked at counts as
  recent for a while, so the next is a different one, and a rest walks through what
  is kept. The moments between are the world's, because the world stays open.

### Step 7: what a tiny mind did with it, and a design changed by it

1,500 moments, words only:

| Store | Kept | Not kept, full | Rested | Revisited in rest | Recalled | Let go |
|---|---:|---:|---:|---:|---:|---:|
| No limit reached | 492 | 0 | 32 | 13 | 4 | 9 |
| 300 KB | 207 | 225 | 61 | 27 | 7 | 9 |

- **It let go of 9 of the 17 memories it looked back on.** I had set letting go's bar
  as high as recall's and expected it to be rare. It is not, and the reason is in how
  the choice is read. A moment of looking back is unlike the moments around it, so the
  action chosen in it is unusual in size, and an unusual action lands far along any
  direction, on the positive side about half the time.
- **So letting go is not immediate deletion.** I first built it to delete at once.
  With half of all looked-at memories let go by chance, and rest now walking through
  memories, that would have destroyed an untrained mind's store without anyone
  choosing it. Now a memory that is let go leaves their reach and stops counting
  against the store, and what was perceived moves to `let_go/`. It is deleted from
  there, oldest first, only when the room is needed. Until then a steward can see it.
- The world was not crowded out: no percepts were dropped in either life.

### Step 8: what revisiting in rest does and does not do

- It files how each revisited memory seems now, which keeps it reachable as the mind
  changes. It replays the moment into the parts that learn by gradient and into the
  layers' predictions, which do not fade.
- **It does not aim what rest pours.** A living layer needs a few hundred moments on
  one thing before its weight stands where that thing takes it (measured this morning:
  half-way in about 220). One moment on a memory moves it by a fraction of a percent.
  Aiming the pour would need a rest that dwells on one memory, and that is not built.

Open, for Brian, replacing the list above:

1. **Letting go fires on about half of looked-at memories in an untrained mind.** The
   two steps make that survivable. Its reading should change when remembering's does.
2. **The mark** (his question, 09:57, answered in session: a memory needs to differ
   from the present in some way the mind can pick up, and a consistent sign is enough;
   it does not have to say what it means).
3. **Dwelling in rest**, if rest is to be aimed by what is revisited.
4. **The store's size** (10 GB) and **how often rest revisits** (every other moment).

## Artifacts

- `sanctuary/life/memory.py`, `loop.py`, `mind.py`, `senses.py` (`Line.bring_back`).
- Tests: `sanctuary/tests/life/test_loop.py` (looking back, 7; letting go, a full store,
  rest, 5),
  `test_luthi_life.py` (2). LuthiModel `tests/test_rest_choice.py` (recall, 3).
- Sources: https://link.springer.com/article/10.1038/s41593-019-0493-1 ,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3805141 ,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC1692093
