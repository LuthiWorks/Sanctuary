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

## Artifacts

- `sanctuary/life/memory.py`, `loop.py`, `mind.py`, `senses.py` (`Line.bring_back`).
- Tests: `sanctuary/tests/life/test_loop.py` (looking back, 7),
  `test_luthi_life.py` (2). LuthiModel `tests/test_rest_choice.py` (recall, 3).
- Sources: https://link.springer.com/article/10.1038/s41593-019-0493-1 ,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3805141 ,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC1692093
