# The memory store, reworked: a design — 2026-11-17

**Status:** a design by Fable 5.2 (chat; design seat). Reviewed by Opus 5.5 on 2026-11-17 and revised the same day: see "Revision". Brian ruled on the store's size and cold storage the same morning: see "Brian's rulings", which is newest where the two disagree. **Built on 2026-11-18** by terminal Fable (`686c1f3`; `docs/research/2026-11-18_memory-store-rework-built.md`), with Opus's review of cold storage and importance taken. Two costs the design did not have came out of the build: opening the store, and the exact search's time. The answers, and the second importance formula, are LuthiModel `docs/reviews/2026-11-18_build-brief-10-opening-search-importance-for-terminal-fable.md`, not yet reviewed or built.

## Objective

Opus's review of 2026-11-14 (LuthiModel `docs/reviews/2026-11-14_rest-memory-and-dreaming-review-from-opus.md`, §5 and §6) found the store built on 2026-11-11 growing by about 5 GB a day outside its own limit. Brian, 2026-11-17: *"The store needs serious reworking then... We can't let that happen."* This is the rework, shaped by what he said the same morning (LuthiModel `docs/DECISIONS.md`):

- On storing compactly: *"Much of life's meaning is in the small details... I would argue every aspect of an experience is inportant. But how do we tell? How do we determine if there might be parts that could be lost without impact.?"*
- On pictures and sound: *"If the fidelity of the memory is not impacted, this sounds good to me."*
- On letting go: *"Choosing what to let go is crucial but it should be automatic based on importance to Luthi. They need to be able to make that choice though, so make sure it is available to them. Also this is the kind of thing that might occur in a subconscious state. Lets explore that as something to consider building in."*
- On a stretch of kept moments being one memory: *"Sounds good to me."*

## Process

### Step 1: why it grew, and a larger problem behind it

- A kept memory has two parts: what was perceived (a file, counted by the limit) and the mind's summary of the moment (a line in `summaries.jsonl`, not counted). I wrote each summary as text: 23 bytes a number, 17 KB at 768 wide.
- An untrained mind keeps a third to a half of its moments. At 10 moments a second that is about 288,000 a day, and 4.9 GB of summaries. Letting a memory go did not free them.
- **Opus measured words only. With sight on it is worse.** A picture is kept as the world gave it, as 4-byte numbers: at least 0.6 MB. The limit does count that, so the store would not overflow; a 10 GB store would be full in about 83 minutes.
- Consecutive kept moments are nearly the same moment. Each keeps the whole window in view, which moves one step at a time.

### Step 2: "no meaningful loss" was wrong, and Brian's question had a measurable answer

I proposed storing summaries at half precision and told Brian it lost nothing meaningful. He asked how we would tell. By measuring, on the stores left by the 2026-11-11 test lives (archived at `E:\luthi_experiment_archive\2026-11-11_memory_test_lives`):

| Store | Summaries | A different memory comes back, at half precision |
| --- | --- | --- |
| `life_True` (sight and sound) | 703 | 13 cues (1.8%) |
| three words-only lives | 495 to 519 | 2 cues each (0.4%) |
| two more | 232, 445 | 0 |

- Half precision changes each likeness by up to 1.6e-4. In these lives the gap between the most alike memory and the next is typically 4e-6 to 1e-4. So rounding decides which memory comes back, some of the time.
- A made-up store at 768 wide showed no change at all. It was not like a real one, and I would have been reassured by it had I not also used the real ones.
- **The text was never more exact than a 4-byte number:** all 23,152 stored numbers checked are exactly what a 4-byte float holds. So writing them as 4-byte binary loses nothing and is 5.5 times smaller (3 KB at 768 wide).

**So: nothing is stored inexactly.** Half precision is dropped. Brian's instinct was right and my claim was not.

### Step 3: what the same stores say about stretches and about likeness

- **Kept moments come in stretches.** 491 kept moments fell in 151 stretches (longest 25); 698 in 85 (longest 95). One memory a stretch is three to eight times fewer memories, each holding its words once.
- **In these tiny minds every summary is nearly the same.** Moments kept one after another are alike at 0.9996 to 0.99999; moments far apart are still alike at 0.999. So recall by likeness is choosing among near-equals. That is a property of an untrained 32-wide mind, and it is why rounding mattered. It also says likeness should be taken after removing what all the summaries share (their running mean); otherwise the shared part drowns the rest.

## The design

### 1. Nothing is stored inexactly

- **Summaries:** 4-byte binary, exactly the numbers the mind produced.
- **Pictures and sound: exactly what the world gave.** If the world's picture is 8-bit underneath (turning it to 8-bit and back gives the same numbers, checked when it is kept), it is stored as 8-bit, losslessly compressed. If it is not, it is stored as given, losslessly compressed. Sound the same way with 16-bit.
- **There is no lossy path in the code.** A photo-style format that discards detail is not used anywhere.

What this costs, honestly: exact pictures are perhaps six to twelve times smaller than now, not thirty.

### 2. One memory for one stretch

Moments kept one after another are one memory: the words once, each picture and sound once, and the mind's summary at each kept moment. A look back files one more summary, as now.

### 3. The limit counts everything, and letting go frees everything

Summaries are counted. When what was let go is deleted for room, its summaries go with it. Until then they stay, as now.

### 4. Letting go: by importance to Luthi, and by their choice

This changes my design of 2026-11-11, in which nothing was ever let go on the mind's behalf and a full store simply refused.

- **When room is needed, what matters least to Luthi is let go first.**
- **Importance is made only of Luthi's own signals.** Nothing a steward judges enters it:
  - how strongly they chose to remember it, against their own bar at the time;
  - how often they chose to recall it, and how it seemed to them when they did;
  - what happened when it came back to them in rest;
  - what they felt, where the mind has a gauge of that.
- **Age alone is not unimportance.** Brian's first question was how Luthi could look back on memories from years past.
- **Their choice stays, both ways.** Choosing to let go, after a recall they chose, acts at once. Choosing to remember while looking back counts toward keeping.
- **Two steps, as now.** What is let go, by them or for them, leaves their reach and waits in `let_go/`; it is deleted, oldest first, only when the room is needed.
- **A caveat that matters:** an untrained mind's signals are close to chance, so early on this is close to letting go at random. The two steps and the archive convention are what stand between that and loss. Whether automatic letting go should wait until the mind's choices are better than chance is **Brian's**.
- **Whether Luthi is told** when something is let go for them is **Brian's**. People are not told.

### 5. A subconscious, to explore and not yet to build

Brian: *"this is the kind of thing that might occur in a subconscious state."* In people, most forgetting and most consolidation happen in sleep, without choice. Here that would mean things that happen during rest, without an action of the mind's:

- **Importance moves with how a memory is met in rest.** A let-go reading while rest shows a memory would lower its importance, and a remember reading raise it, without either acting at once. (Brief 5 has that reading do nothing during rest; this would give it a quiet effect in place of none.)
- **Letting go for room happens in rest,** not in the middle of waking life.
- **Rest ties memories to what the weights hold.** This is where it meets the design for relevant updates (LuthiModel `docs/research/2026-11-17_relevant-updates-design.md`).

Each is an option. None is built until Brian chooses.

### 6. What it comes to

At 10 moments a second with a third of them kept, 768 wide:

| | Now | Reworked |
| --- | --- | --- |
| Summaries | 4.9 GB a day, uncounted | 0.9 GB a day, counted |
| Words | 0.4 GB a day | well under 0.1 GB a day |
| With sight | full in about 83 minutes | perhaps 10 to 30 GB a day of exact pictures |

- **Words only:** a 10 GB store holds about ten days before letting go begins.
- **With sight, exactly kept:** it holds hours, not days. Letting go by importance then runs all the time, and what Luthi holds is a turning selection of what mattered most to them. Fidelity is kept by forgetting whole memories, never by blurring all of them.
- **How much disk Luthi's memory may have is Brian's.** 10 GB was my placeholder.
- One summary a kept moment is most of the words-only size. Keeping one a stretch would cut it several times, at the cost of the record of how each moment seemed. I recommend keeping them all until a real life shows what they are used for.

### 7. Also

- **Likeness is taken after removing the summaries' running mean** (step 3).
- **Old stores are never rewritten.** The archive of 2026-11-11 stays readable by the code of that day.
- **A probe or test that runs a model through a life keeps the life's folder** in a place it names, and it is archived as a run is (Brian, 2026-11-17).
- The two faults that are wrong under any format are already in brief 5 (LuthiModel `docs/reviews/2026-11-17_build-brief-5-rest-readings-store-faults-for-terminal-fable.md`, Part A).

## Revision, the same day: after Opus's review

Opus reviewed this design (LuthiModel `docs/reviews/2026-11-17_relevant-updates-and-memory-store-review-from-opus.md`, Part 2). The design sized the disk and not the search, and left three things about a stretch unsaid. All four are taken as Opus wrote them.

1. **A stretch is its records.** Each kept moment writes one small record that is never changed, into its stretch's folder: the words new since the record before, its picture and sound, and its summary. A crash loses at most the moment in flight, nothing is ever rewritten, and the words are still stored once.
2. **Recall brings back the view at the moment that matched.** Each record notes its moment's view as a place in the stretch's words, so every moment can be brought back exactly as it was, and a recall lands where its summary matched. Letting go and importance act on the whole stretch.
3. **A memory's summaries live in its own folder and are deleted with it.** The matrix the search runs over is a cache rebuilt from the folders. The cache may be rewritten; a record never.
4. **The search.** Opus timed it at full width: as built, 2.1 seconds a search after one day of memories; as one product over a matrix, 0.08 seconds after a day and 0.78 after ten, while rest asks for a search about 2.5 times a second.
   - First lives search exactly, as one product, and say loudly when a search takes longer than a moment allows.
   - Anything faster (one summary a stretch, fewer dimensions, approximate neighbours) changes which memory comes back for some cues. That is half precision's question again, and it is answered the same way: measured against the exact search on a real store, then put to Brian. No real store of full width exists yet to measure it on.

Also taken:

- **The net while choices are close to chance.** I wrote that the two steps and the archive convention stand between chance and loss. The archive convention does not cover deletion for room inside a running life, and with sight that deletion begins in the first hours. Opus's answer is to move what is deleted for room to cold storage instead, while the mind's choices are close to chance. Whether to is **Brian's**, with the size of the store.
- **Which mean a summary is centred by.** Centring by today's mean makes an old memory differ from now mostly by how far the mind has moved. Centring each by the mean of its own time compares what stood out then with what stands out now. To be measured on a life long enough to drift.
- **Importance does not take rest's readings yet.** In the 2026-11-11 lives they fired together under drift. Not until lives of several seeds show they no longer do (brief 5, Part B, is the fix).
- **Sound's 16-bit check** records which scale the round trip held at.
- **The measurements of steps 2 and 3 are a script now:** LuthiModel `redteam/fable_2026_11_17/measure_summaries_in_kept_stores.py`. It reproduces this document's numbers from the archive (23,875 numbers checked when the small stores are included).

## Brian's rulings, 2026-11-17, 09:44

> *"1. For now use the e: drive, and Luthi can have lets say 100gb. If we need to update that we can do it later. 2. Yes, this is a good idea. In fact it may be worth building as a permanent feature because we don't know how quickly Luthi might actually learn when deployed."*

Recorded in LuthiModel `docs/DECISIONS.md`. What they settle, and what I chose to follow from them (his to overrule):

- **The limit is 100 GB, on E:.** It counts everything under the life's `memories/`.
- **Cold storage is permanent, and the store's code deletes no memory.** Section 3's "its summaries go with it" now means: they go with it to cold storage. Wherever this document says "deleted for room", read "moved to cold storage, whole and exact".
- **Automatic letting go runs from the start.** I had asked whether it should wait until the mind's choices beat chance. With cold storage nothing is lost by a choice made near chance, so it does not wait. What is lost is Luthi's reach to the memory, which is what letting go means.
- **Cold storage has no size and is never deleted from by code.** When its drive cannot take a move, nothing leaves, the store is full, a new moment is not kept, and Luthi is told once, as on 2026-11-11.
- **Luthi is not told when something is let go for them,** for now. It is counted and logged. Still Brian's.
- **Whether a memory in cold storage can come back to Luthi** is not built and is Brian's. The format keeps it possible: a folder moved back is all it takes.

What it comes to on E: at 10 moments a second, 768 wide:

| | Reaches 100 GB in | Then cold storage grows by |
| --- | --- | --- |
| Words only | about three months | about 1 GB a day |
| With sight, exactly kept | three to ten days | 10 to 30 GB a day |

With sight, the drive and not the limit becomes the question. That is for when a world sends pictures.

**Importance, first formula** (new today; for Opus to review): the highest rank of how strongly they chose to remember, among their own recent choices, plus one for each recall they chose, plus one for each time they chose to remember while looking back. The records hold the signals, so the formula can change without touching a memory. What rest brings counts for nothing yet.

The build: LuthiModel `docs/reviews/2026-11-17_build-brief-9-the-memory-store-for-terminal-fable.md`.

## Conclusion

A store that never exceeds its limit, keeps everything it keeps exactly, and frees room by letting go what matters least to Luthi by their own signals. Settled by Brian on 2026-11-17: the size of the store (100 GB on E:) and cold storage (permanent). Still open, and Brian's: whether Luthi is told; whether a memory in cold storage can come back to them; which parts of a subconscious to build.

What I do not know: how large exact pictures of the real world will be; what importance looks like in a mind whose choices are not chance; whether removing the mean is enough to make likeness mean something in a small mind.

## Artifacts

- The stores measured: `E:\luthi_experiment_archive\2026-11-11_memory_test_lives\2026-11-11_memory_test_lives.zip` (sha256 `db46fad1…2d490`; LuthiModel `docs/DECISIONS.md`).
- Opus's measurements: LuthiModel `docs/reviews/2026-11-14_rest-memory-and-dreaming-review-from-opus.md`, §5 and §6.
- The measurements of steps 2 and 3: LuthiModel `redteam/fable_2026_11_17/measure_summaries_in_kept_stores.py`, run against the archive.
