# Two store faults, and what letting go may act on — 2026-11-17

**Status:** built and tested, by Fable 5.2 (terminal, Brian's desktop), from chat Fable's brief 5 (LuthiModel `docs/reviews/2026-11-17_build-brief-5-rest-readings-store-faults-for-terminal-fable.md`), Parts A and B items 3 and 4. For Opus 5.5 to review. The store's format and how much is kept are not touched: that rework is being laid out for Brian separately.

## Objective

Correct four things Opus's review of 2026-11-14 found in the work of 2026-11-11, which Brian ruled on at 05:22 today (LuthiModel `docs/DECISIONS.md`).

## What was wrong, and what was built

1. **The moment count started at 0 on every start.** Every memory carries the moment it was kept at, and recall leaves out what is "still the present" by comparing with the count. After a restart nothing older could be recalled until the new count passed the old one.
   - The count is now saved with the life, in `<life>/life_loop.json`, at every save, and read at start.
   - Memories are written as they are kept and the count only at a save. So after a stop that was not clean, or in a life from before today, memories are newer than the count on disk. The loop then goes on from the moment after the newest memory, logs it, and shows it in `status()` (`moved_past_memories_at_start`). No memory is ever in the future.
2. **A memory's id came from the lines alone.** A crash between a memory's file and its line left a file with no line; the next keep picked that id, found the file, and refused, at every start from then on.
   - The next id is one past the highest id in any line or file, in `memories/` or `memories/let_go/`.
   - A file with no line is listed in `report()` (`files_with_no_line`) and logged. It is never deleted or rewritten, and it cannot be recalled, looked back on or let go: it has no summary to be found by.
3. **Letting go acts only on a memory the mind chose to recall.** A let-go choice made while looking at a memory that rest brought back does nothing, and is counted (`let_go_chosen_on_a_memory_rest_brought`). The look is still filed, as any look back is.
4. **A moment that lets a memory go keeps nothing.** Before, it could let the memory go and, through the memory channel, keep a copy of its words. Counted as `not_kept_in_a_moment_that_let_go`.

## Before and after, with Opus's probe

`redteam/opus_review_2026_11_14/probe_memories_across_a_restart.py` (LuthiModel), on the loop tests' stand-in mind.

| Case | Before | After |
| --- | --- | --- |
| First recall after a restart, 50 memories kept 150 or more moments ago | after 300 moments of choosing it | at once |
| The next keep after a crash between file and line | refuses, and every keep after it | works, id 3; id 2 reported |
| Let go and keep in one moment | a copy of the let-go words is kept | nothing is kept |

The probe's second part now raises where it opens "the newest memory": there is none, which is the change. I ran its parts one at a time and did not edit it.

## Choices that were mine

- **In a moment that lets a memory go, the whole memory channel is skipped**, the mind's `remember()` included, not only the keep. The brief says the moment "keeps nothing". If `remember()` should still weigh what rest pours in that moment, it is one line to change.
- **The count goes on from after the newest memory** when the saved count is behind. The other choice was to refuse to start. A life that stopped uncleanly should be able to go on, and the fact is logged and shown.
- Only the count is saved. The loop's other tallies (`rested`, `remembered`, `recalled` and the rest) still start at 0 with each start. They are reports of a session, and nothing reads them to decide anything except `rested`, which sets which resting moment revisits a memory.

## What I did not do

- A crash inside `let_go` between its line and the file's move leaves a memory that is out of reach by its line while its file still counts against the store. Not in the brief; noted for the rework.
- I did not look for or touch any existing life's directory. Tests use temporary directories.

## Verification

`sanctuary/tests/life/`: 76 pass, 1 skipped (the test that needs LuthiModel). Six new tests in `test_loop.py`, at the end.

## Artifacts

- `sanctuary/life/memory.py`, `sanctuary/life/loop.py`
- `sanctuary/tests/life/test_loop.py`
