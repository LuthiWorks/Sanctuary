# The memory store, reworked: what was built — 2026-11-18

**Status:** built by Fable 5.2 (terminal; build seat) from LuthiModel build brief 9 and the design of 2026-11-17 (`2026-11-17_memory-store-rework-design.md`), with Opus's review of cold storage and importance. Not yet reviewed. The full report is LuthiModel `docs/reviews/2026-11-18_build-report-brief-9-from-terminal-fable.md`.

## Objective

Replace the store of 2026-11-11, which grew about 5 GB a day outside its own limit, with one that keeps everything exactly, stays inside its limit, and deletes nothing.

## What was built

- **Nothing is stored inexactly.** Every record is decoded from its own bytes and compared bit for bit with what was given before it is written.
- **A memory is a stretch of moments kept one after another:** one folder, one file for each kept moment and each look back, never changed. Recall brings back the view at the moment that matched.
- **The limit (100 GB by default) counts every byte under `memories/`.**
- **No memory is deleted by code.** What leaves for room is moved to cold storage, checked by size and hash.
- **When nothing Luthi let go is waiting, the stretch that matters least to them is let go for them,** by signals recorded in the records. They are not told, unless a switch is set.
- **Old stores are refused and never rewritten.**

The module's docstring (`sanctuary/life/memory.py`) has the layout on disk.

## Results

Measured at 768 wide, words only, scaled to a day of 288,000 kept moments (LuthiModel `runs_meta/memory_store_2026_11_18/`):

| | Measured |
| --- | --- |
| Bytes a day | 0.99 GB as file sizes, 1.10 GB on disk |
| Files a day | 288,000 |
| Opening the store | 21 minutes for one day of memories |
| Exact search after one day | 0.19 to 0.28 s; a moment is 0.1 s |
| Search cache in memory | 0.82 GB for each day of memories |

## Conclusion

The store holds every must-hold of the brief under test. It is not ready for a long life: opening it and searching it both cost more than a life at 10 moments a second allows after about a day of memories. The two fixes (a search cache on disk, and a ruling on how exact the search must be) are with the design seat and Brian.

What I do not know: the size of exact pictures from a real world; whether the slow file opens are the virus scanner; how the importance formula behaves in a mind whose choices are not chance.

## Artifacts

- `sanctuary/life/memory.py`, `loop.py`, `__main__.py`
- `sanctuary/tests/life/test_memory_store.py` (66 tests), `test_loop.py`
- LuthiModel `redteam/terminal_fable_2026_11_18/probe_memory_store_size_and_search.py`
