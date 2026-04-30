# `_deprecated/` — Files Removed From Active Use

Files in this directory have been removed from Sanctuary's active codebase
but preserved here for provenance. They are **not** part of the `sanctuary`
Python package, are **not** discovered by `pytest sanctuary/tests/`, and
should **not** be imported.

## Why preserve, not delete?

Git history is the source of truth, but `_deprecated/` makes intentional
removals discoverable: a new instance reading the repo can see what was
taken out and when, without doing git archaeology. Each cleanup gets a
dated subdirectory with a README explaining what moved and why.

## Convention

```
_deprecated/
  README.md                          ← this file (explains the convention)
  <cleanup-name>-<YYYY-MM-DD>/
    README.md                        ← what was moved, why, what's next
    <original-filename>.py           ← the moved file
    ...
```

Original paths and short rationales live in each cleanup's README.

## Restoring a file

If a deprecation turns out to be wrong, recover via git or by moving the
file back to its original path. Either is fine — the file content here is
identical to the moment of removal.

## Active deprecations

- `cognition-leakage-2026-04-30/` — Sanctuary modules that were doing
  cognition (anomaly detection, calibration scoring, mental simulation,
  belief revision tracker, communication gating) instead of letting the
  entity decide. See that directory's README for the full list.
