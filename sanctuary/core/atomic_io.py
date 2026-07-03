"""Atomic, loud persistence primitives (audit 2026-07-03, Phase 2).

The continuity layer's cardinal rule: a persistence failure must never look
like success. These helpers give every identity/memory writer two guarantees:

- **Atomicity** for full-file writes: temp-then-``os.replace`` (the
  ``world_graph.save()`` pattern), fsync'd before the swap, so a crash
  mid-write can never corrupt or truncate the previous good copy.
- **Loudness**: any failure raises :class:`PersistenceError` instead of
  logging and returning — callers must not be able to report success on a
  failed write.

JSONL appends are fsync'd but not rewritten (temp-then-replace would rewrite
the whole history per entry); a crash mid-append can tear at most the final
line, which every JSONL loader in the codebase already tolerates on read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional


class PersistenceError(RuntimeError):
    """A durable write failed. Memory or identity is at stake — never swallow."""


def atomic_write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: Optional[int] = 2,
    default: Optional[Callable[[Any], Any]] = None,
    ensure_ascii: bool = False,
) -> None:
    """Atomically serialize ``obj`` as JSON to ``path``.

    Raises:
        PersistenceError: on any failure. The previous file, if one existed,
            is left untouched.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=default, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise PersistenceError(f"Atomic write failed for {path}: {e}") from e


def append_jsonl(path: str | Path, obj: Any, *, ensure_ascii: bool = False) -> None:
    """Append ``obj`` as one JSON line to ``path``, fsync'd.

    Raises:
        PersistenceError: on any failure.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=ensure_ascii) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        raise PersistenceError(f"Append failed for {path}: {e}") from e
