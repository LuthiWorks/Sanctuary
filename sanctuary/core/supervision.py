"""Background-task supervision: make silent task death loud (audit 2026-07-03,
item 14).

An asyncio task created with ``create_task`` and no done-callback swallows any
exception that escapes its coroutine — the task simply vanishes and nothing
observes the failure. ``supervise`` attaches a done-callback that logs the
death loudly and, for critical tasks, invokes a handler so the owner can mark
itself unhealthy. ``CancelledError`` is treated as normal shutdown, never a
failure.

Note on scope: the long-lived loops in this codebase self-heal internally
(their bodies are ``while ...: try/except Exception: continue``), so
supervision here surfaces the *unexpected* death — an exception outside the
loop body, or a task that ends without being asked to — rather than restarting
a healthy loop. Auto-restart would duplicate the existing self-healing and
fight each owner's stop()/cancel lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def supervise(
    task: asyncio.Task,
    *,
    name: str,
    fatal: bool = False,
    on_death: Optional[Callable[[BaseException], None]] = None,
) -> asyncio.Task:
    """Attach a done-callback that makes a task's unexpected death observable.

    Args:
        task: the task to supervise (typically ``asyncio.create_task(...)``).
        name: human label for the log line.
        fatal: log at CRITICAL rather than ERROR — for tasks whose death is
            catastrophic (e.g. the cognitive runner itself).
        on_death: optional handler invoked with the exception when the task
            dies from something other than cancellation (e.g. flip an owner's
            ``_started`` flag so health checks report the loop as down).

    Returns the same task, so call sites can wrap in place:
        ``self._task = supervise(asyncio.create_task(coro), name="...")``
    """

    def _on_done(t: asyncio.Task) -> None:
        if t.cancelled():
            logger.debug("Supervised task %r cancelled (normal shutdown)", name)
            return
        exc = t.exception()
        if exc is None:
            # Clean return is treated as healthy. For a fatal task this assumes
            # the only exit is an intended one (the cognitive runner exits only
            # via stop() today — max_cycles is test-only). If a fatal task ever
            # gains a non-stop() clean-exit path, its death would pass silently
            # here at DEBUG with the owner still marked started; revisit then.
            logger.debug("Supervised task %r exited cleanly", name)
            return
        logger.log(
            logging.CRITICAL if fatal else logging.ERROR,
            "Supervised task %r died: %r", name, exc, exc_info=exc,
        )
        if on_death is not None:
            try:
                on_death(exc)
            except Exception:
                logger.exception("on_death handler for %r itself raised", name)

    task.add_done_callback(_on_done)
    return task
