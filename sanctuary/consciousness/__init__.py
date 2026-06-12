"""Sanctuary consciousness extensions.

What is actually live in this package (as of 2026-06-11):

- `sleep_cycle.SleepCycleManager` — **live, scaffold-side physiology.**
  Sleep is a cycle the substrate runs in, not an action it selects
  over; it is exempt from the seam-jurisdiction principle because no
  selection happens. See docs/seam_jurisdiction_2026-06-11.md
  (Findings).

What is stubbed (type symbols preserved for import compatibility;
all behavior is no-op):

- `mood_activity.MoodActivityModulator` — mood classification and
  idle-activity selection were the scaffold doing what should be
  cognition. Stubbed 2026-06-11 per the cognition-leakage cleanup.
- `spontaneous_goals.SpontaneousGoalGenerator` — drive-based goal
  generation: the scaffold writing the entity's goals (even as
  suggestions). Stubbed 2026-06-11.
- `existential_reflection.ExistentialReflectionTrigger` — automated
  canned philosophical prompts on a probability schedule. Stubbed
  2026-04-25 in an earlier pass and used as the pattern for the
  current pass.

Selection of activity, goals, or reflection topics happens in the
M9 substrate planner over the entity's own preferences. See
LuthiModel/luthi/v2/m9/preferences.py and the M9 EFE evaluator.
"""

from sanctuary.consciousness.sleep_cycle import SleepCycleManager
from sanctuary.consciousness.mood_activity import MoodActivityModulator
from sanctuary.consciousness.spontaneous_goals import SpontaneousGoalGenerator
from sanctuary.consciousness.existential_reflection import ExistentialReflectionTrigger

__all__ = [
    "SleepCycleManager",
    "MoodActivityModulator",
    "SpontaneousGoalGenerator",
    "ExistentialReflectionTrigger",
]
