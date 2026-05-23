"""
Language Interfaces: Peripheral language I/O adapters.

LLMs are used here for parsing input and generating output, but they
are NOT the cognitive core. The canonical cognitive loop lives at
``sanctuary.core.cognitive_cycle.CognitiveCycle``.
"""

from __future__ import annotations

from .language_input import LanguageInputParser
from .language_output import LanguageOutputGenerator

__all__ = [
    "LanguageInputParser",
    "LanguageOutputGenerator",
]
