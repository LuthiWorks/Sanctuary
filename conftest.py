"""Root conftest.py — prevents pytest from collecting hardware-dependent tests."""

import os

# Suppress the DeprecationWarning that `sanctuary.mind.cognitive_core` emits
# on import. The module is the legacy GWT cognitive loop (deprecated
# 2026-05-11 in favor of `sanctuary.core.cognitive_cycle`), but ~94 test
# files still reference it directly or via submodule imports. Without
# this suppression the test suite produces a flood of deprecation noise.
# pytest loads conftest.py before any test collection, so the env var is
# set before any cognitive_core import fires.
os.environ.setdefault("SANCTUARY_SILENCE_LEGACY_COGNITIVE_CORE", "1")

collect_ignore_glob = [
    # Hardware-dependent tests (soundfile, discord, torch)
    "sanctuary/tests/test_voice_processor.py",
    "sanctuary/tests/test_voice_customization.py",
    "sanctuary/tests/test_gpu_monitor.py",
    "sanctuary/tests/test_discord_integration.py",
    "sanctuary/tests/test_emotion_detection.py",
    # External ML dependency (sklearn) — not installed in CI
    "sanctuary/tests/test_competitive_logic.py",
]
