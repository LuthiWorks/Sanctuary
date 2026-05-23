"""Root conftest.py — prevents pytest from collecting hardware-dependent tests."""

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
