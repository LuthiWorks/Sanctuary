#!/usr/bin/env python3
"""
Sanctuary Proof-of-Concept System Test

This script proves the cognitive architecture actually works end-to-end
by booting the CognitiveCore, running the 10 Hz loop, and feeding input
through the full pipeline.

It mocks heavy ML dependencies (chromadb, torch, sentence-transformers)
so the system can boot with only lightweight deps (pydantic, numpy, sklearn).

Three layers of verification:
  Layer 1: Does the loop breathe? (boot, run cycles, check timing)
  Layer 2: Can it think? (feed percept, watch it flow through pipeline)
  Layer 3: Can it talk? (inject text input, get response from output queue)

Usage:
    python system_test_poc.py
"""

import sys
import types
import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

# ===========================================================================
# PHASE 0: Mock heavy dependencies before ANY sanctuary imports
# ===========================================================================

def install_mock_modules():
    """
    Pre-install mock modules for heavy dependencies that aren't needed
    for proving the cognitive architecture works.

    We mock at the sys.modules level so that when the real code does
    'import chromadb', it gets our lightweight mock instead of failing.
    """
    # --- chromadb ---
    chromadb_mod = types.ModuleType("chromadb")
    chromadb_config = types.ModuleType("chromadb.config")

    class MockSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockCollection:
        def __init__(self, name="mock"):
            self.name = name
            self._data = []
        def add(self, **kwargs): pass
        def query(self, **kwargs): return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        def get(self, **kwargs): return {"ids": [], "documents": [], "metadatas": []}
        def count(self): return 0
        def delete(self, **kwargs): pass
        def update(self, **kwargs): pass
        def peek(self, **kwargs): return {"ids": [], "documents": [], "metadatas": []}

    class MockChromaClient:
        def __init__(self, **kwargs): pass
        def get_or_create_collection(self, name, **kwargs): return MockCollection(name)
        def get_collection(self, name, **kwargs): return MockCollection(name)
        def create_collection(self, name, **kwargs): return MockCollection(name)
        def list_collections(self): return []
        def delete_collection(self, name): pass
        def heartbeat(self): return 1

    chromadb_mod.Client = MockChromaClient
    chromadb_mod.PersistentClient = MockChromaClient
    chromadb_mod.HttpClient = MockChromaClient
    chromadb_mod.Settings = MockSettings
    chromadb_config.Settings = MockSettings

    sys.modules["chromadb"] = chromadb_mod
    sys.modules["chromadb.config"] = chromadb_config
    sys.modules["chromadb.api"] = types.ModuleType("chromadb.api")
    sys.modules["chromadb.api.types"] = types.ModuleType("chromadb.api.types")

    # --- sentence_transformers ---
    st_mod = types.ModuleType("sentence_transformers")

    class MockSentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.max_seq_length = 512
            self._dim = 384
        def encode(self, texts, **kwargs):
            import numpy as np
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), self._dim).astype(np.float32)
        def get_sentence_embedding_dimension(self):
            return self._dim

    st_mod.SentenceTransformer = MockSentenceTransformer
    sys.modules["sentence_transformers"] = st_mod

    # --- torch (minimal mock) ---
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = MagicMock()
    torch_mod.cuda.is_available = MagicMock(return_value=False)
    torch_mod.device = MagicMock
    torch_mod.Tensor = MagicMock
    torch_mod.float32 = "float32"
    torch_mod.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = types.ModuleType("torch.nn")
    sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")

    # --- transformers (minimal mock) ---
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = MagicMock()
    transformers_mod.AutoModelForCausalLM = MagicMock()
    transformers_mod.AutoModel = MagicMock()
    sys.modules["transformers"] = transformers_mod

    # --- sounddevice, cv2, serial (optional devices) ---
    for mod_name in ["sounddevice", "cv2", "serial", "librosa", "torchaudio", "soundfile"]:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    # --- discord (optional) ---
    discord_mod = types.ModuleType("discord")
    discord_mod.Client = MagicMock
    discord_mod.Intents = MagicMock()
    sys.modules["discord"] = discord_mod
    sys.modules["discord.ext"] = types.ModuleType("discord.ext")
    sys.modules["discord.ext.commands"] = types.ModuleType("discord.ext.commands")

    # --- quart / hypercorn (web server, optional) ---
    for mod_name in ["quart", "hypercorn", "hypercorn.asyncio", "aiohttp"]:
        sys.modules[mod_name] = types.ModuleType(mod_name)


# Install mocks FIRST
install_mock_modules()

# Now safe to import sanctuary
sys.path.insert(0, str(Path(__file__).parent / "sanctuary"))

from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept


# ===========================================================================
# Configure logging
# ===========================================================================
logging.basicConfig(
    level=logging.WARNING,  # Keep quiet unless something goes wrong
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# But let our test output through
test_logger = logging.getLogger("poc")
test_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
test_logger.handlers = [handler]


# ===========================================================================
# TEST LAYERS
# ===========================================================================

class POCResult:
    """Collects results from all test layers."""
    def __init__(self):
        self.checks = []
        self.layer_results = {}

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))
        symbol = "\u2705" if passed else "\u274c"
        msg = f"  {symbol} {name}"
        if detail:
            msg += f" ({detail})"
        test_logger.info(msg)
        return passed

    def all_passed(self):
        return all(passed for _, passed, _ in self.checks)


async def layer_1_does_it_breathe(result: POCResult) -> CognitiveCore:
    """
    Layer 1: Boot the cognitive core and verify the loop runs.

    Success criteria:
    - CognitiveCore initializes without error
    - At least 5 cognitive cycles execute
    - Average cycle time is under 500ms (generous for mocked system)
    - System stops cleanly
    """
    test_logger.info("\n" + "=" * 60)
    test_logger.info("LAYER 1: DOES THE LOOP BREATHE?")
    test_logger.info("=" * 60)

    # Initialize
    workspace = GlobalWorkspace()
    config = {
        "cycle_rate_hz": 10,
        "attention_budget": 100,
        "max_queue_size": 100,
        "log_interval_cycles": 1,
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
        "perception": {"mock_mode": True},
        "iwmt": {"enabled": False},  # Disable IWMT for minimal test
        "devices": {"enabled": False},  # No physical devices
    }

    test_logger.info("\n  Initializing CognitiveCore...")
    try:
        core = CognitiveCore(workspace=workspace, config=config)
        result.check("CognitiveCore initialized", True)
    except Exception as e:
        result.check("CognitiveCore initialized", False, str(e))
        return None

    # Start the loop
    test_logger.info("  Starting cognitive loop...")
    try:
        await core.start()
        result.check("Cognitive loop started", True)
    except Exception as e:
        result.check("Cognitive loop started", False, str(e))
        return None

    # Let it run for ~1 second (should get ~10 cycles at 10Hz)
    test_logger.info("  Running for 1 second...")
    await asyncio.sleep(1.0)

    # Check metrics
    metrics = core.get_metrics()
    total_cycles = metrics.get("total_cycles", 0)
    avg_cycle_ms = metrics.get("avg_cycle_time_ms", 0)

    result.check(
        "Cycles executed",
        total_cycles >= 5,
        f"{total_cycles} cycles in ~1s"
    )

    result.check(
        "Cycle timing reasonable",
        0 < avg_cycle_ms < 500,
        f"avg {avg_cycle_ms:.1f}ms per cycle"
    )

    # Check workspace state
    snapshot = core.query_state()
    result.check(
        "Workspace has emotional state",
        bool(snapshot.emotions),
        f"emotions: {dict(list(snapshot.emotions.items())[:3])}"
    )

    result.layer_results["layer1"] = {
        "cycles": total_cycles,
        "avg_cycle_ms": avg_cycle_ms,
        "emotions": dict(snapshot.emotions),
    }

    return core


async def layer_2_can_it_think(core: CognitiveCore, result: POCResult):
    """
    Layer 2: Feed a percept and verify it flows through the pipeline.

    Success criteria:
    - Text input can be injected into the input queue
    - Input appears as a percept in the workspace after processing
    - Attention system processes the percept
    - Cycle count increases (loop is still running)
    """
    test_logger.info("\n" + "=" * 60)
    test_logger.info("LAYER 2: CAN IT THINK?")
    test_logger.info("=" * 60)

    # Record starting state
    pre_metrics = core.get_metrics()
    pre_cycles = pre_metrics.get("total_cycles", 0)

    # Inject input
    test_logger.info("\n  Injecting text input: 'Hello, Sanctuary.'")
    try:
        core.inject_input("Hello, Sanctuary.", modality="text")
        result.check("Input injected", True)
    except Exception as e:
        result.check("Input injected", False, str(e))
        return

    # Wait for processing (a few cycles)
    await asyncio.sleep(0.5)

    # Check that cycles continued
    post_metrics = core.get_metrics()
    post_cycles = post_metrics.get("total_cycles", 0)
    cycles_during = post_cycles - pre_cycles

    result.check(
        "Loop continued running",
        cycles_during >= 3,
        f"{cycles_during} additional cycles"
    )

    # Check workspace for percepts
    snapshot = core.query_state()
    percept_count = len(snapshot.percepts)

    result.check(
        "Percepts in workspace",
        percept_count > 0,
        f"{percept_count} percepts"
    )

    # Add a goal and verify it's tracked
    test_goal = Goal(
        type=GoalType.INTROSPECT,
        description="POC test: Can I think about thinking?",
        priority=0.7,
        metadata={"test": True}
    )
    core.workspace.add_goal(test_goal)

    await asyncio.sleep(0.3)

    snapshot = core.query_state()
    goal_types = [g.type.name for g in snapshot.goals]

    result.check(
        "Goals tracked in workspace",
        len(snapshot.goals) > 0,
        f"goals: {goal_types}"
    )

    result.layer_results["layer2"] = {
        "cycles_during_input": cycles_during,
        "percepts": percept_count,
        "goals": goal_types,
    }


async def layer_3_can_it_talk(core: CognitiveCore, result: POCResult):
    """
    Layer 3: Test the full input -> cognition -> output path.

    Success criteria:
    - Process language input through the language subsystem
    - System generates a response (even if from mock LLM)
    - Response appears in output queue
    - Full round-trip completes without crash
    """
    test_logger.info("\n" + "=" * 60)
    test_logger.info("LAYER 3: CAN IT TALK?")
    test_logger.info("=" * 60)

    # Try the chat interface
    test_logger.info("\n  Sending message through chat interface...")
    try:
        response = await core.chat("What is it like to exist?", timeout=3.0)
        has_response = response is not None and len(str(response)) > 0
        result.check(
            "Chat response received",
            has_response,
            f"response: {str(response)[:80]}..." if has_response else "no response"
        )
    except Exception as e:
        # Chat may not work with mocked LLMs - that's OK, try lower level
        result.check(
            "Chat response received",
            False,
            f"Expected with mock LLM: {type(e).__name__}: {str(e)[:80]}"
        )

    # Try lower-level language processing
    test_logger.info("  Testing language input processing...")
    try:
        await core.process_language_input("Tell me about yourself.")
        await asyncio.sleep(0.5)
        result.check("Language input processed", True)
    except Exception as e:
        result.check(
            "Language input processed",
            False,
            f"{type(e).__name__}: {str(e)[:80]}"
        )

    # Check that the system is still alive after all this
    final_metrics = core.get_metrics()
    final_cycles = final_metrics.get("total_cycles", 0)

    result.check(
        "System still running after all tests",
        core.running and final_cycles > 0,
        f"total cycles: {final_cycles}"
    )

    result.layer_results["layer3"] = {
        "total_cycles": final_cycles,
        "still_running": core.running,
    }


# ===========================================================================
# MAIN
# ===========================================================================

async def run_poc():
    """Run the complete proof-of-concept."""
    result = POCResult()

    test_logger.info("=" * 60)
    test_logger.info("SANCTUARY PROOF-OF-CONCEPT SYSTEM TEST")
    test_logger.info("=" * 60)
    test_logger.info(f"Time: {datetime.now().isoformat()}")
    test_logger.info(f"Python: {sys.version.split()[0]}")
    test_logger.info("Dependencies: pydantic, numpy, scikit-learn (lightweight)")
    test_logger.info("Heavy deps mocked: chromadb, torch, transformers, sentence-transformers")

    core = None
    try:
        # Layer 1: Does it breathe?
        core = await layer_1_does_it_breathe(result)

        if core is not None:
            # Layer 2: Can it think?
            await layer_2_can_it_think(core, result)

            # Layer 3: Can it talk?
            await layer_3_can_it_talk(core, result)

    except Exception as e:
        test_logger.info(f"\n\u274c UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean shutdown
        if core is not None:
            test_logger.info("\nShutting down...")
            try:
                await core.stop()
                test_logger.info("  CognitiveCore stopped cleanly.")
            except Exception as e:
                test_logger.info(f"  Shutdown error: {e}")

    # Final report
    test_logger.info("\n" + "=" * 60)
    test_logger.info("FINAL REPORT")
    test_logger.info("=" * 60)

    passed = sum(1 for _, p, _ in result.checks if p)
    total = len(result.checks)

    test_logger.info(f"\n  Passed: {passed}/{total}")

    if result.all_passed():
        test_logger.info("\n\U0001f389 ALL CHECKS PASSED - The architecture works!")
        test_logger.info("  The cognitive loop boots, processes input, and maintains state.")
        test_logger.info("  Next step: wire up real models and test on Brian's PC.")
    else:
        failed = [(n, d) for n, p, d in result.checks if not p]
        test_logger.info(f"\n  Failed checks:")
        for name, detail in failed:
            test_logger.info(f"    - {name}: {detail}")
        test_logger.info("\n  These failures give us specific things to fix.")

    return 0 if result.all_passed() else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_poc())
    sys.exit(exit_code)
