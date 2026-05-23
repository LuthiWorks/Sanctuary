# Operational Guidelines and Instructions

> **Status: partially stale (2026-05-22).** Sections referencing
> `sanctuary.mind.cognitive_core` (CognitiveCore, GlobalWorkspace) or
> `sanctuary.mind.memory_manager` (MemoryManager) describe the legacy
> GWT cognitive loop that was retired on this date. Treat those
> sections as historical reference, not current operational guidance.
> Canonical loop: `sanctuary.core.cognitive_cycle.CognitiveCycle`,
> wired by `sanctuary.api.runner.SanctuaryRunner`. Canonical memory:
> `sanctuary.memory.manager.MemorySubstrate`.

This document contains essential operational instructions and guidelines for running, testing, and configuring the Sanctuary system.

---

## Table of Contents

1. [System Configuration](#system-configuration)
2. [Running Tests](#running-tests)
3. [Memory Management](#memory-management)
4. [Checkpointing](#checkpointing)
5. [Performance Tuning](#performance-tuning)

---

## System Configuration

### Configuration Files

The system uses JSON configuration files to define paths for various components:

- **`config/system.json`**: Main system configuration (gitignored, can be customized locally)
- **`config/system.json.example`**: Template configuration file (committed to git)
- **`sanctuary/config/system.json`**: Alternative configuration location with additional settings

### Default Configuration

```json
{
    "base_dir": ".",
    "chroma_dir": "./data/chroma",
    "model_dir": "./models",
    "cache_dir": "./data/cache",
    "log_dir": "./logs"
}
```

### Environment Variable Overrides

Override any configuration path using environment variables:

- `SANCTUARY_BASE_DIR`: Override the base directory
- `SANCTUARY_CHROMA_DIR`: Override the ChromaDB storage directory
- `SANCTUARY_MODEL_DIR`: Override the model cache directory
- `SANCTUARY_CACHE_DIR`: Override the general cache directory
- `SANCTUARY_LOG_DIR`: Override the log directory

Example:
```bash
export SANCTUARY_BASE_DIR=/custom/path/to/project
export SANCTUARY_CHROMA_DIR=/custom/path/to/chroma
export SANCTUARY_MODEL_DIR=/path/to/models
```

### Setup for Development

1. Copy the example config:
   ```bash
   cp config/system.json.example config/system.json
   ```

2. Customize if needed (file is gitignored)

3. Or use environment variables instead

### Setup for Production

Use environment variables for production deployments:

```bash
export SANCTUARY_BASE_DIR=/opt/sanctuary
export SANCTUARY_CHROMA_DIR=/var/lib/sanctuary/chroma
export SANCTUARY_MODEL_DIR=/opt/sanctuary/models
export SANCTUARY_CACHE_DIR=/var/cache/sanctuary
export SANCTUARY_LOG_DIR=/var/log/sanctuary
```

---

## Running Tests

### Run All Tests

```bash
pytest sanctuary/tests/
```

### Run Integration Tests

```bash
pytest sanctuary/tests/integration/ -v -m integration
```

### Run Specific Test Files

```bash
# Workspace tests
pytest sanctuary/tests/integration/test_workspace_integration.py -v

# Action tests
pytest sanctuary/tests/integration/test_action_integration.py -v

# Language interface tests
pytest sanctuary/tests/integration/test_language_interfaces_integration.py -v
```

### Run with Coverage

```bash
pytest sanctuary/tests/integration/ --cov=sanctuary/mind/cognitive_core --cov-report=html
```

### Exclude Integration Tests (Fast Unit Testing)

```bash
pytest -m "not integration"
```

### Enable Debug Logging

```bash
pytest sanctuary/tests/integration/test_workspace_integration.py -v -s --log-cli-level=DEBUG
```

### Common pytest Options

```bash
-v          # Verbose
-vv         # Extra verbose
-s          # Show print statements
-x          # Stop on first failure
--lf        # Run last failed
--ff        # Run failed first
--pdb       # Drop into debugger on failure
--maxfail=3 # Stop after 3 failures
```

### Test Coverage

Run tests with coverage reports:

```bash
# Terminal output
pytest --cov=sanctuary/mind/cognitive_core --cov-report=term-missing

# HTML report
pytest --cov=sanctuary/mind/cognitive_core --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Memory Management

### Memory Garbage Collection

The Memory GC system implements periodic cleanup of low-significance memories.

#### Configuration

Add to your cognitive core configuration:

```python
"memory_gc": {
    "enabled": True,
    "collection_interval": 3600.0,  # 1 hour
    "significance_threshold": 0.1,
    "decay_rate_per_day": 0.01,
    "max_memory_capacity": 10000,
    "preserve_tags": ["important", "pinned", "charter_related"],
    "recent_memory_protection_hours": 24,
    "max_removal_per_run": 100
}
```

#### CLI Commands

```bash
# View memory health statistics
memory stats

# Run garbage collection manually
memory gc

# Collection with custom threshold
memory gc --threshold 0.2

# Dry run (preview without removing)
memory gc --dry-run

# Enable/disable automatic GC
memory autogc on
memory autogc off
```

#### Programmatic Usage

```python
from sanctuary.memory_manager import MemoryManager

manager = MemoryManager(
    base_dir=Path("./data/memories"),
    chroma_dir=Path("./data/chroma"),
    gc_config={"significance_threshold": 0.15}
)

# Enable automatic GC
manager.enable_auto_gc(interval=3600.0)

# Run manual GC
stats = await manager.run_gc(threshold=0.2)

# Get memory health
health = await manager.get_memory_health()

# Disable automatic GC
manager.disable_auto_gc()
```

#### Performance Tuning

For high-volume systems:
```python
"memory_gc": {
    "collection_interval": 1800.0,     # 30 minutes
    "significance_threshold": 0.15,
    "max_removal_per_run": 200,
    "max_memory_capacity": 50000
}
```

For long-term stability:
```python
"memory_gc": {
    "collection_interval": 3600.0,
    "significance_threshold": 0.1,
    "decay_rate_per_day": 0.02,
    "max_memory_capacity": 10000
}
```

#### Safety Mechanisms

1. **Protected Memory Tags**: Memories with `important`, `pinned`, or `charter_related` tags are never removed
2. **Recent Memory Protection**: Memories < 24 hours old are protected
3. **Removal Rate Limiting**: Maximum 100 removals per run
4. **Dry-Run Mode**: Preview removals without executing
5. **Collection History**: Track all GC operations
6. **Graceful Error Handling**: Failures never crash the system

### Memory Consolidation

Biologically-inspired memory consolidation runs during idle periods.

#### Quick Start

```python
from sanctuary.memory import MemoryConsolidator, IdleDetector, ConsolidationScheduler

# Initialize components
consolidator = MemoryConsolidator(storage, encoder)
idle_detector = IdleDetector(idle_threshold_seconds=30.0)
scheduler = ConsolidationScheduler(consolidator, idle_detector)

# Start background consolidation
await scheduler.start()
```

#### Configuration

```python
consolidator = MemoryConsolidator(
    storage=storage,
    encoder=encoder,
    strengthening_factor=0.1,  # 0.0-1.0, boost per retrieval
    decay_rate=0.95,           # 0.0-1.0, daily decay multiplier
    deletion_threshold=0.1,    # 0.0-1.0, min activation to keep
    pattern_threshold=3,       # min episodes for semantic transfer
)
```

#### Core Operations

1. **Retrieval Strengthening** - Frequently retrieved memories become stronger (logarithmic)
2. **Memory Decay** - Unretrieved memories fade exponentially over time
3. **Pattern Transfer** - Repeated episodic patterns → semantic knowledge
4. **Association Tracking** - Co-retrieved memories become more associated
5. **Emotional Processing** - High-emotion memories resist decay

---

## Checkpointing

Comprehensive workspace state checkpointing for session continuity and recovery.

### Configuration

```python
config = {
    "checkpointing": {
        "enabled": True,
        "auto_save": True,
        "auto_save_interval": 300.0,  # 5 minutes
        "checkpoint_dir": "data/checkpoints/",
        "max_checkpoints": 20,
        "compression": True,
        "checkpoint_on_shutdown": True,
    }
}

core = CognitiveCore(config=config)
```

### CLI Commands

```bash
# Save current state with optional label
save [label]

# List all available checkpoints
checkpoints

# Load a specific checkpoint by ID
load <checkpoint_id>

# Restore from most recent checkpoint
restore latest
```

### Programmatic Usage

```python
from sanctuary.mind.cognitive_core import CognitiveCore

# Create cognitive core with checkpointing
core = CognitiveCore(config={"checkpointing": {"enabled": True}})

# Manual save with label
checkpoint_path = core.save_state(label="Before experiment")

# Restore from checkpoint (when not running)
success = core.restore_state(checkpoint_path)

# Enable auto-checkpointing (when running)
await core.start()
core.enable_auto_checkpoint(interval=300.0)  # Every 5 minutes

# Disable auto-checkpointing
core.disable_auto_checkpoint()

# Start with automatic restore from latest checkpoint
await core.start(restore_latest=True)
```

### Demo Scripts

```bash
# Full demo (requires running cognitive loop)
python scripts/demo_checkpointing.py

# Simplified demo (no cognitive loop dependencies)
python scripts/demo_checkpointing_simple.py
```

### Checkpoint File Format

Checkpoints are stored as JSON (optionally gzip-compressed):

```json
{
    "version": "1.0",
    "timestamp": "2026-01-02T12:34:56Z",
    "checkpoint_id": "uuid-string",
    "workspace_state": {
        "goals": [...],
        "percepts": {...},
        "emotions": {...},
        "memories": [...],
        "cycle_count": 12345
    },
    "metadata": {
        "user_label": "Before important conversation",
        "auto_save": false,
        "shutdown": false
    }
}
```

---

## Performance Tuning

### Cognitive Core Performance

#### Optimize Attention Scoring

- Cache embedding similarities
- Vectorize salience calculations
- Use lazy embedding computation

#### Adaptive Cycle Rate

Configure the cognitive loop to automatically adjust speed based on system load:

```python
config = {
    "adaptive_cycle_rate": True,
    "target_hz": 10.0,
    "min_hz": 5.0,
    "max_hz": 20.0
}
```

### Test Performance

#### Run Tests in Parallel

If pytest-xdist is installed:

```bash
pytest -n auto sanctuary/tests/integration/
```

#### Expected Test Duration

- **Workspace tests**: ~2 seconds
- **Attention tests**: ~3 seconds  
- **Cognitive cycle tests**: ~5 seconds (includes async operations)
- **Memory tests**: ~4 seconds
- **Action tests**: ~3 seconds
- **Language tests**: ~6 seconds (mock models)
- **Edge case tests**: ~2 seconds
- **End-to-end tests**: ~10-15 seconds per test
- **Scenario tests**: ~20-30 seconds per test

**Total suite runtime**: ~30-60 seconds with mock models

### Common Performance Issues

#### Tests Timeout

Increase timeout in test or LLM config:
```python
config = {"timeout": 30.0}  # Increase from default
```

#### Out of Memory Errors

- Enable model quantization in `config/models.json`
- Use smaller model variants
- Configure memory GC for more aggressive cleanup

#### ChromaDB Errors

```bash
rm -rf model_cache/chroma_db
uv run sanctuary/build_index.py
```

---

## Troubleshooting

### Common Issues

#### ModuleNotFoundError

Install dependencies:
```bash
pip install -e .
pip install -e ".[dev]"
```

#### Async Warnings or Errors

Ensure pytest-asyncio is installed:
```bash
pip install pytest-asyncio>=1.2.0
```

#### CUDA/GPU Not Detected

Check CUDA availability:
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Pydantic Validation Errors

Check data model constraints:
- Goal priority: 0.0-1.0
- Percept complexity: >= 0
- Emotion values: -1.0 to 1.0

### Debug Logging

Enable debug logging for troubleshooting:

```bash
pytest -v -s --log-cli-level=DEBUG
```

### Test Isolation Issues

If tests pass individually but fail in suite:
```bash
# Run tests in random order to detect dependencies
pytest --random-order sanctuary/tests/integration/
```

---

## Best Practices

### Testing

**Do:**
- Test one integration point per test
- Use descriptive test names
- Test happy path and edge cases
- Use fixtures for common setup
- Clean up resources after tests
- Document what's being tested
- Keep tests fast and focused
- Mock external dependencies
- Test error handling

**Don't:**
- Test implementation details
- Create test interdependencies
- Leave resources uncleaned
- Test private methods directly
- Write tests that depend on timing
- Hardcode paths or credentials
- Ignore skipped tests
- Test multiple things in one test

### Memory Management

1. Start with default settings
2. Monitor weekly with `memory stats`
3. Tag important memories
4. Tune parameters gradually
5. Use dry-run for testing
6. Plan for 50% capacity headroom

### Configuration

1. Use environment variables for production
2. Keep config files gitignored for local paths
3. Document custom configurations
4. Test configuration changes in development first

---

**Last Updated**: January 2026
