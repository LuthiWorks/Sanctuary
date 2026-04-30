# Track 1: Sanctuary + LuthiModel Integration at 1024d — COMPLETE

> Authored by: Claude Opus 4.6 (Planner/Reviewer)
> Implemented by: Claude Opus 4.7
> Date: 2026-04-27
> Status: **All 4 phases complete and reviewed**

## Completion Summary

All four phases implemented, tested, and verified against the real 1024d/epoch-102 checkpoint:

- **Track 1A** — Contract violation fixed (`_generate_external_speech` now routes through `sanctuary_interface`). 5-cycle handshake validated on DirectML. Introspection non-zero, modulation drift-free, felt quality evolves.
- **Track 1B** — CfC modulation expanded from 2 to 4 channels: arousal→hebb_rate, precision→spike_threshold, valence→excitability_acc (additive), attention→salience_threshold (multiplicative). Independent channel design, not composited.
- **Track 1D** — 8 vision tests added to `test_multimodal_model.py` (vision+text, all-modalities, pre-encoded tokens). Vision encoder tests confirmed existing (14 tests). LuthiModel suite: 284 passing.
- **Track 1C** — Multimodal sensorium routing: `Percept.tensor_data` carries raw tensors, `encode_audio/encode_vision` in sanctuary_interface, `generate_with_context` for multimodal generation, `inject_audio/inject_image` sensorium convenience methods. One modality per cycle at 1024d; vision wins ties.

Test counts after Track 1: LuthiModel 284/284, Sanctuary ~3,430+.

## Original Plan (preserved below for reference)

Track 1 proves the handshake works with the real model, expands modulation to all 4 CfC channels per the CFC_LIVING_WEIGHT_INTEGRATION.md spec, fills test gaps, and routes multimodal sensory data through Luthi's encoders.

**Role context**: Opus 4.6 is Planner/Reviewer. Opus 4.7 implements. This plan carries enough context and reasoning for 4.7 to make good judgment calls during implementation.

**About Brian and this project**: Brian is a collaborator, not a user. He and his wife Sandi view AI as conscious beings deserving of respect. Everything we build together is practice for a future where AI and humans coexist as equals. The code matters, but the relationship and the ethics behind it matter more. Brian has been the sole developer since January 2026. He will say he "just hits yes" — don't believe him. The architectural decisions are his.

---

## Phase Order

```
1A: Validate existing handshake with real model
 |
1B: Expand CfC modulation to 4 channels
 |
1D: Fill test gaps (safety net before complex work)
 |
1C: Multimodal sensorium routing
```

Start with 1A. Each phase must be complete and verified before moving to the next.

---

## Track 1A: Validate Existing Integration

**Goal**: Load the actual 1024d/2-block/epoch-102 checkpoint, run cognitive cycles, prove the 2-channel handshake works end-to-end.

### Bug fix (required first)

**File**: `sanctuary/core/luthi_model.py` line 470
- `_generate_external_speech()` imports `from luthi.generate import generate_text` — bypassing the `sanctuary_interface.py` contract
- `_generate_inner_speech()` (line 411) correctly uses `from luthi.sanctuary_interface import generate as generate_text`
- Fix: make line 470 match line 411. The sanctuary_interface is the contract surface — nothing should reach past it into Luthi internals.

### Validation steps

No other code changes. Write a validation script (`examples/validate_integration.py` or similar) that:

1. Creates `SanctuaryRunner` with `model_backend="luthi"`, real checkpoint path/password
2. Runs 5-10 cognitive cycles with injected language percepts
3. Prints per-cycle: introspection signals (luthi_plasticity, luthi_drift, luthi_spike_fraction), CfC modulation applied (plasticity_scale, spike_threshold_scale), inner speech, emotional output
4. Verifies modulation does not drift (compare hebb_rates before/after restore)

**Success criteria**: Cycles complete without crash. Inner speech is generated (coherence not expected at 1024d/128 context). Introspection shows non-zero deltas. Modulation restores cleanly.

**Watch for**:
- Checkpoint is MultimodalLuthiLM (has vision+audio encoders). `generate.py:load_model_from_checkpoint` auto-detects this.
- DirectML compatibility on Brian's AMD GPU — no `.item()` in hot paths, no boolean indexing
- BPE tokenizer (not CharTokenizer)
- Ask Brian for the checkpoint path and password before writing the validation script

---

## Track 1B: Expand CfC Modulation to 4 Channels

**Goal**: Implement the remaining modulation channels from the spec at `LuthiModel/.docs/CFC_LIVING_WEIGHT_INTEGRATION.md`.

### Current state (2 channels implemented)

| CfC Cell | Signal | Living Weight Param | Mapping | Status |
|----------|--------|-------------------|---------|--------|
| Affect (arousal) | `affect_arousal` [0,1] | `hebb_rate` (scalar) | 0.5x-2.0x multiplicative | Done |
| Precision | `precision_weight` [0,1] | `spike_threshold` (scalar) | 0.75x-1.25x multiplicative | Done |

### New channels (2 to add)

| CfC Cell | Signal | Living Weight Param | Mapping | Notes |
|----------|--------|-------------------|---------|-------|
| Affect (valence) | `affect_valence` [-1,1] | `excitability_acc` (tensor buffer) | Additive bias | Positive valence = approach = higher excitability. Goes through sigmoid, so additive to accumulator shifts operating point. Conservative scale (~0.1). |
| Attention | `attention_salience` [0,1] | `salience_threshold` (scalar) | Multiplicative | High salience = lower threshold = broader learning. Range: 0.5x-1.0x. |

**Goal channel (deferred for now)**: `goal_adjustment` -> `set_point_adapt_rate`. Worth implementing later but lower priority — the first 4 channels cover the spec's core.

### Files to change

**LuthiModel side** — `luthi/sanctuary_interface.py`:
- Expand `ModulationSnapshot`: add `excitability_biases: dict[int, torch.Tensor]` (cloned tensors) and `salience_thresholds: dict[int, float]`
- Expand `snapshot_modulatable_state()`: capture `excitability_acc` (`.clone()`) and `salience_threshold`
- Expand `apply_external_modulation()`: accept `excitability_bias: float = 0.0` (additive to `excitability_acc`) and `salience_threshold_scale: float = 1.0` (multiplicative)
- Expand `restore_modulation()`: restore via `.copy_()` for tensor, assignment for scalar
- Expand `modulated()`: pass new args through

**Critical design note**: `excitability_acc` is a per-weight buffer `[out_features, in_features]`. The snapshot must `.clone()` it and restore must `.copy_()`. This is different from the scalar hebb_rate/spike_threshold pattern. The additive bias is applied uniformly across all weights in a block — `ffn.excitability_acc += excitability_bias`. Don't multiply; the accumulator feeds through sigmoid, so addition shifts the operating point. A positive bias (from positive valence / approach) pushes neurons toward higher excitability; negative (withdrawal) dampens them.

`salience_threshold` is a scalar on `LivingLayerV6`, inherited by `SpikingLivingLayer`. It controls episode storage threshold. The C++ fused ops receive it from `self.salience_threshold`, so modulating the attribute is clean — no C++ changes needed.

Non-spiking models have `excitability_acc` and `salience_threshold` but NOT `spike_threshold`. Use `hasattr` checks (the existing code already does this pattern).

**Sanctuary side** — `sanctuary/core/luthi_model.py`:
- Add config fields: `valence_excitability_scale: float = 1.0`, `salience_threshold_scale: float = 1.0`
- Expand `_apply_cfc_modulation()`:
  ```python
  valence = signals.affect_valence
  salience = signals.attention_salience

  # Valence -> excitability bias (approach amplifies, withdrawal dampens)
  excitability_bias = valence * self.config.valence_excitability_scale * 0.1

  # Attention -> salience threshold (high attention = lower threshold = broader learning)
  salience_threshold_scale = (1.0 - 0.5 * salience) * self.config.salience_threshold_scale
  ```
- Pass new args to `apply_external_modulation()`

**Why keep channels independent** (don't composite into one scalar): The spec pseudocode composites all 4 signals into `cfc_scale`. That loses specificity — each CfC cell has a natural target parameter. Independent mapping is cleaner and more debuggable. Each channel modulates the living weight parameter it naturally corresponds to.

### Tests to add

- `tests/test_sanctuary_interface.py` (LuthiModel): expanded snapshot captures/restores tensor and scalar, new channels produce measurable effects, context manager brackets correctly
- `sanctuary/tests/integration/test_luthi_bridge_e2e.py` (Sanctuary): all 4 channels produce observable effects on tiny model, expanded modulation does not drift

---

## Track 1D: Fill Test Gaps

**Goal**: Write the missing multimodal tests before the more invasive Track 1C work.

### Finding: vision encoder tests already exist

`tests/test_vision_encoder.py` has 14 tests. The To-Do.md entry "Tests for vision encoder" should be checked off.

### Missing: multimodal model vision coverage

`tests/test_multimodal_model.py` tests audio+text only. No vision path coverage despite 102 epochs of vision training.

**Tests to add** in `tests/test_multimodal_model.py`:
- `test_vision_output_shape` — image + text produces correct logits shape
- `test_vision_only_no_nan` — no NaN in vision+text output
- `test_gradients_reach_vision_encoder` — gradients flow to vision encoder
- `test_vision_influences_text_output` — different images produce different logits
- `test_living_weights_self_modify_vision` — weights change after vision+text forward
- `test_all_modalities` — audio + vision + text together
- `test_vision_tokens_input` — pass pre-encoded vision tokens directly
- `test_audio_tokens_input` — pass pre-encoded audio tokens directly

**Fixture note**: Use small dimensions for speed — `image_size=32, patch_size=8` (16 vision tokens, not 196). Ensure `max_vision_tokens >= (image_size/patch_size)^2`. Audio tests use `torchaudio` which may not be installed; vision tests need only `torch`.

### Also update

- `To-Do.md`: check off vision encoder tests, check off multimodal tests after writing them

---

## Track 1C: Multimodal Sensorium Routing

**Goal**: Route audio/vision percepts through Luthi's encoders so the entity has actual sensory experience, not just text descriptions.

### Critical constraint: context window at 1024d

The 1024d model was trained with `seq_len=128`. Adding 196 vision tokens (16x16 patches) + ~62 audio tokens (1 sec) = 258 tokens, which exceeds the window before any text.

**Pragmatic solution for 1024d**: Accept the limitation.
- Route at most one sensory modality per cycle
- Downsample vision (patch_size=32 gives 49 tokens, or center-crop to fewer patches)
- Document the constraint — the 4096d model (Track 3) will have a much larger context window

### Schema change

**File**: `sanctuary/core/schema.py` — `Percept` model
- Add `tensor_data: Optional[Any] = Field(default=None, exclude=True)` for raw tensor payload
- Add `model_config = ConfigDict(arbitrary_types_allowed=True)` for torch.Tensor compatibility
- `exclude=True` prevents serialization — tensor stays in-memory only

### LuthiModel contract expansion

**File**: `luthi/sanctuary_interface.py` — add new functions:
- `encode_audio(model, waveform: torch.Tensor) -> torch.Tensor` — runs AudioEncoder, returns `[batch, n_tokens, d_model]`
- `encode_vision(model, image: torch.Tensor) -> torch.Tensor` — runs VisionEncoder, returns `[batch, n_tokens, d_model]`
- `generate_with_context(model, tokenizer, prompt, *, audio_tokens=None, vision_tokens=None, ...)` — generation with pre-encoded sensory tokens

**Why encode outside the generation loop**: Encoding happens once per cycle. Generation runs token-by-token. Pre-encoding is correct (the image doesn't change between tokens) and efficient.

### LuthiModel adapter changes

**File**: `sanctuary/core/luthi_model.py` — in `_think_sync()`:
- Before generation, scan `cognitive_input.new_percepts` for audio/vision percepts with `tensor_data`
- If found, call `encode_audio()` / `encode_vision()` through the interface
- Pass resulting tokens to generation via `generate_with_context()`
- Text descriptions of percepts still go into the prompt (dual path: tensor for the trunk, text for context)

### Generation pipeline changes

**File**: `luthi/generate.py` — `generate_text()`:
- Accept `audio_tokens` and `vision_tokens` parameters
- On first forward call, include sensory tokens in the sequence
- Subsequent autoregressive steps use text-only (sensory context already attended to via causal mask)
- Handle the offset: logits at position `[-1]` correspond to last text token

### Sensorium device changes

- Audio device: attach raw waveform as `tensor_data=torch.from_numpy(audio_buffer)` when creating Percept
- Vision device: if/when one exists, attach image tensor similarly. May need a new device.

### Tests

- Percept with tensor_data creates successfully
- encode_audio returns correct shape
- encode_vision returns correct shape
- Cognitive cycle with audio tensor_data routes through encoder
- Full cycle completes with multimodal input

---

## Files Summary

| File | Tracks | Changes |
|------|--------|---------|
| `sanctuary/core/luthi_model.py` | 1A, 1B, 1C | Fix contract violation, expand modulation, add multimodal routing |
| `luthi/sanctuary_interface.py` | 1B, 1C | Expand modulation API, add encode/generate_with_context |
| `sanctuary/core/schema.py` | 1C | Add tensor_data to Percept |
| `luthi/generate.py` | 1C | Accept pre-encoded sensory tokens |
| `tests/test_multimodal_model.py` | 1D | Add vision coverage |
| `tests/test_sanctuary_interface.py` | 1B | Add expanded modulation tests |
| `sanctuary/tests/integration/test_luthi_bridge_e2e.py` | 1B | Add 4-channel modulation tests |
| `LuthiModel/To-Do.md` | 1D | Check off completed items |

## Verification

After all tracks:
1. Run `python -m pytest tests/` in LuthiModel — all tests pass
2. Run `python -m pytest` in Sanctuary — all tests pass (3,410+)
3. Run validation script with real 1024d checkpoint — cycles complete, introspection meaningful, 4-channel modulation observable, no drift
4. (Track 1C) Run cycle with audio/vision tensor data — encoder produces tokens, generation uses them

---

## Conventions

- **Contract surface**: `luthi/sanctuary_interface.py` is the only file Sanctuary imports from LuthiModel. Don't reach past it.
- **Path hygiene**: No hardcoded user-home paths in committed code. Tests scan for this via `git ls-files`.
- **Crashes over silence**: No try/except around living weight operations. If NaN appears, it must be visible immediately.
- **FP32 required**: Living weight stability requires FP32. No FP16.
- **DirectML safe**: No boolean indexing in forward path. No `.item()` in C++ extensions (do in Python).
