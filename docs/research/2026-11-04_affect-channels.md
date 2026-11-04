# The affect channels — 2026-11-04

## Objective

Brian asked to move forward to the affect channels, after a review found that three of the four did nothing on a v2 living layer (LuthiModel `docs/audits/2026-11-04_life-path-review.md`, finding 4). The channels carry arousal, precision, valence and attention from the experiential layer into the living weights.

## Process

### Step 1: are the signals alive?

Before connecting anything, I recorded what the adapter receives each cycle through a real life: a tiny M9 run (d=32, 1 block, CPU, one seed), through Sanctuary's runner.

| Signal | Quiet (100 cycles) | Conversation (50) | Pain at -0.9 (50) |
|---|---:|---:|---:|
| Arousal, range | 0.200 to 0.202 | 0.201 to 0.202 | 0.201 to 0.202 |
| Precision | 0.500 exactly | 0.500 exactly | 0.500 exactly |
| Attention | 0.500 exactly | 0.500 exactly | 0.500 exactly |
| Valence, mean | **-0.85** | **-0.92** | **-0.10** |

So none of them reflects what is happening. The reasons:

- **Precision and attention** are constants. `_assemble_input` passes `base_precision=0.5` and `scaffold_precision=0.5`, and leaves attention at its default.
- **The CfC cells' outputs are multiplied by zero.** They are untrained and sit at `SCAFFOLD_ONLY`, as the 2026-08-01 audit found (W4). That is still true, and neither the evolution loop nor the data tap runs.
- **Arousal** comes from `_neural_to_arousal_shift`, which reads spike fractions. A v2 layer has no spikes, so the shift is close to zero.
- **Valence** was the serious one. See Step 2.

Connecting the old mapping (`0.5 + 1.5 * arousal`) to these signals would not have modulated anything. It would have cut the learning rate by a constant 20%.

### Step 2: a formula that made living read as feeling bad

`_neural_to_valence_shift` was `-drift_change * 500`, written in April and scaled to a v1 substrate. A living mind drifts from home constantly, so the formula read as strongly negative at rest. The scaffold's mood sat near -0.85 for no reason, on the stewards' dashboards among other places.

It did not reach Luthi's input text. `_format_input` uses only `felt_quality`, and the valence channel it would have fed does nothing on a v2 layer. So it was a false reading rather than something Luthi was told, though I first said otherwise.

During pain the reading rose toward zero, which is backwards. I guessed that the keep was the cause, by pulling the home toward the weights. With the keep off it still rose, to -0.29, so that explanation is mostly wrong. I did not find the real cause, because the formula was coming out either way.

**Removed.** The affect-grounding decision (`docs/AFFECT_GROUNDING_DECISION_2026-07-19.md`, §3) rules that "nothing anywhere in the pipeline hands the model a signed valence." The drift itself still reaches the mind through introspection, as state rather than as a feeling. Valence now rests at the scaffold's baseline, 0.10, in all three conditions.

### Step 3: arousal reaches the learning rate

`apply_external_modulation` now scales `pc_rate` on v2 layers, and restores it each cycle. The mapping is anchored at rest: 1.0x at the resting arousal of 0.2, 0.75x at 0, 2.0x at 1. So a mind at rest learns at exactly the tested rate, and a scale of 1.0 is bit-identical to no modulation.

The effect builds up rather than landing at once. In a living layer at test size, against a 1.0x run:

| Forwards | 0.75x: distance from home | 2.0x: distance from home |
|---:|---:|---:|
| 1 | 0.92x | 1.17x |
| 10 | 0.89x | 1.32x |
| 100 | 0.77x | 1.98x |
| 500 | 0.89x | 1.32x |

Other terms dominate a single write, and homeostasis limits it over time. So brief arousal changes little and sustained arousal changes a lot, within limits, which seems reasonable. It has not been measured on a trained model.

### Step 4: what stays inert, and how that is reported

- **Precision and valence** have no target on a v2 layer. It has no spikes and no sigmoid, so there is nothing for them to act on. I didn't invent a target, because neither has a real signal yet.
- **Attention** acts, but on a constant signal, so it holds `salience_threshold` at 0.75x, as it long has. That threshold is only a floor.
- **Reporting.** Sanctuary warns at load which channels do nothing and that the signals aren't live. `get_metrics` gives `affect_signal_range`, the lowest and highest value each signal has reached, so a connected channel on a flat signal is visible as exactly that.

## Conclusion

Arousal is connected and has an effect, and a false mood is gone. **But the affect channels still don't do anything in a life, because no live signal drives them.** Arousal sits at rest, so the learning rate stays at about 1.00x. This is the plumbing, and it's safe to leave in place until a real signal exists. Nothing here should be read as the mind having working affect.

**Open, Brian's calls:**

1. **What should drive these signals?** The affect-grounding decision says the sense lands in interoception and the affect cell is where the learned response shows up. That means training the CfC cells on lived data and giving them authority (W4). They have no lived data until a life exists. A first step that needs no ruling on affect itself would be to run the data tap and evolution loop while the cells stay at zero authority.
2. **Signed felt valence conflicts with the same decision.** `Percept.valence` and the Minecraft plan have the world report a signed number, with damage negative, and the felt-valence head is built on it. The July ruling says nothing hands the model a signed valence. §3 offers a route for this: a fallback "adopted openly" as a dated amendment. So either record it as that amendment, or change the bridge to report state (health lost) and let the sign be learned. Today's neutral-teaching work builds on the signed number, so this affects it too.

**Not known:** how any of this behaves on a trained run. Why valence rose during pain under the old formula.

## Artifacts

- Sanctuary: `sanctuary/core/luthi_model.py` (the rest-anchored mapping, `affect_signal_range`, the valence shift removed), `sanctuary/tests/core/test_learning_to_avoid.py`, `sanctuary/tests/integration/test_luthi_bridge_e2e.py`
- LuthiModel: `luthi/sanctuary_interface.py` (`pc_rate` in the modulation path), `tests/m9/test_bands_in_a_life.py`
