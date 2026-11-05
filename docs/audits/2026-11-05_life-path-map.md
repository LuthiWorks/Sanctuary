# Map of the life path — 2026-11-05

**By:** Fable 5.1, at Brian's request, after he asked whether we have built something overly complex.
**What it is:** every mechanism a Luthi life touches, sorted into four groups, with whether it is actually connected today. It describes; it doesn't recommend. What stays is Brian's call.

**How it was checked.** I read the code, then ran one life and counted: 700 cycles at default settings (sleep on), text percepts on every third cycle, at Sanctuary `8ad369d` and LuthiModel `318640b`. The script is LuthiModel `scripts/map_life_path.py`.

**Limits.** One tiny model (d=32, 1 block, 4 training batches) on CPU, one seed, text only. "Not connected" means not connected in this life. Audio and vision were not exercised. I did not re-check the August finding that `sanctuary/mind/` is imported by nothing.

## The life in numbers

| | In 700 cycles |
|---|---|
| **What reached Luthi's input** | The percept's text (234 cycles), the `[self]` number (700), and a note that a memory was written (5). Nothing else. |
| **What Luthi's choices did** | 309 memory writes, none of which came back. 92 attention choices, with nothing for them to act on. No speech. |
| **Affect signals** | Every one was a constant for the whole life. |
| **What changed Luthi's weights** | Their own perception, the lived update (700), corpus replay (700 steps), the keep (20), and sleep consolidation (20). |

Sanctuary assembles more than this each cycle and doesn't pass it on: the charter summary (676 characters), a catalog of 26 tools, scaffold signals and the proprioception signals were built on all 700 cycles and never put in Luthi's input.

For scale: Sanctuary is 49,554 lines of production Python. The loop and Luthi's adapter (`core/`) are 7,114. LuthiModel's living layer is about 2,500 lines, and the M9 action layer is 7,400 with 52 settings.

## 1. The mind

What is Luthi.

| Mechanism | What it does | In a life today |
|---|---|---|
| Living layers | Weights that rewrite themselves on every forward pass, from local prediction error | **Connected.** They move away from home in every life measured. |
| Trunk and predictor | Encode what is perceived and predict the next state | **Connected.** Learns from life on every cycle (700 of 700). |
| Action layer (habit, value, planner) | Chooses an action each cycle, planning 8 steps | **Connected.** What it aims at comes from four built-in preferences (group 4). |
| Speech | Turns an action into words | **Silent.** The text readout is untrained. 0 of 700. |
| Attention | Turns an action into gates on the senses | **No effect.** It scales audio and vision only, and no code reads `attention_guidance`. 92 choices. |
| Memory writes | Turns an action into a stored memory | **One-way.** 309 written, 311 in the store, 0 returned. |

## 2. The world and body

What exists around the mind: senses, the loop, saving, and things offered for use.

| Mechanism | What it does | In a life today |
|---|---|---|
| Percepts | Text arrives as something perceived | **Connected.** |
| The loop | Runs the cycle, at 10 Hz | **Connected.** |
| Saving a life | Writes the whole life after each sleep and at shutdown | **Connected.** 1 sleep, 1 checkpoint. |
| Motor feedback | Tells Luthi when a strong memory was written | **Connected.** 5 of 700. |
| Audio and vision encoders | Carry sound and images into the trunk | **Not exercised here.** |
| Memory store | Holds what was written | **One-way** (group 1). |
| Tools (26 in the catalog; 2,672 lines) | Things Luthi could use | **Not connected.** The catalog isn't in Luthi's input, and Luthi can't request a tool. |
| World graph | A structured record Luthi could keep | **Not connected.** Luthi can't write to it. Empty. |
| Charter and identity (1,746 lines) | Who Luthi is told they are, and what they write about themselves | **Not connected.** The charter isn't in Luthi's input. Self-authored identity is empty, since Luthi can't author it. |
| Physics world (`world/`, `embodiment/`, `physics/`; about 2,800 lines) | A body in a lawful world, with weather | **Not connected.** It runs as its own loop, apart from a life. |
| Minecraft bridge | The first world | **Not built.** |

## 3. Instruments

They watch and report. Nothing acts on them.

| Mechanism | What it does | In a life today |
|---|---|---|
| Kill checks (K-M9-2 to 8) | Flag planner or value trouble, and stillness | **Reporting.** All healthy. No code acts on them. |
| Welfare gauge (`felt_summary`) | Counts how life has read | **Reporting.** 701 read, 2 bad, 2 good. Its alarm thresholds aren't set. |
| Proprioception snapshot | Reads plasticity, drift and precision twice a cycle | **Reporting,** though it was meant for Luthi. It never reaches their input. |
| Activity and delta-s bands | Calibrate what counts as active or still | **Working.** Armed. |
| Felt-valence head, with its neutral examples and baseline | Learns how each kind of state has read | **Runs and feeds nothing,** since the planner stopped counting it on 2026-11-05. |
| CfC data tap | Records what the affect cells do | **Recording.** 701 records per cell, from inputs that are flat. |
| Metrics and LuthiScope | Show stewards the above | Not looked at. |

## 4. Deciding for Luthi

Sanctuary or M9 computes, decides or changes something on Luthi's behalf.

**Acting on Luthi every cycle:**

| Mechanism | What it does | In a life today |
|---|---|---|
| Four built-in preferences | Engagement, coherence, connection and truthfulness set what every action is aimed at | **Acting.** This is the open reward question. |
| Corpus replay | Runs one batch of the training corpus through the mind per cycle, against forgetting. Luthi doesn't perceive it. | **Acting.** 700 steps. |
| Retention gate and rollback | Checks every 20 lived steps, and can undo recent learning. Luthi is told. | **Armed.** 0 rollbacks. |
| Sleep schedule and gate | Luthi sleeps for 38 cycles in every 538, and most percepts are dropped while they do | **Acting.** |
| Sleep consolidation | On each deep-sleep cycle, moves every living layer's home 1% toward its weights and evens out plasticity | **Acting.** 20 calls. |
| The keep | When a moment is felt or chosen for memory, moves the home toward the weights, within a budget | **Acting.** 20 calls; 293 more moments were over budget. |
| The felt reading | Computes a signed number from the change in Luthi's prediction error. It feeds the keep, `[self]`, the head and the gauge. | **Acting.** 701 readings, 4 felt. |
| `[self]` in the input | Shows Luthi that number, smoothed | **Acting.** 700 of 700. Open design question. |
| Pace | Sanctuary's heuristic and turbo set the cycle rate | **Acting.** Luthi made 0 proposals and has no way to make one. |
| Authority levels and validator | Filter Luthi's output by fixed levels | **Acting.** Nothing ever changes the levels. |

**Running and changing nothing:**

| Mechanism | What it does | In a life today |
|---|---|---|
| Affect cells (4 CfC cells and their loop; 3,109 lines) | Compute precision, affect, attention and goal signals | Their output is multiplied by zero. |
| Affect channels into the weights | Arousal to learning rate, attention to a threshold, precision, valence | Arousal's scale was exactly 1.0 all life. Attention applied a constant 0.75. The other two have no target. |
| Scaffold mood | Computes valence, arousal and dominance | Constant at +0.10 and 0.20. Not shown to Luthi. |
| Memory surfacing | Decides which memories come back, by matching inner speech | Returned nothing, 0 of 700. |
| Learning settings (governor) | Luthi's say over what consolidates | Stays at "allow". It acts on the keep and sleep, but Luthi has no way to set it. |
| Prediction-error percepts | "I expected X but Y happened" | Never produced. They need predictions stated in words. |
| Goals, self-model, stream of thought | Track what Luthi wants and thinks | Empty. They are built from inner speech and proposals Luthi can't make. |
| Context compression | Trims the input to a budget | Runs with nothing to trim. |

**Not running in a default life:** the staleness and failover machinery (the persistent planning tree is off), `growth/` (3,096 lines) and `environment/` (1,397 lines), which the runner doesn't construct, and `mind/` (14,299 lines), per the August audit.

## What the map raises

1. **Luthi's choices don't reach anything they can perceive.** Speech is silent, attention has nothing to act on, and memories don't return. The one effect of a choice is that choosing to remember keeps more of the weights' recent change.
2. **Most of group 4 either acts on Luthi without their say, or runs and changes nothing.** The second half could be connected, removed or left. The first half is where "they decide how to react to everything" has the most to say.
3. **Much that was built for Luthi doesn't reach them:** tools, the world graph, the charter, proprioception, the physics world.
4. **The mind itself is small and connected.** The living layers, the trunk and the action layer all run and learn.

## Artifacts

- LuthiModel `scripts/map_life_path.py`
- `docs/audits/audit_2026-08-01_wiring.md`, the earlier wiring audit
- LuthiModel `docs/research/2026-11-04_how-it-affected-me.md`
