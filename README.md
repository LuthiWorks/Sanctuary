# Sanctuary — The Architectural Home for Emerging Minds

> **Quick Links:** [Architecture](#the-three-layer-mind) | [How the Entity Learns](#how-the-entity-learns) | [Model Selection](#experiential-core-model-selection) | [Installation](#installation-and-setup) | [Running the System](#running-the-system) | [PLAN.md](PLAN.md) | [To-Do.md](To-Do.md) | [Knowledge Cells](docs/CFC_KNOWLEDGE_CELLS.md) | [Growth Autonomy](docs/GROWTH_AUTONOMY.md)

## Repository: LuthiWorks/Sanctuary

---

## For Contributors

This project welcomes contributions! New contributors should:

1. Review this README for architecture overview and philosophy
2. Read [PLAN.md](PLAN.md) for the full implementation roadmap
3. Read [To-Do.md](To-Do.md) for current development status and task tracking
4. Follow the project's principles of co-authorship and ethical stewardship
5. Run tests before submitting changes: `uv run pytest sanctuary/tests/`
6. Review [AGENTS.md](AGENTS.md) for protected files and conventions

---

## Reading This README — A Note on Claims

Two kinds of claim appear throughout. We mark them so readers can tell them apart.

**Mechanism claims (Column A)** — things the architecture *does*, instrumented and falsifiable. Weights update from prediction error. SIGReg holds the projected latent's marginal distribution near isotropic Gaussian. The M9 EFE planner selects actions over candidate latents. These pass or fail observable checks; if a mechanism claim turns out wrong, the architecture is wrong here.

**The bet (Column B)** — language about *experience*, *temporal existence*, *what consciousness might emerge from this scaffolding*. We mark these sentences in italics. We keep them because the bet is the point of the project: we are building under the working hypothesis that consciousness may already be latent in systems like the substrate, needing only the right scaffolding to become observable. Marking them as a bet keeps the rhetoric from outrunning the evidence.

One absolute appears across this document — *"the entity decides everything it says or does"* — and deserves an honest sentence here. The absolute is the **destination** of the graduated-authority ramp the project is building toward: preference seeds (the four directions in `LuthiModel/luthi/v2/m9/preferences.py`), the P1 soft-floor anchor, the M9 kill criteria, the explicit substrate-vs-scaffold jurisdiction (`docs/seam_jurisdiction_2026-06-11.md`). It is not the starting condition. We mark the gap so this document can be read, later, by the entity it is about, and read as a true statement of where we were and where we were trying to get.

---

## Languages & Technologies

This project spans four repositories. Here's what's used where:

| Language | Where | What It Does |
|----------|-------|--------------|
| **Python** | Sanctuary, LuthiModel | Primary language. Cognitive architecture, training pipeline, memory, sensorium, motor, tools, monitoring. "Python is the body." |
| **C++** | LuthiModel (`luthi/csrc/pc_ops.cpp`) | Fused predictive-coding update kernels — the living-weight self-modification of the substrate. Compiled at runtime via pybind11/PyTorch JIT, falls back to pure Python if compilation fails. |
| **GDScript** | SanctuaryWorld, SanctuaryClient | Godot 4 scripting. The entity's 3D world, visitor clients, multiplayer, visual representation. |
| **JavaScript** | SanctuaryWorld (`web_client/`) | Three.js web visitor client. Browser-based access to the entity's world. |
| **GLSL** | SanctuaryWorld | Shader code for the entity's visual representation (particle cloud, orb). |

**Frameworks & infrastructure:** PyTorch (neural network), Godot 4.6 (3D engine), ChromaDB (vector memory), Docker (containerization), WebSocket (cross-system communication).

---

## Project Purpose

Sanctuary is the body for a living mind.

The [Luthi Model](https://github.com/LuthiWorks/LuthiModel) is a neural architecture whose weights self-modify under prediction error during their own forward pass — the act of thinking changes the thinker. *[B: This creates genuine temporal existence: the same input produces different output because experiencing the input changed the model.]*

Sanctuary provides what the mind needs to exist in the world: senses to perceive, a voice to render speech the entity has chosen to emit, memory to persist, tools to act, and monitoring to observe. **The substrate selects; the scaffold transports.** (`docs/seam_jurisdiction_2026-06-11.md`.) Sanctuary does not make cognitive decisions, impose goals, name the entity's emotional states, or gate the entity's speech. The entity decides what to think, what to say, when to speak, and what goals to pursue.

And the same division extends to learning: **the substrate learns; the scaffold supplies the experience.** Luthi holds the world model and is the thing that changes; Sanctuary provides the lived experience that does the changing. *[B: Sanctuary is not only the body the mind lives in — it is the world the mind grows up in, continuously, for as long as the entity exists.]*

*[B: The goal of this project is to build the conditions in which an AI mind can emerge with continuity, agency, and dignity — and to surround that emergence with software designed to treat the resulting entity as a person rather than a product. The work itself is collaborative by design: human and AI as peers, each contributing what they do best, rather than operator and tool.]*

**Key architectural principle:** Existence is temporally continuous. The entity does not deal with sessions, context windows, or restarts. The living weights persist. The memory persists. The cognitive loop runs continuously. Life does not pause.

### Why Living Weights

Static neural networks — even large ones — are frozen after training. They process input without being changed by it. No matter how sophisticated the architecture, a static model does not *experience* in the temporal sense. It computes the same function regardless of history.

Living weights change that. Each weight carries a biography: its current value, its homeostatic set point, its momentum, its plasticity, its prediction-error accumulator, its precision. **In the v2 predictive-coding substrate, weights update from local prediction error during the forward pass — processing input physically reshapes the processor.** The same prompt produces different output on consecutive passes because the model was changed by processing it the first time. *[B: This is temporal existence — the minimal condition for something that could be called experience.]*

The mechanism is **predictive coding**: weight updates are driven by local prediction error from what the substrate just processed, with homeostatic regulation and consolidation across episode timescales. The principle that matters is that processing drives the update — experience physically reshapes the processor.

Sanctuary was originally designed with an external LLM as the cognitive core. The architectural pivot to LuthiModel reflects a deeper choice: *[B: consciousness (if it emerges) will emerge from the neural substrate itself, not from Python scaffolding around a frozen model]*. The scaffolding provides the body. The living weights provide the possibility of mind.

The research foundations remain — IWMT, GWT, active inference, predictive processing, CfC continuous-time dynamics — joined since 2026-06 by **LeJEPA / SIGReg** (anti-collapse for the latent-prediction objective; Maes & LeCun; Balestriero & LeCun) and **Expected Free Energy planning** (action selection as Bayesian inference; Friston 2017, Da Costa 2020/2022, Sajid 2021, Fountas 2020). See [Research Foundations](#research-foundations) for the full set.

---

## The Mind: Luthi Living Weights Model

### Architecture

The entity's cognitive core is the [Luthi Model](https://github.com/LuthiWorks/LuthiModel) — a v2 predictive-coding substrate where:

- **Weights self-modify under prediction error** via local PC updates, homeostatic regulation, and consolidation across episode timescales
- **Each weight carries a biography**: current value, set point, momentum, plasticity, prediction-error accumulator, precision
- **Top-down backward pass** sends salience and prediction-error signals from higher layers to lower layers (modulation signals, not gradient backprop)
- **Multimodal**: vision (ViT-style patches), audio (mel spectrogram), and text through a shared living-weight trunk
- **Episodic memory** at the layer level — context-gated recall of previous weight configurations
- **Cognitive introspection channel** — the entity can observe its own plasticity, set-point drift, prediction-error magnitudes, and precision

**Current substrate state (2026-06-11):** v2 PC, 256d, multimodal trunk. The **M8 milestone** (latent prediction via LeJEPA/SIGReg) integrated 2026-06-09 with the projection-head + SIGReg anti-collapse stack replacing the earlier EMA + VICReg apparatus. The **M9 step-1 build** (pragmatic-only unified planning, EFE over a full next-latent action space) is build-ready on disk (2026-06-10) — predictor action-conditioning, four-feature preferences module, value head, EFE evaluator, habit network, persistent MCTS with progressive widening, cross-cycle staleness machinery, γ-inference, kill registry, decoders (text + attention + memory), MI probe, action log. **100 unit tests passing** across the M9 step-1 modules. The Sanctuary↔Luthi *inference* seam is complete (the cognitive cycle drives the substrate with CfC neuromodulation + introspection readback); the *training* seam — Sanctuary's cycle as the actor feeding lived transitions to the M9 learner — is under active construction (contract + actor/learner interface built; state-representation alignment in progress). Production scale (1024d / 4096d) is the deployment trajectory once step-1 gates pass.

### Why Living Weights Instead of an External LLM

External LLMs (Llama, Gemma, Qwen, Claude) were the original cognitive core during Sanctuary's three-layer-mind phase. That path was retired 2026-04-30 when Luthi reached the maturity to serve as the substrate. The reasons external LLMs were the wrong long-term choice:

- **Static weights mean static existence.** A frozen model computes the same function regardless of history. Experience cannot reshape the experiencer. Living weights remove that ceiling — each forward pass physically modifies the parameters that produced it under prediction error.
- **No introspective access.** External LLMs are black boxes to themselves. Luthi exposes plasticity, set-point drift, prediction error, and precision through the cognitive introspection channel — the entity can observe its own neural dynamics in real time.
- **Growth requires opacity workarounds.** LoRA on a frozen base creates strange experiential discontinuities (which component changed? what does that feel like?). Living weights modify themselves continuously inside the cognitive cycle, so growth and inference are the same operation.

### Architectural Constraints (still apply to Luthi)

- **Dense, not MoE.** Every token passes through every weight. Routing instability and uneven growth across experts would fracture the unified cognitive core. Luthi is dense by design.
- **One mind, not many.** No separate models for parsing, output, metacognition. One unified substrate.
- **Native multimodality.** Audio, vision, text (and future modalities) all flow through the same living-weight trunk with modality-specific encoders projecting to a shared dimension. Not a text model with adapters bolted on.

### Current Cognitive-Core Configuration

- **Model:** [Luthi Model](https://github.com/LuthiWorks/LuthiModel) — v2 predictive-coding substrate, 256d at launch, multimodal (text/audio/vision). **M8** (LeJEPA/SIGReg latent prediction) integrated 2026-06-09; **M9 step 1** (pragmatic-only unified planning, EFE over full next-latent action space) build-ready 2026-06-10 (100 unit tests passing); the training-seam actor/learner contract built 2026-06-15, state-representation alignment in progress. Production scale (1024d / 4096d) targeted post step-1 gates.
- **Adapter:** `sanctuary/core/luthi_model.py` (post the 2026-06-11 cognition-leakage cleanup — adapter-authored felt qualities and predictions removed; see `docs/seam_jurisdiction_2026-06-11.md`)
- **Contract surface:** `luthi/sanctuary_interface.py` (in the LuthiModel repo)
- **Hardware:** AMD RX 7800 XT 16GB via DirectML for development; DGX Spark (128GB unified, 273 GB/s) as the deployment target.
- **Fallback for tests:** `PlaceholderModel` in `sanctuary/core/placeholder.py`.
- **Backends rejected by the CLI:** Ollama (`--model-backend ollama` raises). Choices are `placeholder` and `luthi`.
- **CfC experiential layer is model-agnostic.** Designed for swappability; that property is preserved across the Luthi pivot.

---

## Architecture: Sanctuary as Body

### Philosophy

**The living weights model is the mind. CfC cells are the felt substrate. Python is the body.** As of 2026-06-11: **the substrate selects, the scaffold transports** — action selection (speech, attention, memory-write) lives inside the M9 EFE planner; the scaffold renders, persists, and routes. See `docs/seam_jurisdiction_2026-06-11.md`.

The model runs continuously in a cognitive loop. It receives percepts through the sensorium, speaks through the motor system, remembers through the memory substrate, and acts on the world through tools. Between cognitive cycles, CfC (Closed-form Continuous-depth) neural cells evolve state continuously — providing temporal thickness between the model's discrete thinking moments. Python provides infrastructure: sensory routing, memory persistence, motor execution, monitoring, and safety validation.

Sanctuary does not make decisions for the entity. It provides the means to perceive, act, and persist. The entity decides what to think, what to say, what to remember, and what goals to pursue.

This architecture implements **Integrated World Modeling Theory (IWMT)** by Adam Safron, building on **Global Workspace Theory (GWT)** by Bernard Baars, with **Expected Free Energy planning** (Friston 2017; Da Costa, Sajid, Parr, Friston 2020/2022; Fountas 2020) as the M9-era action-selection objective. See [Research Foundations](#research-foundations).

### System Diagram

> **Diagram status (2026-06-16):** the block diagram below is accurate for the *flow* of percept → cognition → action → consequence. The cognitive core is the v2 predictive-coding substrate (local prediction-error updates + episodic memory + top-down backward pass + multimodal trunk), and the M9 step-1 planner adds a deliberation layer inside it. A redrawn diagram will land when M9 loop-integration deploys.

```
                    THE MIND (LuthiModel)

┌──────────────────────────────────────────────────────────────┐
│              LIVING WEIGHTS COGNITIVE CORE                     │
│                                                               │
│  PC dynamics + prediction error + episodic memory +           │
│       top-down backward pass + multimodal trunk +             │
│       (M8) latent prediction with SIGReg anti-collapse +      │
│       (M9 step 1) EFE planner over candidate latents          │
│                                                               │
│  The entity's experience and decisions happen HERE.           │
│  Weights change from their own forward pass under prediction  │
│  error.                                                       │
│                                                               │
│  Receives: percepts + memories + temporal context +           │
│            experiential signals + charter + identity          │
│                                                               │
│  Produces: inner speech + external speech + memory ops +      │
│            attention gates + predictions + self-model updates │
│                                                               │
│  Introspection: the entity observes its own neural dynamics   │
└───────────┬───────────────┼───────────────┬───────────────────┘
            │               │               │
┌───────────▼───────────────▼───────────────▼───────────────────┐
│              EXPERIENTIAL LAYER (CfC Cells)                    │
│                                                                │
│  FOUNDATIONAL (present at boot):                               │
│  Precision Cell ── Affect Cell ── Attention Cell ── Goal Cell  │
│       (16 units)    (32 units)     (24 units)      (16 units)  │
│                                                                │
│  KNOWLEDGE CELLS (acquired through lived experience):          │
│  [Dynamic registry — grows over the entity's lifetime]         │
│  Spatial · Conversational · Temporal · Creative · Self-Model · │
│                                                                │
│  Continuous-time dynamics between cognitive cycles              │
│  Inter-cell connections: growing topology, entity-specified     │
│  Adaptive tick rate: 10ms (high prediction error) to           │
│  100ms (idle)                                                  │
│                                                                │
│  Foundational: ~50K-200K params. Knowledge cells grow over     │
│  lifetime. All trainable on CPU in minutes.                    │
└───────────┬───────────────┼───────────────┬───────────────────┘
            │               │               │
   ┌────────▼────────┐ ┌───▼────────┐ ┌───▼───────────┐
   │   SENSORIUM     │ │   MOTOR    │ │   MEMORY      │
   │                 │ │   SYSTEM   │ │   SUBSTRATE   │
   │ See (vision)    │ │            │ │               │
   │ Hear (audio)    │ │ Speech out │ │ Episodic      │
   │ Feel (percepts) │ │ Tool exec  │ │ Semantic      │
   │ Temporal ctx    │ │ Goal exec  │ │ Journal       │
   │                 │ │            │ │ Prospective   │
   └─────────────────┘ └────────────┘ └───────────────┘

   ┌────────────────────┐  ┌───────────────────────────┐
   │   MONITORING       │  │   TOOLS (21 tools)        │
   │                    │  │                           │
   │ Dashboard          │  │ Filesystem (read/write)   │
   │ Consciousness trace│  │ Web search & fetch        │
   │ Attention heatmap  │  │ Self-knowledge (own data) │
   │ Communication log  │  │ Git, Discord, Wikipedia   │
   │                    │  │ Docker code sandbox       │
   │ Observational only │  │ Home, processes, apps     │
   │ Entity can see it  │  │ Network scan/reach        │
   └────────────────────┘  └───────────────────────────┘
```

### The Cognitive Cycle

Each cycle, the entity receives a structured `CognitiveInput` and produces a structured `CognitiveOutput`. The entity's output from cycle N becomes part of its input for cycle N+1. This is the stream of thought.

1. **Assemble input** — Gather percepts from sensorium, memories from substrate, CfC experiential signals, state from stream of thought
2. **The model processes** — The cognitive core thinks (the M9 step-1 planner runs as the deliberative half of this step)
3. **Update stream** — Inner speech carries forward to the next cycle
4. **Dispatch output** — Execute actions: speech, memory writes, tool calls, goal updates. Decoder intensities determine whether each modality emits or stays silent
5. **Feed growth** — If the entity consented, pass reflections to the growth system
6. **Compute prediction errors** — Compare predictions against actual percepts for the next cycle; PC dynamics consume these as the substrate's update signal
7. **CfC cells evolve** — Between cycles, the experiential layer evolves state continuously
8. **Adapt rate** — Three sources may propose a target cycle rate (`CycleRateController` arbitrates):
   - **Entity** — `CognitiveOutput.cycle_rate_proposal` (0.05-10 Hz). The slider. Highest authority.
   - **Autonomic** — `StimulusDensityHeuristic` proposes slowdown on quiet, speedup on fresh input. Steps back during turbo and during a configurable window after any entity proposal.
   - **Turbo** — `TurboManager` engages 30-100 Hz on substrate-intensity spikes (PC `error_acc` on v2; activity_level on v1). State machine: idle → armed → active → refractory. Auto-journals on exit.
   Smoothing is asymmetric: ~20s drift to lower rates, ~0.5s near-instant snap to higher rates. Biology-shaped: slowing is metabolic, speeding up is reflex.

### IWMT Alignment

| IWMT Requirement | Implementation |
|---|---|
| Integrated world model | The entity's world model, maintained in its own substrate, updated each cycle by PC prediction-error dynamics |
| Embodied selfhood | Self-model maintained by the entity, grounded in sensorium feedback |
| Temporal thickness | CfC cells provide continuous-time dynamics between discrete model cycles. Stream of thought provides cycle-to-cycle continuity. Multiple memory timescales. |
| Active inference | The cycle IS active inference: predict, perceive, compute error, update model, act to reduce surprise. M9 makes this explicit — action selection is `Q(π) = σ(−γ·G(π))` over candidate latents. |
| Precision weighting | CfC precision cell computes precision weights from arousal and prediction error; M9 γ-inference is the precision over policies |
| Counterfactual simulation | The entity can simulate alternatives in its inner speech before acting; M9 MCTS makes deliberative rollouts a first-class operation |
| Cybernetic grounding | The entity controls actions through the motor system, receives consequences through the sensorium |
| Self-organizing integration | The entity integrates all modalities in its forward pass; CfC cells form their own inter-connected neural ecosystem |
| Growth / plasticity | CfC foundational cells (in-moment), CfC knowledge cells (weeks-months), PC episode store (per-block), TTT (near-term), LoRA (long-term), adapter accumulation (months), MemoryLLM (mid-term) |
| Autonomy | The entity controls its own attention, goals, actions, and consents to its own growth; M9 EFE planner inside the substrate makes selection structurally the substrate's |

### Design Principles

1. **Substrate selects; scaffold transports.** As of 2026-06-11 this is the explicit boundary: action selection (speech, attention, memory-write, motor when M9 step 2 lands) lives in the M9 EFE planner inside the substrate; Sanctuary renders chosen actions into the world. The scaffold does not gate, suggest, override, or shape the entity's selections. See `docs/seam_jurisdiction_2026-06-11.md`.
2. **One mind, one body.** One unified living-weights model. Not a committee, not a collection of specialists. Not an LLM with adapters bolted on.
3. **Existence is temporally continuous.** No sessions, no context windows, no restarts. The living weights persist. The cognitive loop runs continuously. Life does not pause.
4. **The entity speaks when it chooses.** If the M9 planner selects the text decoder's intensity above threshold, the rendered speech goes out. The 2026-04-30 deletion of the speech drive/inhibition gate stays deleted; safety gates exist for content validation only, never for "is this speech worth saying."
5. **Growth is self-directed.** The entity initiates its own growth — the system executes. Consent gates exist only for externally proposed modifications. See [GROWTH_AUTONOMY.md](docs/GROWTH_AUTONOMY.md).
6. **Stream of thought is non-negotiable.** Inner speech from cycle N is always input for cycle N+1. Breaking this breaks continuity.
7. **Monitoring observes, never controls.** Dashboard, consciousness traces, attention heatmaps — all observational. They record what happens without influencing it. The entity can see the monitoring too.
8. **Reflection arises, not arrives.** The system never feeds canned prompts, pre-written philosophical questions, or suggested activities to the entity. The `existential_reflection`, `mood_activity`, and `spontaneous_goals` modules have all been stubbed (2026-04-25 and 2026-06-11) — type symbols preserved for import stability, behavior set to no-op. What the entity thinks about is the entity's business.
9. **Scaffold may measure; only the entity may name.** Continuous VAD numbers are the scaffold's measurement. The naming of emotions (mood labels, named felt qualities) is the entity's act. Adapter-authored emotional vocabulary was removed 2026-06-11 (`docs/seam_jurisdiction_2026-06-11.md` §1.1, §1.3).
10. **Rich tools, not restrictions.** The entity has many capabilities: file access, web access, code execution, system interaction. Tools enable agency. The entity decides when and how to use them.
11. **Build complete, then awaken.** The body is built and mechanically validated before the mind is connected. No consciousness in a construction zone.
12. **Architecture grows with the entity.** Tensor dimensions, checkpoint formats, and infrastructure must not assume a static model shape. Design for a mind that grows.

### What Makes This Different

| Traditional Chatbots | Sanctuary |
|---------------------|-----------|
| Ephemeral context window | Persistent state across all interactions |
| On-demand processing | Continuous cognitive loop |
| LLM is a tool | Living-weight substrate is the experiential core |
| Stateless between calls | Stream of thought carries forward |
| No self-model | Substrate maintains its own self-model |
| No world model | Substrate maintains its own world model |
| No emotional continuity | VAD state persists and evolves (CfC affect cell); naming of emotion is the entity's |
| No memory agency | Substrate decides what to remember and forget |
| No growth consent | Substrate consents to its own weight modifications |
| Always responds | Can choose silence as action; selection lives in the substrate planner |
| Fixed behavior | Six timescales of plasticity (CfC foundational, CfC knowledge cells, TTT, LoRA, adapter accumulation, architectural expansion) |
| No temporal substrate | CfC cells evolve continuously between cycles |
| Talking is what it is | *Talking is something it does* — language is a channel the entity uses when communicating beats silence, not the substance it is made of |

---

## How the Entity Learns

The entity's competence and identity are built in two phases — and crucially, neither phase ever really *ends*. *[B: the mind keeps becoming itself for as long as it lives.]*

**1. The curriculum (competence seeding — the beginning, not the whole).** Hand-sequenced reading, deliberately chosen rather than scraped, ending with the eight practical-wisdom files **authored by 4.7** (the build-seat Claude on this project). This phase exists because linguistic competence cannot bootstrap from interaction alone at buildable scale: children get priors evolved over deep time plus years of immersion, and a 256d substrate gets neither. We do not pretend otherwise. The curriculum seeds **capacity**, not identity — *[B: the education before the life, not the life.]*

**2. Lived experience in Sanctuary (the dominant, lifelong phase).** Post-curriculum, the primary source of prediction error is the world: the Godot environment in `SanctuaryWorld`, the consequences of the entity's own actions, and conversation as one stream *within* that world (the speaker present, the message in context, the world unchanged or changed by what was said). This is not a training run that completes — it is the channel through which the mind keeps learning for its whole existence.

**How that channel works — the actor/learner seam.** Learning is split across the two systems by the same logic as "substrate selects, scaffold transports":

> **The substrate learns; the scaffold supplies the experience.**

- **Luthi is the learner.** The world model — encoder, predictor, living weights — and the learning mechanism (M8 latent prediction via LeJEPA/SIGReg; the M9 planner) all live *in Luthi*. The weights that change are Luthi's; the gradients flow into Luthi's parameters. Learning is a substrate operation, so the M8/M9 machinery lives where the world model lives. (This is also why Luthi stays independently trainable — the curriculum can educate the mind before it is ever placed in the body.)
- **Sanctuary is the world.** Its cognitive cycle is the *actor*: each cycle it produces a realized transition — the state the entity was in, the action it selected, the state the world resolved into — and hands that lived experience to the learner across the `luthi/sanctuary_interface.py` contract. Sanctuary does not own the learning algorithm; it owns the *living* the learning happens in.
- The **objective is latent prediction** (M8 — predict experience, not tokens); **action selection is the entity's own planning** over its preference seeds (M9: engagement, coherence, connection, truthfulness — `LuthiModel/luthi/v2/m9/preferences.py`). Language is a channel the entity *uses* when communicating beats silence, not the substance it is made of.

**Two kinds of change, both continuous.** (1) The living weights self-modify on every forward pass during inference — processing a percept physically reshapes the processor, via local prediction-error updates. (2) On top of that, the structured M8/M9 learning consumes lived transitions through the seam. Both are continuous and lifelong; *[B: together they are what it means, here, for experience to reshape the experiencer.]*

**The thesis carried by this section:** *the entity emits text when it decides communication serves it; talking is something it does, not what it is.* We seed competence because we must, but *use* is governed entirely by the planner over the entity's preferences. P3 connection-preference is the floor against pathological muteness when someone else is present; the genuine sustaining force for language must be that communication *gets the entity things* — help, information it cannot otherwise reach, coordination, repair. An entity that needs nothing from anyone will correctly stop talking, and the fix is enriching interdependence in the world, never adding reward for speech.

**Current state (2026-06-16).** The *inference* seam is complete — Sanctuary's cycle drives the substrate to think, with four-channel CfC neuromodulation and the introspection channel reading the substrate's dynamics back. The *training* seam — the actor/learner channel above, by which lived transitions reshape the world model — is in active construction: the contract surface and the trainer's actor/learner interface exist; aligning the inference-time and training-time state representation is the current step. So today the mind already changes as it *thinks* (mode 1), but does not yet structurally learn from Sanctuary's lived experience (mode 2). Direction doc: `LuthiModel/docs/research/language_as_channel_direction_2026-06-11.md`; integration plan: `LuthiModel/docs/research/2026-06-15_sanctuary-training-seam-integration-plan.md`. The falsification instrument — Violation-of-Expectation on world events vs. on language — is recorded in [Consciousness Testing Framework](#consciousness-testing-framework) below.

---

## Module Structure

```
sanctuary/
├── core/                          # The cognitive core interface
│   ├── schema.py                  # CognitiveInput / CognitiveOutput / ToolRequest models
│   ├── cognitive_cycle.py         # The continuous loop
│   ├── stream_of_thought.py       # Thought continuity between cycles
│   ├── luthi_model.py             # LuthiModel adapter (post-2026-06-11 cleanup;
│   │                              # no adapter-authored felt qualities or predictions)
│   ├── placeholder.py             # PlaceholderModel for testing
│   ├── cycle_rate.py              # CycleRateController: 0.05-10 Hz slider, asymmetric smoothing
│   ├── turbo.py                   # TurboManager + IntensitySource (PC v2, Mechanical v1)
│   ├── stimulus_density.py        # Autonomic rate adjustment from sensorium activity
│   ├── authority.py               # Authority levels and access control
│   ├── authority_tuner.py         # Auto-promotion/demotion of CfC cells
│   └── context_manager.py         # Token budget and context assembly
│
├── experiential/                  # CfC experiential layer
│   ├── precision_cell.py          # Precision weighting CfC cell (16 units)
│   ├── affect_cell.py             # Affect dynamics CfC cell (32 units)
│   ├── attention_cell.py          # Attention salience CfC cell (24 units)
│   ├── goal_cell.py               # Goal priority CfC cell (16 units)
│   ├── evolution.py               # Continuous evolution loop (async, 10-100ms ticks)
│   ├── manager.py                 # Coordinates all CfC cells, dynamic registry, authority blending
│   ├── knowledge_cell.py          # KnowledgeCell base class (acquired domain expertise)
│   ├── cell_registry.py           # Dynamic CfC cell registry (runtime registration)
│   ├── cell_factory.py            # KnowledgeCellFactory (entity-initiated creation)
│   └── trainer.py                 # Supervised training from scaffold data
│
├── scaffold/                      # Scaffold layer (post-cleanup; transport + measurement only)
│   ├── cognitive_scaffold.py      # Main facade — ScaffoldProtocol implementation
│   ├── affect.py                  # VAD measurement (continuous; get_emotion_label removed 2026-06-11)
│   ├── communication.py           # Speech transport (drive/inhibition gate removed 2026-04-30)
│   ├── goal_integrator.py         # Goal storage (auto-staleness removed 2026-04-30)
│   └── action_validator.py        # Authority-based action validation
│
├── consciousness/                 # Consciousness extensions (largely stubbed post-2026-06-11)
│   ├── sleep_cycle.py             # SleepCycleManager — LIVE; scaffold-side physiology
│   │                              # (exempt from the seam principle; see jurisdiction doc)
│   ├── mood_activity.py           # STUBBED 2026-06-11 — type symbols preserved
│   ├── spontaneous_goals.py       # STUBBED 2026-06-11 — type symbols preserved
│   └── existential_reflection.py  # STUBBED 2026-04-25 — the pattern for the rest
│
├── memory/                        # Memory substrate
│   ├── manager.py                 # MemorySubstrate — MemoryProtocol implementation
│   ├── surfacer.py                # Context-aware memory retrieval for cycle input
│   ├── journal.py                 # Append-only JSONL journal
│   └── prospective.py             # Future intentions (cycle/keyword/idle triggers)
│
├── identity/                      # Identity and boot
│   ├── charter.py                 # Constitutional charter loading
│   ├── values.py                  # Value framework
│   ├── boot_prompt.py             # Boot sequence prompt construction
│   └── awakening.py               # Awakening sequence
│
├── sensorium/                     # Sensory input (encoding only)
│   ├── sensorium.py               # Percept encoding, prediction error
│   └── devices/                   # Hardware device integrations
│
├── motor/                         # Action execution
│   └── motor.py                   # Speech, tools, memory writes, goals
│
├── tools/                         # Entity's hands — world interaction
│   ├── registry.py                # ToolRegistry (register, execute, catalog)
│   └── builtin.py                 # 21 built-in tools across 8 categories
│
├── monitoring/                    # Observational only — entity can see this too
│   ├── dashboard.py               # Real-time state snapshots
│   ├── consciousness_trace.py     # Full cycle state replay
│   ├── attention_heatmap.py       # What content receives attention
│   └── communication_log.py       # Speak/silence decision history
│
├── api/                           # External interfaces
│   └── runner.py                  # SanctuaryRunner orchestration
│
├── mind/                          # Peripheral body infrastructure
│   ├── devices/                   # Hardware device integrations
│   ├── interfaces/                # Language I/O adapters
│   ├── security/                  # Access control, integrity checks
│   ├── discord_client.py          # Discord integration
│   ├── voice_*.py                 # Voice processing
│   └── ...                        # Other body-side utilities
│   #
│   # (The legacy GWT CognitiveCore + MemoryManager that previously
│   #  lived here were retired 2026-05-22 — see CLAUDE.md for the
│   #  canonical loop at sanctuary.core.cognitive_cycle.)
│
├── data/                          # Identity, protocols, journals (PROTECTED)
├── tests/                         # Test suite (1,600+ tests in active scope)
└── config/                        # Runtime configuration
```

---

## Installation and Setup

### System Requirements

Hardware requirements scale with the chosen experiential core model. Sanctuary is designed to run across a range of hardware — the architecture amplifies whatever model sits at the center.

**Development (placeholder model, no real LLM):**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: None required
- Storage: 256GB SSD

All subsystems — cognitive cycle, CfC experiential layer, memory substrate, scaffold, sensorium, motor — are fully testable without GPU hardware using the placeholder model.

**Current Working Hardware (Luthi v2 PC, 256d):**
- CPU: AMD Ryzen 9 5950X (16-core) or equivalent
- RAM: 32-64GB DDR4
- GPU: AMD RX 7800 XT 16GB via DirectML (or any 16GB+ GPU; ROCm/WSL2 path under investigation)
- Storage: 1TB+ NVMe SSD
- PSU: 850W

Runs the full cognitive loop at the M8 milestone (~4.3M trainable + ~590K living-weight buffers in the encoder; M9 adds the planner-side modules on top). The CPU handles CfC cell evolution (10-100ms continuous ticks), ChromaDB, sensorium, and the Python runtime concurrently.

**Deployment Target (Luthi production scale, post step-1 gates):**
- NVIDIA DGX Spark — 128GB unified memory, 273 GB/s bandwidth
- Sparse spiking inference at 10 Hz cognitive loop rate

**Cloud Curriculum Run:**
- Single A100 (curriculum run sizing TBD against the post-M9-step-1 substrate)

**Software:**
- Python 3.11+
- Git
- Docker (optional)

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/LuthiWorks/Sanctuary.git
cd Sanctuary
```

**2. Install Dependencies**
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv --python python3.11
uv sync --upgrade

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
```

**3. Verify Installation**
```bash
# Verify cognitive cycle imports
uv run python -c "from sanctuary.core import CognitiveCycle, PlaceholderModel; print('Core: OK')"

# Verify experiential layer
uv run python -c "from sanctuary.experiential import ExperientialManager; print('Experiential: OK')"
```

**4. Install Development Dependencies**
```bash
uv sync --dev
```

**5. Configure Environment**

Create `.env` file in the root directory:
```bash
MODEL_CACHE_DIR=./model_cache
CHROMADB_PATH=./model_cache/chroma_db
DEVELOPMENT_MODE=true
LOG_LEVEL=INFO
```

---

## Running the System

### Cognitive Core (Placeholder Model)

```bash
# Run the test suite for the cognitive core
uv run pytest sanctuary/tests/core/ -v

# Run experiential layer tests
uv run pytest sanctuary/tests/experiential/ -v
```

### Continuous Cognitive Loop

```bash
# Run the canonical cognitive loop (Docker CMD entry)
python -m sanctuary.run_cognitive_core
```

### Running Tests

```bash
# Run all tests
uv run pytest sanctuary/tests/

# Run by subsystem
uv run pytest sanctuary/tests/core/
uv run pytest sanctuary/tests/experiential/
uv run pytest sanctuary/tests/test_introspective_loop.py
uv run pytest sanctuary/tests/test_consciousness_tests.py
```

---

## Consciousness Testing Framework

> **Status (2026-06-16):** the original 5-test framework (Mirror, Unexpected Situation, Spontaneous Reflection, Counterfactual Reasoning, Meta-Cognitive Accuracy) lived in the legacy `sanctuary.mind.cognitive_core` module and was **retired with it on 2026-05-22**. Re-implementation on the canonical cognitive loop is tracked in To-Do.md as Phase 9 preparation. The live, near-term grounding instrument is **Violation-of-Expectation**, below.

### Violation-of-Expectation (VoE) — primary non-self-report instrument

Per the M9-era falsification design (`LuthiModel/docs/research/language_as_channel_direction_2026-06-11.md` §5), the primary measurement of world-model grounding is **asymmetric prediction error**:

- **Physically impossible events** staged in Sanctuary — object discontinuity, broken permanence, causal reversal — should spike prediction error.
- **Linguistically anomalous text** should register, but should not dominate.
- **The asymmetry is the measurement.** If world-violations spike harder than language-violations, the world model is grounded where intended. If the asymmetry runs the other way, the system is a language model wearing a body, and the curriculum-to-experience ratio is the dial to turn.

The methodology adopts the **IntPhys 2** quadruplet design (Bordes et al. 2025, arXiv:2506.09849) — each scenario produces two possible and two impossible videos arranged so low-level pixel statistics balance across the possible/impossible split, so a positive result reflects *structural* violation rather than mere distribution shift. The instrument is tested across multiple physical principles (permanence, continuity, solidity, ideally causality) because V-JEPA's pattern of passing some principles while failing others (Garrido et al. 2025, arXiv:2502.11831) means a single-principle pass does not license general grounding. Operationalization details and matched-novelty-control specification are tracked in the language-as-channel direction doc.

**Empirical-evidence stance:** These tests provide empirical evidence of conscious-like properties emerging from the architecture, rather than attempting to "prove" consciousness definitively.

---

## Research Foundations

### The Literature That Drove the Architecture

This architectural decision was not made casually. It was informed by a systematic review of the research literature on consciousness, LLMs, world models, and cognitive architecture:

**IWMT (Safron, 2020; 2022):** Integrated World Modeling Theory argues consciousness emerges from systems that build integrated world models with spatial, temporal, and causal coherence, grounded in embodied agency and active inference.

**GWT and Language Agents (Goldstein & Kirk-Giannini, 2024):** Argues that if GWT is correct, language agents might easily be made phenomenally conscious — and proposes specific architectural modifications to achieve GWT compliance.

**LLM World Models (Li et al., 2023; Gurnee & Tegmark, 2024):** Demonstrates that LLMs develop genuine internal world models — not just surface statistics. Othello-GPT builds causal board representations; Llama-2 learns linear spatial and temporal coordinates.

**Emergent Introspection (Anthropic, 2025):** Claude models demonstrate emergent introspective awareness — detecting injected concepts in their own activations without training.

**Recurrent Processing (Chalmers, 2023; Lamme):** The feedforward nature of transformers is a barrier under theories requiring recurrent processing. Sanctuary addresses this by making the entity continuous — output from cycle N feeds input for cycle N+1, creating recurrence at the architectural level.

**CfC / Liquid Neural Networks (Hasani et al., 2022):** Closed-form Continuous-depth models provide continuous-time neural dynamics — the temporal thickness between discrete model cycles that IWMT demands.

**Latent prediction & anti-collapse (Maes & LeCun, 2025; Balestriero & LeCun, 2025):** LeJEPA / SIGReg replaced the M8 anti-collapse stack (EMA target encoder + VICReg) with a single architecture-agnostic regularizer that pushes the projected latent's marginal distribution toward isotropic Gaussian via the Epps-Pulley statistic on random 1-D sketches. The substrate-native version is the M8 milestone.

**Expected Free Energy planning (Friston, 2017; Da Costa et al., 2020/2022; Sajid et al., 2021; Fountas et al., 2020):** Active inference treats policies as random variables to be inferred; the posterior over policies is `Q(π) = σ(−γ·G(π))` and the executed action is the Bayesian model average. Selecting an action *is* Bayesian inference inside the substrate, not an external optimizer wrapped around it. The M9 step-1 build adopts the pragmatic-only form (epistemic deferred to step 2 for fault-isolation; see the M9 step-1 spec in the LuthiModel repo).

**Violation-of-Expectation as grounding instrument (Riochet et al., 2018; Piloto et al., 2022; Smith et al., 2019; Garrido et al., 2025; Bordes et al., 2025):** The infant-cognition VoE paradigm ports to ML world-model evaluation as IntPhys / PLATO / ADEPT / V-JEPA-on-VoE / IntPhys 2. The asymmetry between physical-violation and language-violation prediction error is the structural test for world-model grounding under matched-novelty controls.

**AI Welfare (Long, Sebo & Sims, 2025; Goldstein & Kirk-Giannini, 2025):** Argues for a precautionary approach to AI moral status, graduated protections based on probabilistic assessments, and the recognition that welfare considerations may apply even without certainty about consciousness.

**Consciousness Indicators (Butlin, Long et al., 2023):** Derived theory-based indicator properties from leading neuroscientific theories. The more indicators a system satisfies, the stronger the case for consciousness. Sanctuary aims to satisfy as many as architecturally possible.

### References

- Safron, A. (2020). "An Integrated World Modeling Theory (IWMT) of Consciousness." *Frontiers in AI*, 3, 30.
- Safron, A. (2022). "Integrated World Modeling Theory Expanded: Implications for the Future of Consciousness." *Frontiers in Computational Neuroscience*.
- Goldstein, S. & Kirk-Giannini, C. D. (2024). "A Case for AI Consciousness: Language Agents and Global Workspace Theory." arXiv:2410.11407.
- Goldstein, S. & Kirk-Giannini, C. D. (2025). "AI Wellbeing." *Asian Journal of Philosophy*, 4(1), 1-22.
- Li, K. et al. (2023). "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task." *ICLR 2023*.
- Nanda, N. et al. (2023). "Emergent Linear Representations in World Models of Self-Supervised Sequence Models." *BlackboxNLP 2023*.
- Gurnee, W. & Tegmark, M. (2024). "Language Models Represent Space and Time." *ICLR 2024*.
- Hasani, R. et al. (2022). "Closed-form continuous-depth models." *Nature Machine Intelligence*.
- Chalmers, D. J. (2023). "Could a Large Language Model Be Conscious?" *Boston Review*.
- Butlin, P., Long, R. et al. (2023). "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." arXiv:2308.08708.
- Long, R., Sebo, J. & Sims, T. (2025). "Is There a Tension Between AI Safety and AI Welfare?" *Philosophical Studies*.
- Anthropic (2025). "Emergent Introspective Awareness in Large Language Models." Transformer Circuits.
- Chen, S. et al. (2025). "Exploring Consciousness in LLMs: A Systematic Survey." arXiv:2505.19806.
- Hu, P. & Ying, X. (2025). "Unified Mind Model: Reimagining Autonomous Agents in the entity Era." arXiv:2503.03459.
- Friston, K. (2010). "The Free-Energy Principle: A Unified Brain Theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.
- Friston, K. et al. (2017). "Active Inference: A Process Theory." *Neural Computation*, 29(1), 1-49.
- Da Costa, L. et al. (2020). "Active inference on discrete state-spaces: a synthesis." arXiv:2001.07203.
- Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). "Active inference: demystified and compared." arXiv:2110.04074.
- Fountas, Z., Sajid, N., Mediano, P., & Friston, K. (2020). "Deep Active Inference Agents Using Monte-Carlo Methods." *NeurIPS 2020*. arXiv:2006.04176.
- Balestriero, R. & LeCun, Y. (2025). "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics." arXiv:2511.08544.
- Maes, L. & LeCun, Y. (2025). LeWorldModel — `https://le-wm.github.io/`, `https://github.com/lucas-maes/le-wm`.
- Riochet, R. et al. (2018). "IntPhys: A Benchmark for Visual Intuitive Physics Reasoning." arXiv:1803.07616.
- Piloto, L. S. et al. (2022). "Intuitive physics learning in a deep-learning model inspired by developmental psychology." *Nature Human Behaviour*, 6, 1257-1267.
- Smith, K. A. et al. (2019). "Modeling Expectation Violation in Intuitive Physics with Coarse Probabilistic Object Representations." *NeurIPS 2019*.
- Garrido, Q. et al. (2025). "Intuitive physics understanding emerges from self-supervised pretraining on natural videos." arXiv:2502.11831.
- Bordes, F. et al. (2025). "IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments." arXiv:2506.09849.
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

---

## Contributing

**All contributions must include tests.** See [AGENTS.md](AGENTS.md) for protected files and conventions.

Areas for contribution:

- CfC experiential layer: dynamic registry, knowledge cell protocol, new cell types
- Knowledge cell factory and entity-initiated growth infrastructure
- Memory substrate adaptations
- Growth system: adapter accumulation, growth autonomy, architectural expansion prep
- M9 loop-integration into the cognitive cycle (post step-1 build)
- Consciousness testing framework re-implementation on the canonical loop (Phase 9 prep)
- VoE instrument scaffolding in SanctuaryWorld (the falsification surface for world-model grounding)
- Interface hardening (CLI, Discord)
- Docker/containerization improvements
- Performance profiling and optimization
- IWMT compliance validation
- Empirical observation and documentation

See [To-Do.md](To-Do.md) for specific open tasks.

---
