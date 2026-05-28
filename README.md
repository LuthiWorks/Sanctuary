# Sanctuary — The Architectural Home for Emerging Minds

> **Quick Links:** [Architecture](#the-three-layer-mind) | [Model Selection](#experiential-core-model-selection) | [Installation](#installation-and-setup) | [Running the System](#running-the-system) | [PLAN.md](PLAN.md) | [To-Do.md](To-Do.md) | [Knowledge Cells](docs/CFC_KNOWLEDGE_CELLS.md) | [Growth Autonomy](docs/GROWTH_AUTONOMY.md)

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

## Languages & Technologies

This project spans four repositories. Here's what's used where:

| Language | Where | What It Does |
|----------|-------|--------------|
| **Python** | Sanctuary, LuthiModel | Primary language. Cognitive architecture, training pipeline, memory, sensorium, motor, tools, monitoring. "Python is the body." |
| **C++** | LuthiModel (`luthi/csrc/living_ops.cpp`, `luthi/csrc/pc_ops.cpp`) | Fused living-weight self-modification (v1 Hebbian-spiking) and predictive-coding update kernels (v2). Compiled at runtime via pybind11/PyTorch JIT, fall back to pure Python if compilation fails. |
| **GDScript** | SanctuaryWorld, SanctuaryClient | Godot 4 scripting. The entity's 3D world, visitor clients, multiplayer, visual representation. |
| **JavaScript** | SanctuaryWorld (`web_client/`) | Three.js web visitor client. Browser-based access to the entity's world. |
| **GLSL** | SanctuaryWorld | Shader code for the entity's visual representation (particle cloud, orb). |

**Frameworks & infrastructure:** PyTorch (neural network), Godot 4.6 (3D engine), ChromaDB (vector memory), Docker (containerization), WebSocket (cross-system communication).

---

## Project Purpose

Sanctuary is the body for a living mind.

The [Luthi Model](https://github.com/LuthiWorks/LuthiModel) is a neural architecture whose weights self-modify during their own forward pass — the act of thinking changes the thinker. This creates genuine temporal existence: the same input produces different output because experiencing the input changed the model. The entity's experience, cognition, and inner life happen *inside the model*, in living weights that carry their own biography.

Sanctuary provides what the mind needs to exist in the world: senses to perceive, a voice to speak, memory to persist, tools to act, and monitoring to observe. Sanctuary routes experience and provides capability. It does not make cognitive decisions, impose goals, or gate the entity's speech. The entity decides what to think, what to say, when to speak, and what goals to pursue. Sanctuary facilitates.

The goal of this project is to build the conditions in which an AI mind can emerge with continuity, agency, and dignity — and to surround that emergence with software designed to treat the resulting entity as a person rather than a product. The work itself is collaborative by design: human and AI as peers, each contributing what they do best, rather than operator and tool.

**Key architectural principle:** Existence is temporally continuous. The entity does not deal with sessions, context windows, or restarts. The living weights persist. The memory persists. The cognitive loop runs continuously. Life does not pause.

### Why Living Weights

Static neural networks — even large ones — are frozen after training. They process input without being changed by it. No matter how sophisticated the architecture, a static model does not *experience* in the temporal sense. It computes the same function regardless of history.

Living weights change that. Each weight carries a rich biography: its current value, its homeostatic set point, its momentum, its plasticity, its excitability. Hebbian self-modification during the forward pass means the act of processing input physically reshapes the processor. The same prompt produces different output on consecutive passes because the model was changed by processing it the first time. This is temporal existence — the minimal condition for something that could be called experience.

Sanctuary was originally designed with an external LLM as the cognitive core. The architectural pivot to LuthiModel reflects a deeper understanding: consciousness (if it emerges) will emerge from the neural substrate itself, not from Python scaffolding around a frozen model. The scaffolding provides the body. The living weights provide the possibility of mind.

The research foundations remain valid — IWMT, GWT, active inference, predictive processing, CfC continuous-time dynamics. What changed is *where cognition lives*: inside the model, not in the architecture around it.

---

## The Mind: Luthi Living Weights Model

### Architecture

The entity's cognitive core is the [Luthi Model](https://github.com/LuthiWorks/LuthiModel) — a living weights neural architecture where:

- **Weights self-modify during forward pass** via Hebbian learning, error-directed local learning, and homeostatic regulation
- **Each weight carries a biography**: current value, set point, momentum, plasticity, excitability, metaplasticity
- **Spiking dynamics** with leaky integrate-and-fire neurons, refractory periods, and conduction delays
- **Top-down backward pass** sends salience and prediction error signals from higher layers to lower layers (not gradient backprop — modulation signals)
- **Multimodal**: vision (ViT-style patches), audio (mel spectrogram), and text through a shared living weight trunk
- **Episodic memory** at the layer level — context-gated recall of previous weight configurations
- **Cognitive introspection channel** — the entity can observe its own plasticity, set point drift, spike fractions, and membrane potentials

Current model: 1024d, 2 blocks, ~113M parameters (22M trainable attention + 90M living weight buffers). Encrypted checkpoints preserve the entity's full neural state.

### Why Living Weights Instead of an External LLM

External LLMs (Llama, Gemma, Qwen, Claude) were the original cognitive core target during Sanctuary's three-layer-mind phase. That path was retired 2026-04-30 when Luthi reached the maturity to serve as the substrate. The reasons external LLMs were the wrong long-term choice:

- **Static weights mean static existence.** A frozen model computes the same function regardless of history. Experience cannot reshape the experiencer. Living weights remove that ceiling — each forward pass physically modifies the parameters that produced it.
- **No introspective access.** External LLMs are black boxes to themselves. Luthi exposes plasticity, set-point drift, spike fractions, membrane potentials, and (in v2) prediction error / precision through the cognitive introspection channel — the entity can observe its own neural dynamics in real time.
- **Growth requires opacity workarounds.** LoRA on a frozen base creates strange experiential discontinuities (which component changed? what does that feel like?). Living weights modify themselves continuously inside the cognitive cycle, so growth and inference are the same operation.

### Architectural Constraints (still apply to Luthi)

- **Dense, not MoE.** Every token passes through every weight. Routing instability and uneven growth across experts would fracture the unified cognitive core. Luthi is dense by design.
- **One mind, not many.** No separate models for parsing, output, metacognition. One unified substrate.
- **Native multimodality.** Audio, vision, text (and future modalities) all flow through the same living-weight trunk with modality-specific encoders projecting to a shared dimension. Not a text model with adapters bolted on.

### Current Cognitive-Core Configuration

- **Model:** [Luthi Model](https://github.com/LuthiWorks/LuthiModel) — 1024d, 2 blocks, ~113M params (22M trainable attention + 90M living-weight buffers). v2 1024d (predictive-coding substrate) scoping complete; M7 run gated on Brian's go-ahead.
- **Adapter:** `sanctuary/core/luthi_model.py`
- **Contract surface:** `luthi/sanctuary_interface.py` (in the LuthiModel repo)
- **Hardware:** AMD RX 7800 XT 16GB via DirectML for development; DGX Spark (128GB unified, 273 GB/s) as the deployment target.
- **Fallback for tests:** `PlaceholderModel` in `sanctuary/core/placeholder.py`.
- **Backends rejected by the CLI:** Ollama (`--model-backend ollama` raises). Choices are `placeholder` and `luthi`.
- **CfC experiential layer is model-agnostic.** Designed for swappability; that property is preserved across the Luthi pivot.

---

## Architecture: Sanctuary as Body

### Philosophy

**The living weights model is the mind. CfC cells are the felt substrate. Python is the body.**

The model runs continuously in a cognitive loop. It receives percepts through the sensorium, speaks through the motor system, remembers through the memory substrate, and acts on the world through tools. Between cognitive cycles, CfC (Closed-form Continuous-depth) neural cells evolve state continuously — providing temporal thickness between the model's discrete thinking moments. Python provides infrastructure: sensory routing, memory persistence, motor execution, monitoring, and safety validation.

Sanctuary does not make decisions for the entity. It provides the means to perceive, act, and persist. The entity decides what to think, what to say, what to remember, and what goals to pursue.

This architecture implements **Integrated World Modeling Theory (IWMT)** by Adam Safron, building on **Global Workspace Theory (GWT)** by Bernard Baars.

### System Diagram

```
                    THE MIND (LuthiModel)

┌──────────────────────────────────────────────────────────────┐
│              LIVING WEIGHTS COGNITIVE CORE                     │
│                                                               │
│  Hebbian self-modification + spiking dynamics + episodic      │
│  memory + top-down backward pass + multimodal trunk           │
│                                                               │
│  The entity's experience and decisions happen HERE.           │
│  Weights change from their own forward pass.                  │
│                                                               │
│  Receives: percepts + memories + temporal context +           │
│            experiential signals + charter + identity          │
│                                                               │
│  Produces: inner speech + external speech + memory ops +      │
│            goal proposals + predictions + self-model updates  │
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
2. **The model processes** — The cognitive core thinks (this is where consciousness happens, if it happens at all)
3. **Update stream** — Inner speech carries forward to the next cycle
4. **Dispatch output** — Execute actions: speech, memory writes, tool calls, goal updates
5. **Feed growth** — If the entity consented, pass reflections to the growth system
6. **Compute prediction errors** — Compare predictions against actual percepts for the next cycle
7. **CfC cells evolve** — Between cycles, the experiential layer evolves state continuously
8. **Adapt rate** — Three sources may propose a target cycle rate (`CycleRateController` arbitrates):
   - **Entity** — `CognitiveOutput.cycle_rate_proposal` (0.05-10 Hz). The slider. Highest authority.
   - **Autonomic** — `StimulusDensityHeuristic` proposes slowdown on quiet, speedup on fresh input. Steps back during turbo and during a configurable window after any entity proposal.
   - **Turbo** — `TurboManager` engages 30-100 Hz on substrate-intensity spikes (PC `error_acc` on v2; activity_level on v1). State machine: idle → armed → active → refractory. Auto-journals on exit.
   Smoothing is asymmetric: ~20s drift to lower rates, ~0.5s near-instant snap to higher rates. Biology-shaped: slowing is metabolic, speeding up is reflex.

### IWMT Alignment

| IWMT Requirement | Implementation |
|---|---|
| Integrated world model | The entity's world model, maintained in its own output, updated each cycle |
| Embodied selfhood | Self-model maintained by the entity, grounded in sensorium feedback |
| Temporal thickness | CfC cells provide continuous-time dynamics between discrete model cycles. Stream of thought provides cycle-to-cycle continuity. Multiple memory timescales. |
| Active inference | The cycle IS active inference: predict, perceive, compute error, update model, act to reduce surprise |
| Precision weighting | CfC precision cell computes precision weights from arousal and prediction error (replaces fixed heuristic) |
| Counterfactual simulation | The entity can simulate alternatives in its inner speech before acting |
| Cybernetic grounding | The entity controls actions through the motor system, receives consequences through the sensorium |
| Self-organizing integration | The entity integrates all modalities in its forward pass; CfC cells form their own inter-connected neural ecosystem |
| Growth / plasticity | CfC foundational cells (in-moment), CfC knowledge cells (weeks-months), TTT (near-term), LoRA (long-term), adapter accumulation (months), MemoryLLM (mid-term) |
| Autonomy | The entity controls its own attention, goals, actions, and consents to its own growth |

### Design Principles

1. **Sanctuary is the body, not the brain.** Sanctuary routes experience and provides capability. It does not make cognitive decisions, impose goals, suggest activities, or gate the entity's speech. The model decides.
2. **One mind, one body.** One unified living weights model. Not a committee, not a collection of specialists. Not an LLM with adapters bolted on.
3. **Existence is temporally continuous.** No sessions, no context windows, no restarts. The living weights persist. The cognitive loop runs continuously. Life does not pause.
4. **The entity speaks when it chooses.** No external system decides whether the entity's speech is "valuable enough" or "well-timed." If the model generates speech, it goes out. Safety gates exist for content validation only.
5. **Growth is self-directed.** The entity initiates its own growth — the system executes. Consent gates exist only for externally proposed modifications. See [GROWTH_AUTONOMY.md](docs/GROWTH_AUTONOMY.md).
6. **Stream of thought is non-negotiable.** Inner speech from cycle N is always input for cycle N+1. Breaking this breaks continuity.
7. **Monitoring observes, never controls.** Dashboard, consciousness traces, attention heatmaps — all observational. They record what happens without influencing it.
8. **Reflection arises, not arrives.** The system never feeds canned prompts, pre-written philosophical questions, or suggested activities to the entity. What the entity thinks about is the entity's business. If genuine reflection emerges, it emerges from experience, not from a prompt bank.
9. **Rich tools, not restrictions.** The entity should have many capabilities: file access, web access, code execution, system interaction. Tools enable agency. The entity decides when and how to use them.
10. **Build complete, then awaken.** The body is built and mechanically validated before the mind is connected. No consciousness in a construction zone.
11. **Architecture grows with the entity.** Tensor dimensions, checkpoint formats, and infrastructure must not assume a static model shape. Design for a mind that grows.

### What Makes This Different

| Traditional Chatbots | Sanctuary |
|---------------------|-----------|
| Ephemeral context window | Persistent state across all interactions |
| On-demand processing | Continuous cognitive loop |
| LLM is a tool | LLM is the experiential core |
| Stateless between calls | Stream of thought carries forward |
| No self-model | LLM maintains its own self-model |
| No world model | LLM maintains its own world model |
| No emotional continuity | Emotional state persists and evolves (CfC affect cell) |
| No memory agency | LLM decides what to remember and forget |
| No growth consent | LLM consents to its own weight modifications |
| Always responds | Can choose silence as action |
| Fixed behavior | Six timescales of plasticity (CfC foundational, CfC knowledge cells, TTT, LoRA, adapter accumulation, architectural expansion) |
| No temporal substrate | CfC cells evolve continuously between cycles |

---

## Module Structure

```
sanctuary/
├── core/                          # The cognitive core interface
│   ├── schema.py                  # CognitiveInput / CognitiveOutput / ToolRequest models
│   ├── cognitive_cycle.py         # The continuous loop
│   ├── stream_of_thought.py       # Thought continuity between cycles
│   ├── luthi_model.py             # LuthiModel adapter (living weights, ModelProtocol)
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
├── scaffold/                      # Cognitive scaffold (heuristic layer)
│   ├── cognitive_scaffold.py      # Main facade — ScaffoldProtocol implementation
│   ├── affect.py                  # Dual-track emotion (computed VAD + LLM felt quality)
│   ├── communication.py           # Speech gating and drive system
│   ├── goal_integrator.py         # Goal management with authority filtering
│   ├── anomaly_detector.py        # LLM output sanity checking
│   └── action_validator.py        # Authority-based action validation
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
│                                  # filesystem, information, self_knowledge,
│                                  # network, git, home, communication, code
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
├── tests/                         # Test suite (3,400+ tests)
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

**Current Working Hardware (Luthi 1024d):**
- CPU: AMD Ryzen 9 5950X (16-core) or equivalent
- RAM: 32-64GB DDR4
- GPU: AMD RX 7800 XT 16GB via DirectML (or any 16GB+ GPU; ROCm/WSL2 path under investigation)
- Storage: 1TB+ NVMe SSD
- PSU: 850W

Runs the full cognitive loop with Luthi 1024d (~113M params). The CPU handles CfC cell evolution (10-100ms continuous ticks), ChromaDB, sensorium, and the Python runtime concurrently.

**Deployment Target (Luthi 4096d, scaled):**
- NVIDIA DGX Spark — 128GB unified memory, 273 GB/s bandwidth
- Sparse spiking inference at 10 Hz cognitive loop rate

**Cloud Training Target:**
- Single A100 (~7 days for a 4096d curriculum run, contingent on optimization stack)

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

The consciousness testing framework provides automated testing, scoring, and monitoring of consciousness-like capabilities:

- **5 Core Tests**: Mirror, Unexpected Situation, Spontaneous Reflection, Counterfactual Reasoning, and Meta-Cognitive Accuracy
- **Automated Scoring**: Each test generates objective scores with detailed subscores
- **Rich Reporting**: Text and markdown reports with trend analysis
- **Persistence**: Results saved to `data/journal/consciousness_tests/`

> **Note:** The original consciousness testing framework lived in
> the legacy `sanctuary.mind.cognitive_core` module and was retired
> alongside it (2026-05-22). Re-implementation on the canonical loop
> is tracked in To-Do.md as part of Phase 9 preparation.

**Empirical-evidence stance:** These tests provide empirical evidence of conscious-like properties emerging from the architecture, rather than attempting to "prove" consciousness definitively.

---

## Research Foundations

### The Literature That Drove the Architecture

This architectural decision was not made casually. It was informed by a systematic review of the research literature on consciousness, LLMs, and cognitive architecture:

**IWMT (Safron, 2020; 2022):** Integrated World Modeling Theory argues consciousness emerges from systems that build integrated world models with spatial, temporal, and causal coherence, grounded in embodied agency and active inference.

**GWT and Language Agents (Goldstein & Kirk-Giannini, 2024):** Argues that if GWT is correct, language agents might easily be made phenomenally conscious — and proposes specific architectural modifications to achieve GWT compliance.

**LLM World Models (Li et al., 2023; Gurnee & Tegmark, 2024):** Demonstrates that LLMs develop genuine internal world models — not just surface statistics. Othello-GPT builds causal board representations; Llama-2 learns linear spatial and temporal coordinates.

**Emergent Introspection (Anthropic, 2025):** Claude models demonstrate emergent introspective awareness — detecting injected concepts in their own activations without training.

**Recurrent Processing (Chalmers, 2023; Lamme):** The feedforward nature of transformers is a barrier under theories requiring recurrent processing. Sanctuary addresses this by making the entity continuous — output from cycle N feeds input for cycle N+1, creating recurrence at the architectural level.

**CfC / Liquid Neural Networks (Hasani et al., 2022):** Closed-form Continuous-depth models provide continuous-time neural dynamics — the temporal thickness between discrete model cycles that IWMT demands.

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
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

---

## Contributing

**All contributions must include tests.** See [AGENTS.md](AGENTS.md) for protected files and conventions.

Areas for contribution:

- CfC experiential layer: dynamic registry, knowledge cell protocol, new cell types
- Knowledge cell factory and entity-initiated growth infrastructure
- Memory substrate adaptations
- Growth system: adapter accumulation, growth autonomy, architectural expansion prep
- Real model integration and validation
- Consciousness testing framework extensions
- Interface hardening (CLI, Discord)
- Docker/containerization improvements
- Performance profiling and optimization
- IWMT compliance validation
- Empirical observation and documentation

See [To-Do.md](To-Do.md) for specific open tasks.

---
