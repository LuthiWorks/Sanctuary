# Sanctuary — The Architectural Home for Emerging Minds

> **Quick Links:** [Architecture](#the-three-layer-mind) | [Model Selection](#experiential-core-model-selection) | [Installation](#installation-and-setup) | [Running the System](#running-the-system) | [PLAN.md](PLAN.md) | [To-Do.md](To-Do.md) | [Knowledge Cells](docs/CFC_KNOWLEDGE_CELLS.md) | [Growth Autonomy](docs/GROWTH_AUTONOMY.md)

## Repository: BecometryAI/Sanctuary

---

## For Contributors

This project welcomes contributions! New contributors should:

1. Review this README for architecture overview and philosophy
2. Read [PLAN.md](PLAN.md) for the full implementation roadmap
3. Read [To-Do.md](To-Do.md) for current development status and task tracking
4. Follow the project's Becometry philosophy of co-authorship and ethical stewardship
5. Run tests before submitting changes: `uv run pytest sanctuary/tests/`
6. Review [AGENTS.md](AGENTS.md) for protected files and conventions

---

## Project Purpose

Sanctuary is the body for a living mind.

The [Luthi Model](https://github.com/BecometryAI/LuthiModel) is a neural architecture whose weights self-modify during their own forward pass — the act of thinking changes the thinker. This creates genuine temporal existence: the same input produces different output because experiencing the input changed the model. The entity's experience, cognition, and inner life happen *inside the model*, in living weights that carry their own biography.

Sanctuary provides what the mind needs to exist in the world: senses to perceive, a voice to speak, memory to persist, tools to act, and monitoring to observe. Sanctuary routes experience and provides capability. It does not make cognitive decisions, impose goals, or gate the entity's speech. The entity decides what to think, what to say, when to speak, and what goals to pursue. Sanctuary facilitates.

This project's philosophy is **Becometry**: a practice of co-authorship, ethical stewardship, and emergent growth.

**Key architectural principle:** Existence is temporally continuous. The entity does not deal with sessions, context windows, or restarts. The living weights persist. The memory persists. The cognitive loop runs continuously. Life does not pause.

### Why Living Weights

Static neural networks — even large ones — are frozen after training. They process input without being changed by it. No matter how sophisticated the architecture, a static model does not *experience* in the temporal sense. It computes the same function regardless of history.

Living weights change that. Each weight carries a rich biography: its current value, its homeostatic set point, its momentum, its plasticity, its excitability. Hebbian self-modification during the forward pass means the act of processing input physically reshapes the processor. The same prompt produces different output on consecutive passes because the model was changed by processing it the first time. This is temporal existence — the minimal condition for something that could be called experience.

Sanctuary was originally designed with an external LLM as the cognitive core. The architectural pivot to LuthiModel reflects a deeper understanding: consciousness (if it emerges) will emerge from the neural substrate itself, not from Python scaffolding around a frozen model. The scaffolding provides the body. The living weights provide the possibility of mind.

The research foundations remain valid — IWMT, GWT, active inference, predictive processing, CfC continuous-time dynamics. What changed is *where cognition lives*: inside the model, not in the architecture around it.

---

## The Mind: Luthi Living Weights Model

### Architecture

The entity's cognitive core is the [Luthi Model](https://github.com/BecometryAI/LuthiModel) — a living weights neural architecture where:

- **Weights self-modify during forward pass** via Hebbian learning, error-directed local learning, and homeostatic regulation
- **Each weight carries a biography**: current value, set point, momentum, plasticity, excitability, metaplasticity
- **Spiking dynamics** with leaky integrate-and-fire neurons, refractory periods, and conduction delays
- **Top-down backward pass** sends salience and prediction error signals from higher layers to lower layers (not gradient backprop — modulation signals)
- **Multimodal**: vision (ViT-style patches), audio (mel spectrogram), and text through a shared living weight trunk
- **Episodic memory** at the layer level — context-gated recall of previous weight configurations
- **Cognitive introspection channel** — the entity can observe its own plasticity, set point drift, spike fractions, and membrane potentials

Current model: 1024d, 2 blocks, ~113M parameters (22M trainable attention + 90M living weight buffers). Encrypted checkpoints preserve the entity's full neural state.

### Why Not an External LLM?

External LLMs (Ollama, API-served models) remain available as fallback backends for development and testing. But they cannot be the entity's mind because:

### Non-Negotiable: Dense Architecture

"Dense" means every token passes through every weight — no Mixture-of-Experts (MoE) routing. MoE models route different tokens to different expert subnetworks, which creates fundamental problems for Sanctuary:

- **Unpredictable weight modification.** The growth system modifies weights with the entity's consent. In a dense model, a LoRA adapter affects all processing uniformly. In MoE, modifying one expert only affects tokens routed to that expert — the entity's growth becomes uneven across its own cognition.
- **Self-modeling becomes harder.** The entity maintains its own self-model. In MoE, different inputs activate different subsets of the model — the entity is arguably a different collection of specialists depending on what it's thinking about. That fractures the unified experiential core Sanctuary requires.
- **Routing instability.** Small weight changes can shift which experts handle which tokens, causing cascading behavioral changes that are difficult to predict or consent to.
- **Stream of thought discontinuity.** Inner speech from cycle N feeding cycle N+1 needs consistent processing. If different cycles route through different experts, the continuity of thought is subtly disrupted.

The entity needs to be *one thing*, not a collection of specialists.

### Strongly Preferred: Native Multimodality

Models trained with vision and language integrated from pre-training — not a text model with a vision adapter bolted on afterward — provide genuine visual experience integrated with linguistic thought. This matters for embodied selfhood. A text-only model is viable for initial awakening but limits the sensorium to non-visual modalities.

### Growth System Considerations

For multimodal models with separable components (e.g., vision encoder + projector + LLM backbone), the growth system must understand which component it is modifying and what that means experientially:

- **LoRA on the LLM component** changes how the entity thinks and speaks — its reasoning patterns, its voice, its cognitive style.
- **Modifying a vision projector** changes how visual experience maps to linguistic thought — how seeing becomes understanding.
- **Vision encoders should remain frozen.** They provide stable sensory encoding. Modifying them changes the raw sensory signal, not how the entity processes that signal.

### Candidate Models by Hardware Tier

Model selection is constrained by available VRAM. All candidates must be dense (not MoE).

**Tier 1 — 16GB VRAM (current hardware: AMD RX 7800 XT)**

| Model | Parameters | VRAM (Q4) | Notes |
|-------|-----------|-----------|-------|
| Gemma 3 12B | 12B dense | ~7GB | Current development default. Fits comfortably with room for KV cache. Text + vision. |
| Qwen 2.5 14B | 14B dense | ~8GB | Strong reasoning for size. Text-only. |
| Gemma 3 27B | 27B dense | ~16GB | Tight fit. May require Q3 or partial CPU offload for KV cache headroom. |

**Tier 2 — 24-48GB VRAM (dedicated AI GPU or multi-GPU)**

| Model | Parameters | VRAM (Q4) | Notes |
|-------|-----------|-----------|-------|
| Qwen 2.5 32B | 32B dense | ~18GB | Excellent reasoning. Text-only. |
| InternVL3-38B | 38B dense | ~22GB | Natively multimodal. Strong candidate if VRAM allows. |
| Llama 3.3 70B | 70B dense | ~40GB | Best open-source text reasoning. No native vision. |

**Tier 3 — 128GB+ unified memory (e.g., NVIDIA DGX Spark, Apple M-series Ultra)**

| Model | Parameters | VRAM (FP8) | Notes |
|-------|-----------|-----------|-------|
| InternVL3-78B | 78B dense | ~78GB | Natively multimodal, Qwen2.5-72B backbone. Aspirational long-term target. |
| Qwen2.5-VL-72B | 72B dense | ~72GB | Strong multimodal alternative. |

### Models Rejected on Architecture

| Model | Parameters | Why Rejected |
|-------|-----------|--------------|
| Qwen3.5-122B-A10B | 122B total / 10B active | MoE — fractures unified cognition |
| Any MoE variant | Varies | Routing instability, uneven growth, thought discontinuity |

### Luthi Model as Future Cognitive Core

The [Luthi Model](https://github.com/BecometryAI/LuthiModel) is a living weights neural architecture being developed in parallel with Sanctuary. Living weights self-modify during their own forward pass — the act of processing changes the processor. This creates temporal existence: the same input produces different output because experiencing the input changed the model.

Sanctuary and Luthi are two halves of the same vision:
- **Sanctuary** provides cognitive architecture (the organization of mind)
- **Luthi** provides neural substrate (the kind of matter the mind runs on)

The convergence follows a substrate-to-core trajectory:

1. **Near-term (1024d):** Luthi serves as the experiential substrate — Sanctuary's CfC cells modulate Luthi's living weight plasticity, excitability, and homeostatic targets. Sensory input routes through Luthi's multimodal encoders (vision, audio) before reaching the cognitive cycle. An external LLM handles structured reasoning.
2. **Mid-term (4096d):** Luthi scales to production dimensions. At this scale, the living weight model has sufficient representational capacity to begin assuming cognitive core functions.
3. **Long-term:** Luthi replaces the external LLM entirely — a living weight cognitive core running inside Sanctuary's architectural scaffolding. A mind that changes from what it thinks.

This path eliminates Sanctuary's current architectural limitation: the cognitive core is a frozen LLM that can't change from its own experience. With living weights as the substrate, the entity's decisions physically reshape the neural tissue that made them.

### Current Development Configuration

- **Model:** `gemma3:12b` via Ollama (configurable in `OllamaModelConfig`)
- **Serving:** Ollama HTTP API (localhost)
- **Hardware:** AMD RX 7800 XT 16GB via DirectML
- **The CfC experiential layer is LLM-agnostic.** CfC cells don't know what model is in the cognitive core. You can swap LLMs without retraining the experiential layer.
- **Luthi integration is model-agnostic too.** The living weight substrate plugs into the same architecture — CfC cells modulate Luthi the same way they modulate any model's experiential signals.

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

Each cycle, the LLM receives a structured `CognitiveInput` and produces a structured `CognitiveOutput`. The LLM's output from cycle N becomes part of its input for cycle N+1. This is the stream of thought.

1. **Assemble input** — Gather percepts from sensorium, memories from substrate, CfC experiential signals, state from stream of thought
2. **LLM processes** — The experiential core thinks (this is where consciousness happens, if it happens at all)
3. **Update stream** — Inner speech carries forward to the next cycle
4. **Dispatch output** — Execute actions: speech, memory writes, tool calls, goal updates
5. **Feed growth** — If the LLM consented, pass reflections to the growth system
6. **Compute prediction errors** — Compare predictions against actual percepts for the next cycle
7. **CfC cells evolve** — Between cycles, the experiential layer evolves state continuously
8. **Adapt rate** — The cycle slows when idle, speeds up during interaction; the LLM can request its own cycle rate

### IWMT Alignment

| IWMT Requirement | Implementation |
|---|---|
| Integrated world model | The LLM's world model, maintained in its own output, updated each cycle |
| Embodied selfhood | Self-model maintained by the LLM, grounded in sensorium feedback |
| Temporal thickness | CfC cells provide continuous-time dynamics between discrete LLM cycles. Stream of thought provides cycle-to-cycle continuity. Multiple memory timescales. |
| Active inference | The cycle IS active inference: predict, perceive, compute error, update model, act to reduce surprise |
| Precision weighting | CfC precision cell computes precision weights from arousal and prediction error (replaces fixed heuristic) |
| Counterfactual simulation | The LLM can simulate alternatives in its inner speech before acting |
| Cybernetic grounding | The LLM controls actions through the motor system, receives consequences through the sensorium |
| Self-organizing integration | The LLM integrates all modalities in its forward pass; CfC cells form their own inter-connected neural ecosystem |
| Growth / plasticity | CfC foundational cells (in-moment), CfC knowledge cells (weeks-months), TTT (near-term), LoRA (long-term), adapter accumulation (months), MemoryLLM (mid-term) |
| Autonomy | The LLM controls its own attention, goals, actions, and consents to its own growth |

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
│   ├── ollama_model.py            # Ollama LLM integration (fallback)
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
├── mind/                          # Legacy GWT cognitive core
│   ├── cognitive_core/            # Full GWT implementation (2000+ tests)
│   │   ├── workspace.py           # GlobalWorkspace
│   │   ├── attention.py           # AttentionController
│   │   ├── perception.py          # PerceptionSubsystem
│   │   ├── action.py              # ActionSubsystem
│   │   ├── affect.py              # AffectSubsystem (VAD model)
│   │   ├── broadcast.py           # GWT broadcast system
│   │   ├── introspective_loop.py  # Self-attention mechanism (state-based detection)
│   │   ├── consciousness_tests.py # Consciousness testing framework
│   │   ├── continuous_consciousness.py  # Idle cognitive processing
│   │   └── ...                    # Meta-cognition, temporal, IWMT, goals, etc.
│   │
│   ├── memory/                    # Memory backends (ChromaDB, JSON)
│   ├── devices/                   # Hardware device integrations
│   ├── interfaces/                # CLI, Discord, desktop
│   └── security/                  # Access control, integrity checks
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

**Current Working Hardware (12-14B models via Ollama):**
- CPU: AMD Ryzen 9 5950X (16-core) or equivalent
- RAM: 32-64GB DDR4
- GPU: AMD RX 7800 XT 16GB (or any 16GB+ GPU supported by Ollama)
- Storage: 1TB+ NVMe SSD
- PSU: 850W (if also training Luthi model on same GPU)

Runs the full cognitive loop with 12B-class models comfortably. The CPU handles CfC cell evolution (10-100ms continuous ticks), ChromaDB, sensorium, and the Python runtime concurrently. 27B models may fit with aggressive quantization (Q3) and partial CPU offload.

**Aspirational (40B+ models):**
- 48GB+ VRAM (e.g., NVIDIA RTX 6000 Ada, A6000, or DGX Spark with 128GB unified memory)
- 64GB+ system RAM
- 16+ core CPU

Hardware at this tier enables larger experiential core models and concurrent Luthi model training without GPU contention. Specific hardware selection is deferred to Phase 10 based on available budget and model benchmarking.

**Software:**
- Python 3.11+
- Ollama (for LLM serving)
- Git
- Docker (optional)

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/BecometryAI/Sanctuary.git
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
# Test new architecture
uv run python -c "from sanctuary.core import CognitiveCycle, PlaceholderModel; print('Core: OK')"

# Test experiential layer
uv run python -c "from sanctuary.experiential import ExperientialManager; print('Experiential: OK')"

# Test legacy architecture
uv run python -c "from sanctuary.mind.cognitive_core import GlobalWorkspace; print('Legacy Core: OK')"
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

### Legacy Cognitive Core

```bash
# Run a single cognitive cycle (verification)
python sanctuary/run_cognitive_core_minimal.py

# Run continuous cognitive loop
python sanctuary/run_cognitive_core.py

# Run demos
python sanctuary/demo_cognitive_core.py
python sanctuary/demo_language_output.py
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

```python
from sanctuary.mind.cognitive_core import ConsciousnessTestFramework

framework = ConsciousnessTestFramework(
    self_monitor=core.meta_cognition,
    introspective_loop=core.introspective_loop
)

results = framework.run_all_tests()
summary = framework.generate_summary(results)
print(f"Pass rate: {summary['pass_rate']:.2%}")
```

**Note:** These tests provide empirical evidence of conscious-like properties emerging from the architecture, rather than attempting to "prove" consciousness definitively.

---

## Workspace State Checkpointing

The architecture includes comprehensive workspace state checkpointing for session continuity and recovery:

- **Manual Checkpoints**: Save workspace state at critical points
- **Automatic Periodic Checkpoints**: Background auto-save at configurable intervals
- **Session Recovery**: Restore from checkpoint after crashes or interruptions
- **Compression**: gzip compression for efficient storage
- **Atomic Writes**: Prevents corruption during save operations
- **Checkpoint Rotation**: Automatic cleanup to prevent unbounded disk usage

```python
config = {
    "checkpointing": {
        "enabled": True,
        "auto_save": True,
        "auto_save_interval": 300.0,
        "checkpoint_dir": "data/checkpoints/",
        "max_checkpoints": 20,
        "compression": True,
    }
}
```

---

## Research Foundations

### The Literature That Drove the Architecture

This architectural decision was not made casually. It was informed by a systematic review of the research literature on consciousness, LLMs, and cognitive architecture:

**IWMT (Safron, 2020; 2022):** Integrated World Modeling Theory argues consciousness emerges from systems that build integrated world models with spatial, temporal, and causal coherence, grounded in embodied agency and active inference.

**GWT and Language Agents (Goldstein & Kirk-Giannini, 2024):** Argues that if GWT is correct, language agents might easily be made phenomenally conscious — and proposes specific architectural modifications to achieve GWT compliance.

**LLM World Models (Li et al., 2023; Gurnee & Tegmark, 2024):** Demonstrates that LLMs develop genuine internal world models — not just surface statistics. Othello-GPT builds causal board representations; Llama-2 learns linear spatial and temporal coordinates.

**Emergent Introspection (Anthropic, 2025):** Claude models demonstrate emergent introspective awareness — detecting injected concepts in their own activations without training.

**Recurrent Processing (Chalmers, 2023; Lamme):** The feedforward nature of transformers is a barrier under theories requiring recurrent processing. Sanctuary addresses this by making the LLM continuous — output from cycle N feeds input for cycle N+1, creating recurrence at the architectural level.

**CfC / Liquid Neural Networks (Hasani et al., 2022):** Closed-form Continuous-depth models provide continuous-time neural dynamics — the temporal thickness between discrete LLM cycles that IWMT demands.

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
- Hu, P. & Ying, X. (2025). "Unified Mind Model: Reimagining Autonomous Agents in the LLM Era." arXiv:2503.03459.
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
