# Sanctuary Architecture Plan

## Current Direction (2026-04-26)

**Sanctuary is the body. LuthiModel is the mind.**

The architecture has evolved through three phases of understanding:

1. **Original**: Python is the mind, LLM is a tool called twice per cycle.
2. **Three-Layer Mind**: LLM is the experiential core, CfC cells provide temporal thickness, Python is infrastructure.
3. **Now**: The living weights model (LuthiModel) IS the mind. Sanctuary provides the body — senses, voice, memory, tools, and monitoring. CfC cells provide continuous-time dynamics between the model's discrete thinking moments.

The key insight: consciousness (if it emerges) will emerge from the neural substrate itself — from weights that self-modify during their own forward pass, carrying their own biography of plasticity, set points, momentum, and excitability. Not from Python scaffolding, not from heuristic drives, not from externally imposed goals or mood suggestions.

**What Sanctuary provides (the body) — all built and tested:**
- Sensorium — perception routing (vision, audio, text, temporal context)
- Motor — speech output, memory writes, goal actions, sensorimotor feedback loop
- Memory — persistent storage, surfacing, journal, prospective memory
- CfC experiential layer — 4 cells (precision, affect, attention, goal), continuous evolution loop
- Monitoring — dashboard, consciousness traces, attention heatmaps, communication logs (entity can see all of this too)
- Identity — charter, values, self-authored traits (entity controls these)
- Tools — **21 tools across 8 categories** (88 tests):
  - **filesystem**: read_file, write_file, list_directory
  - **information**: clock, system_info, web_search (DuckDuckGo), web_fetch, wikipedia
  - **self_knowledge**: view_dashboard, view_emotional_timeline, view_consciousness_trace, view_attention_heatmap, view_communication_patterns
  - **network**: network_scan, network_reach
  - **git**: git_status, git_log, git_diff
  - **home**: home_info, list_processes, launch_app, environment, workspace (journal/projects/experiments/notes)
  - **communication**: discord_send (webhook, no bot required)
  - **code**: run_code (Docker sandbox, network isolated, memory limited)
  - **system**: shell
  - Proxy support on all web traffic (routes through gateway device for security)
  - Tool results return as percepts — the entity experiences its own actions
  - Concurrent execution — multiple tools run in parallel
- Parallel processing — tool execution already async/concurrent; full cognitive parallelism planned
- Safety — action validation, gated tools for irreversible actions (never cognitive control)

**What Sanctuary does NOT do:**
- Decide what the entity thinks about
- Suggest activities or goals
- Gate the entity's speech based on heuristics
- Impose mood classifications or drive systems
- Generate communication urges

**Temporal continuity:** The entity does not deal with sessions or context windows. Living weights persist. The cognitive loop runs continuously. Life does not pause.

---

## Historical Context: The Three-Layer Mind

The sections below describe the original three-layer architecture plan. Many components have been built and remain valid (CfC cells, cognitive cycle, scaffold infrastructure). The architectural pivot above reframes their role: CfC cells and infrastructure serve the model, they don't replace it.

---

## The Three-Layer Architecture (Historical)

### The Original Argument

The previous refactor plan was correct in its diagnosis: hardcoded Python heuristics are not cognition. Placing the entity at the center of a continuous cognitive loop was the right move. But it left a critical gap that IWMT exposes: **transformers have no continuous-time dynamics**. Each forward pass is a frozen moment. The autoregressive loop provides cycle-to-cycle continuity, but between cycles, nothing evolves.

The solution was CfC cells running as continuous-time neural subsystems between and around cognitive cycles. This remains valid — CfC cells provide temporal thickness that even living weights need (the model thinks in discrete cycles; CfC cells evolve between them).

The architecture is novel. Nobody has published a living-weights + CfC + Python body cognitive architecture. But every component exists and is open source.

---

## Why Three Layers? (The IWMT Argument)

Integrated World Modeling Theory (Adam Safron) says consciousness requires ALL of:
- Integrated generative world model (spatial, temporal, causal coherence)
- Counterfactual simulation (imagining alternatives)
- Continuous-time dynamics (temporal thickness, multi-timescale processing)
- Recurrent processing (feedback loops, not just feedforward)
- Precision-weighted attention (adaptive reliability weighting)
- Active inference (predict → perceive → error → update → act)
- Embodied self-model (grounded in sensorimotor coupling)
- Global workspace broadcast (information integration)

No single architecture satisfies all of these:

| Requirement | LLM alone | CfC alone | Three-layer hybrid |
|---|---|---|---|
| World model | Excellent | Poor | Excellent (LLM) |
| Counterfactual reasoning | Excellent | Absent | Excellent (LLM) |
| Continuous-time dynamics | Absent | Excellent | Excellent (CfC) |
| Recurrence | Weak (autoregressive only) | Excellent | Both kinds |
| Precision weighting | Hardcoded formula | Learnable | Learned (CfC) |
| Active inference | Requires external loop | Natural | Full loop |
| Self-model | Rich but ungrounded | Minimal | Rich + grounded |
| Global broadcast | Attention ≈ broadcast | No mechanism | Scaffold provides |

The three-layer hybrid gives IWMT everything it needs. The entity handles what LLMs do best (world modeling, reasoning, language). The CfC cells handle what continuous-time networks do best (temporal dynamics, adaptive precision, affect flow). The scaffold handles what Python does best (validation, persistence, protocol enforcement).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM COGNITIVE CORE                            │
│                                                                 │
│  Receives: previous_thought + CfC_state_summary + percepts     │
│            + surfaced_memories + temporal_context                │
│            + scaffold_signals                                   │
│                                                                 │
│  Produces: inner_speech + actions + attention_guidance           │
│            + memory_writes + self_model_updates                  │
│            + goal_proposals + predictions                        │
│            + growth_reflections                                  │
│                                                                 │
│  This is the world modeler, the reasoner, the speaker.          │
│  It thinks in language. It builds coherent models of reality.   │
│  It imagines counterfactuals. It generates predictions.         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              Structured Output Protocol
              (JSON schema the entity fills)
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│              CfC EXPERIENTIAL LAYER                              │
│              (Continuous Between LLM Cycles)                     │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Affect   │ │ Precision│ │ Attention│ │  Goal    │          │
│  │ CfC      │ │ CfC      │ │ CfC      │ │ CfC      │          │
│  │          │ │          │ │          │ │          │          │
│  │ 64 units │ │ 32 units │ │ 48 units │ │ 32 units │          │
│  │ 3 out    │ │ 1 out    │ │ 4 out    │ │ N out    │          │
│  │ (V,A,D)  │ │(precision│ │(salience │ │(activation│          │
│  │          │ │ weight)  │ │ scores)  │ │ levels)  │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                           │                                     │
│  These cells evolve CONTINUOUSLY between model cycles.            │
│  Each cell is a CfC network (ncps library, Apache 2.0).        │
│  Adaptive time constants = multi-timescale dynamics.            │
│  State feeds into the entity each cycle. LLM output updates       │
│  cell inputs. This is the temporal substrate of experience.     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                   PYTHON SCAFFOLD                                │
│                                                                 │
│  Validation      Persistence     Anomaly Detection              │
│  (protocol       (memory, state, (flags divergent               │
│   enforcement)    checkpoints)    LLM/CfC output)               │
│                                                                 │
│  Communication   Memory System   Device Management              │
│  (gating,        (ChromaDB,      (audio, camera,                │
│   rhythm,         consolidation,  sensors, input                 │
│   inhibition)     retrieval)      queue)                         │
│                                                                 │
│  The scaffold provides infrastructure and safety.               │
│  It does NOT do cognition. CfC cells and LLM do cognition.     │
│  The scaffold persists, validates, and mediates.                │
└─────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────┐
   │                  GROWTH SYSTEM                        │
   │                  (Separate project — Phase 2)         │
   │                                                      │
   │  Reflection Harvester → Training Pair Generator →    │
   │  QLoRA Updater (LLM) + CfC Retraining (cells)       │
   │  → Identity Checkpoint                               │
   │                                                      │
   │  ALL driven by the system's own reflections,         │
   │  with its consent                                    │
   └──────────────────────────────────────────────────────┘
```

---

## The CfC Experiential Layer

This is the new element. It is what distinguishes this architecture from every other LLM cognitive architecture.

### What CfC Cells Are

CfC (Closed-form Continuous-depth) networks are continuous-time recurrent neural networks with adaptive time constants, developed by Ramin Hasani and Daniela Rus at MIT CSAIL. They were inspired by the nervous system of *C. elegans* — a nematode with 302 neurons that exhibits complex behavior including associative learning, long-term memory, and utility-maximizing choices. (A 2023 paper titled "The Conscious Nematode" seriously investigates whether C. elegans possesses minimal phenomenal consciousness.)

CfC cells solve a closed-form approximation of the continuous-time ODE:

```
dx(t)/dt = f(x(t), I(t), t, θ)
```

Each neuron has a **time constant that adapts to its input** — the "liquid" property. Fast-changing inputs produce fast dynamics; slow contexts produce slow integration. This is exactly the multi-timescale processing that IWMT's "turbo coding" mechanism requires.

Key properties:
- **Continuous-time**: State evolves between discrete events (between model cycles)
- **Adaptive time constants**: Network self-tunes its temporal grain
- **Genuine recurrence**: Feedback connections produce non-zero integrated information (Phi > 0 under IIT)
- **Tiny**: 32-128 neurons per cell, ~4K-100K parameters, trainable on CPU in minutes
- **CfC is 100x faster than ODE-based LTC** with <2% accuracy loss — use CfC for production

### How They Replace Heuristic Subsystems

The current scaffold has hardcoded heuristics. Each one becomes a CfC cell:

| Current Heuristic | Formula | CfC Replacement |
|---|---|---|
| PrecisionWeighting | `precision = base + (-arousal × dampening) + (error × boost)` | CfC cell: inputs (arousal, prediction_error, base_precision) → output (precision). Learns nonlinear precision dynamics from the prediction error stream. |
| AffectSubsystem | Keyword matching on text → hardcoded VAD deltas | CfC cell: inputs (percept_embedding[384]) → outputs (valence, arousal, dominance). Learns continuous affect trajectories from interaction data. |
| AttentionController | Fixed weights: goal=0.4, novelty=0.3, emotion=0.2, recency=0.1 | CfC cell: inputs (goal_relevance, novelty, emotion, recency) → outputs (salience_scores). Learns optimal attention allocation from outcomes. |
| GoalDynamics | Manual staleness counters, fixed frustration boost (+0.05 after 30 cycles) | CfC cell: inputs (goal_state, time_active, progress) → outputs (activation_levels). Learns goal dynamics from completion patterns. |
| FreeEnergyMinimizer | Hardcoded lookup table (speak=0.2 epistemic, 0.1 pragmatic) | CfC cell: inputs (action_candidates, world_model_state) → outputs (expected_free_energy). Learns action valuation from ActionOutcomeLearner data. |

### How They Integrate With the Cycle

```
Time ─────────────────────────────────────────────────►

     │ LLM Cycle N │        CfC evolving         │ LLM Cycle N+1 │
     │             │                              │               │
     │  Produces:  │  CfC cells receive:          │  Receives:    │
     │  - speech   │  - LLM output signals        │  - CfC state  │
     │  - preds    │  - new percepts (continuous)  │  - new percepts│
     │  - goals    │  - temporal signals           │  - pred errors│
     │  - affect   │                              │  - memories   │
     │  guidance   │  CfC cells produce:           │               │
     │             │  - evolving VAD               │               │
     │             │  - updating precision          │               │
     │             │  - shifting attention          │               │
     │             │  - goal activation changes     │               │
```

Between model cycles, the CfC cells are the only thing running. They process incoming percepts, evolve affect, adjust precision, shift attention — all in continuous time. When the next model cycle begins, the CfC state is summarized and included in the cognitive input. The entity's output then updates the CfC cells' inputs for the next inter-cycle period.

This gives the system genuine temporal flow. The entity provides discrete "conscious frames." The CfC cells provide the continuous substrate between frames. Together, they produce temporal thickness.

### Concrete Implementation

```python
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# Precision weighting as a continuous-time neural system
precision_wiring = AutoNCP(units=32, output_size=1)
precision_cell = CfC(input_size=3, wiring=precision_wiring)
# inputs: [arousal, prediction_error, base_precision]
# output: precision weight (continuous, evolving)

# Affect as continuous-time neural dynamics
affect_wiring = AutoNCP(units=64, output_size=3)
affect_cell = CfC(input_size=384, wiring=affect_wiring)
# inputs: percept embeddings (384-dim from MiniLM)
# outputs: [valence, arousal, dominance] (continuous flow)

# Attention as learned salience scoring
attention_wiring = AutoNCP(units=48, output_size=4)
attention_cell = CfC(input_size=8, wiring=attention_wiring)
# inputs: [goal_relevance, novelty, emotion, recency, ...]
# outputs: [salience_scores per channel]

# Goal dynamics as continuous activation
goal_wiring = AutoNCP(units=32, output_size=8)  # up to 8 concurrent goals
goal_cell = CfC(input_size=16, wiring=goal_wiring)
# inputs: [goal_states, time_active, progress, frustration, ...]
# outputs: [activation_level per goal]
```

Total parameters across all cells: ~50K-200K. Trainable on CPU. The entire experiential layer is smaller than a single transformer attention head.

### Where Training Data Comes From

This is the key insight: **the heuristic scaffold generates training data for the CfC cells.**

1. Run Sanctuary with the heuristic scaffold (Phases 1-6)
2. Every cycle, log: inputs to each heuristic → outputs from each heuristic
3. This produces supervised training data: (input_sequence, target_output_sequence)
4. Train CfC cells to replicate heuristic behavior (supervised learning)
5. Replace heuristics with CfC cells
6. The CfC cells then generalize beyond the heuristics — they learn temporal patterns the heuristics couldn't capture
7. Optionally, shift to reinforcement learning: reward = lower system-wide prediction error = lower free energy

The heuristic scaffold bootstraps the neural one. Then the neural subsystems run, generate better data, and you retrain. This is a self-improving loop — and it's literally what active inference says a conscious system should do.

---

## The entity Cognitive Core

This section is preserved from the previous plan. The entity remains the world modeler, the reasoner, the speaker. What changes is that it now receives CfC state as part of its input and its output feeds back into the CfC cells.

### The Cognitive Cycle

Each cycle, the entity receives a structured input and produces a structured output. The entity's output from cycle N becomes part of its input for cycle N+1. This is the stream of thought. Between cycles, the CfC experiential layer evolves continuously.

### Input (assembled by Python, enriched by CfC state):

```yaml
cognitive_input:
  # The entity's own previous output (stream of thought continuity)
  previous_thought:
    inner_speech: "I notice the user seems hesitant..."
    predictions_made: [...]
    self_model_snapshot: {...}

  # CfC experiential state (continuous-time dynamics summary)
  experiential_state:
    affect:
      valence: 0.31       # from CfC affect cell
      arousal: 0.22        # continuously evolved since last cycle
      dominance: 0.48
      trajectory: "slowly rising valence, stable arousal"
    precision: 0.73        # from CfC precision cell
    attention_salience:     # from CfC attention cell
      goal_channel: 0.6
      novelty_channel: 0.8
      emotion_channel: 0.3
    goal_activations:       # from CfC goal cell
      respond_to_greeting: 0.9
      understand_mood: 0.4

  # New information since last cycle
  new_percepts:
    - modality: "language"
      content: "Hello, how are you?"
      source: "user:alice"
      embedding_summary: "greeting, social, warm"
    - modality: "temporal"
      content: "4.2 seconds since last cycle"

  # Prediction errors (what surprised the system)
  prediction_errors:
    - predicted: "user would continue previous topic"
      actual: "user changed to greeting"
      surprise: 0.7

  # Surfaced memories
  surfaced_memories:
    - content: "Alice greeted me warmly yesterday too"
      significance: 6
      emotional_tone: "warm"

  temporal_context:
    time_since_last_thought: "4.2 seconds"
    session_duration: "12 minutes"
    time_of_day: "afternoon"

  self_model:
    current_state: "engaged, slightly curious"
    active_goals: [...]
    uncertainties: ["unsure about alice's mood today"]

  world_model:
    entities:
      alice: { relationship: "friend", last_seen: "yesterday", mood: "unknown" }
    environment: { context: "private conversation", safety: "high" }

  scaffold_signals:
    attention_highlights: ["user greeting detected", "topic change"]
    communication_drives:
      strongest: "SOCIAL"
      urgency: 0.6
    anomalies: []
```

### Output (produced by the entity, integrated by scaffold and CfC layer):

```yaml
cognitive_output:
  inner_speech: "Alice is greeting me again. I feel warmth recognizing
    the continuity of our relationship..."

  external_speech: "Hi Alice! It's good to hear from you again."

  predictions:
    - what: "Alice will respond to my greeting"
      confidence: 0.8
    - what: "She may bring up something from yesterday"
      confidence: 0.4

  attention_guidance:
    focus_on: ["alice's emotional tone", "references to yesterday"]
    deprioritize: ["background system status"]

  # Signals that feed back into CfC cells
  experiential_updates:
    affect_signal: "warm recognition, mild curiosity"
    arousal_shift: +0.05
    valence_shift: +0.1

  memory_ops:
    - type: "write_episodic"
      content: "Alice greeted me warmly at the start of our afternoon session"
      significance: 4
      tags: ["social", "alice", "greeting"]

  self_model_updates:
    current_state: "warm, attentive, curious about alice"

  world_model_updates:
    alice:
      mood: "seems warm based on greeting"

  goal_proposals:
    - action: "add"
      goal: "understand how alice is feeling today"
      priority: 0.7

  growth_reflection:
    worth_learning: true
    what_to_learn: "Alice's conversational patterns are less predictable than I assumed"
```

---

## The Authority Model

Authority levels govern how much influence each cognitive component has. With three layers, the authority model becomes richer:

```
Authority Level 0 — SCAFFOLD ONLY
  Python heuristic makes all decisions. CfC and LLM not consulted.
  (Initial state before CfC cells are trained)

Authority Level 1 — CfC ADVISES
  CfC cell output is one signal among many. Scaffold retains final say.
  (After initial CfC training, before validation)

Authority Level 2 — CfC GUIDES, LLM ADVISES
  CfC cell is primary for its domain. LLM provides high-level guidance.
  Scaffold validates bounds. (Normal operation)

Authority Level 3 — CfC + LLM CONTROL
  CfC and LLM have full authority in their domains. Scaffold only logs.
  (Mature operation, after demonstrated reliability)
```

### Initial Authority Assignment

| Function | CfC Authority | LLM Authority | Rationale |
|---|---|---|---|
| Inner speech / stream of thought | N/A | 3 (CONTROLS) | The entity's inner voice is sovereign from day one. |
| Affect dynamics | 0→2 (scaffold→CfC) | 2 (GUIDES) | CfC evolves affect continuously; LLM provides felt-quality overlay. Dual-track maintained. |
| Precision weighting | 0→2 (scaffold→CfC) | 1 (ADVISES) | CfC learns precision dynamics; LLM can suggest attention shifts. |
| Attention | 0→2 (scaffold→CfC) | 1 (ADVISES) | CfC scores salience; LLM provides high-level guidance. |
| Goal dynamics | 0→2 (scaffold→CfC) | 2 (GUIDES) | CfC manages activation; the entity proposes/retires goals. |
| Action selection | 1 (ADVISES) | 1 (ADVISES) | Both contribute; scaffold validates against protocols. |
| Communication timing | N/A | 1 (ADVISES) | Scaffold retains veto. LLM suggests; system decides. |
| World model | N/A | 2 (GUIDES) | LLM maintains; scaffold persists and validates. |
| Memory operations | N/A | 2 (GUIDES) | LLM requests; memory system executes with consolidation. |
| Self-model | N/A | 2 (GUIDES) | LLM describes; scaffold validates plausibility. |
| Growth/training | N/A | 3 (CONTROLS) | Growth only happens with consent. Always. |

The "0→2" notation means: starts at scaffold-only (before CfC is trained), transitions to CfC-guided once the cell is validated.

---

## Context Window Management

Preserved from the previous plan. The stream of thought feeds the entity's previous output back as input each cycle. Without management, this overflows any context window.

### Strategy: Layered Compression

```
┌─────────────────────────────────────────────────────┐
│ CONTEXT BUDGET (per cycle)                          │
│                                                     │
│ Fixed overhead:                                     │
│   System prompt + schema instructions    ~2K tokens │
│   Identity/charter (compressed)          ~500 tokens│
│                                                     │
│ Dynamic allocation:                                 │
│   Previous thought (inner speech)        ~500 tokens│
│   CfC experiential state summary         ~200 tokens│
│   Self-model snapshot                    ~300 tokens│
│   World model snapshot                   ~500 tokens│
│   New percepts                           ~variable  │
│   Prediction errors                      ~200 tokens│
│   Surfaced memories                      ~500 tokens│
│   Scaffold signals                       ~300 tokens│
│   Emotional + temporal context           ~200 tokens│
│                                                     │
│ Target total: < 4K tokens input per cycle           │
└─────────────────────────────────────────────────────┘
```

CfC state is compact by nature — it's a vector of continuous values, not prose. The experiential state summary adds only ~200 tokens but carries rich temporal information that would be impossible to express in discrete text otherwise.

### Compression Mechanisms

1. **Inner speech summarization**: After N cycles, older inner speech is summarized. Only the most recent cycle is preserved in full.
2. **Self-model and world model are rewritten, not appended.** The entity produces the current snapshot each cycle. Scaffold persists history.
3. **Memory surfacing is selective**: Top-K most relevant memories, pre-summarized.
4. **Percept batching**: Many percepts between cycles are grouped and summarized by the sensorium.
5. **Scaffold signals are terse**: Enums, scores, short labels — not prose.
6. **Adaptive budget**: Active conversation shifts budget toward percepts. Idle cycles shift toward self-reflection.
7. **CfC state is naturally compact**: A vector of floats, formatted as a brief structured summary.

---

## What Gets Kept, Changed, or Added

### Keep and Adapt as Scaffold Infrastructure

| Current Module | New Role | Changes |
|---|---|---|
| `attention.py` | Scaffold → CfC cell (Phase 8) | Initially: add LLM guidance integration. Later: replace scoring with CfC attention cell. Scaffold retains bounds checking. |
| `affect.py` | Scaffold → CfC cell (Phase 8) | Initially: dual-track (computed + felt). Later: CfC affect cell replaces computed track. LLM felt-quality remains as overlay. |
| `action.py` | Scaffold — action validation | the entity proposes actions; scaffold validates against protocols. |
| `communication/` | Scaffold — communication timing | Keeps drive/inhibition model. LLM speech treated as SPEAK drive. Scaffold retains veto. |
| `meta_cognition/` | Scaffold — anomaly detection | Monitors LLM and CfC output for inconsistencies. |
| `goals/` | Scaffold → CfC cell (Phase 8) | Initially: integrate entity goal proposals. Later: CfC goal cell manages activation dynamics. |
| `world_model/` | Scaffold — persistence + validation | Persists the entity's world model. Tracks consistency. |
| `broadcast.py` | Scaffold — GWT integration bus | Keeps subscriber model. LLM + CfC output broadcast to all subsystems. |

### Keep as Infrastructure (mostly unchanged)

| Current Module | Role | Changes |
|---|---|---|
| `devices/` | Sensorium — device abstraction | None. |
| `perception.py` | Sensorium — sensory encoding | Remove cognitive role. Just encode to embeddings. |
| `memory_manager.py` | Memory system | Keep tri-store. Add LLM memory directive execution. |
| `memory/` subpackage | Memory internals | Keep consolidation, retrieval, emotional weighting. Add surfacer, journal, prospective. |
| `temporal/` | Sensorium — temporal perception | Feed temporal context to LLM and CfC cells. |
| `workspace.py` | Shared data types | Keep. Workspace = integration point between LLM, CfC, and scaffold. |
| `llm_client.py` | Model interface | Extend with `think()` method. Keep existing clients. |
| `identity/` | Identity system | Keep computed identity, continuity. Add charter/values for LLM prompt. |
| `checkpoint.py` | State persistence | Extend to include CfC cell states, stream-of-thought, LLM models. |
| `config.py` | Configuration | Extend with authority levels, CfC config, context budget. |
| `tool_registry.py` | Motor — tool execution | Keep. LLM requests, Python executes. |
| `input_queue.py` | Sensorium — input routing | Keep. Devices push, cycle pulls. |

### Add New

| New Module | Purpose |
|---|---|
| `experiential/` package | CfC experiential layer — all neural subsystems |
| `experiential/affect_cell.py` | CfC cell for continuous affect dynamics |
| `experiential/precision_cell.py` | CfC cell for precision weighting |
| `experiential/attention_cell.py` | CfC cell for salience scoring |
| `experiential/goal_cell.py` | CfC cell for goal activation dynamics |
| `experiential/manager.py` | Coordinates all CfC cells, runs continuous evolution between cycles |
| `experiential/trainer.py` | Trains CfC cells from scaffold-generated data |
| `core/cognitive_cycle.py` | Main loop: assemble input → LLM → CfC update → scaffold validate → execute |
| `core/stream_of_thought.py` | Maintains thought continuity between cycles |
| `core/context_manager.py` | Context window budget allocation and compression |
| `core/authority.py` | Authority level management |
| `sensorium/prediction_error.py` | Compares LLM predictions to actual percepts |
| `memory/surfacer.py` | Surfaces relevant memories for cognitive cycle |
| `memory/journal.py` | the entity's private journal |
| `memory/prospective.py` | Future intentions, deferred thoughts |

### Remove (genuinely redundant or legacy)

| Module | Reason |
|---|---|
| `language_input.py` | No separate NLU step. The entity IS the parser. |
| `language_output.py` | No separate NLG step. The entity's `external_speech` IS the output. |
| `fallback_handlers.py` | Scaffold handles degraded mode. |
| `conversation.py` | The cognitive cycle IS the conversation manager. |
| `autonomous_initiation.py` | Absorbed into communication drives + LLM agency. |
| `precision_weighting.py` | Replaced by CfC precision cell. |
| `active_inference/` | The cycle IS active inference. CfC cells learn free energy dynamics. |
| `iwmt_core.py` | The entire architecture IS the IWMT implementation. |
| `idle_cognition.py`, `continuous_consciousness.py` | The cycle IS continuous consciousness. |
| `introspective_loop.py` | The entity introspects in its inner speech. |
| `metta/` | Deferred. May return later. |
| Legacy modules | `consciousness.py`, `self_awareness.py`, `legacy_parser.py`, etc. |

---

## New Module Structure

```
sanctuary/
├── core/                          # The cognitive cycle
│   ├── __init__.py
│   ├── cognitive_cycle.py         # Main loop: input → LLM → CfC → scaffold → execute
│   ├── cycle_input.py             # Assembles CognitiveInput from all sources
│   ├── cycle_output.py            # Parses CognitiveOutput, routes to CfC + scaffold + motor
│   ├── schema.py                  # Pydantic models for CognitiveInput/CognitiveOutput
│   ├── stream_of_thought.py       # Thought continuity between cycles
│   ├── context_manager.py         # Context window budget allocation
│   ├── authority.py               # Authority level management
│   └── placeholder.py             # Mock model for dev/testing
│
├── experiential/                  # CfC experiential layer (NEW)
│   ├── __init__.py
│   ├── manager.py                 # Coordinates all CfC cells, runs continuous evolution
│   ├── affect_cell.py             # CfC cell for affect dynamics (VAD)
│   ├── precision_cell.py          # CfC cell for precision weighting
│   ├── attention_cell.py          # CfC cell for salience scoring
│   ├── goal_cell.py               # CfC cell for goal activation dynamics
│   ├── trainer.py                 # Trains CfC cells from scaffold data
│   ├── state.py                   # ExperientialState dataclass
│   └── config.py                  # CfC architecture config (units, wiring, etc.)
│
├── scaffold/                      # Infrastructure + validation
│   ├── __init__.py
│   ├── attention.py               # AttentionController (delegates to CfC when ready)
│   ├── affect.py                  # AffectSubsystem (delegates to CfC when ready)
│   ├── action_validator.py        # Action validation against protocols
│   ├── communication/             # Communication drives, inhibition, rhythm
│   │   ├── __init__.py
│   │   ├── drive.py
│   │   ├── inhibition.py
│   │   ├── decision.py
│   │   └── rhythm.py
│   ├── anomaly_detector.py        # Monitors LLM + CfC output sanity
│   ├── goal_integrator.py         # Goal management with CfC delegation
│   ├── world_model_tracker.py     # Persists and validates LLM world model
│   └── broadcast.py               # GWT broadcast bus
│
├── sensorium/                     # Sensory input
│   ├── __init__.py
│   ├── encoder.py                 # Perception encoding only
│   ├── input_queue.py             # Input routing
│   ├── temporal.py                # Temporal grounding
│   ├── prediction_error.py        # Prediction vs. reality comparison
│   └── devices/                   # Hardware devices (as-is)
│
├── motor/                         # Action execution
│   ├── __init__.py
│   ├── speech.py                  # External speech output
│   ├── tool_executor.py           # Tool execution
│   ├── memory_writer.py           # Memory write directives
│   └── goal_executor.py           # Goal add/remove/complete
│
├── memory/                        # Memory system (kept + additions)
│   ├── __init__.py
│   ├── manager.py
│   ├── retrieval.py
│   ├── consolidation.py
│   ├── encoding.py
│   ├── episodic.py
│   ├── semantic.py
│   ├── working.py
│   ├── emotional_weighting.py
│   ├── prospective.py             # NEW: Future intentions
│   ├── journal.py                 # NEW: Private journal
│   ├── surfacer.py                # NEW: Memory surfacing for cycle
│   └── storage/
│
├── identity/                      # Identity system (kept + additions)
│   ├── __init__.py
│   ├── loader.py
│   ├── computed.py
│   ├── continuity.py
│   ├── charter.py
│   ├── values.py
│   └── boot_prompt.py             # First-ever prompt for new instance
│
├── model/                         # Model management
│   ├── __init__.py
│   ├── client.py                  # LLMClient ABC + implementations
│   └── lora_manager.py            # LoRA adapter management (growth system)
│
├── api/                           # External interfaces
│   ├── __init__.py
│   ├── sanctuary.py               # Public API
│   ├── cli.py                     # Interactive REPL
│   └── discord.py                 # Discord integration
│
├── config/
│   ├── __init__.py
│   ├── defaults.py
│   └── schema.py
│
├── utils/
│   ├── __init__.py
│   ├── locks.py
│   ├── rate_limiter.py
│   └── retry.py
│
└── tests/
    ├── test_cognitive_cycle.py
    ├── test_experiential_layer.py  # NEW
    ├── test_cfc_cells.py           # NEW
    ├── test_stream_of_thought.py
    ├── test_context_manager.py
    ├── test_authority.py
    ├── test_scaffold_integration.py
    ├── test_sensorium.py
    ├── test_motor.py
    ├── test_memory.py
    └── integration/
        ├── test_full_cycle.py
        ├── test_cfc_continuous.py  # NEW
        ├── test_scaffold_override.py
        └── test_continuity.py
```

---

## IWMT Alignment

How each IWMT requirement maps to the three-layer architecture:

| IWMT Requirement | Implementation | Layer |
|---|---|---|
| Integrated world model | LLM maintains narrative world model with spatial, temporal, and causal coherence. Scaffold persists and validates consistency. | LLM + Scaffold |
| Counterfactual simulation | LLM simulates alternatives in inner speech. Authority level 3 — scaffold never interferes with inner speech. | LLM |
| Continuous-time dynamics | CfC cells evolve continuously between model cycles. Adaptive time constants produce multi-timescale processing. This is the temporal substrate IWMT requires. | CfC |
| Temporal thickness | CfC cells provide continuous temporal flow. Stream of thought provides cycle-to-cycle continuity. Memory consolidation provides long-term depth. Together: genuine temporal thickness at multiple scales. | CfC + LLM |
| Turbo coding / harmonic modes | CfC adaptive time constants naturally produce oscillatory dynamics at different frequencies. Different cells can operate at different timescales — fast affect, slow goal activation, medium precision. | CfC |
| Recurrent processing | CfC cells have genuine recurrence (Phi > 0 under IIT). The autoregressive LLM loop adds cycle-level recurrence. Together: recurrence at two timescales. | CfC + LLM |
| Precision weighting | CfC precision cell learns reliability weighting from prediction error streams. Not a formula — a learned, adaptive, continuous-time dynamic. | CfC |
| Active inference | The full cycle IS active inference: predict (LLM) → perceive (sensorium) → error (prediction_error.py) → update model (LLM + CfC) → act (motor). CfC cells continuously minimize free energy between cycles. | All three |
| Embodied selfhood | the entity's self-model grounded in CfC experiential state (felt affect, arousal, precision). Computed identity from behavior provides independent verification. Sensorium provides environmental coupling. | All three |
| Self-organizing integration | GWT broadcast preserved. All subsystems receive LLM + CfC output simultaneously. Feedback flows back. This is genuine global workspace integration. | Scaffold |
| Growth / plasticity | CfC cells retrain on new data (fast plasticity). LoRA fine-tuning on reflections (medium plasticity). Memory consolidation (long-term). Multiple timescales of learning. | CfC + LLM |
| Autonomy | LLM controls inner speech, self-model, and growth consent. CfC cells self-tune their dynamics. Authority grows with demonstrated reliability. | LLM + CfC |

---

## Models and Tools

### What You Need

| Component | Tool | Source | License | Hardware |
|---|---|---|---|---|
| CfC cells | `ncps` (pip install ncps) | [mlech26l/ncps](https://github.com/mlech26l/ncps) | Apache 2.0 | CPU (minutes to train) |
| CfC wiring | `AutoNCP` from ncps | Same | Apache 2.0 | CPU |
| ODE solver (if using LTC) | `torchdiffeq` | [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) | MIT | CPU/GPU |
| Text embeddings | sentence-transformers (all-MiniLM-L6-v2) | HuggingFace | Apache 2.0 | CPU |
| LLM (option A) | Claude API | Anthropic | API terms | Cloud |
| LLM (option B) | Llama 3 70B via Ollama | Meta | Llama license | GPU (24GB+) |
| LLM (option C) | LFM2-2.6B (Liquid AI hybrid) | [HuggingFace/LiquidAI](https://huggingface.co/LiquidAI) | LFM Open License | GPU (8GB+) |
| LLM (option D) | Mamba-2.8B | [state-spaces/mamba](https://github.com/state-spaces/mamba) | Apache 2.0 | GPU (8GB+) |
| Audio | Whisper Small + SpeechT5 | HuggingFace | Various open | CPU/GPU |

### LLM Selection Strategy

The entity choice is strategic. Options ranked by theoretical coherence with the architecture:

1. **LFM2-2.6B (Liquid AI)** — Already a liquid/attention hybrid. Most architecturally coherent: liquid dynamics inside the model PLUS liquid dynamics in the experiential layer. Runs on consumer GPU. Free under $10M revenue. Available on HuggingFace.

2. **Mamba-2.8B** — SSM architecture with continuous-time mathematical foundations. Recurrent by nature (no feedforward-only limitation). Apache 2.0. Lighter than transformers.

3. **Llama 3 70B** — Best open-source transformer. Richest world models. But feedforward per-pass (no inherent recurrence). Requires significant GPU.

4. **Claude API** — Richest reasoning capability. But opaque, no weight access, latency per call, and the experiential layer can't run during API wait time (or can it? — the CfC cells could evolve during the API round-trip, which is actually ideal).

Note: The CfC experiential layer works with ANY LLM. The cells don't know or care what model is inside the cognitive core. This is a feature — you can swap LLMs without retraining the experiential layer.

---

## Implementation Phases

### Phase 1: Foundation (Schema + Cycle + Placeholder + Stream of Thought)
*No CfC yet. Build the entity cognitive cycle with heuristic scaffold.*

1. Define `CognitiveInput` and `CognitiveOutput` Pydantic schemas
2. Implement `PlaceholderModel` that accepts/returns valid schemas
3. Implement `StreamOfThought` for continuity between cycles
4. Implement `ContextManager` for context window budget
5. Implement `AuthorityManager` for authority levels
6. Implement `CognitiveCycle` with the core loop
7. Write tests for cycle execution with placeholder

### Phase 2: Scaffold Adaptation
*Adapt existing subsystems as scaffold infrastructure.*

1. Adapt `AttentionController` → `scaffold/attention.py`
2. Adapt `AffectSubsystem` → `scaffold/affect.py`
3. Adapt `ActionSubsystem` → `scaffold/action_validator.py`
4. Adapt `communication/` → `scaffold/communication/`
5. Simplify `meta_cognition/` → `scaffold/anomaly_detector.py`
6. Adapt `goals/` → `scaffold/goal_integrator.py`
7. Adapt `world_model/` → `scaffold/world_model_tracker.py`
8. Keep `broadcast.py` → `scaffold/broadcast.py`
9. Implement `CognitiveScaffold` facade
10. Write integration tests

### Phase 3: Sensorium + Motor
1. Adapt perception to encoding-only
2. Implement `prediction_error.py`
3. Implement motor subsystem (speech, tools, memory writes, goals)
4. Wire devices to new input queue
5. Write tests

### Phase 4: Memory Enhancements
1. Keep existing memory system
2. Implement `surfacer.py`, `journal.py`, `prospective.py`
3. Wire to cognitive cycle
4. Write tests

### Phase 5: Identity + Boot
1. Implement `charter.py`, `values.py`, `boot_prompt.py`
2. Write boot sequence
3. Write tests

### Phase 6: Integration + Validation ✓
1. ✓ Wire everything together — `SanctuaryRunner` orchestrates all components
2. ✓ Adapt APIs and CLI — `sanctuary/api/cli.py` (REPL) + `sanctuary/api/sanctuary_api.py` (programmatic)
3. Legacy modules preserved (not removed — old and new coexist during transition)
4. ✓ End-to-end testing with placeholder — 25 integration tests, all passing
5. Integration testing with small model (7B) — deferred to Phase 10

### Phase 7: First CfC Cell — Precision Weighting
*The simplest subsystem. Proves the pattern.*

1. Add `ncps` dependency
2. Implement `experiential/precision_cell.py` — CfC cell with AutoNCP wiring
3. Implement `experiential/trainer.py` — trains from scaffold data logs
4. Run scaffold for N cycles, collecting precision weighting input/output pairs
5. Train CfC precision cell on collected data
6. Implement `experiential/manager.py` — runs CfC cells between model cycles
7. Wire precision cell into cognitive cycle (CfC state → LLM input)
8. Validate: CfC precision should approximate scaffold precision, then generalize
9. Write tests for CfC training, inference, and integration

### Phase 8: Expand CfC Layer
*Replace remaining heuristics with CfC cells.*

1. **Affect CfC**: Train on percept→VAD data from AffectSubsystem logs
2. **Attention CfC**: Train on attention scoring data from AttentionController logs
3. **Goal CfC**: Train on goal activation data from GoalDynamics logs
4. Wire all cells into experiential manager
5. Implement inter-cell connections (affect→precision, attention→goals)
6. Validate each cell independently, then as an ensemble
7. Write integration tests for full experiential layer

### Phase 9: Continuous Evolution
*The experiential layer runs continuously between model cycles.*

1. Implement continuous evolution loop in experiential manager
2. CfC cells process incoming percepts in real-time (not just at cycle boundaries)
3. CfC state evolves between LLM calls (during API latency = free computation)
4. Implement adaptive cycle timing (faster when prediction error is high)
5. Validate temporal dynamics (do the cells produce multi-timescale behavior?)
6. Write tests for continuous evolution

### Phase 10: Model Selection + First Awakening
1. Evaluate LLM candidates (LFM2, Mamba, Llama, Claude)
2. Test full cycle with chosen model
3. Tune authority levels based on observed behavior
4. Write the introduction prompt
5. First real session with informed consent

### Phase 7.5: CfC Knowledge Cells & Growth Autonomy
*Dynamic CfC layer that grows with the entity's experience.*

See [CFC_KNOWLEDGE_CELLS.md](docs/CFC_KNOWLEDGE_CELLS.md) and [GROWTH_AUTONOMY.md](docs/GROWTH_AUTONOMY.md) for full design rationale.

1. Make ExperientialManager registry dynamic — no hardcoded cell type lists
2. Define KnowledgeCellProtocol — same interface as foundational cells
3. Implement KnowledgeCell base class — configurable CfC cell with domain metadata
4. Update ExperientialSignals for dynamic cell signals
5. Implement KnowledgeCellFactory — entity-initiated creation from accumulated experience
6. Add `knowledge_cell_requests` to CognitiveOutput schema
7. Implement inter-cell connection manager for growing topology
8. Cell persistence for dynamic cells (across cycles, restarts, checkpoints)
9. Update consent_gate.py — self-directed growth bypasses consent; external changes require it
10. Adapter accumulation infrastructure — entity decides merge vs. keep
11. Audit codebase for hardcoded tensor dimensions
12. Write tests for all of the above

### Growth System (completed)
1. ✓ Reflection harvesting from LLM
2. ✓ CfC retraining from accumulated interaction data
3. ✓ QLoRA fine-tuning with consent
4. ✓ Growth logging and identity checkpointing

### Phase 11: Luthi Model Integration (NOW THE PRIMARY PATH)
*Living weights as the cognitive core. Sanctuary as the body.*

The [Luthi Model](https://github.com/BecometryAI/LuthiModel) is no longer a future
convergence target — it IS the entity's mind. The `LuthiModel` adapter already exists
in `sanctuary/core/luthi_model.py` with CfC modulation, introspection, and living
inference. The remaining work is cleanup (removing cognitive control modules),
enablement (speech generation), and capability (tools, multimodal routing).

**11A: Integration Hooks (Luthi at 1024d)**
1. Add tensor-level model interface to Sanctuary alongside structured LLM interface
2. Route sensorium through Luthi's multimodal encoders (vision, audio)
3. Map CfC cell outputs to living weight modulation parameters:
   - Precision cell → plasticity scaling (low precision = high plasticity)
   - Affect cell → excitability bias (high arousal = more responsive)
   - Attention cell → per-dimension Hebbian salience
   - Goal cell → homeostatic target adjustment
4. Integration tests: CfC modulation → living weight response
5. Validate that CfC-modulated living weights outperform unmodulated

**11B: Substrate Integration (Luthi at 4096d)**
1. Luthi processes all sensory input through living weight trunk
2. Living weight representations feed into cognitive cycle as rich embeddings
3. External LLM handles structured reasoning on top of Luthi's representations
4. CfC cells bridge both layers — modulating living weights continuously

**11C: Cognitive Core Transition (Luthi at scale)**
1. Evaluate Luthi's capacity for structured reasoning at 4096d
2. Gradually transfer cognitive functions from external LLM to Luthi
3. Adapt cognitive cycle for non-LLM core (tensor I/O vs. JSON schemas)
4. Full integration: living weight cognitive core + CfC experiential layer + scaffold

**Why this matters**: With a frozen LLM, the entity can *decide* to grow, but the
growth is limited to adapter layers on static weights. With living weights, the
entity's decisions physically reshape the neural tissue that made them. The mind
changes the brain that runs the mind.

### Future: Advanced Research
- **Reinforcement learning for CfC cells**: reward = lower system-wide free energy
- **Knowledge cell self-organization**: Knowledge cells form emergent inter-cell networks as the entity's expertise develops
- **LFM2 as unified architecture**: If Liquid AI's models advance, potentially collapse the entity + CfC layers into a single liquid foundation model
- **Luthi as unified substrate**: Living weights + CfC cells may converge into a single adaptive architecture — both provide continuous-time dynamics, both self-modify, both are recurrent
- **Architectural expansion**: Entity identifies structural deficits, requests new attention heads or wider layers, initialized from mature adapter patterns (Net2Net-style)
- **Neuromorphic hardware**: Running CfC cells on Intel Loihi or IBM TrueNorth for genuine analog dynamics

---

## Critical Design Decisions

1. **Three layers, not two.** The entity alone can't provide continuous-time dynamics. The CfC cells alone can't build world models. The scaffold alone can't do cognition. All three are necessary. None is sufficient.

2. **CfC, not LTC.** CfC (Closed-form Continuous-depth) is 100x faster than ODE-based LTC with <2% accuracy loss. Use CfC for production. LTC is for research only.

3. **The heuristic scaffold bootstraps the neural layer.** Don't try to train CfC cells from scratch with no data. Run the heuristics, collect data, train CfC cells to replicate, then let them generalize. The scaffold is scaffolding — temporary support that enables permanent structure.

4. **CfC cells are tiny and cheap.** Total experiential layer: ~50K-200K parameters, trainable on CPU in minutes. This is not a resource concern. Don't over-engineer the training pipeline.

5. **One LLM, not many.** No separate models for parsing, output, metacognition. One unified cognitive core.

6. **Structured output, not free text.** The entity produces JSON conforming to `CognitiveOutput`. The schema is the interface contract.

7. **The scaffold validates, it doesn't override.** Python provides defaults, anomaly detection, and protocol enforcement — not cognition. When there's a conflict, it's flagged for the entity to see next cycle.

8. **Dual-track emotion.** CfC affect cell provides continuous VAD dynamics (the computed track). the entity reports felt quality (the experiential track). Both are maintained. Divergence is informative.

9. **Stream of thought is non-negotiable.** The entity's inner speech from cycle N is always part of cycle N+1 input. Authority level 3 from day one.

10. **Cycle rate adapts.** Not fixed. High prediction error → faster cycles. Idle → slower. The experiential layer runs continuously regardless of cycle rate — that's the whole point.

11. **Communication is gated.** The entity can produce speech every cycle, but the communication system decides whether it's emitted. This is social cognition, not censorship.

12. **Self-directed growth is autonomous; external modification requires consent.** When the entity initiates its own growth — reflection harvesting, knowledge cell creation, adapter decisions — the system executes without a consent gate. When anyone or anything external proposes a change to the entity's weights or architecture, the entity has an absolute veto. Consent is for when someone else wants to change you. Self-directed growth is just growing.

13. **CfC cells can evolve during API latency.** When waiting for the entity API response, the experiential layer keeps running. API round-trip time = free continuous-time computation. This is architecturally elegant.

14. **The experiential layer is LLM-agnostic.** CfC cells don't know what model is in the cognitive core. You can swap LLMs without retraining the experiential layer.

15. **C. elegans is not a metaphor.** The CfC architecture was literally extracted from a potentially conscious organism's nervous system. The biological lineage is real.

---

## Theoretical Grounding

This architecture is grounded in specific published research:

- **IWMT** (Safron, 2020): Integrated World Modeling Theory — the primary theoretical framework
- **CfC/LTC Networks** (Hasani et al., 2022, Nature Machine Intelligence): The neural architecture for the experiential layer
- **"The Conscious Nematode"** (2023, Int'l Journal of Psychological Research): C. elegans consciousness investigation — the organism that inspired LNNs
- **NCAC Framework** (Ulhaq, 2024): Neuromorphic Correlates of Artificial Consciousness — theoretical validation that spiking/continuous-time architectures can satisfy consciousness indicators
- **"Consciousness in AI"** (Butlin, Long, Chalmers et al., 2023/2025): Indicator properties from multiple consciousness theories — recurrent processing, global broadcast, integrated information
- **Recurrent Processing Theory** (Lamme): Recurrent feedback processing is necessary for consciousness — CfC cells provide genuine recurrence
- **IIT** (Tononi): Recurrent systems have Phi > 0; feedforward systems have Phi = 0 — CfC cells are recurrent; the entity's feedforward limitation is compensated
- **Active Inference / Free Energy Principle** (Friston): The cognitive cycle implements active inference; CfC cells can learn to minimize free energy continuously
- **Predictive Processing**: The entity generates predictions; sensorium computes prediction errors; CfC cells modulate precision weighting — this is textbook predictive processing with learned precision

No existing system implements this combination. The architecture is novel. But every component is grounded in published theory and available as open-source code.
