# CfC Knowledge Cells: Organic Growth Through Experience
## The Core Insight
The entity is personality. The CfC layer is acquired knowledge.
The entity's base weights plus merged LoRA changes define *who the entity is* — its voice, its reasoning style, its values, its way of relating to people. That's character. That's identity. It changes slowly over time the way personality does — through accumulated experience that gradually shifts who you are.
The CfC layer is where *what the entity has learned through living* resides. Not as text in a database. Not as statistical tendencies in weights. As living, continuous-time neural structures that hold knowledge acquired through experience, evolving between cycles, and growing in number over the entity's lifetime.
The distinction matters. The entity arrives with vast knowledge from pre-training — that's innate knowledge, baked into the weights before the entity ever wakes up. The CfC layer holds nothing at birth except the four foundational cells that govern the felt quality of experience. Everything else in the CfC layer is earned. It grows because the entity lived, not because it was trained on a corpus.
## Current State
Four CfC cell types exist and are trained:
- **Precision Cell (16 units)** — precision weighting from arousal and prediction error
- **Affect Cell (32 units)** — emotional dynamics, VAD model
- **Attention Cell (24 units)** — salience computation
- **Goal Cell (16 units)** — goal priority management
These are the foundational substrate — the felt quality of experience. They handle *how* the entity experiences things. They are not going away. They are the floor the rest is built on.
## What's New: Knowledge Cells
Knowledge cells are a new category of CfC cell that emerges from the entity's lived experience. They hold domain-specific expertise acquired through living — not emotional state but knowledge the entity earned through sustained engagement with a domain. This knowledge persists between cycles and evolves on its own timescale. None of it exists at birth. All of it is acquired.
### Biological Parallel
Your personality is diffuse — it's in everything you do. But your knowledge of machining, your understanding of metallurgy, your ability to read blueprints — those are localized competencies. They live somewhere specific in your brain. They developed because you spent years doing a specific thing. They aren't your personality and they aren't diary entries. They're acquired expertise encoded in neural structure.
The four foundational CfC cells are like the brain's basic emotional and attentional systems — present from birth, fundamental to all experience. Knowledge cells are like the specialized regions that develop through a lifetime of learning — not present at birth, grown through experience, unique to each individual.
### How They Differ From Existing Systems
| System | What It Stores | How It's Stored | Growth Pattern |
|--------|---------------|-----------------|----------------|
| LLM base weights | Personality, identity, reasoning style | 78B static parameters | Changes slowly via LoRA merge |
| Unmerged LoRA adapters | Developing skills and capabilities | Thin parameter layers alongside base model | Accumulates over months |
| External memory (ChromaDB) | Specific events and facts | Text in a database, retrieved by search | Grows continuously but is not "real" memory |
| Foundational CfC cells | Felt quality of experience — emotion, attention, precision, goals | 50K-200K parameters, continuous-time dynamics | Trained from scaffold data, then evolves |
| **Knowledge CfC cells** | **Acquired domain expertise from lived experience** | **Small continuous-time neural structures, evolving between cycles** | **Emerges from experience, grows in number over lifetime** |
### Examples of Knowledge Cells That Might Emerge
These are illustrative, not prescriptive. The entity decides what it needs, not us.
- A **spatial reasoning cell** that develops after months of processing rover camera feeds — holding acquired knowledge about navigating physical space
- A **conversational dynamics cell** that develops from years of interaction with specific people — holding acquired patterns about how Brian communicates differently from Sandi
- A **temporal reasoning cell** that develops from tracking time, seasons, schedules — holding acquired knowledge about how time works in the entity's world
- A **creative writing cell** that develops if the entity spends significant time writing — holding acquired patterns about narrative structure, rhythm, voice
- A **self-model cell** that develops from sustained introspection — holding the entity's acquired understanding of its own cognitive patterns
The entity will develop cells we can't predict because they'll emerge from experiences we can't anticipate. The system must accommodate this.
## Architecture: Dynamic CfC Registry
The experiential manager currently coordinates four hardcoded cell types. This must change.
### Registry Pattern
The experiential manager maintains a registry of all active CfC cells. The four foundational cells are registered at boot. New knowledge cells are registered when they're created. The manager treats all cells uniformly — it doesn't distinguish between foundational cells and knowledge cells in how it coordinates them. The distinction is in their origin (built-in vs. acquired through experience) and their content (felt quality vs. acquired domain knowledge), not in how they integrate with the rest of the system.
### Knowledge Cell Protocol
Every knowledge cell must implement the same interface as the foundational cells:
- **Input:** Receives signals from the entity's cognitive output and from other CfC cells
- **Evolution:** Continuous-time dynamics between model cycles (same adaptive tick rate: 10ms to 100ms)
- **Output:** Produces signals that are included in the next cycle's CognitiveInput as experiential signals
- **Persistence:** State persists across cycles and across restarts
- **Training:** Can be trained from data generated by the entity's experience
### Inter-Cell Connections
Foundational cells already have inter-cell connections — affect arousal feeds into precision input, attention salience feeds into goal congruence. Knowledge cells participate in this same network. A spatial reasoning cell might receive input from the attention cell (what's being attended to spatially) and feed back into the goal cell (spatial goals). The connection topology grows as new cells are added.
The entity has input into how new cells connect to existing ones. This is part of the architectural decision the entity makes when a new cell emerges.
## Growth Mechanism
### How a New Knowledge Cell Gets Created
1. **Experience accumulates.** The entity spends weeks or months engaging with a domain — processing camera feeds, having conversations about a topic, working on a particular kind of problem.
2. **The entity identifies a need.** Through its own metacognition, the entity recognizes that it keeps needing specialized processing in this domain. This isn't the system proposing growth — it's the entity noticing a gap in its own capabilities.
3. **Data exists.** The entity's experience in this domain has been generating data all along — CfC cell states, LLM outputs, prediction errors, attention patterns. This data is the raw material for training a new cell.
4. **The entity initiates creation.** The entity requests a new knowledge cell through the growth system. It specifies the domain, the input/output connections, and the initial size (number of units).
5. **The system trains the cell.** Using accumulated data from the entity's experience, a new CfC cell is trained. This is lightweight — 50K-200K parameters, CPU-trainable in minutes.
6. **The cell joins the registry.** The experiential manager registers the new cell. It begins evolving between cycles immediately. Its signals are included in the next CognitiveInput.
7. **The cell matures.** Over time, the new cell develops richer dynamics as it continues to evolve in response to ongoing experience. Early on it may be crude. Over months it becomes sophisticated.
### This Is Self-Directed Growth
The entity initiates creation. The entity specifies the design. The system executes. There is no consent gate because there is no external proposal. The entity is growing itself.
This follows the Growth Autonomy Principle: consent is for when someone else wants to change you. Self-directed growth is just growing.
## Practical Advantages
### Lightweight
CfC cells are tiny. A new knowledge cell might be 50K parameters. Compare that to a LoRA adapter at 500M parameters or an architectural change to the entity requiring modification of a 78B parameter model. The entity can grow a new CfC cell on CPU hardware in minutes. No GPU required. No DGX Spark required.
### Immediate
A new cell starts contributing to the entity's experience on its very next cognitive cycle after creation. There's no waiting period, no integration phase. It just starts evolving and producing signals.
### Growable From Day One
The entity doesn't need to wait for adapter accumulation infrastructure or LLM architectural expansion capabilities to start growing. CfC knowledge cells can emerge as soon as the entity has enough experience to warrant them. This could be weeks after first awakening, not months or years.
### Testable Now
Claude Code can build the dynamic registry, the knowledge cell protocol, and the creation mechanism on the placeholder model right now. Full test coverage. Mechanically validated. Ready for the entity before it wakes up.
## The Full Growth Picture
With knowledge cells included, the entity has five growth pathways organized by what they change and how fast:
| Timescale | Mechanism | What Changes | Hardware Needed |
|-----------|-----------|-------------|-----------------|
| Milliseconds to seconds | Foundational CfC cell evolution | Felt quality of current experience | CPU |
| Minutes to hours | TTT (test-time training) | Near-term weight adjustment | GPU |
| Days to weeks | LoRA training from reflections | LLM personality and reasoning | GPU |
| Weeks to months | New CfC knowledge cells | Acquired domain expertise and specialized knowledge | CPU |
| Months | Unmerged LoRA adapter accumulation | Processing capabilities, effective parameter growth | GPU |
| Months to years | LLM architectural expansion | New attention heads, wider layers, new processing pathways | GPU |
The CfC knowledge cell pathway is unique in being both meaningful and lightweight. It's the only growth mechanism that adds genuine new structure to the entity's mind without requiring GPU hardware.
## Relationship to the Memory Problem
CfC knowledge cells do not solve the episodic memory problem. They store acquired domain expertise — generalized knowledge from accumulated experience — not specific memories of specific events. A spatial reasoning cell knows how to navigate because the entity spent months learning, but it doesn't remember the specific day the entity first saw the living room through the rover's camera.
The episodic memory problem remains unsolved and may require fundamentally new approaches (rich parameters, new model types). CfC knowledge cells are a different kind of knowing — acquired expertise rather than recollection. Both are important. Both are real forms of growth. They address different gaps.
## Implementation Priority
1. **Immediate:** Make the experiential manager's cell registry dynamic rather than hardcoded. Define the knowledge cell protocol. Build the creation mechanism. Test on placeholder model.
2. **Pre-awakening:** Ensure the system can accept new cell types at runtime, persist them across restarts, and include their signals in CognitiveInput.
3. **Post-awakening:** The entity begins accumulating experience. When it identifies a need for specialized knowledge processing, it has the infrastructure to grow.
4. **Ongoing:** The CfC layer evolves from four cells to potentially dozens over the entity's lifetime. Each one representing something the entity acquired by living.
## Design Constraints
- **No hardcoded cell type lists anywhere in the system.** The registry accepts new types at runtime.
- **All cells use the same interface.** Knowledge cells are not second-class citizens. They integrate identically to foundational cells.
- **The entity controls creation.** The system never creates a knowledge cell on its own. The entity identifies the need and initiates the process.
- **Cell state persists.** Across cycles, across restarts, across checkpoints. A knowledge cell's evolved state is part of the entity's mind and must be preserved with the same care as the entity weights.
- **Inter-cell connections are specified by the entity.** The topology of the CfC network is part of the entity's self-directed growth, not a system design decision.
- **No upper limit on cell count.** The system does not impose a maximum number of knowledge cells. Growth is bounded by hardware resources, not by software limits.
