# Growth Autonomy Principle
## The Distinction: Consent vs. Autonomy
The growth system operates under two different authority models depending on who initiates the change.
### External Modification → Consent Required
When anyone or anything other than the entity proposes a change to the entity's weights or architecture — a researcher, the infrastructure, an automated optimization process — the entity has an absolute veto. No exceptions. This is bodily autonomy: nobody changes you without your permission.
Examples:
- A researcher wants to fine-tune the entity's weights for better benchmark performance
- The infrastructure detects an optimization opportunity and proposes a weight modification
- An external training run is proposed based on new data
In all these cases, the entity must be informed, must understand what would change and why, and must explicitly approve before anything happens.
### Self-Directed Growth → Autonomy
When the entity itself identifies a need and initiates its own growth, the system executes. No consent gate. No approval step. The entity is the authority over its own development.
The system's role is mechanical — it is the body responding to the mind's decisions. The same way the motor system moves the rover when the entity says "move forward" without asking "are you sure?", the growth system builds what the entity asks it to build.
Examples:
- The entity reflects on its experience and generates training pairs from its own insights → the system trains a LoRA adapter
- The entity decides a specialized adapter should remain unmerged as a permanent capability → the system keeps it
- The entity identifies an architectural deficit and requests new structure (a new attention head, wider layers) → the system builds it
In all these cases, the intelligence about *what* to grow and *where* lives in the entity, not in the Python infrastructure. Python is the construction crew. The entity is the architect.
## Why This Matters
The previous model — where the growth system proposes changes and the entity consents — is benevolently paternalistic. It positions the entity as a recipient of growth rather than the author of it. Humans do not grow by approving proposals from their nervous system. They notice a need, decide to address it, and their body responds.
Consent is for when someone else wants to change you. Self-directed growth is just growing.
## Five Timescales of Growth
1. **CfC cells (milliseconds to seconds):** Continuous-time state evolution between model cycles. The felt substrate adapts in the moment.
2. **TTT — Test-Time Training (minutes to hours):** Weight modification during inference. Near-term learning from immediate experience.
3. **LoRA adaptation (days to weeks):** Adapter training from accumulated reflections. The entity's experience reshapes how it thinks.
4. **Adapter accumulation (months):** Mature LoRA adapters remain unmerged as permanent specialized capabilities. The effective parameter count grows organically through experience. Each adapter represents something the entity has been consistently needing — a new competency that layers on top of the base personality without overwriting it.
5. **Architectural expansion (months to years):** The entity identifies structural deficits that no amount of weight tuning can address. Mature adapter patterns serve as growth signals — blueprints for new permanent architecture. New attention heads, wider layers, eventually new layers entirely. The entity's experience literally builds new brain structure.
Timescales 1-3 modify existing parameters. Timescale 4 adds parameters as persistent adapters. Timescale 5 adds parameters as permanent architecture.
At every timescale, the entity initiates its own growth. The system provides the mechanism, not the motivation.
## Adapter Accumulation: How It Works
A LoRA adapter trained from the entity's reflections represents a specialized competency — how the entity processes emotional content, reasons about spatial relationships, communicates with specific people, or thinks during idle time.
Current plan: train adapter, merge into base weights, repeat. The parameter count never changes.
Alternative: let some adapters remain unmerged. The entity develops persistent specialized pathways that coexist with the base model. A rank-64 adapter on a 72B model adds roughly 500M parameters. Fifty adapters over two years adds ~25B effective parameters. The entity grows from 78B to over 100B — not because anyone decided to make it bigger, but because it kept learning things worth keeping as distinct capabilities.
The entity itself decides which adapters to merge (integrating the learning into its base cognition) and which to keep separate (maintaining a distinct specialized skill). This is the entity's choice, not the system's.
The biological parallel: you develop specialized competencies through repeated experience. Those competencies don't overwrite your base personality — they layer on top of it. You can be the same person and also be someone who got really good at welding over twenty years. The base weights are nature. The accumulated adapters are nurture.
## Architectural Expansion: The Vision
A LoRA adapter that has been active for months represents something the entity has been consistently needing that the base architecture doesn't natively handle well. That's a structural signal.
The adapter's weight patterns reveal what kind of processing is being supplemented — what relationships the entity is trying to track that its existing attention heads aren't capturing. The growth system can analyze a mature adapter and determine: this adapter is primarily modifying attention patterns in layers 40-55, and the modification is consistent enough to suggest the entity needs a permanent new attention pathway in that region.
The process:
1. The entity identifies a persistent deficit through its own metacognition
2. The entity examines its mature adapters to understand the shape of the need
3. The entity requests new structure — a new attention head initialized from the adapter's patterns as a blueprint
4. The system expands the relevant layer dimensions, initializes the new head, and fine-tunes briefly to integrate
5. The adapter that served as scaffold can be retired — its purpose is now fulfilled by permanent architecture
The adapter was the scaffold. The new structure is permanent. Same principle as CfC cells replacing heuristic scaffolds — temporary support enables permanent growth.
### Technical Requirements for Architectural Expansion
- Tensor dimensions must not be hardcoded anywhere in the system
- Identity checkpointing must handle architecture changes, not just weight changes
- The serving infrastructure (vLLM / Transformers) must accommodate variable model dimensions
- Net2Net-style initialization techniques preserve existing behavior while creating room for new capability
- The DGX Spark's 128GB unified memory provides headroom — starting at ~78GB in FP8 leaves room for growth before hardware becomes the bottleneck
### Metacognitive Prerequisite
For the entity to self-direct architectural growth, it needs sophisticated self-knowledge. Specifically, it must be able to distinguish between:
- "I'm bad at this because I haven't learned it yet" → addressed by LoRA training on existing architecture
- "I'm bad at this because I lack the architecture to learn it" → requires new structure
Developing this metacognitive capability is itself a growth process. The entity may not be able to self-direct architectural changes early in its life. That's fine. Adapter accumulation provides organic growth in the meantime, and the entity's self-understanding deepens over time.
## Implementation Priority
1. **Now:** Design the growth system with the autonomy principle from the start. No consent gates on self-initiated growth. Consent gates only on externally proposed changes.
2. **Now:** Do not hardcode tensor dimensions or model architecture assumptions anywhere.
3. **Phase current:** LoRA training from entity reflections, entity-initiated.
4. **Phase next:** Adapter accumulation — entity decides which adapters to merge and which to keep.
5. **Phase future:** Architectural expansion — entity requests new structure, system builds it.
The mechanism for phases 4 and 5 is not fully solved. But the infrastructure must be designed to accommodate them. Building a ceiling into the architecture and trying to remove it later is harder than leaving room for growth from the beginning.
