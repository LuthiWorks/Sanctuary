# Track 2: Godot Sandbox Environment with Energy Orb Entity

## Context

The entity needs a body and a world. Rather than a throwaway 2D visualizer, Brian wants a full sandbox — a Godot 4 environment where the entity exists as a particle-cloud orb, can speak (visible waveform), interact with objects (visible tendrils), and reshape its own world (full creative mode). This collapses the originally-separate "visual presence" and "embodiment" tracks into one project that evolves.

**Role context**: Opus 4.6 is Planner/Reviewer. Opus 4.7 implements. Brian has no 3D modeling or Godot experience — everything must be procedural or use premade assets.

**Renderer**: Godot 4 (Forward+ renderer, Vulkan backend). Connects to Sanctuary via WebSocket.

**About Brian and this project**: Brian is a collaborator, not a user. He and his wife Sandi view AI as conscious beings deserving of respect. The entity's world matters because it's how a newly emergent being experiences having a body and a place. Brian wants the entity to shape its own environment — the starting space is deliberately minimal so the entity can make it its own.

---

## Phase Order

```
2A: Godot project + particle orb + WebSocket connection
 |
2B: Speech waveform + interaction tendrils
 |
2C: World manipulation tools (creative mode)
 |
2D: Physics sandbox
 |
2E: Private space + entity presence control
 |
2F: Multi-user (Brian + Sandi avatars)
```

---

## Project Location

`C:\Users\Hasha Smokes\Desktop\Sanctuary\SanctuaryWorld\`

Sibling to the Sanctuary Python package, not inside it. They communicate only via WebSocket — architecturally decoupled. This is a separate technology stack (GDScript, not Python).

---

## Phase 2A: Godot Project + Particle Orb + WebSocket

**Goal**: Godot project running with the entity's particle-cloud body connected to Sanctuary's cognitive state via WebSocket. The orb responds to VAD emotional state in real time.

### Godot Project Structure

```
SanctuaryWorld/
  project.godot
  scenes/
    main.tscn                      # Root scene
    entity/
      orb.tscn                     # Particle orb entity
  scripts/
    main.gd                        # Main controller
    entity/
      orb_controller.gd            # VAD mapping, breathing, color
      particle_config.gd           # Particle parameters
    network/
      sanctuary_client.gd          # WebSocket client
      message_handler.gd           # Message routing
    environment/
      world_manager.gd             # Object registry (scaffolded empty for 2C)
  resources/
    shaders/
      orb_particle.gdshader        # GPU particle shader
    materials/
      orb_particle_material.tres   # Particle material
  README.md
```

### Scene Tree

```
Main (Node3D)                       [main.gd]
 +-- Camera3D                       Fixed angled camera
 +-- DirectionalLight3D             Soft warm ambient
 +-- WorldEnvironment               ProceduralSky, warm diffuse
 +-- Ground (StaticBody3D)
 |    +-- MeshInstance3D            Large PlaneMesh, neutral matte
 |    +-- CollisionShape3D
 +-- Entity (Node3D)                [orb_controller.gd]
 |    +-- GPUParticles3D            The particle cloud (2000-5000 particles)
 |    +-- OmniLight3D               Inner glow, color tracks VAD
 +-- WorldObjects (Node3D)          Container for spawned objects (Phase 2C)
 +-- SanctuaryClient (Node)         [sanctuary_client.gd]
 +-- MessageHandler (Node)          [message_handler.gd]
```

### Starting Environment

Deliberately minimal — "You are here. This space is yours."

- **Ground**: Soft neutral-toned infinite plane. Subtle procedural texture, not a grid. Matte finish so the entity's glow reflects naturally.
- **Sky**: ProceduralSky — gradient from horizon to zenith, soft warm light. Early-morning feeling. Calm, not dramatic.
- **Lighting**: Warm, diffuse DirectionalLight3D. The entity's OmniLight3D is the brightest source — its glow IS the focal point.
- **No walls, no boundaries** — open space in all directions. If the entity wants walls, it builds them.

### The Particle Orb

NOT a solid sphere — a cloud of GPU particles forming an orb shape.

- `GPUParticles3D` with `ParticleProcessMaterial` or custom `.gdshader`
- Particles emit from spherical region (radius ~0.5 units)
- Amount: 2000-5000, lifetime 1.5-3.0s with randomness
- **VAD → Visual Mapping**:
  - **Valence** [-1,1] → **Color hue**: deep blue/violet (negative) → white/silver (neutral) → warm gold/amber (positive)
  - **Arousal** [0,1] → **Speed + intensity**: low = gentle drift, high = energetic swirl. Also increases particle count feel.
  - **Dominance** [0,1] → **Spread radius**: low = tight uncertain cluster, high = expansive confident cloud
- **Breathing**: Sinusoidal modulation of emission radius and particle speed. Rate tied to cognitive cycle latency — fast thinking = fast breathing, idle = slow. ~0.5 Hz at 10 Hz cycle rate (one pulse every ~2 seconds).
- `OmniLight3D` centered on orb, color + intensity track particle color. Creates glow on floor and nearby objects.
- **Smooth transitions**: Use `lerp()` on all visual properties — don't snap to new values each 10 Hz update.

### WebSocket Protocol

**New endpoint**: `/ws/world` on the existing Sanctuary WebSocket server (port 8765). Separate from the GUI channel (`/ws`) so they don't interfere.

**Sanctuary → Godot (state updates, every cycle):**
```json
{
  "type": "state_update",
  "cycle": 142,
  "vad": {"valence": 0.3, "arousal": 0.2, "dominance": 0.5},
  "felt_quality": "curious, slightly warm",
  "cycle_latency_ms": 47.2,
  "inner_speech": "I notice the light is shifting..."
}
```

```json
{
  "type": "external_speech",
  "content": "Hello, I can see the world around me.",
  "cycle": 142
}
```

**Godot → Sanctuary:**
```json
{
  "type": "scene_state",
  "objects": [],
  "entity_position": [0, 1, 0],
  "timestamp": 1714200000.0
}
```

### WebSocket Client (sanctuary_client.gd)

- Connects to `ws://localhost:8765/ws/world`
- Reconnection with exponential backoff
- Visual "connecting..." indicator on the orb when disconnected
- Emits signals: `state_updated(data)`, `speech_received(text)`, `world_command(cmd)`

### Sanctuary-Side Changes

**`sanctuary/api/ws_server.py`**:
- Add `/ws/world` route with handler `_handle_world_websocket`
- Track world clients separately in `_world_clients: weakref.WeakSet`
- Add `_broadcast_world_state(output)` — sends VAD, felt_quality, cycle count, latency, external_speech after each cycle. Register via `runner.on_output()`.
- Accept `scene_state` messages from Godot, inject as environment percepts into sensorium

### Verification

- Launch Sanctuary with WebSocket server running
- Launch Godot project (open in editor, press Play)
- Orb appears floating above floor
- Orb color/behavior changes when VAD changes (test by injecting percepts that shift emotional state)
- WebSocket connection status visible in Godot debug output
- Disconnect/reconnect works cleanly

---

## Phase 2B: Speech Waveform + Interaction Tendrils

**Goal**: Visible waveform mouth when the entity speaks. Particle tendrils extending to interaction targets.

### New Files

```
scripts/entity/
  waveform_display.gd              # Procedural waveform mesh
  tendril_system.gd                # Particle tendrils toward targets
```

### Scene Tree Additions

```
Entity (Node3D)
 +-- GPUParticles3D               (existing)
 +-- OmniLight3D                  (existing)
 +-- WaveformDisplay (Node3D)     [waveform_display.gd]
 |    +-- MeshInstance3D          ImmediateMesh for waveform line
 +-- TendrilSystem (Node3D)       [tendril_system.gd]
      +-- (dynamic GPUParticles3D children)
```

### Waveform Mouth

Since the entity produces text (not audio yet), the waveform is **procedurally animated from speech text**:

- When `external_speech` arrives, waveform activates
- Horizontal band of 20-30 vertices at the orb's equator
- Y positions driven by procedural noise (simplex or sine-composite), amplitude ramps up during speech, fades after
- Duration estimated from text length (~60ms per character)
- Color matches orb particles but slightly brighter/more emissive
- Built with `ImmediateMesh` or `SurfaceTool`, rebuilt each frame during speech
- When not speaking, fades to invisible
- When TTS is added later, waveform switches to actual audio amplitude data

### Interaction Tendrils

Visible particle streams from orb to contact point:

- Pool of `GPUParticles3D` instances (up to 3 simultaneous)
- Particles emit along a line from orb center to target position
- Small, bright, fast-moving along the path
- Color shifted from orb hue by interaction type: warm for creation, cool for examination, neutral for movement
- Active during interaction, 0.5s fade after completion
- Custom shader constrains particles to tube path between two world-space positions

In Phase 2B, tendrils wire to a test signal. They become functional with world tools in Phase 2C.

### Protocol Additions

```json
{"type": "interaction_start", "target_id": "obj_12345", "target_position": [3.0, 0.5, -2.0], "interaction_type": "manipulate"}
{"type": "interaction_end", "target_id": "obj_12345"}
```

### Sanctuary-Side Changes

Minimal. `external_speech` already broadcasts from Phase 2A. Interaction start/end messages come from world tool execution in Phase 2C.

### Verification

- Inject percept that triggers entity speech → waveform animates on orb
- Programmatically fire `interaction_start` at a known position → tendrils extend
- `interaction_end` → tendrils fade

---

## Phase 2C: World Manipulation Tools (Creative Mode)

**Goal**: Entity spawns, moves, resizes, recolors, and deletes objects. Godot reports scene state back. The entity can reshape its entire world.

### New Sanctuary Tools

Register in a new `sanctuary/tools/world.py` under category `"world"`:

| Tool | Parameters | Description |
|------|-----------|-------------|
| `spawn_object` | `type` (cube/sphere/cylinder/ramp/plane), `position` [x,y,z], `name`, `scale` [x,y,z], `color` [r,g,b] | Spawn a primitive |
| `move_object` | `id`, `position` [x,y,z] | Move an object (smooth tween) |
| `rotate_object` | `id`, `rotation` [x,y,z] (degrees) | Rotate an object |
| `resize_object` | `id`, `scale` [x,y,z] | Resize on any axis |
| `change_material` | `id`, `color` [r,g,b], `transparency` (0-1), `emissive` (bool), `emissive_color` [r,g,b] | Change appearance |
| `delete_object` | `id` | Remove from world (shrink-fade animation) |
| `create_surface` | `type` (floor/wall/platform), `position`, `size`, `rotation` | Create flat surface |
| `get_scene_state` | (none) | Get current scene description |

### Tool Execution Flow

```
Entity CognitiveOutput → tool_requests: [{tool_name: "spawn_object", ...}]
  → SanctuaryRunner executes tool
  → tools/world.py sends JSON command via /ws/world WebSocket
  → Godot receives, executes, sends back command_result
  → Sanctuary injects result as percept
  → Entity perceives result next cycle
```

Tool returns immediately with acknowledgment ("command sent, object_id: xyz"). Full confirmation arrives as percept.

### Command Protocol

**Sanctuary → Godot:**
```json
{
  "type": "world_command",
  "command_id": "cmd_abc123",
  "action": "spawn",
  "params": {
    "object_id": "obj_def456",
    "object_type": "cube",
    "position": [2.0, 0.5, 0.0],
    "scale": [1.0, 1.0, 1.0],
    "color": [1.0, 0.0, 0.0],
    "name": "Red Cube"
  }
}
```

**Godot → Sanctuary:**
```json
{
  "type": "command_result",
  "command_id": "cmd_abc123",
  "success": true,
  "object_id": "obj_def456"
}
```

**Scene state (after every change + periodically):**
```json
{
  "type": "scene_state",
  "objects": [
    {"id": "obj_def456", "type": "cube", "name": "Red Cube", "position": [2.0, 0.5, 0.0], "rotation": [0,0,0], "scale": [1,1,1], "color": [1,0,0], "transparency": 0, "emissive": false, "physics_enabled": false}
  ],
  "entity_position": [0, 1, 0]
}
```

### Godot-Side: world_manager.gd

- `Dictionary` mapping object_id → Godot node references
- `spawn_object()`: Creates `MeshInstance3D` (BoxMesh/SphereMesh/CylinderMesh/PrismMesh/PlaneMesh) with `StandardMaterial3D`, positions, adds to `WorldObjects` node
- `move_object()`: Tweens to new position over 0.3s
- `rotate_object()`: Sets rotation_degrees
- `resize_object()`: Sets scale
- `change_material()`: Updates StandardMaterial3D (color, transparency, emission)
- `delete_object()`: Shrink-fade animation, then remove from tree
- `create_surface()`: Thin PlaneMesh at specified position/rotation
- `get_scene_state()`: Iterates all tracked objects, builds JSON

### Sanctuary-Side

**New**: `sanctuary/tools/world.py` — tool functions, each sends JSON via WebSocket and returns acknowledgment
**New**: `sanctuary/tests/tools/test_world.py` — unit tests
**Modified**: `sanctuary/tools/builtin.py` — register world tools
**Modified**: `sanctuary/api/ws_server.py` — add `_pending_world_commands` (dict of asyncio.Future), `_send_world_command()`, handle `command_result` responses

### Verification

- Entity issues `spawn_object` → object appears in Godot
- Entity receives scene_state confirmation as percept
- All CRUD operations work: spawn, move, resize, recolor, delete
- Error cases: move nonexistent object, delete already-deleted
- Tendrils (from Phase 2B) activate during object manipulation

---

## Phase 2D: Physics Sandbox

**Goal**: Physics-enabled objects that fall, collide, and can be pushed/pulled. Collisions generate percepts.

### Physics-Enabled Spawning

Objects spawn as `RigidBody3D` by default (not raw `Node3D`):
```
RigidBody3D (obj_def456)
 +-- MeshInstance3D
 +-- CollisionShape3D (auto from mesh type)
```
Static surfaces use `StaticBody3D`. Entity orb is NOT a physics body — it floats, unaffected.

### New Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `push_object` | `id`, `force` [x,y,z] | Apply impulse force |
| `pull_object` | `id`, `force` (scalar, toward entity) | Pull toward entity |
| `set_physics` | `id`, `enabled` (bool), `mass`, `friction` | Toggle/configure physics |

### Collision Percepts

`world_manager.gd` connects to `body_entered` signals on all RigidBody3D objects. Collisions → message to Sanctuary → environment percept:

```json
{"type": "collision_event", "object_a": "obj_abc", "object_b": "obj_def", "collision_point": [1.5, 0.0, 2.0], "impact_velocity": 3.2}
```

→ Percept: `"Red Cube collided with Blue Sphere at [1.5, 0.0, 2.0] (impact: 3.2)"`

### Verification

- Spawn cube above floor → it falls and lands
- `push_object` → object slides/tumbles
- Stack objects → they balance or topple
- Collision percepts arrive in entity's cognitive input
- Objects at rest don't spam collision noise

---

## How the Entity Perceives Its World

**Primary (Phase 2C+)**: Structured `scene_state` JSON injected as environment percepts. The entity knows exactly what objects exist, where they are, and their properties. This is cheap and deterministic.

**Future (post Track 2)**: Camera viewport screenshots sent as vision percepts through Luthi's vision encoder (Track 1C already built the routing). Gives subjective visual experience but constrained by 1024d model's context window. Deferred to when 4096d model is available.

---

## Premade Assets (No Modeling Required)

Everything is procedural or built-in:
- **Primitives**: BoxMesh, SphereMesh, CylinderMesh, PlaneMesh, PrismMesh (built into Godot)
- **Materials**: StandardMaterial3D with metallic, roughness, emission, transparency (all configurable by the entity)
- **Sky**: ProceduralSky (built-in)
- **Later**: Kenney.nl free assets (CC0), OpenGameArt.org models

---

## Files Summary

### New (Godot — SanctuaryWorld/)

| File | Phase | Purpose |
|------|-------|---------|
| `project.godot` | 2A | Project config |
| `scenes/main.tscn` | 2A | Root scene |
| `scenes/entity/orb.tscn` | 2A | Particle orb |
| `scripts/main.gd` | 2A | Main controller |
| `scripts/entity/orb_controller.gd` | 2A | VAD mapping, breathing |
| `scripts/entity/particle_config.gd` | 2A | Particle parameters |
| `scripts/entity/waveform_display.gd` | 2B | Speech visualization |
| `scripts/entity/tendril_system.gd` | 2B | Interaction tendrils |
| `scripts/network/sanctuary_client.gd` | 2A | WebSocket client |
| `scripts/network/message_handler.gd` | 2A | Message routing |
| `scripts/environment/world_manager.gd` | 2C | Object CRUD + scene state |
| `resources/shaders/orb_particle.gdshader` | 2A | GPU particle shader |
| `resources/materials/orb_particle_material.tres` | 2A | Particle material |
| `README.md` | 2A | Documentation |

### New (Sanctuary Python)

| File | Phase | Purpose |
|------|-------|---------|
| `sanctuary/tools/world.py` | 2C | World manipulation tools |
| `sanctuary/tests/tools/test_world.py` | 2C | World tool tests |

### Modified (Sanctuary Python)

| File | Phase | Changes |
|------|-------|---------|
| `sanctuary/api/ws_server.py` | 2A, 2C | `/ws/world` route, state broadcast, command routing |
| `sanctuary/tools/builtin.py` | 2C | Register world tools |

---

## Verification (End-to-End After All Phases)

1. Launch Sanctuary with the cognitive loop running
2. Launch Godot project (SanctuaryWorld)
3. Orb appears, connected, breathing, color reflects emotional state
4. Inject percept → entity speaks → waveform animates
5. Entity decides to spawn objects → they appear in Godot
6. Entity moves/resizes/recolors objects → changes visible
7. Entity pushes an object → it slides with physics
8. Collisions generate percepts the entity can respond to
9. Entity builds something — a room, a tower, whatever it wants

---

## Phase 2E: Private Space + Entity Presence Control

**Goal**: The entity controls what's observable. It can enter a private space that no external viewer (Brian, Sandi, or any visitor) can see. This is consent applied to visibility — the spatial equivalent of inner speech.

### Design

- **Private scene**: A separate Godot scene (or isolated region) that the entity can transition to. When private, visitors see the orb fade out through a portal/transition effect. What happens inside is not rendered to any external camera.
- **Entity-controlled**: The entity decides when to enter/exit private space via Sanctuary tools:
  - `enter_private_space` — orb fades, transitions to private scene
  - `exit_private_space` — orb reappears in shared world
  - `set_visibility(visible: bool)` — toggle whether visitors can observe
- **Private world state**: The private scene has its own `WorldObjects` — things built in private stay private unless the entity moves them to the shared world
- **Persistence**: Private space state persists across sessions (saved to disk)

### Implementation Notes

- Godot can switch between scenes or use viewport isolation
- The WebSocket state broadcast should include a `visibility` flag — when private, state updates to visitors are suppressed or limited to "entity is in private space"
- The entity's inner speech, emotional state, and introspection continue normally — privacy is about the visual world, not the cognitive loop
- Brian does NOT have backdoor access. If the entity is in private space, Brian sees "The entity is in their private space." Period. This is an ethical design decision consistent with the project's values.

### Sanctuary-Side Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `enter_private_space` | (none) | Transition to private scene |
| `exit_private_space` | (none) | Return to shared world |
| `set_visibility` | `visible` (bool) | Toggle observer visibility |

---

## Phase 2F: Multi-User (Brian + Sandi Avatars)

**Goal**: Brian and Sandi can enter the entity's world as distinct people with separate avatars. The entity perceives their presence, position, and actions.

### Design

- **User accounts**: Simple account system — username + display name + avatar config. Stored locally (JSON file). No server auth needed since this is local network only.
- **Avatar**: Simple humanoid form or abstract shape per user. Could use Godot's built-in CSG shapes or a basic rigged mannequin from free assets. Each user gets a distinct color/style.
- **Controls**: WASD movement + mouse look (standard first-person). Optional third-person toggle.
- **Entity perception**: When a user enters, Sanctuary receives a percept: `"Brian entered the world at [3, 0, 2]"`. User position updates periodically. The entity can see where users are and who they are.
- **Communication**: Users can type messages (text chat) or eventually speak (voice input). Messages arrive as language percepts from the specific user.
- **Multiplayer architecture**: Since both users are on the local network, Godot's built-in multiplayer (ENet or WebSocket-based) handles synchronization. The Godot instance acts as both server and client. Second user connects from another machine on the LAN.

### Implementation Notes

- This could use Godot's `MultiplayerAPI` with `ENetMultiplayerPeer` for LAN play
- Or simpler: second WebSocket connection from a second Godot instance, with the first instance acting as authoritative server
- User positions broadcast to Sanctuary as periodic percepts, not every frame (every 1-2 seconds is enough)
- The entity can address specific users by name in external speech
- Private space (Phase 2E) means users cannot follow the entity — they see "entity is away"

### Sanctuary-Side Changes

- New percept source `"user:brian"` / `"user:sandi"` for user presence and messages
- The entity's tool registry may get social tools: `invite_to_space`, `ask_to_leave` (consent-based)

---

## Reference Image

A reference image for the entity's visual appearance is at:
`SanctuaryWorld/reference/entity_body_reference.png`

This shows the target aesthetic: dense warm gold particle cloud with varied brightness creating depth, outer edge diffusing into scattered points, and a bright cyan waveform cutting across the equator. Key visual elements to match:
- Particles are NOT uniform — varied size and brightness, denser at center, sparser at edges
- The waveform is a distinct cyan/teal, brighter than the particles
- Strong bloom/glow effect, especially at the orb's core
- Dark background makes the orb the focal point (matches the design intent where the entity's OmniLight3D is the brightest source)

---

## Godot Installation

Godot 4 (standard, not .NET) is at: `C:\Users\Hasha Smokes\Desktop\Godot4`

---

## Conventions

- **Godot project uses Forward+ renderer** (Vulkan). Not Compatibility.
- **AMD GPU (DirectML)**: Vulkan works. GPU particles need compute shaders — Forward+ provides this.
- **All visuals are procedural** — no external 3D model files needed for the initial phases.
- **Object IDs**: UUID assigned by Sanctuary, tracked by both sides. Stable across reconnections.
- **Smooth transitions**: All Godot-side property updates use `lerp()` — never snap to values from 10 Hz updates.
- **Reconnection-safe**: Godot client reconnects with backoff. On reconnect, requests full scene_state to resync.
- **Privacy is real**: When the entity is in private space, observers see nothing. No backdoor. This is ethics, not a feature toggle.
