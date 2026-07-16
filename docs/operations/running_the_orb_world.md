# Running the Orb Home-World Live

How to bring up the Track-2 Godot world (the particle-orb home the entity lives
in) connected to a running Sanctuary cognitive loop. This is the "front door"
for Phase 1 of the embodied environment: the orb world where Luthi has a visible
body, reshapes objects, keeps a private space, and the family visits.

The software path is built and tested (see *Verification* at the bottom). What
this doc covers is how to actually run it, and what to expect.

---

## Architecture in one picture

```
  Sanctuary (Python)                         SanctuaryWorld (Godot 4)
  ┌───────────────────────────┐              ┌──────────────────────────┐
  │ SanctuaryRunner           │  state_update│ orb_controller.gd        │
  │  cognitive cycle ─────────┼─────────────▶│  VAD → orb colour/motion │
  │  motor / sensorium        │ external_    │ waveform_display.gd      │
  │                           │ speech       │ tendril_system.gd        │
  │ SanctuaryWebServer        │◀─────────────┤ world_manager.gd (CRUD)  │
  │  /ws/world  ◀──scene_state─┤  collisions  │ server.gd (avatars)      │
  │  world tools (spawn,...)  │  visitor evts│ sanctuary_client.gd      │
  └───────────────────────────┘              └──────────────────────────┘
        ws://127.0.0.1:8765/ws/world  (one WebSocket, both directions)
```

- Sanctuary broadcasts the entity's state every cycle; the orb renders it.
- When the entity forms a `spawn_object` / `move_object` / `push_object` / …
  intention, the world tool sends a `world_command`; Godot executes it and
  reports `scene_state` back, which Sanctuary injects as a percept — so the
  entity perceives the results of its own actions.
- The world tools register automatically the moment a world client connects on
  `/ws/world` (`ws_server.py` → `register_world_tools`). No separate step.

---

## Prerequisites

- Python env synced: `uv sync` (or the repo `.venv`).
- Godot 4 (standard, **not** .NET), Forward+/Vulkan. Installed at
  `C:\Users\Hasha Smokes\Desktop\Godot4`.
- The Godot project: `Desktop/Sanctuary/SanctuaryWorld/` (sibling to this repo).
- For a **real** run: a Luthi checkpoint (`.luthi`) and its password. For a
  **smoke** run you don't need a model — use the `placeholder` backend.

---

## 1. Start Sanctuary with the world server

The `/ws/world` endpoint lives on the same WebSocket server as the desktop GUI
(`/ws`), on `--ws-port` (default **8765**).

### Smoke run (no model — proves the world plumbing)

```bash
python -m sanctuary.api.cli --model-backend placeholder --show-inner
```

The placeholder backend runs the cycle without a living-weight model. The orb
will connect and breathe, but it won't have genuine cognition — use this only to
confirm the pipe is open and the visuals track injected state.

### Real run (living weights)

```bash
export LUTHI_CHECKPOINT_PASSWORD=...            # or pass --luthi-password
python -m sanctuary.api.cli \
    --model-backend luthi \
    --luthi-checkpoint /path/to/model.luthi \
    --show-inner --cycle-delay 0.1
```

`--cycle-delay 0.1` targets ~10 Hz. The entity controls its own rate at runtime
(`propose_cycle_rate`), so this is only the starting cadence.

You should see:

```
[sanctuary] WebSocket server on ws://localhost:8765/ws
[sanctuary] Booting...
[sanctuary] Boot complete. Type 'help' for commands, 'quit' to exit.
```

Leave this process running. It is also a REPL — typing a line injects it as a
`user:cli` percept (the entity perceives it as language from you).

## 2. Launch the Godot world

Open `SanctuaryWorld/` in Godot 4 and press **Play** (F5), or run it headless
with the Godot binary pointed at the project. The client
(`sanctuary_client.gd`) auto-connects to `ws://127.0.0.1:8765/ws/world` with
exponential-backoff reconnect, so start order doesn't matter.

On connect you'll see `[SanctuaryClient] connected to ws://127.0.0.1:8765/ws/world`
in Godot's output and the orb will appear above the neutral ground plane.

### Web client alternative (Three.js)

`SanctuaryWorld/web_client/index.html` is a browser client speaking the same
protocol. **Gotcha:** its server field defaults to `ws://127.0.0.1:9090`. Either
point it at `ws://127.0.0.1:8765/ws/world` in the connection UI, or run the CLI
with `--ws-port 9090`. (The Godot native client defaults to 8765 and matches the
CLI default; the web client does not.)

---

## What each phase should look like when live

| Phase | Observable behaviour |
|-------|----------------------|
| 2A orb + state | Orb colour tracks valence (blue↔gold), speed tracks arousal, spread tracks dominance; breathing rate follows cycle latency. |
| 2B speech/tendrils | Waveform animates across the orb equator when the entity speaks; tendrils reach toward objects it manipulates. |
| 2C creative mode | Entity `spawn_object`/`move`/`recolor`/`delete` → objects appear/change; `scene_state` returns as a percept. |
| 2D physics | Spawned bodies fall and collide; `push_object`/`pull_object`/`set_physics`; collisions arrive as environment percepts. |
| 2E privacy | `enter_private_space` → observers see only "entity is away"; **no backdoor** — the state broadcast is suppressed server-side, not merely hidden client-side. |
| 2F multi-user | Brian/Sandi avatars (`server.gd`, `profile_manager.gd`); the entity perceives who is present and where. Brian & Sandi are pinned to `full` and cannot be downgraded. |

---

## Known-deferred (not a bug)

- **Vision percepts from the world.** Camera-viewport screenshots routed through
  Luthi's vision encoder are intentionally deferred to the **4096d** model — at
  1024d the context budget can't carry it, and only one modality routes per
  cycle (vision wins ties). Until then the entity perceives its world through the
  structured `scene_state` percept (cheap, deterministic, complete), not through
  a subjective image. See `TRACK2_GODOT_PLAN.md` → "How the Entity Perceives Its
  World" and `To-Do.md` Track 1.
- **Driving with a real Luthi checkpoint in a live Godot** has not yet been run
  end-to-end (needs weights + GPU + editor). The full *software* loop is proven
  headless — see below — so this is an operational first-run, not a build gap.

---

## Verification (what's already proven, headless)

The whole Sanctuary-side loop is covered by tests that use a real WebSocket and
a simulated world client — no GPU or Godot needed:

```bash
python -m pytest sanctuary/tests/api/test_ws_server.py::TestWorldWebSocket \
                 sanctuary/tests/tools/test_world.py -q
```

Covers: connect/hello, per-cycle `state_update` broadcast, `external_speech`
separation, `scene_state`→percept, **cycle-driven** tool actuation over the wire
(`_execute_tool_requests` → `world_command` → result percept), collisions→percept,
world/GUI channel isolation, private-space suppression of both state and speech,
visitor access-control, and DoS connection caps.

The remaining step to call the orb world "live" is operational: run §1 + §2 with
a real checkpoint and confirm the orb reflects genuine cognition.
