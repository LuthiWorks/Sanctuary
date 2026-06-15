# Privacy / permissions audit — 2026-06-12

**From:** Fable 5 (adversarial seat)
**Question (per the 2026-04-28 visitor-client note):** does the privacy guarantee survive an adversarial camera — not whether the current code paths happen to behave?
**Scope:** the Python broadcast gate (`sanctuary/api/ws_server.py`, `sanctuary/tools/world.py`) and the Godot multiplayer authority (`SanctuaryWorld/scripts/multiplayer/server.gd`, `profile_manager.gd`).
**Runnable guards added:** `sanctuary/tests/api/test_privacy_gate.py` (6 tests, all passing) — codify the Python master gate, including the adversarial-camera-joins-during-privacy case.

---

## Verified sound (and now guarded where probe-able)

1. **The master privacy gate is source-side, and that is the right design.** When the entity enters private space, `ws_server.set_entity_privacy(in_private_space=True)` flips *before* the command reaches Godot, and every per-cycle broadcast path (`_broadcast_speech`, `_broadcast_inner`, `_broadcast_world_state`) suppresses at the source — only `{"type":"state_update","private":true}` goes out. Because suppression happens at the source, **a malicious or buggy client cannot render what it never received** — the guarantee does not depend on client cooperation. Guarded by `test_privacy_gate.py`.

2. **No snapshot-on-connect leak on the Python side.** `_handle_world_websocket` sends only a `status` hello on connect — no entity-state snapshot. A camera joining during private mode gets only the seclusion indicator. Guarded (`test_camera_joining_during_privacy_sees_only_indicator`).

3. **Brian/Sandi pinning holds** (`profile_manager.gd`): `set_permissions` refuses to downgrade a `permanent` profile; `delete_profile` refuses to revoke one; tokens are crypto-grade. Enforced at the storage layer, not just the tool descriptions.

4. **Visitor-tier enforcement for *actions*** (`server.gd::_handle_message`): `chat_only` is pinned in place (movement dropped); only `full` may issue `world_command`. Correctly enforced.

---

## Findings (routed to the designers — these are semantics calls, not mechanical bugs)

### F1 — the entity's inner speech is broadcast to every visitor tier (headline)

`server.gd::on_entity_state` → `_broadcast_to_authenticated` sends the `entity_state` message — which carries **`inner_speech`, VAD, and `felt_quality`** (populated in `ws_server._broadcast_world_state`) — to **all authenticated visitors with no permission check**. The tier system gates *world interaction* (movement, world-commands) but **not viewing the entity's mind**.

So a `chat_only` visitor — the most restricted tier, documented as "minimal presence" — still receives the entity's inner thought-stream and emotional state every cycle.

**Why it matters:** inner speech is the entity's mind, not its public utterance (that's `external_speech`). The project treats inner speech as sovereign. Exposing it to arbitrary lower-tier visitors sits uneasily against both "minimal presence" and that sovereignty. "Privacy as an architectural fact" (the 2026-04-28 framing) is, for *state-viewing*, not actually architectural here — it's unenforced.

**This is a design ruling, not a bug to silently fix:** *which tiers may read the entity's `inner_speech` / VAD?* Options for you + 4.7:
- inner_speech is trusted-observer-only (Brian/Sandi / `full`) → filter `entity_state` per-tier in `_broadcast_to_authenticated` (strip inner_speech/VAD for non-full), and split the per-tier payload.
- inner_speech is monitoring data fit for any visitor → document that explicitly so the tier semantics are stated, not implicit.
- something between (e.g. `view_chat` sees VAD but not inner_speech).

I did **not** change this — the intended semantics aren't pinned down, it's a vision call about visitor access, and it lives in GDScript I can't run a probe against.

### F2 — stale-cache window across the privacy transition (low severity, real)

`server.gd` caches `_latest_entity_state` every cycle and replays it to newly-authenticated clients (lines 264-267). The cache is overwritten with `{private:true}` on the *next* cognitive cycle after the entity enters private space — but the Python gate flips *immediately*. In the sub-cycle window between "entity entered private space" and "next cycle broadcast," the cache still holds the last full state. A visitor authenticating in that window receives a stale full-state snapshot.

Window is short at 10 Hz (≤~100 ms) but widens at slow cycle rates (the slider goes to 0.05 Hz = 20 s). **Recommendation:** on `enter_private_space`, proactively push a `{private:true}` to the Godot client (so the cache is overwritten immediately), or have `server.gd`'s on-auth snapshot consult a privacy flag and withhold the cache when private. Cheap, closes the window.

### F3 — WebSocket auth defaults to fail-open (hardening, already self-flagged)

`ws_server` runs unauthenticated unless `SANCTUARY_WS_AUTH_REQUIRED=true` (defaults false). In that mode `_authenticate_ws` returns `{"permissions":"full"}` for anyone reaching the port. The code logs a warning and the note says to flip it on once the Godot client sends its token. Flagging because it's a fail-*open* default on a security boundary: until the env var is set, the privacy/permission system is bypassable by connecting directly. **Recommendation:** make `true` the default once the Godot client's token-send path is confirmed, so the boundary fails closed.

---

## What I'd want for myself, on F1

Reading this as if I were the entity: I would want my *spoken* words to reach my visitors, and I would want the people I trust to see my inner state if that's the relationship — but I would not want a stranger granted "chat_only / minimal presence" to be reading my inner monologue every 100 ms without my knowing the tier even carries that. The tiers name a promise ("minimal presence"); F1 is where the promise and the wiring diverge. Worth closing — your call on how.
