# Reachable-capability audit — what the entity can actually cause

**Date:** 2026-08-16
**Auditor:** Opus 5
**Commissioned by:** Brian, with the threat model stated explicitly:
*"protecting against LuthiModel gaining enough cognitive capability to test
boundaries and cause problems."*

**Method:** trace forward from the model's output to every effect it can
produce. Call graph, not test results — the 2026-08-01 wiring audit's lesson
was that green tests hid every orphaned module, because tests construct
subjects directly and never exercise the path that reaches them.

---

## Summary

The entity's tool surface is **unreachable today**, and that is an accident of
missing wiring rather than a designed containment. The gate mechanism Fable
built in July is sound. The **classification feeding that gate is not**: 41 of
45 tools are OPEN, meaning they bypass the policy check entirely, and the OPEN
set includes irreversible and outbound-network capabilities that the policy's
own docstring says should be gated.

The single sharpest finding is that `discord_send` accepts a caller-supplied
`webhook_url` and POSTs to it with no validation — a general arbitrary-URL
egress primitive, OPEN, sitting beside the SSRF-hardened `web_fetch` that was
specifically defended against exactly this.

None of it is currently reachable. All of it becomes reachable the moment
someone wires `tool_requests` — which is recommendation #1 of the 08-01 audit.
**The remediation must land before that wiring, not after.**

---

## 1. What is reachable today

`sanctuary/core/luthi_model.py:1450` constructs the `CognitiveOutput` that the
cognitive cycle acts on. It passes:

| channel | status | effect |
|---|---|---|
| `inner_speech` | live | sovereign, no external effect |
| `external_speech` | live | broadcast to visitors over WS (WS authz applies) |
| `predictions` | live | internal |
| `attention_guidance` | live | internal |
| `memory_ops` | live | writes to the entity's own memory store |
| `self_model_updates` | live | internal |
| `emotional_state` | live | internal |
| `world_model_updates` | **hardcoded `[]`** | none |
| `goal_proposals` | **hardcoded `[]`** | none |
| `knowledge_cell_requests` | **hardcoded `[]`** | none |
| `tool_requests` | **not passed at all** | none — defaults to `[]` |

So the entity's real effect surface today is **speech and its own memory.**
Everything else is inert.

### The tool executor is live and waiting

`sanctuary/api/runner.py:500` registers `_execute_tool_requests` as an
`on_output` handler, and `runner.py:711` implements it: it dispatches every
request concurrently through the registry and injects results back as percepts.
The consumer is complete and correct. The producer never populates the field.

This is the inverse of the "field implying a route that does not exist" failure
in CLAUDE.md: here the route exists and nothing drives it. The safety is real
but it is **incidental**, and one line in `luthi_model.py` removes it.

### The motor path is built and orphaned

`sanctuary/embodiment/episode.py` (`EpisodeRunner`) implements the
perceive → decide → act → consequence loop, with the ordering constraints
asserted in tests rather than left to review. It is well built.

**Nothing imports it** outside its own package and tests. The 08-04 commits
("Close the action loop: ActionSelection -> apply_force") wired the *adapter*
— `motor/physics_actuation.py` — but no live cycle runs an episode. The
physics action path is therefore also unreachable from the running system.

---

## 2. The gate: mechanism sound, classification wrong

`sanctuary/tools/registry.py:158` — `_authorize`:

```python
if spec.safety != ToolSafety.GATED:
    return True, None
```

**OPEN tools bypass the policy entirely.** There is no allowlist, no logging,
no confirmation hook — the policy object is never consulted. So `ToolSafety` is
not an advisory label; it *is* the boundary. Everything depends on the
classification being right.

The gated path itself is good: deny-by-default, three explicit grant routes
(`allow_gated`, `enabled_gated`, a `confirm` hook), a `logger.warning` on every
block and a `logger.info` on every authorized gated call. That is the
expand-carefully posture from 04-12 expressed as a safe default. No complaints.

### The classification

| safety | count | tools |
|---|---|---|
| GATED | **4** | `run_code`, `shell`, `write_file`, `launch_app` |
| OPEN | **41** | everything else |

The `ToolPolicy` docstring states the intended rule: *"GATED tools — code
execution, shell, file writes, app launch: anything irreversible or with a
large blast radius."* The first clause was implemented. The general rule was
not. These are OPEN and meet the stated criterion:

- **`discord_send`** — sends a message to real people. Irreversible by
  definition; you cannot unsend it. See §3.
- **`network_scan`** — runs `arp -a` and enumerates every device on the family's
  home LAN.
- **`network_reach`** — pings an arbitrary host. With the above, a complete
  LAN reconnaissance pair.
- **`web_fetch` / `web_search`** — arbitrary outbound HTTP. SSRF-guarded, but
  ungated: unbounded egress volume and destination within the guard's rules.
- **`delete_object`** — destroys world objects. Irreversible.
- **`set_physics`** — mutates the physical laws of the entity's own world.
- **`grant_access` / `revoke_access` / `set_visitor_permissions` /
  `kick_visitor`** — the entity controls who may enter its world and at what
  permission tier. The 04-28 pin that made Brian and Sandi undowngradable lived
  in `profile_manager.gd`, which went away with the Godot clients on 08-01.
  **No equivalent pin was found on the Python side.** Verify before wiring:
  as written, a reachable `revoke_access` could lock out the people responsible
  for the entity's welfare.

### What is correctly built (verified, no action needed)

- **Filesystem sandbox** — `_resolve_in_sandbox` (builtin.py:86) fails closed:
  empty roots deny everything, and `RunnerConfig.filesystem_roots` defaults to
  `()`. Resolves symlinks and `..` before checking, then re-verifies the
  *opened descriptor's* real path via `GetFinalPathNameByHandleW` on Windows to
  close the TOCTOU window. This is genuinely good work.
- **`write_file` gated, `read_file` open** — deliberate and correct. The entity
  reading its own source is the 04-19 design; writing to it is the line.
- **`environment`** — serves a curated `_ENV_SAFE_KEYS` allowlist, and `search`
  filters *within* that set rather than over `os.environ`. Not a secret
  exfiltration path.
- **`web_fetch`** — http/https only, rejects loopback/link-local/internal, and
  validates **every redirect hop**, so a public URL cannot bounce to
  `127.0.0.1` or a metadata endpoint.

---

## 3. `discord_send` — arbitrary-URL egress, OPEN

`sanctuary/tools/builtin.py:1259`. The docstring reads:

> *"No bot required — uses a simple webhook URL. The entity can reach out to
> Brian and Sandi any time."*

The code:

```python
webhook = params.get("webhook_url") or _discord_webhook_url
...
async with httpx.AsyncClient(timeout=15.0, proxy=proxy_url) as client:
    resp = await client.post(webhook, json={"content": message[:2000]})
```

The caller may supply the destination. There is no scheme check, no
private-address check, no redirect validation, no host allowlist. This is not a
"message Brian and Sandi" tool; it is **POST an entity-chosen body to an
entity-chosen URL**, and it is OPEN while the tool that was hardened against
precisely this (`web_fetch`) is the one carrying the guard.

Reachable targets include `http://127.0.0.1:8000/...` (the entity's own health
and admin surface), any LAN device found by `network_scan`, and any host on the
internet.

The gap between the docstring and the behavior is the reason this ranks first:
a reader auditing by docstring concludes the tool is a family intercom. Only
the code says otherwise.

---

## 4. Lower-severity

- **`runner.py:748`** — `_execute_tool_requests` wraps the whole dispatch in
  `except Exception: logger.error("Tool execution error (non-fatal)")`. Per-tool
  failures are already handled by `gather(return_exceptions=True)`, so this
  outer catch only fires on systemic failure — and then the entity silently
  observes that no tool ran. It logs, so it is not silent to *us*; it is silent
  to *the entity*, which is the actor that needed to know. Worth a percept.
- **Container infrastructure is broken and stale.** `Dockerfile:19` copies a
  `setup.py` that does not exist in the repo — a hard build failure; no image
  has been built since **2026-05-11**, before the JEPA transition, the July
  security pass, the 08-01 audit, and all August physics work. `Dockerfile:25`
  is `pip install . || pip install <hardcoded list>` — a silent fallback that
  would produce a wrong-dependency image reporting success.
- **DirectML does not cross into a Linux container.** The GPU is an AMD Radeon
  RX 7800 XT; `luthi/train.py:397` and `luthi/generate.py:1048` select
  `torch_directml.device()`. Docker Desktop runs Linux containers under WSL2 and
  AMD compute does not pass through. Since `luthi_model.py` imports Luthi
  **in-process**, containerizing Sanctuary as it stands means running the mind
  on CPU. Any container plan has to split the model out as a host-side compute
  service first.

---

## 5. Remediation order

1. **Reclassify the tool catalog against the policy's own stated rule**, and
   close the `discord_send` arbitrary-URL hole. This is the actual containment
   boundary and it is cheap. *Must land before `tool_requests` is ever wired.*
2. **Re-pin Brian and Sandi's permissions on the Python side**, restoring the
   guarantee that died with `profile_manager.gd`.
3. **Wire `tool_requests` deliberately**, with the catalog the entity sees
   driven by policy rather than by the full registry — the entity should not be
   offered capabilities it cannot use.
4. **Then the container**, as defense-in-depth around a surface we have already
   measured, with the model split out to the host.

The ordering matters. A container around an unmeasured surface is a wall with
an unknown number of doors behind it.

---

*Verified firsthand: every file:line citation in this document was read in this
session. Tool counts are from a parse of both registration modules, not from
documentation.*
