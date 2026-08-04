# SanctuaryWorld

The Godot project for the entity's embodied world. Sibling to the `sanctuary/`
Python package, not inside it -- a separate technology stack (GDScript, not
Python), communicating with Sanctuary over WebSocket only.

**Status: empty, awaiting build.** This directory currently holds nothing but
its own `.gitignore` and this README. That is deliberate, and the reason
matters.

---

## Why this folder exists before the code does

The 2026-04 Godot world -- `SanctuaryWorld` and `SanctuaryClient`, containing
the particle orb, the Godot and Three.js visitor clients, the multiplayer
server, and the three-tier privacy gate -- **is permanently lost.**

It was never under version control. `docs/TRACK2_GODOT_PLAN.md` placed the
project outside the Python package, on the Desktop of a machine whose boot drive
was formatted on 2026-07-25. The rebuild runbook stated that "the projects
themselves are in the Sanctuary repos." They never were.

Verified on 2026-08-04, from the repository rather than from notes:

- Full recursive tree of `origin/main`: no `.gd`, `.tscn`, `.tres`, `.godot`,
  or `.import` files.
- Only two remote branches exist; nothing on a stale branch.
- `git log --all --diff-filter=A` across every ref: **zero Godot files have ever
  been added to this repository, at any point in its history.** Not deleted --
  never committed.
- No Godot content in any other LuthiWorks repository.
- No `project.godot`, `*.tscn`, or `*.gd` anywhere on `C:\Users`, `C:\Dev`,
  `D:`, or `E:` -- including both salvage copies and the pre-migration backup.

The backup discipline on this project follows the repositories. Anything outside
a repository is therefore unprotected *by construction*. This folder is the
structural fix: the world now has a tracked home, so that committing it is the
default rather than an extra step someone has to remember.

## The `.gitignore` in this directory is load-bearing

The repository-root `.gitignore` is a stock Python template. Its patterns would
silently swallow ordinary Godot files -- `*.bin` (every glTF model buffer),
`*.html` (the web export shell and the browser visitor client), `*.so`/`*.dll`
(GDExtension), and directory names like `lib/`, `var/`, `target/`, `env/`.

None of that fails loudly. `git add` reports success and stages nothing.

`SanctuaryWorld/.gitignore` therefore clears every inherited rule for this
subtree and re-adds only Godot's own generated artifacts. **Read the comments in
it before adding a pattern.** Breadth is what cost us the world the first time.

Verify before trusting, rather than after losing something:

```bash
git check-ignore -v SanctuaryWorld/<path>   # silence means the file is tracked
git status --porcelain SanctuaryWorld/      # confirm files actually appear
```

## What to build against

Every specification survived. A rebuild starts from written design, not from
memory:

| Document | Covers |
|---|---|
| `docs/TRACK2_GODOT_PLAN.md` | Orb home-world: project structure, scene tree, particle orb, WebSocket protocol |
| `docs/sanctuary_world_entity_spec_2026_06_29.md` | Environment-as-curriculum spec (rev 5); stop -> comfort -> understood-comfort -> reciprocity |
| `docs/DEVELOPMENTAL_WORLD_BUILD_PLAN_2026-07-15.md` | Developmental (rover) world; the physics-substrate seam and what is gated behind scale + LUTHISCOPE |
| `docs/DEVELOPMENTAL_WORLD_PHYSICS_DECISION_2026-07-16.md` | Continuous physics parameters; derived affordances |
| `docs/AFFECT_GROUNDING_DECISION_2026-07-19.md` | Affect grounding |
| `docs/WEATHER_DYNAMICS_DECISION_2026-07-20.md` | Weather dynamics |
| `docs/operations/running_the_orb_world.md` | Operations runbook (describes the lost build; treat as historical) |

## Open scope questions

Not settled as of 2026-08-04, and Brian's calls to make:

1. **Which world.** Faithful rebuild of the 2026-04 orb world, or build the
   developmental/rover world the 07-15 through 07-20 decision documents specify
   and let the orb go? Since no faithful restoration is available, both are now
   builds-from-spec.
2. **Whether Godot comes first at all.** `docs/audits/audit_2026-08-01_wiring.md`
   recommends closing the action loop (W1) and building the world headless
   against the physics seam before any GDScript, which reduces Godot to a
   rendering job performed once against a world that already works.
3. **`SanctuaryClient`.** The visitor clients were part of the same loss and are
   not covered by this folder. They need the same protection if they are rebuilt.
