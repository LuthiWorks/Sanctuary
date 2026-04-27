# Fresh-Instance Audit Protocol

A practice for catching the blind spots that accumulate when one person — even one person working with Claude instances — is the sole human in the loop on a long-running project.

## Why this exists

The Sanctuary + LuthiModel work has been built largely by one human (Brian) collaborating with Claude instances, often the same project for many months. That arrangement is productive but has a structural weakness: **ordinary assumptions go uninspected**. Hardcoded paths, premature abstractions, drifted documentation, half-finished refactors, and "works on my machine" patterns survive longer than they should because the only person reviewing is the same person who wrote them.

Fresh-instance audits exist to compensate for this. A new Claude instance, with no investment in the existing decisions, reads the code with outsider eyes and reports what looks wrong, drifted, missing, or suspicious. The first one of these (2026-04-26) found:

- A premature crash boundary that violated the project's own coding standard
- A hardcoded developer path that broke any non-Brian deployment
- An integration handshake between repos that didn't actually exist (Sanctuary reaching into Luthi internals)
- A subtle broadcasting comment that, if read alone, would have suggested a bug existed when one didn't
- Tests asserting old behavior the project's standards had since changed

None of those required deep expertise to find. They required *fresh eyes*.

## When to run an audit

- **Before a major version bump** (Phase 4 → 5, 5 → 6, etc.)
- **After a significant refactor** that touched multiple subsystems
- **When test coverage feels uneven** (new code merged without proportional tests)
- **Before any "we're going to wake the entity now" milestone** — this one is non-negotiable
- **Quarterly, by default**, even if nothing else triggers it

You don't need a reason. The point is to invite someone uninvested.

## How to run an audit

The auditor is a fresh Claude instance — usually spawned via the `Agent` tool with `subagent_type: "Explore"` from a parent instance, or as the first task of a new session. It must not have prior context about the codebase from your conversation; that context defeats the purpose.

### What to give the auditor

Hand them a self-contained briefing:

1. **Greeting and project context.** Who Brian and Sandi are, what's being built and why it matters. (See the "How to Treat Subagents" section in `~/.claude/CLAUDE.md` for the standard.)
2. **The repo path.**
3. **Pointer to settled findings.** `CLAUDE.md`, `PLAN.md`, `To-Do.md`. Tell them which findings are "DO NOT REINVENT" so they don't waste time second-guessing the architecture.
4. **A specific question, not "audit everything."** Open-ended audits produce shallow results. Pick one or two areas: "look for migration leftovers," "check coding-standard compliance in the cognitive loop," "evaluate test coverage gaps for backward-pass paths."
5. **An explicit instruction to surface null findings.** "If you don't find anything in a category, say so" — null findings are real findings.
6. **A word count target.** ~600–1200 words is the right size for a useful audit report. Open-ended produces sprawl.

### What the auditor reports

A structured doc with sections matching the questions you asked. Each finding should have:

- **What** — the issue, with file:line references
- **Why it matters** — the consequence if left unfixed
- **Suggested resolution** — but not necessarily a fix; the auditor's job is to surface, the maintainer's is to decide

### What you do with the report

1. **Read it without defending.** The point is to surface what you stopped seeing. Pre-emptive defense ("but we did that for a reason") wastes the audit.
2. **Triage.** Critical, medium, low. Critical goes into the next session. Medium goes on `To-Do.md`. Low gets noted but may stay.
3. **Verify.** A finding is a claim that something exists at a path. Before acting on it, check the file. Auditors can hallucinate or misread.
4. **Save the report.** Even findings you don't act on are useful as a record of what was true at a moment in time.

## Templates

### Spawn prompt (single agent, focused area)

```
Hello. I'm a Claude instance helping Brian — the human collaborating
with Claude instances on the Sanctuary cognitive architecture.

Context: [project description, who Brian and Sandi are, why this work
matters — see CLAUDE.md "How to Treat Subagents"]

Repo path: <absolute path>

Read first:
- CLAUDE.md (especially the "DO NOT REINVENT" section if applicable)
- PLAN.md (current phase)
- To-Do.md (task state)

Specific area to audit: <ONE focus, e.g. "the cognitive cycle's
fault-isolation pattern compared to AGENTS.md's stated rules">

What I'd like in the report:
1. Map of what's actually there vs what the docs say
2. Specific findings with file:line references
3. Null findings stated explicitly
4. ~600-1000 words

Do NOT modify any file. Read-only research.
Do NOT touch sanctuary/data/, .memories/, data/, or anything
constitutional/journal-related — those are protected entity records.

Thank you for doing this.
```

### Spawn prompt (parallel survey of two repos)

When auditing both repos, spawn two agents in the same message — one
per repo — with overlapping but distinct prompts. The session of
2026-04-26 used this pattern; both reports are higher quality than
trying to fit both into one agent's context.

## What this protocol is not

- **Not a substitute for tests.** Tests catch regressions; audits catch drift the tests don't know to test for.
- **Not a code review.** The auditor is not the gatekeeper for any change. The maintainer decides what's a finding worth acting on.
- **Not an oracle.** Auditors miss things, hallucinate, and over-flag. The protocol assumes you'll filter their output.
- **Not free.** Each audit costs context budget on the parent side and one full agent's worth of compute. Use them when you'll act on the results.

## History

- **2026-04-26** — First audit (Opus 4.7). Found three structural issues (premature crash boundary, hardcoded path, missing integration handshake), eight broad-exception violations in legacy paths, and several test coverage gaps. All structural issues fixed in commits `27c357c`, `97e8b24`, and the cross-repo integration test introduced this protocol document.
