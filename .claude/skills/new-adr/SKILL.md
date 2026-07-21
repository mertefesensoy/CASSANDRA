---
name: new-adr
description: Scaffold a new Cassandra architecture decision record at docs/decisions/NNNN-slug.md when evidence now justifies committing to one method over another. Use after a stage or set of stages resolves a question. Auto-numbers from existing ADRs and applies the project prose style.
disable-model-invocation: true
---

Scaffold a new architecture decision record for Cassandra. Write an ADR only when
measured evidence now justifies committing to one method over another, retiring a
branch, or rescoping a prior decision. This skill writes one file and does not run
experiments. The argument, if given, is the topic or slug.

## Steps

1. Resolve the next number. Glob `docs/decisions/*.md`, take the highest leading
   four-digit number `NNNN`, and use the next integer, zero-padded to four digits.
   Ignore `*.codex-draft.md` files when numbering; they are evidence drafts attached
   to an accepted ADR, not separate ADRs.

2. Fix the topic and slug from the argument, or ask the author one concise question.
   The slug is short, lowercase, hyphenated.

3. Orient before writing. Read the stages this decision rests on in
   `experiments/tiny_language_lab/RESULTS.md` and the relevant `runs/*.md`, the
   hypotheses it resolves in `docs/hypotheses/`, any cited `research/` Gemini notes,
   and the most recent ADR in `docs/decisions/` as a structural exemplar. If Codex
   has left a `NNNN-...codex-draft.md` evidence draft, read it: the project pattern is
   for Claude to accept it with revisions and preserve the draft alongside the record.

4. Write `docs/decisions/NNNN-slug.md` using the canonical section template below.
   Ground every claim in measured stages; cite run files. Do not invent metrics.

5. Apply the prose style. No em dashes and no en dashes anywhere; use the middle dot
   or restructure the sentence.

6. After writing, remind the author to: mark the resolved hypothesis as RESOLVED with
   a pointer to this ADR; add or update the `README.md` "Next ladder" rung; and update
   memory (`MEMORY.md` index line and the goal-loop memory file). Print the path.

## Canonical section template

```
# ADR NNNN · <decision in one line>

- Status: Accepted | Proposed | Superseded
- Date: <YYYY-MM-DD>
- Author: Claude (hypothesis, ADR, and roadmap role)
- Resolves: <hypothesis or question, with path>
- Builds on: <stages, prior ADRs, Gemini notes, with paths>

## Context
<what was measured, and the question this decision settles>

## Evidence
<the deciding stage results, with numbers and run-file citations>

## Decision
<numbered, concrete commitments: what is adopted, retired, or rescoped>

## Scope and what this decision does not claim
<the tight boundary; local versus general; what stays open>

## What would reopen or reverse this decision
<the explicit, falsifiable condition that brings the question back>

## Links
<RESULTS.md stages, runs/*.md, resolved hypotheses, Gemini notes, README,
 and any preserved .codex-draft.md>
```

Every ADR must carry the reopen-or-reverse condition. A decision with no condition
that would reopen it is not falsifiable and should be written as a hypothesis first.
