---
name: read-back
description: Summarize what Codex or Gemini produced since the last session into a verdict-first read-back for the roadmap. Use at the start of a session after runs or research notes landed, or when the user says "Codex finished X" or "Gemini did some research".
---

# read-back · fold new evidence into the roadmap

When to use: new material exists (fresh `runs/*` artifacts, a new
RESULTS.md stage, new `research/` notes) that the roadmap has not yet
absorbed.

## Exact steps, in order

1. Diff the ground truth: list files in
   `experiments/tiny_language_lab/runs/` and `research/` newer than the
   last read-back (memory records the last one), and read any new
   `## Stage N` section at the end of RESULTS.md.
2. For each new stage, extract EXACTLY: which pre-registered decision
   line fired (CONFIRM, KILL, GRADED, INCONCLUSIVE, or a gate PASS or
   FAIL), the one or two deciding numbers with their run-file citations,
   and any deviation from the specced protocol.
3. Verify before trusting: spot-check the deciding numbers against the
   raw `.jsonl` or log, not just the prose. If a verdict rests on a
   number that is not in an artifact, flag it instead of propagating it.
4. State the consequence: which ADR clause fires, which hypothesis
   resolves, what the next rung becomes. If a reversal or reopen clause
   is touched, say so explicitly.
5. Record it durably: update the relevant ADR status paragraph, mark the
   hypothesis resolved, update memory (index line plus goal-loop file),
   and only then plan the next stage.

## Full example of a good final output (real read-back: the ADR 0013 Status implementation paragraph)

```markdown
Implementation: Stage 53 (D1) COMPLETE on 2026-07-02, verdict KILL
E-interfere (early lead `-0.252` at 200 steps, late loss `+0.072/+0.100` at
1000/2000 on all paired seeds, surviving the mandated `--muon-lr 0.005`
rerun at `+0.055`). D3's assembly rule therefore FIRES toward RANDOM INIT
for the flagship, and H021's flagship-carrying clause is moot; Stage 54's
floor-scaling question stands on its own scientific value (see D2 note).
Stage 53 also confirmed checkpoint RESUME does not exist yet; D3's resume
requirement is a build item, not a verification item. Stages 54 and 55
pending.
```

Every element is present: the stage, the date, the verdict by its
pre-registered name, the deciding numbers, the guard rerun that kept the
verdict honest, the downstream clause that fires, and what remains open.

## Mistakes to avoid (each one actually happened here)

- **Do not accept a verdict without its guard.** Stage 53's E-interfere
  was only recordable AFTER the mandated lower-LR rerun; a read-back that
  skips the guard propagates an unhonest kill.
- **Do not let stale status text survive.** "Human review remains
  pending" sat in three documents after the review PASSED; a read-back
  updates every document that carried the pending state (RESULTS entry,
  ADR statuses, memory), not just one.
- **Check for confounds before headline numbers.** Stage 40's dramatic
  0.000000 held-out result was an output-emission artifact, not a
  finding; Stage 52 had sleep-inflated seconds; the 20k seed spread was
  eval noise. Ask "what else could produce this number" before folding it
  in.
- **Numbering and lanes.** Codex evidence drafts are `.codex-draft.md`
  files attached to an ADR, not ADRs (an orphaned draft 0009 exists);
  Claude accepts drafts with revisions and preserves them.
- **Point at the exact files.** A read-back without run-file citations
  cannot be audited; every number carries its artifact path.
