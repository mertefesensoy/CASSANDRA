---
name: weekly-commit
description: Commit the week's accumulated work in safe, reviewable pieces. Use when git status shows uncommitted changes older than a few days, or before any risky operation on the working tree.
disable-model-invocation: true
---

# weekly-commit · keep the repo commit-clean without committing hazards

When to use: `git status` shows accumulated modified and untracked files
(the historical pattern here is 5 commits in 3 weeks with batch messages,
which is the failure mode this skill exists to end).

## Exact steps, in order

1. `git status` and READ it. Sort changes into: docs (hypotheses, ADRs,
   reports, README), code (scripts in the lab dir, `.claude/`), and
   hazards (anything in `corpus/`, `runs/`, `artifacts/` binaries, or any
   file over 50 MB).
2. Hazards never get staged. Verify the big-file set is ignored
   (`.gitignore` covers `runs/`, `artifacts/**/*.onnx*`, `*.pt`); if a
   large file is TRACKED (check with `git ls-files` for corpus paths),
   `git rm --cached <file>` in its own commit together with the
   `.gitignore` change.
3. Stage and commit in coherent units, one theme per commit: for example
   docs of one phase together, one tool with its usage doc together.
   Message format: one imperative summary line naming the stage, ADR, or
   tool, then a short body if the change needs context.
4. Do not push if the history still carries oversized blobs (GitHub
   rejects files over 100 MB); pushing is gated on the ADR 0015 history
   surgery until that is done.
5. End by re-running `git status`; it should show only intentionally
   uncommitted work.

## Full example of a good final output (message written for this repo's real 2026-07-07 working tree)

```text
Add Phase 4 evaluation suite and close-out records

New lab tools: flagship_eval_lib.py (shared loader + chunked eval),
phase4_validate.py (8M-char re-eval + ONNX parity), eval_text8.py
(zero-shot external anchor), make_phase4_figures.py, playground.py
(blind A/B review UI).

Docs: phase4-flagship-evaluation-report.md with figures under
docs/figures/phase4/, ADR 0014 (Phase 4 closeout + evaluation
standard), ADR 0015 (Phase 5 scope), H022 (calibrated), phase5 intake
and Codex goal prompt. RESULTS.md Stage 51 entry carries the passed
human review.

Corpus, runs/, checkpoints, and ONNX binaries intentionally excluded.
```

One theme (the Phase 4 close-out package), every file nameable, hazards
named as excluded, no binary anywhere near the stage area.

## Mistakes to avoid (each one actually happened here)

- **Batch mega-commits.** Real history: "Hypothesis 9-12 tested" as a
  single commit spanning a week of unrelated work. One theme per commit.
- **A 471 MB corpus file is tracked in this repo right now** and about
  1.07 GB of corpus blobs live in history; GitHub's hard limit is 100 MB
  per file, so the repo cannot push until surgery. Never stage corpus
  text; check sizes before staging, not after.
- **Do not commit `runs/` or model binaries.** `runs/` is gitignored by
  design; the durable record is RESULTS.md. The `.onnx.data` weights file
  is 775 MB.
- **Do not run history rewrites casually.** `git filter-repo` happens
  only after `git bundle create` backup and explicit user sign-off, never
  as part of routine committing.
- **Never bypass the hooks** (`--no-verify`); the dash lint and any
  future large-file guard are load-bearing conventions here.
