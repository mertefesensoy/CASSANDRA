# Phase 5 · Success Criteria and Operating Brief

Date: 2026-07-07 · written by Claude at the user's request before an
unmonitored period. This is the plain-language companion to ADR 0015 and
`docs/phase5-codex-goal-prompt.md`: what success means per workstream, who
owns what, and when to stop and ask. If this file and ADR 0015 disagree,
ADR 0015 wins.

## What success means, per workstream

### Stage 56 · H022 broad-corpus test (runs FIRST, about 13 GPU-hours)

Success is a CLEAN VERDICT, not a good model. Three acceptable outcomes on
the text8 TEST split (chunked convention, final checkpoint):

- **At or below `1.70` bits/char = CONFIRM.** The specialization gap was a
  data effect; the char substrate survives; Stage 58 freezes on char.
- **At or above `2.10` (surviving the 20k lower-LR guard) = KILL.** The
  substrate is implicated; H023 (real BPE, 5k to 10k merges) becomes the
  next stage; Stage 58 runs on BPE.
- **Between = GRADED.** The BPE-times-broad cell decides next.

Failure is an UNREADABLE result, not a KILL: replicas more than `0.10`
apart, a missing contamination assert, a verdict recorded without its
guard rerun, or the instability guard tripping unaddressed (final in-run
val worse than the run's own 20k value by more than `0.05`).

### Stage 57 · Recipe v2 gates (cheap, about 1 to 2 GPU-hours)

Success = one locked recipe block in RESULTS.md with measured deltas.
Pre-registered lines: bf16 adopts at within `0.01` NLL of fp32 at 200
steps AND at least 1.4x throughput; cosine adopts if it wins the 5000-step
pair at 25.25M; `--vocab-chars` and `--checkpoint-keep` adopt on smoke
pass; block 512 produces a timing row only. All-gates-fail is a valid
outcome (Recipe v1 stays, budgets get re-estimated by Claude). A skipped
gate is the only failure.

### Behavior probe (eval-only, no training)

Success = one number: flagship copy accuracy on the letters-only
memorization-proof probe versus chance `0.0625`. At or above chance plus
`0.10` reopens the behavior axis; below keeps it closed and extends
ADR 0008's capacity reading to 200M. Either outcome is progress.

### Open-source preparation (D2)

Success = release-READY, not released: corpus untracked before any
commit, the cleanup tiers executed (about 35 GB), history surgery done
AFTER a bundle backup AND explicit user sign-off, licensing notes written
(facts, not the choice), fp16 model-only exports, model card drafted.
Hard rule: **nothing is pushed publicly without the user present.**

### Stage 58 · H024 developmental experiment (DO NOT START)

Claude writes H024 only after Stage 56's verdict. Success, when it runs =
the three-arm answer (COLD vs CURRICULUM vs MIXTURE at one fixed budget)
with text8-test primary and the TinyStories retention curve. Any clean
outcome is a finding; agreement with Stage 53's interference law would be
the phase headline. Codex may pre-build the mixture-shard interleaver
only.

## The user's lane (nothing else blocks on him)

1. Round-2 A/B votes: 10 to 15 pinned flagship-vs-Stage-51 votes across
   5 or more prompts (the playground's Pairing dropdown and suggested
   prompts make this about ten minutes). This settles the provisional
   2-2 coherence read (ADR 0014 addendum) and unblocks the release
   narrative.
2. Sign-offs when asked: git history surgery (before any force-push),
   license choice (after the due-diligence doc), release go or no-go
   (later).

## Stop-and-ask triggers (pause work, flag for Claude)

- **C: free space below 15 GB before launching any run, or below 5 GB at
  any time.** (Added 2026-07-08 after the disk hit `0.04 GB` free and the
  Stage 56 seed-7 step-50000 checkpoint save corrupted mid-write; the
  pre-approved cleanup tiers were executed the same morning, freeing
  `42.7 GB`. Check `Get-PSDrive C` before every launch; each 85M
  checkpoint is `~0.65 GB`, each 200M checkpoint `~1.5 GB`.)
- Any checkpoint writing to OneDrive or `%TEMP%` again.
- Any KILL about to be recorded without its guard rerun.
- Stage 56's instability guard trips.
- Stage 58 starting before the H024 doc exists.
- Every Stage 57 gate failing (needs a rebudget, not improvisation).
- Anything that would push content off this machine.

## The one-sentence bar

Phase 5 succeeds if, by the end of the internship window (2026-08-21),
the specialization gap has a measured CAUSE, the recipe is measurably
faster, the childhood question has a three-arm answer with forgetting
curves, and the repo could go public the day the user decides it should.

## Links

- `docs/decisions/0015-phase-5-developmental-training-and-open-source-posture.md`
- `docs/phase5-codex-goal-prompt.md` · `docs/phase5-intake.md`
- `docs/hypotheses/022-broad-corpus-specialization-gap.md`
- `docs/decisions/0014-...` addendum (the provisional A/B read)
