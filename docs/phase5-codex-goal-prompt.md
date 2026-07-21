# Phase 5 Goal Prompt for Codex · Confirm and Execute

Drafted 2026-07-07 by Claude from ADR 0015. Scope, in order: Stage 56
(H022 broad-corpus test, already fully specced), Stage 57 (Recipe v2
gates), the D4 behavior probe (eval-only), the D2 open-source preparation
tasks, and Stage 58 (H024, which Claude writes only after Stage 56 reads
out). Read ADR 0015 and H022 first. Success definitions, the user's lane,
and the stop-and-ask triggers live in `docs/phase5-success-criteria.md`;
consult it before recording any verdict, especially while the user is not
monitoring. If this prompt and a hypothesis doc
disagree, the hypothesis doc wins and the disagreement gets flagged in
stage notes.

## Ground rules for the whole phase

- Confirm before you code; every failed assumption is a stage-notes line.
- Every run through the ADR 0012 visible-launch protocol; checkpoints go
  DIRECTLY to `C:\cassandra_runs\...` (ADR 0014 D1: never OneDrive, never
  `%TEMP%`).
- ADR 0014 evaluation conventions bind: closeout numbers come from the
  chunked deterministic eval (`phase4_validate.py` convention or
  `eval_text8.py`); `--eval-mode sampled --eval-batches 16` is in-run
  monitoring only.
- Preserve failed evidence with suffixed artifact names; never delete a
  decision row.
- Reproducibility rule (ADR 0015 D6): any number that could appear in a
  release-facing document must regenerate from a command in the repo.

## Stage 56 · H022 broad-corpus specialization-gap test

Fully specced in `docs/hypotheses/022-broad-corpus-specialization-gap.md`
(CONFIRM at or below `1.70` text8-test bits/char after the 2026-07-07
calibration; KILL at or above `2.10` surviving the lower-LR guard). The
command shape, corpus-prep script contract
(`make_text8_shards.py` with the test-range contamination assert), and
confirm-first items are in the hypothesis doc. Run this FIRST; its verdict
freezes Stage 58's substrate.

## Stage 57 · Recipe v2 gates (ADR 0015 D3)

Adopt only what passes. Report every gate in RESULTS.md whether it passes
or fails.

1. **bf16 autocast.** Add a `--precision {fp32,bf16}` flag wrapping the
   training forward and loss in `torch.autocast("cuda", dtype=torch.bfloat16)`
   (optimizer steps and eval logits stay fp32; no GradScaler needed for
   bf16). Gate: 200 steps at 25.25M, same seed, fp32 vs bf16. ADOPT if
   the bf16 sampled NLL is within `0.01` of fp32 AND steps per second
   improve by at least 1.4x. Report peak CUDA for both.
2. **Cosine LR warmdown.** Add `--lr-schedule {constant,cosine}` and
   `--lr-final-frac` (default `0.1`). Gate: 5000 steps at 25.25M, seed 7,
   constant `0.01` vs cosine `0.01 -> 0.001`. ADOPT if the cosine arm's
   final sampled NLL is at or below the constant arm's.
3. **`--checkpoint-keep N`** (rolling window, optimizer state only needed
   in the newest) and an fp16 model-only archival save mode. Plumbing:
   smoke-test save, prune, resume; no science gate.
4. **`--vocab-chars` override** so a checkpoint can carry the union
   33-char alphabet while training on a 27-char corpus. Required by
   Stage 58's retention metric; smoke-test that a text8-trained model
   with the 33-char vocab scores TinyStories text without error.
5. **Block 512 row.** One 200-step timing-and-VRAM row at 85M block 512
   under whatever precision passed gate 1. Decision INPUT for Stage 58
   sizing, not an adoption.

Lock the passing set as Recipe v2 and state it in one RESULTS.md block:
flags, measured deltas, the locked command fragment.

## D4 behavior probe (eval-only, no training)

The Stage 42 memorization-proof copy corpus uses digits, which the
33-char vocab lacks. Build a letters-only generator variant (payload
alphabet drawn from `a` to `p`, case ids spelled with letter sequences,
same verified-answer construction), then score the flagship checkpoint
zero-shot with the existing copy-probe machinery at block 256. Report
copy accuracy against chance (`1/16 = 0.0625`). The reopen line (Claude
reads it, not Codex): flagship at or above chance plus `0.10`.

## D2 open-source preparation (order matters, nothing goes public)

The detailed execution plan with owners and the post-Phase-5 disk
end-state budget lives in `docs/open-source-release-and-disk-plan.md`;
this section is the Codex-lane subset.

1. Before ANY further commit: `git rm --cached` the tracked corpus files
   (`tinystories_char_seed.txt` and companions) alongside the already
   modified `.gitignore`.
2. Storage cleanup tiers per `docs/phase4-flagship-midrun-report.md`
   Section 3 plus addendum (now unblocked by ADR 0014 D5).
3. History surgery, gated on explicit user sign-off in the session log:
   `git bundle create ..\cassandra-backup.bundle --all`, then
   `git filter-repo` stripping `experiments/tiny_language_lab/corpus/`
   blobs, then verify `git count-objects -vH` shrank accordingly. Do NOT
   force-push anywhere without the user present.
4. Licensing due-diligence doc (`docs/phase5-licensing-notes.md`): code
   license candidates, the TinyStories dataset license and what it
   implies for released weights, text8's Wikipedia lineage. Facts and
   citations only; the choice is the user's.
5. fp16 model-only archival export of the three finals (uses the Stage 57
   item-3 save mode).
6. Model card draft grown from `docs/phase4-flagship-evaluation-report.md`
   (architecture, data, eval numbers with regeneration commands, intended
   use, limitations incl. the `2.07`-bit specialization gap).

## Stage 58 · H024 developmental experiment (DO NOT START)

Claude writes the H024 hypothesis doc after Stage 56's verdict freezes
the substrate. Pre-registered design lives in ADR 0015 D1. What Codex can
prepare ahead: the mixture-shard interleaver (deterministic ratio mixing
of two shard dirs) behind a `--train-shard-dirs A:B:ratio` style flag,
dark until H024 lands.

## Reporting back to Claude

After each item: the RESULTS.md entry, run artifacts, and one line naming
which decision line fired. Claude folds Stage 56's verdict into H024
before Stage 58 launches; do not start any 10-hour-class run while a gate
verdict that shapes it is unread.

## Links

- `docs/decisions/0015-phase-5-developmental-training-and-open-source-posture.md`
- `docs/hypotheses/022-broad-corpus-specialization-gap.md`
- `docs/phase5-intake.md` · `docs/phase4-flagship-evaluation-report.md`
- `experiments/tiny_language_lab/flagship_eval_lib.py`,
  `phase4_validate.py`, `eval_text8.py`, `playground.py`,
  `make_phase4_figures.py`
