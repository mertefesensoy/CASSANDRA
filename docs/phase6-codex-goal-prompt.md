# Phase 6 Goal Prompt for Codex · Confirm and Execute

Drafted 2026-07-21 by Claude from ADR 0017. Scope, in order: Stage 59
(H025, fully specced and audited) and, ONLY if its pre-registered gate
fires, Stage 60 (the dual-register flagship scale-up and its Hugging Face
packaging). Read ADR 0017 and H025 first. If this prompt and
`docs/hypotheses/025-rehearsal-dose-response-and-mixing-law.md` disagree,
the hypothesis doc wins and the disagreement gets flagged in stage notes.
The user's lane and stop-and-ask triggers from
`docs/phase5-success-criteria.md` carry over unchanged; the internship
window (weekdays autonomous until 2026-08-21) makes the stop-and-ask rule
MORE binding, not less: when a trigger fires, park the branch and continue
elsewhere.

## Ground rules for the whole phase

- Confirm before you code; every failed assumption is a stage-notes line.
- Every run through the ADR 0012 visible-launch protocol with the
  keep-awake helper; checkpoints go DIRECTLY to `C:\cassandra_runs\...`
  (never OneDrive, never `%TEMP%`); the 15 GiB free-space launch gate
  binds every launch.
- ADR 0014 evaluation conventions bind: decision numbers come from the
  deterministic chunked evals (`eval_text8.py`,
  `eval_tinystories_retention.py`); `--eval-mode sampled` is in-run
  monitoring only.
- Determinism: seeds are `7 11 19` for the proxy sweep, `11 19` for
  Part 2 (a REGISTERED deviation, H025 explains it), corpus generators
  carry fixed seeds, every mixture directory gets a `.meta.json` with a
  SHA-256.
- Preserve failed evidence with suffixed artifact names; never delete a
  decision row. Every stage entry records command shape, corpus, seeds,
  trainable parameters, val NLL plus bits/char, task metrics, and what the
  result does and does not prove.
- Reproducibility (ADR 0015 D6): any number that could reach a
  release-facing document must regenerate from a command in the repo.
- The 02:00 MUSAHIT task collides with overnight GPU runs. For any run
  that will span 02:00, either plan completion before it or drop
  `SKIP_NEXT_RUN.flag` next to MUSAHIT's `run_nightly.ps1` (one night per
  flag; verify the guard still exists in that script before relying on
  it).

## Stage 59 · H025, the rehearsal dose-response (rung 69)

The hypothesis doc carries the full design, decision lines, and risk
register. This section is the execution ordering and the build list.

### Part 0 first (minutes, no training)

```powershell
python .\experiments\tiny_language_lab\eval_letters_copy_probe.py `
  --checkpoint C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt `
  --device cuda `
  --out .\experiments\tiny_language_lab\runs\stage59_cold_letters_probe.json `
  --summary .\experiments\tiny_language_lab\runs\stage59_cold_letters_probe.md
```

Remaining flags at their Phase 5 probe defaults; record them in the
summary. Report accuracy against chance `0.0625`. The `0.1625` reopen
line is Claude's to read, not Codex's; just record the row either way and
do not branch on it.

### Part 1 build list (in dependency order)

1. **Proxy config registration.** Add a 3.2M-class Recipe v2 mixture
   config family to `cassandra_compare.py` (`config_args` branch plus
   `--configs` choices, per the standing convention). Requirements: the
   H019 3.2M size lineage, union 33-character vocabulary via the Stage 57
   `--vocab-chars` override, Muon, fp32, cosine schedule, block 256,
   `random_full` surface (no prior; nothing here touches frozen-prior
   machinery). CONFIRM FIRST: print and record the actual parameter count
   of this never-trained config in the smoke log; H025 deliberately does
   not assert it.
2. **Mixture shard directories.** Six doses via the verified generator,
   weight pairs from H025: `w = 0.05 -> 1:19`, `0.10 -> 1:9`,
   `0.20 -> 1:4`, `0.30 -> 3:7`, `0.50 -> 1:1`, plus `w = 0` which is the
   existing Stage 58 broad shard directory used as-is (no new generation).
   Tiny source is `corpus\tinystories_char_shards_500mb`, broad source is
   the exact broad shard directory the Stage 58 COLD launcher used
   (recorded in `runs/phase5_stage58-cold_20260714_002318_launcher.log`;
   confirm it there rather than assuming a name). Chars-per-step arithmetic
   is pinned at 4,096 (block 256 times batch 8 times grad-accum 2, the
   Stage 58 convention); `--total-chars` = the smoke-set step budget times
   4,096 with headroom so no shard repeats inside a run. One `.meta.json`
   per directory, SHA-256 included, dose arithmetic stated.
3. **Throughput smoke, then freeze the sweep budget.** One short timed run
   (200 steps is fine) of the proxy config sets seconds-per-step; pick the
   per-run step count so the full 18-run sweep (6 doses times 3 seeds)
   fits one GPU-evening, and REGISTER the chosen count in the launcher log
   BEFORE the sweep starts. Every sweep run records sampled broad val NLL
   and sampled TinyStories val NLL, same eval batch conventions across all
   18 runs. Output: `runs/stage59_proxy_sweep.jsonl` and `.md`.
4. **The fit script.** New committed tool
   `experiments/tiny_language_lab/make_mixing_law_fit.py`, beside the
   trainer. Contract: reads `stage59_proxy_sweep.jsonl`; fits BOTH the
   exponential and the power-law form of the mixing law to mean broad val
   loss across seeds as a function of `w`; reports both fits with
   residuals; registers ONE as primary in its output; writes
   `runs/stage59_mixing_law_fit.json` and `.md` containing the two
   pre-registered predictions (the predicted 85M cost at `w = 0.10`, and
   `w*` under the provisional retention bound from H025). This file MUST
   exist, with the predictions in it, before any Part 2 launch: that
   ordering is the falsifiability mechanism, treat it like a launch gate.

### Part 2 (two 85M arms, about 5.6 GPU-hours each)

1. **Indicative pre-step** (three minutes, feeds no decision line): score
   `stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step020000.pt`
   on text8 TEST; record it as the only obtainable 20k-step toll reading
   at the 29.8 percent dose, mid-cosine caveat stated. If it lands wildly
   off the `+0.028` anchor, record the discrepancy in the launcher log
   before the arms start (do not silently proceed).
2. **Retention baselines** (one-minute evals, registered in H025): run
   `eval_tinystories_retention.py` on the existing COLD b20000 seed-11 and
   seed-19 finals under `C:\cassandra_runs\stage58_dev_cold_checkpoints`,
   same corpus and conventions as Stage 58.
3. **The arms.** MIXTURE `w = 0.10`, 85.11M, Recipe v2, 20,000 steps,
   `--checkpoint-every 5000`, seeds 11 then 19, Stage 58 launcher pattern
   with the `w = 0.10` shard directory. Then per final: deterministic
   text8 TEST and the retention series, exactly the H025 command shapes.
4. **Verdict application.** Apply H025's partition in its stated
   precedence order (INCONCLUSIVE checked first, then CONFIRM, then KILL,
   then GRADED; a seed is CONFIRM-side at `d <= +0.010` and `r >= 1.0`,
   KILL-side at `d >= +0.020`). The instability guard is pinned: each
   arm's final sampled broad val NLL must improve on its own step-10,000
   value or that arm reruns before any verdict. On a seed split, the
   escalation is the registered seed-7 pair (COLD 20k plus MIXTURE 20k,
   about 11.2 GPU-hours); do not improvise a different tie-break.
5. **Record.** RESULTS.md stage entry in the standing format, one line
   naming which decision line fired, run files cited per number. Claude
   reads the verdict back before anything Stage 60 happens.

## REDESIGN NOTICE (2026-07-22) · read this before the sections below

Stage 59 resolved E-partial and the ADR 0017 D2 gate is SPENT: the
mixture flagship section below never launches and stays only as the
historical record. The redesigned Phase 6 execution order is now:

1. **Stage 60 = H026 eval-only circuit mapping** (rung 71). Authority:
   `docs/hypotheses/026-diverse-data-circuit-formation.md`. Probe every
   surviving Stage 58/59 checkpoint with the frozen letters probe at
   Part 0 defaults through the new committed wrapper
   `make_stage60_circuit_matrix.py` (H025-Part-0 probe tool computes no
   hashes itself): one row per checkpoint with checkpoint and probe
   SHA-256, consolidated into `runs/stage60_circuit_matrix.jsonl` and
   `.md`. Trains NOTHING; about one GPU-hour total. Hash rule per H026:
   verify where a recorded hash exists, exclude-and-flag mismatches,
   mark `hash_unverified` where no record exists. First row is the
   determinism check: re-probe the COLD seed-7 final and require exact
   reproduction of `0.194336` before trusting any other row. Record
   missing or incompatible rows with reasons.
2. **Stage 61 = pure-broad instrumented 200M flagship** (rung 72).
   Authority: ADR 0018 D3 to D5. Reuse the gate sequence written for the
   old Stage 60 below (sizing gate, sustained 5,000-step throughput
   measure, resume drill, launch protocol, evaluation battery, packaging)
   with these substitutions: the corpus is PURE text8 broad shards (no
   mixture build, no dose rule, `w_flag` machinery void); checkpoint
   retention is `--checkpoint-keep 12` or higher (keep-all for the run;
   the old section's `--checkpoint-keep 2` is OVERRIDDEN because it
   would erase the instrumentation ladder), and EVERY 5,000-step
   checkpoint survives until its probe and retention rows are recorded
   in `runs/stage61_instrumentation.jsonl` (intermediates deletable only
   after that, finals never); the letters
   probe and the TinyStories retention eval run per checkpoint within a
   day of it landing; the publish bars are ADR 0018 D5, not ADR 0017 D4
   (no retention bar; the (a) text8 line `1.357318` and the sample-review
   gate stand).
3. Stage 61 may launch before Stage 60's verdict is read (it is
   measurement either way), but no packaging step proceeds before both
   the Stage 60 read-back and the D5 bars.

Everything in "Ground rules for the whole phase" still binds, including
the MUSAHIT night rule and the stop-and-ask triggers.

## Stage 60 · The Phase 6.1 flagship (rung 70) · SUPERSEDED 2026-07-22, GATE SPENT, DO NOT RUN

Launches ONLY on H025 = E-cheap-rehearsal (ADR 0017 D2). If any other
line fired, stop after Stage 59 and hand back to Claude for the
retrospective intake. Everything below is conditional.

### Sizing and budget gates (confirm-first, all recorded before launch)

1. **D3a-analog sizing gate.** Instantiate the Stage 55 configuration
   lineage (201,609,249 parameters, block 256, RoPE, Muon) with the union
   33-character vocabulary; print and verify the parameter count (the
   vocab change moves it slightly off the Stage 55 number; record the new
   exact count). Verify activation memory at block 256 fits with
   headroom on the 8 GB card at the training batch size.
2. **Sustained throughput measure.** 5,000 timed steps at the full
   configuration on the mixture corpus. The measured seconds-per-step
   sets the budget: 50,000 steps if it prices at or under the 40 GPU-hour
   envelope, otherwise the largest 5,000-multiple that fits 40 hours. If
   even 30,000 steps does not fit, STOP and ask the user (ADR 0017
   reopen item 3): flagship size versus budget is a user fork.
3. **Dose and corpus.** `w_flag` follows ADR 0017 D3's registered rule:
   `w*` ONLY IF the transfer read registered USEFUL AND the fit's
   predicted `d(w*)` is at or below `+0.010` AND `|w* - 0.10| <= 0.10`;
   otherwise `0.10`, the only dose directly confirmed at 85M. Build ONE
   continuous mixture at `w_flag` sized to the full budget (at 50,000
   steps times 4,096 chars that is 204,800,000 characters; at
   `w_flag = 0.10` that is 20,480,000 TinyStories and 184,320,000 broad
   characters, about 2.05 epochs of the 90M-character text8 train split;
   redo this arithmetic for the actual budget and dose and put it in the
   `.meta.json`).
4. **Resume drill.** Kill and resume a short run of the exact
   configuration and verify checkpoint lineage (step count,
   `formation_forward_passes`, SHA-256) before the long run starts
   (ADR 0013 precedent; Stage 58's crash history is the reason).

### The run

Visible launcher, keep-awake, `--checkpoint-every 5000`,
`--checkpoint-keep 2`, fp32, cosine to `--lr-final-frac 0.1`, one
continuous schedule over the full budget, checkpoints to a fresh
`C:\cassandra_runs\stage60_flagship_checkpoints`. Plan MUSAHIT nights.
Any pause or crash goes through the run-pitstop protocol with a dated
pitstop note in `runs/`.

### The evaluation battery (all deterministic, all recorded)

1. text8 TEST bits/char (`eval_text8.py --split test`, full 5M chars).
2. TinyStories retention series over the checkpoint ladder
   (`eval_tinystories_retention.py --checkpoint-dir`, the 1,499,904-char
   protocol), plus the final's retention number.
3. The letters copy probe on the final.
4. Generation samples for the user's review (playground conventions,
   proxy scoring recorded, no coherence claim before the human review,
   ADR 0011).
5. Success bars are ADR 0017 D4 and are Claude's to read: (a) strictly
   below `1.357318` text8 TEST; (b) retention at least `1.0` bits/char
   better than `3.556502`; (c) probe recorded; (d) sample review passes.
   Codex records the numbers and the margins; Claude writes the verdict
   and the closeout ADR.

### Packaging (prepare everything, publish nothing)

1. fp16 model-only safetensors export of the final
   (`export_model_only_checkpoint.py`), plus config json (layers, heads,
   dims, block, alphabet), the 33-char codec as a plain list, and a
   minimal `load_model` snippet built on `flagship_eval_lib.py`.
2. Model card grown from `docs/phase5-model-card-draft.md`: architecture,
   data disclosures (TinyStories CDLA-Sharing-1.0, text8 Wikipedia
   lineage, links), every number with its regenerating command, the
   cross-scale retention-anchor caveat from ADR 0017 D4, limitations
   stated first.
3. fp32 resume-capable checkpoint staged for the PRIVATE Hub archive
   repo; checksums computed and recorded. The upload-verified-BEFORE-
   local-delete rule binds.
4. The user presses every public button (Hub repo creation, uploads,
   visibility). Stage 60 ends at "prepared and verified".

## Reporting back to Claude

After Part 0, after Part 1's fit file exists, after each Part 2 arm, and
after any Stage 60 gate: the RESULTS.md entry or gate record, the run
artifacts, and one line naming which decision line fired or which gate
passed. Do not start any 10-hour-class run while a gate verdict that
shapes it is unread. Claude's read-backs happen at: the Stage 59 verdict
(decides Stage 60), the Stage 60 gate records (before the long run), and
the Stage 60 evaluation battery (decides publication-worthiness and the
Phase 6 closeout ADR).

## Links

- `docs/decisions/0017-phase-6-rehearsal-dose-flagship-gate.md` (the authorizing ADR)
- `docs/hypotheses/025-rehearsal-dose-response-and-mixing-law.md` (the Stage 59 authority)
- `docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md`
- `docs/phase5-success-criteria.md` (stop-and-ask triggers, user lane)
- `docs/open-source-release-and-disk-plan.md` · `docs/phase5-licensing-notes.md` · `docs/phase5-model-card-draft.md`
- `experiments/tiny_language_lab/eval_text8.py` · `eval_tinystories_retention.py` · `eval_letters_copy_probe.py` · `make_mixture_shards.py` · `export_model_only_checkpoint.py` · `flagship_eval_lib.py` · `playground.py`
