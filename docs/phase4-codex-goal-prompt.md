# Phase 4 Goal Prompt for Codex · Confirm and Execute

Drafted 2026-07-02 by Claude from ADR 0013. Scope, in order: Stage 53
(H020 free-accelerator test), Stage 54 (H021 order-5 floor scaling, gated),
Stage 55 (flagship build). Read ADR 0013, H020, and H021 first; this prompt
adds the architectural plan and the confirmation checklist. Nothing here
overrides the pass/fail lines in the hypothesis docs; if this prompt and a
hypothesis doc disagree, the hypothesis doc wins and the disagreement gets
flagged in the stage notes.

## Ground rules for the whole phase

- Confirm before you code. Section "Confirm first" below lists every
  assumption Claude made from reading, not running. Any assumption that
  fails becomes a stage-notes line and, if it changes a decision line, a
  stop-and-report.
- Every run through the ADR 0012 visible-launch protocol (extend
  `run_phase3_visible.ps1` with phase 4 modes or add a sibling script).
  `--eval-mode sampled --eval-batches 16` everywhere; the default full eval
  stalls on this corpus.
- Preserve failed evidence (the Stage 52 OOM JSONL convention). Never delete
  a decision row; supersede with suffixed artifact names.
- Do not use the two sleep-inflated Stage 52 `seconds` rows (25.25M/1000/seed
  11 and 85.11M/2000/seed 19) for any throughput estimate. Use Stage 51 Gate
  C's `0.169` s/step at 25.25M and the Gate B smoke's `75.1` s per 200 steps
  at 85M.
- Every stage lands in `RESULTS.md` with the standard schema: command shape,
  corpus and split, seeds, trainable and frozen parameter counts, val NLL and
  bits/char, wall clock, and an interpretation stating what it does and does
  not prove.

## Confirm first (one short pass before Stage 53 code)

1. `cassandra_compare.py` forwards `--n-layer --n-head --n-embd
   --block-size --batch-size --grad-accum-steps --pos-encoding
   --activation-checkpoint --optimizer --muon-lr --eval-mode --eval-batches
   --train-shard-dir` into the per-config args the same way the Stage 52
   launches used them.
2. RESOLVED by the pre-handoff audit: the prior cache is keyed by corpus,
   shard dir, train chars, order, alpha, and backoff, so it WILL hit for
   `count_prior_ng4_all`, BUT only when `--prior-cache-dir` is passed;
   `count_ngram_prior_cache_path` returns `None` when the dir is unset and
   the prior silently rebuilds from shards. Every Stage 53 and 54 command
   must therefore carry
   `--prior-cache-dir .\experiments\tiny_language_lab\runs\stage52_prior_cache`
   (the hypothesis command shapes now include it).
3. With `--residual-base count-ngram --train-scope all`, the frozen
   `base_logits` tensor is excluded from the optimizer (buffer or
   `requires_grad=False`), Muon's parameter grouping picks up the full body,
   and `--zero-residual-head` still applies so step-0 output equals the
   prior exactly. Verify with a 10-step CUDA smoke: step-0 sampled NLL of
   `count_prior_ng4_all` must match the frozen floor to eval precision.
4. MOSTLY RESOLVED by the pre-handoff audit at code level: sampled eval
   draws from one generator seeded per run with `args.seed`, and arms
   consume it identically when steps, grad-accum, and eval interval match,
   so Stage 52 `random_full` rows ARE row-adjacent to new same-seed runs.
   Keep the cheap fallback anyway: if anything in the new config changes
   generator consumption (it should not), rerun the 25.25M `random_full`
   cells instead of reusing, and say so.
5. Checkpoint save exists on the Stage 51 path; RESUME has never been
   exercised. Confirm a save/kill/resume round-trip before the flagship
   (details in Stage 55).

## Stage 53 · H020 free-accelerator test

Architecture: this is deliberately near-zero new surface. One branch in
`config_args`:

```python
elif name == "count_prior_ng4_all":
    args.update({
        "residual_base": "count-ngram",
        "prior_order": 4,
        "train_scope": "all",
    })
```

No LoRA keys, no copy flags. Register the name in the `--configs` choices.
The known project gotcha (copy-arm branches hardcode `lora_alpha`) does not
bite here because there is no LoRA at all; state that in the stage notes so
the audit trail is explicit.

Runs: 25.25M (`--n-layer 8 --n-head 8 --n-embd 512`), rescaled corpus with
`--train-shard-dir ...tinystories_char_shards_500mb` AND
`--prior-cache-dir ...runs\stage52_prior_cache` (mandatory, see Confirm
item 2), budgets 200, 500, 1000, 2000, seeds `7 11 19`, modern recipe flags
exactly as the command shape in H020. Artifacts
`stage53_prior_all_25m_b{budget}.jsonl/.md`.

Baseline: Stage 52 `random_full` 25.25M rows (means `1.353240 / 1.106377 /
0.999363 / 0.922147`), reused per Confirm item 4. Context anchor: Stage 52
prior-arm rows, no rerun.

Decision per H020, applying its four clauses IN ORDER (KILL, CONFIRM,
E-wash, GRADED; first match wins, and the KILL margin rides on PAIRED
per-seed deltas while the early-advantage clauses use the unpaired seed
spread). If the E-interfere line trips (all 3 paired deltas worse than
`random_full` at 1000 or 2000 by more than `0.01`), run ONE extra cell,
2000 steps, `--muon-lr 0.005`, before recording the verdict; if
interference vanishes, record GRADED with an LR-sensitivity note.

Budget: about one GPU-hour. Do this stage first and report before starting
Stage 54 code; its verdict sets the flagship's initialization.

## Stage 54 · H021 order-5 floor scaling, Phase A gate then Phase B

### Phase A architecture (the only real new machinery in Phase 4)

Goal: a shard-native order-5 count prior with backoff to the existing
order-4 table, built in one streaming pass, cached to disk, and cheap to
look up on GPU during forward.

Recommended design (Codex may deviate with measurements, not taste):

1. **Counting pass.** Stream the existing shards exactly like the Stage 52
   order-4 builder. For each position, pack the 5-char context plus next
   char into a single int64 key: `key = ((((c1*33 + c2)*33 + c3)*33 +
   c4)*33 + c5)*33 + next` (all values are vocab indices, V=33, so the max
   key is 33^6 - 1, about 1.29e9, well inside int64 and int32 range fits
   too but use int64 for safety). Per shard, build the key tensor, run
   `torch.unique(keys, return_counts=True)` (or a numpy sort-reduce), and
   merge into a running dict or sorted-array accumulator. This keeps peak
   memory at shard scale, never corpus scale, which is the mistake that
   OOMed the first Stage 52 prior pass.
2. **Pruning / backoff threshold.** After counting, group keys by context
   (`key // 33`). Keep only contexts whose TOTAL count is at least K
   (`--prior5-min-count`, default 10, recorded in the report dict). The
   kept set is expected to be far below the 39.1M possible contexts because
   natural text is concentrated; report the kept-context count.
3. **Storage.** Two aligned tensors: `ctx_keys` (int64, sorted, one entry
   per kept context) and `ctx_logits` (float16, shape `[n_kept, 33]`,
   holding `log(alpha + count)` rows with the SAME smoothing alpha
   convention as the existing `build_count_logits`; do not tune alpha).
   Save both plus metadata (order, K, alpha, corpus hash) as the disk cache
   beside the Stage 52 cache, new hash.
4. **Forward lookup.** At each position: pack the last-5-chars context key,
   `torch.searchsorted(ctx_keys, key)`, verify the hit, gather the fp16 row
   and cast to fp32; on miss, fall back to the existing dense order-4
   lookup. Both `searchsorted` and `gather` are CUDA-native, so keep
   `ctx_keys` and `ctx_logits` on GPU if the smoke shows room, else CPU
   with a per-batch gather and measure the step-time cost. Positions with
   fewer than 5 chars of history use the order-4 path (same convention the
   dense builder uses for short history, whatever it is; confirm and
   match).
5. **Cap removal.** Raise the `--prior-order` cap to 5 ONLY for the
   backoff path; the dense path stays capped at 4 with a clear error, so
   nobody accidentally allocates a 4.8 GiB dense table.
6. **Equivalence check.** On a small sample corpus, brute-force order-5
   counts with the naive method and compare to the backoff builder at K=1:
   require `max_abs_diff = 0.0` on kept contexts, mirroring the Stage 52
   verification pattern.

Gate measurements to report: build wall-clock, `ctx_keys`/`ctx_logits`
bytes, kept-context count, validation BACKOFF RATE (fraction of eval
positions that fell back to order 4), peak CUDA in a 20-step smoke, and the
PAIRED floor comparison: `count_prior_ng5_lora_r2_floor` (new) and
`count_prior_ng4_lora_r2_floor` (already registered) in the SAME
invocation, seeds `7 11 19`, sampled eval, `--prior-cache-dir` set, so the
per-seed eval windows are identical and the deltas are paired. Stage 52
never ran a floor arm; its trained rank-2 means (`1.104469`, `1.113529`)
are sanity context for the measured order-4 floor, not the baseline. Hold
`count_alpha 0.1` and `ngram_backoff 1.0` at the Stage 52 values.

Gate line (from H021): mean paired delta (order-5 floor minus order-4
floor) at or below `-0.02` NLL with all three paired deltas negative. Fail
= KILL at Phase A: record everything, write the RESULTS entry, do NOT run
Phase B, and skip the order-5 option in the flagship recipe.

### Phase B (gate-conditional)

New configs `count_prior_ng5_lora_r2` and `count_prior_ng5_lora_r2_floor`
mirroring the ng4 pair (rank 2, `lora_alpha 2.0`, matching the existing r2
convention). 25.25M column, budgets 200/500/1000/2000, seeds `7 11 19`,
`--prior-cache-dir` set, reuse `random_full` rows per Confirm item 4.
Decision: does the crossover move at least one rung later than Stage 52's
(full model at-or-below prior at 1000)? Note the implied magnitude from
H021: that requires the new floor to undercut `random_full`'s 1000-step
mean `0.999363`, a drop near `0.105`, so GRADED (floor drop in roughly
`[0.02, 0.10)` with no rung move) is the most likely positive outcome and
should be reported as such, not rounded up to CONFIRM.
Artifacts `stage54_ng5_25m_b{budget}.jsonl/.md` plus a Phase A gate report
`stage54_gateA_order5_prior.md`. Optional 85M column only if the window has
slack after the flagship is scheduled.

## Stage 55 · Flagship build

### D3a sizing gate (run before the long run, report before launching)

Probe matrix, 200-step visible CUDA smokes, recording s/step and peak CUDA:

- 85.11M (`--n-layer 12 --n-head 12 --n-embd 768`) at block 128 and block
  256.
- One larger candidate, about 200M params (`--n-layer 16 --n-head 16
  --n-embd 1024`), at block 128 first; only try block 256 if the first
  smoke leaves at least 2 GiB headroom.
- If Stage 53 chose a prior base, run the chosen top candidate once WITH
  the prior attached to catch any lookup overhead or VRAM delta.

Selection rule (from ADR 0013 D3): the largest configuration whose peak
CUDA stays at least 1 GiB under the 8,188 MiB limit AND whose projected
wall-clock fits the window: token target at least 200M training chars seen
(chars/step = batch 8 x block x grad-accum 2; block 256 gives 4,096
chars/step, so 50k steps is about 205M chars), stretch goal one epoch
(420M). Prefer 3 seeds (`7 11 19`) if the projected total is under about 36
GPU-hours; otherwise one flagship seed at full budget plus two replicate
seeds at a reduced budget, stated plainly.

### Checkpoint-resume proof (required, ADR 0013 accepted costs)

Before the long run: launch the chosen config for 400 steps with periodic
checkpointing, kill it at about step 200, resume from the checkpoint, and
confirm (a) it resumes at the right step, (b) optimizer state (Muon) and
the data-shard cursor are restored or their non-restoration is understood
and recorded, (c) final val NLL is sane against an unkilled 400-step run.
Stage 53's confirm pass established that resume does NOT exist (only
checkpoint save via `--checkpoint-dir`), so this is a BUILD item, not a
verification item; a 10-hour run without resume on a laptop that sleeps is
how Stage 52 got its two bad timing rows, except worse.

### The run

Recipe assembled from verdicts, no taste: base = frozen count prior if
Stage 53 returned E-accel or E-wash, else random init; prior order = 5 if
Stage 54's gate passed, else 4. Modern recipe otherwise (RoPE, Muon,
activation checkpointing, sampled eval). Checkpoints and samples saved per
seed; Stage 46 generation score sheet extended per ADR 0011 D3's bar;
artifacts `stage55_flagship_*.jsonl/.md` plus a checkpoint directory.
NO coherence or quality claim in any doc: write the proxy scores and
"human review pending," same as Stage 51. Mert reviews after 2026-08-21 if
the run lands inside the window.

### Nsight DL Designer closeout artifact

After Phase 4 succeeds, create one NVIDIA Nsight Deep Learning Designer ONNX
export from the final successful flagship checkpoint. This is a closeout
artifact, not a training input and not a substitute for the `.pt` checkpoint.
Put it under:

```text
experiments/tiny_language_lab/artifacts/phase4/nsight_dld/
```

Use the exporter added for this purpose, replacing the checkpoint path and name
with the final successful checkpoint step:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\export_stage55_nsight_dld.ps1 `
  -Checkpoint C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt `
  -OutDir .\experiments\tiny_language_lab\artifacts\phase4\nsight_dld `
  -Name stage55_seed7_final_success_b1_s256 `
  -SeqLen 256 `
  -BatchSize 1 `
  -Device cpu
```

Record the `.onnx` path and `.manifest.json` path in the Stage 55/Phase 4
RESULTS.md closeout. Do not create this artifact after a failed or partial
Phase 4 run; in that case, keep only the normal checkpoint and run evidence.

## Scheduling

Before 2026-07-20: Confirm-first pass, Stage 53 complete and reported,
Stage 54 Phase A (gate) complete and reported. Inside 2026-07-20 to
2026-08-21: Stage 54 Phase B if gated in, D3a sizing gate,
checkpoint-resume proof, flagship long run. The Stage 51 human sample
review is Mert's own pre-window item and is not Codex work, but if it
returns a verdict before the flagship launches, fold its quality-bar
implications into the Stage 46 score-sheet extension.

## Reporting back to Claude

After each stage: the RESULTS.md entry, the runs artifacts, and one short
note stating which decision line fired (E-accel/E-wash/E-interfere/GRADED
for Stage 53; gate pass or Phase-A KILL plus crossover read for Stage 54;
chosen config, projected and actual wall-clock, and proxy scores for Stage
55). Claude folds Stage 53 and 54 verdicts into the flagship recipe
confirmation before the long run launches; do not start the flagship long
run without that confirmation if the verdicts came out GRADED or
contradictory.

## Update from Claude · 2026-07-06 · post-seed-7 facts, some supersede text above

Read this section as authoritative where it conflicts with earlier sections.

1. Seed 7 flagship is COMPLETE (exit 0): final val NLL `0.556410`,
   `0.802730` bits/char, peak CUDA `4,007.55 MiB`, summary at
   `runs/stage55_flagship_200m_b50000_seed7.md`. The seed-11 20k replica
   launched 2026-07-06 14:32. Stage 51 human review PASSED the same week
   (recorded in the Stage 51 RESULTS.md entry and the ADR 0011/0013
   statuses); the flagship review bar is "clearly better than the Stage 51
   samples."

2. Checkpoint locations changed mid-run and SUPERSEDE the paths above. The
   OneDrive checkpoint save failed at step 35000 (`WinError 5` on the
   `.pt.tmp` rename; error rows preserved in
   `runs/stage55_flagship_200m_b50000_seed7_onedrive_checkpoint_error_20260706.*`),
   and the run was recovered with `--checkpoint-dir` pointed at
   `%TEMP%\cassandra_runs\stage55_flagship_checkpoints`. Windows Storage
   Sense is ENABLED with temp cleaning on this machine, so `%TEMP%` is a
   purgeable location. The two final seed-7 artifacts (`..._step050000.pt`
   and the no-suffix final) were copied with byte-count verification to the
   durable, non-synced directory:

   ```text
   C:\cassandra_runs\stage55_flagship_checkpoints
   ```

3. Action items for Codex, in order:
   - Seed 11: when the run finishes, copy its final checkpoint from the
     `%TEMP%` directory to `C:\cassandra_runs\stage55_flagship_checkpoints`.
   - Seed 19: launch with `--checkpoint-dir
     C:\cassandra_runs\stage55_flagship_checkpoints` directly (not OneDrive,
     not `%TEMP%`). Update the checkpoint-dir constant in
     `run_phase4_visible.ps1`'s flagship mode accordingly.
   - ONNX closeout: the command in "Nsight DL Designer closeout artifact"
     above names a `runs\stage55_flagship_checkpoints` checkpoint path that
     no longer holds the final model. Export from the secured copy instead:
     `C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`.
     The `-OutDir` (`artifacts/phase4/nsight_dld`) and naming are unchanged.
     Also note `export_stage55_nsight_dld.ps1` and `export_nsight_dld.py`
     default to the stale OneDrive step-15000 path; either update the
     defaults or always pass `-Checkpoint` explicitly.
   - Stage 55 RESULTS.md entry: include the OneDrive failure and recovery as
     run evidence (the error jsonl/md, the split resume legs, and the fact
     that wall-clock is a sum of legs because elapsed resets per
     invocation).
   - After the ONNX is exported and verified, the storage cleanup tiers in
     `docs/phase4-flagship-midrun-report.md` (Section 3 plus the 2026-07-06
     addendum) apply: the OneDrive flagship checkpoint dir (steps 5k-30k
     plus the stranded `step035000.pt.tmp`, `10.8 GB`) and the resume-proof
     dir (`9.27 GB`) become deletable; retire the `%TEMP%` dir after its
     final artifacts are moved.

4. Overnight context only: the 02:00 `MUSAHIT_Nightly` Ollama task was
   suppressed for 2026-07-07 only (one-shot skip flag consumed by its own
   launcher), so tonight's replica window is protected. No Codex action.

## Codex closeout note - 2026-07-07

Seed 11 and seed 19 completed their 20,000-step reduced replicas. Their final
and no-suffix checkpoints were copied from `%TEMP%` to
`C:\cassandra_runs\stage55_flagship_checkpoints` with byte-count verification.
Seed 19 also trained in `%TEMP%` before this note was read; the durable copies
and the updated launcher default are the remediation.

The final DL Designer export was created from
`C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`
and verified with `onnx.checker`. The closeout artifacts are:

- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.onnx`
- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.onnx.data`
- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.manifest.json`

`run_phase4_visible.ps1`, `export_stage55_nsight_dld.ps1`, and
`export_nsight_dld.py` now default to the durable final checkpoint path for
Stage 55 flagship continuations and exports.

## Links

- `docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md`
- `docs/hypotheses/020-frozen-prior-free-accelerator.md`
- `docs/hypotheses/021-prior-order-floor-scaling.md`
- `docs/phase4-intake.md`
- `experiments/tiny_language_lab/runs/stage52_h019_crossover_scaling_summary.md`
- `docs/decisions/0012-visible-terminal-experiment-launches.md`
