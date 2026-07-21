# Phase 4 Flagship Mid-Run Report and Post-Run Storage Plan

Date: 2026-07-05
Status: seed 7 paused at step 32,000 of 50,000; durable checkpoint at step
30,000 (`stage55_seed7_before35000_pitstop_20260704.md` is the resume
authority). Written by Claude while training is paused, per the user's
request for a progress report, an architecture retrospective, and a storage
cleanup strategy for after the ONNX export is ready.

## 1 · Where the project stands

Fifty-five stages, twenty-one hypotheses, and thirteen ADRs in, the lab has
moved from "can a frozen bigram prior replace gradient training" to training
its first hardware-scale model with a recipe assembled entirely from measured
verdicts. Phase 4 so far:

- Stage 53 (H020) KILL E-interfere: the frozen order-4 prior helps early but
  is a late-training handicap under full-body Muon training (`+0.100` NLL at
  2000 steps, surviving the lower-LR rerun at `+0.055`). This fired ADR 0013's
  assembly rule toward random init for the flagship.
- Stage 54 (H021) CONFIRM: the sparse order-5 backoff prior beats the order-4
  floor by `~0.117` NLL and moves the 25.25M crossover from 1000 to 2000
  steps. It improves the tiny-surface recipe but does not touch the flagship.
- Stage 55 D3a gates PASS: sizing gate chose 201.61M params (L16 · H16 ·
  D1024, block 256) with `3,233 MiB` peak CUDA and `~4,954 MiB` headroom;
  checkpoint-resume proven exactly (resumed run matched the unbroken run's
  step-400 eval to full precision).

### Flagship live status (seed 7)

Recipe: `random_full`, 201,609,249 trainable params, vocab 33 (char), block
256, batch 8 × grad-accum 2 (4,096 chars/step), RoPE, activation
checkpointing, Muon at constant `0.01`, dropout `0.0`, fp32 throughout,
sampled eval (16 batches), checkpoint every 5,000 steps. Corpus: 494.09M
chars TinyStories, 419.98M train / 74.11M val.

| Step | Train NLL | Val NLL | Bits/char |
| ---: | ---: | ---: | ---: |
| 20,000 | 0.633789 | 0.569873 | 0.822153 |
| 25,000 | 0.615162 | 0.575585 | 0.830394 |
| 30,000 | 0.621661 | 0.578902 | 0.835180 |

Log reached step 32,000 (64% of the 50k budget); steps past 30,000 are
discarded on resume by design. Observed throughput is `~1.9 s/step` against
the D3a projection of `1.146 s/step` (eval, checkpoint writes, thermals, and
OneDrive sync churn all land on top of the gate's clean number). Remaining
cost: `~9.5 h` for seed 7, then `~10.5 h` each for the 20k-step replicas
(seeds 11 and 19).

For scale: Stage 51's 25.25M checkpoint reached 1.181 bits/char at 5,000
steps. The flagship at step 30,000 sits at 0.835 bits/char, the best NLL the
lab has ever produced, at 8x the parameters and 42x the corpus of the Phase 2
era.

### The plateau worth watching

Val NLL has drifted UP `+0.009` from step 20k to 30k while train NLL stayed
flat. Two readings, not yet distinguishable: sampled-eval noise (16 batches
resample each eval; the drift is the same order as seed spread), or a
constant Muon LR of `0.01` that is too hot for the late phase of a run 25x
longer than the horizon it was tuned on (Stage 53's own LR rerun proved this
recipe is LR-sensitive). Recommendation: do NOT change the recipe mid-run;
the protocol's integrity is worth more than a possible late NLL crumb. Watch
the 35k and 40k evals. If the drift persists to 50k, a cheap follow-up arm
(`stage55b`: resume from the 30k checkpoint with a decayed LR, run to 50k,
compare final NLL) turns the observation into a falsifiable claim instead of
a regret.

## 2 · Architecture retrospective

The user asked: how would Claude have designed the flagship, what is good,
what is different, what can be done better later.

### What is good and should be kept

1. Evidence-assembled recipe. Every choice traces to a measured verdict:
   random init from Stage 53, size and block from D3a measurements, resume
   support proven before committing 16 GPU-hours. The resume proof has
   already paid for itself three times over across the pause/resume cycles
   visible in the launcher logs.
2. The modern minimal stack at this scale: RoPE, Muon, activation
   checkpointing, shard-streamed corpus, grad accumulation. Nothing
   fashionable that the evidence did not ask for.
3. Ops discipline. The visible-launch protocol (ADR 0012), pit-stop notes
   with a durable-checkpoint pointer and exact resume command, and the
   discard-past-checkpoint rule make the long run interruption-proof and
   auditable.
4. The honest seed policy (one full seed plus two reduced replicas, stated
   plainly) instead of pretending three full seeds fit the budget.

### What Claude would have designed differently, with the same evidence

1. Match sizing candidates on wall-clock, not step count. The D3a gate
   compared four candidates at 200 steps each. At `1.146 s/step` (200M·b256)
   versus `0.525 s/step` (85M·b256), an equal-TIME comparison gives the 85M
   model 2.2x the steps and data. Fixed-step comparison structurally favors
   capacity; that is exactly Stage 52's monotone-capacity fact restated. The
   flagship will see about 205M chars, roughly 1 char per parameter, far
   below the `~20` tokens-per-param compute-optimal band, so the missing
   control is real: an 85M model given the same 16 hours might match or beat
   the 200M's final NLL. Not a claim the current run is wrong, but the
   equal-compute arm is the cheapest important experiment Phase 4 has not
   run (one overnight run).
2. Reconsider the substrate for a prior-free build. Char-level was fixed by
   ADR 0011 D1 because the analytic prior needed it. Stage 53 then removed
   the prior from the flagship, but the char constraint stayed inherited.
   Stage 50 killed the BPE bigram PRIOR, not BPE under `random_full`. A BPE
   flagship would fit roughly 4x more text into block 256, and coherence is
   bounded by context. The counterargument is comparability with Stages
   44-54, which is real; but a BPE `random_full` candidate deserved a row in
   the D3a gate.
3. A learning-rate schedule. Constant `0.01` was tuned at 2000-step horizons
   and is now run 25x longer. A cosine or linear warmdown over the final
   third is free, standard, and the observed 20k-30k val drift is exactly
   the symptom it prevents.

### What can be done better later, ranked

1. bf16 autocast. The trainer has no `torch.autocast` or mixed-precision
   path; the flagship trains in full fp32 on an Ada-generation RTX 4070.
   bf16 would give roughly 1.5-2x step throughput and cut activation memory,
   with none of fp16's loss-scaling fragility. Biggest single win for any
   future long run.
2. Checkpoint policy. Each checkpoint is `1.51 GiB` because model AND Muon
   momentum are saved fp32 and every checkpoint is kept. Add
   `--checkpoint-keep N` (rolling window; optimizer state only needed in the
   newest) and an archival mode that saves model-only fp16 (`~400 MB`). This
   is also the root cause of the storage problem in Section 3.
3. `--lr-schedule cosine --lr-final <x>` flags for any run past a few
   thousand steps.
4. A block-512 probe. TinyStories stories run roughly 750-1500 chars; block
   256 never sees a whole story in one window, so the coherence ask exceeds
   the context. D3a measured `~4.9 GiB` of headroom; a 200-step gate row
   would price block 512 the same way block 256 was priced.
5. Full (not sampled) final eval for the closeout claim. Sampled-16 noise is
   the same order as the drift currently being interpreted.
6. Optional: weight EMA for the final artifact; cheap and usually buys a
   small NLL and smoother samples.

## 3 · Storage: measured map, projection, cleanup strategy

Context: C: has `135.6 GB` free, so the local disk is not the binding
constraint. The repo lives inside OneDrive (`Masaüstü` is synced), so every
checkpoint write is also a `1.5 GB` cloud upload against the OneDrive quota,
and sync churn overlaps `torch.save` writes during training.

### Measured map (2026-07-05)

| Item | Size | Verdict |
| --- | ---: | --- |
| `runs/stage55_resume_checkpoints` | 9.27 GB | Delete. The resume proof is recorded (PASS) in RESULTS.md and the jsonl/md summaries. Nothing references these files. Safe even before training resumes. |
| `runs/stage55_flagship_checkpoints` | 9.27 GB now, ~17 GB at seed-7 completion | Prune after ONNX is verified. Keep the final checkpoint only (optionally plus step 25000 as a half-run reference). During the run keep the newest two. |
| `corpus/tinystories_raw` | 524 MB | Delete or archive after Phase 4. Re-downloadable; derivation is scripted; the `.meta.json` records provenance. |
| `corpus/tinystories_char_seed.txt` + shards | 950 MB | KEEP. The launcher requires both (seed file provides val split and vocab, shards feed training). A later code change could derive both from shards and drop the monolith. |
| `runs/stage52_prior_cache` | 338 MB | Keep until the publication fork (ADR 0013 D5) is decided; rebuildable in minutes if desperate. |
| `runs/stage51_checkpoints` | 291 MB | Review PASSED 2026-07-06, no longer blocked. Low priority: keep through Phase 4 closeout as the 25M reference model, then optional delete. |
| `.git` | 1.26 GB | Corpus blobs are in history (476.8 MB raw slice, 471.2 MB seed snapshot). See git subsection. |
| All jsonl / md / log evidence | < 10 MB | Keep forever. This is the durable record. |

Projection if nothing is pruned: seed 7 finishing adds checkpoints at 35k,
40k, 45k, 50k plus a final file (`+7.7 GB`); each 20k replica adds `~7.7 GB`
more. Peak around `45 GB` of checkpoints alone. The plan below caps that.

### During the remaining run (optional but recommended)

- Keep the newest TWO flagship checkpoints, delete older ones as new ones
  land. Never delete the newest (it is the resume anchor).
- Run the seed 11 and 19 replicas either with `--checkpoint-every 10000` or
  prune each replica to its final checkpoint as it finishes.

### After training is done and the ONNX file is ready

Ordering matters: nothing is deleted until the ONNX is exported from the
FINAL checkpoint and verified.

1. Export. Per the pit-stop closeout constraint, export only after Phase 4
   succeeds, from the final seed-7 checkpoint, into
   `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/`. Codex updated
   the export defaults to the durable final seed-7 checkpoint and artifact
   folder:

   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\export_stage55_nsight_dld.ps1 `
     -CheckOnnx
   ```

2. Verify before deleting anything: `-CheckOnnx` passes, the file opens in
   Nsight DL Designer, and (recommended) a short logits parity spot-check
   between the `.pt` model and the ONNX on one prompt.
3. Keep-set: the final `.pt` checkpoint (ONE copy; the trainer writes both a
   `_step050000.pt` and a no-suffix final file with identical content, keep
   the explicit step file), the ONNX plus its metadata json, every jsonl,
   md, and log, and the RESULTS.md stage entry.
4. Tier A delete: `runs\stage55_resume_checkpoints` entirely (`9.27 GB`).
5. Tier B delete: all flagship intermediates except the keep-set
   (`~14 to 15.4 GB` at that point). Same rule for the replica seeds.
6. Tier C delete: `corpus\tinystories_raw` (`524 MB`).
7. Recovery if a delete goes wrong: only the final checkpoint is
   irreplaceable for continuing training; the ONNX is irreplaceable for
   inference. Copy both somewhere outside the repo before Tier B.

Expected end state: repo around `3 to 4 GB` with every piece of evidence
intact.

### Git housekeeping (separate decision, do not rush)

`git status` shows `corpus/tinystories_char_seed.txt` as tracked-and-
modified. If it is committed as-is, a fresh `471 MB` blob enters history.
Before the next commit:

```powershell
git rm --cached experiments/tiny_language_lab/corpus/tinystories_char_seed.txt
git rm --cached experiments/tiny_language_lab/corpus/tinystories_char_seed.meta.json  # only if intended
# commit together with the already-modified .gitignore
```

History already holds `~1.07 GB` of corpus blobs. A remote exists
(`github.com/mertefesensoy/CASSANDRA`), and GitHub rejects files over
100 MB, so what the remote actually holds needs verification before any
surgery (`git fetch origin` then compare, or check the repo on github.com).
If the fat blobs need purging: make a full backup first
(`git bundle create ..\cassandra-backup.bundle --all`), then use
`git filter-repo` to strip `experiments/tiny_language_lab/corpus/` blobs
from history, then force-push. This is the only step in this plan that
rewrites shared state; treat it as its own task with its own confirmation.

### Structural fix: get run artifacts out of OneDrive

The durable record is RESULTS.md plus the small run summaries; `runs/` is
already gitignored and was never meant to be synced. For the next long run,
point the launcher's checkpoint directory outside OneDrive (for example
`C:\cassandra_runs\...`), either by editing `$CheckpointDir` in
`run_phase4_visible.ps1` or adding a `-CheckpointDir` parameter. Benefits:
no quota pressure, no sync churn during `torch.save`, no risk of OneDrive
uploading a half-written checkpoint. Archived keep-set files that must stay
in OneDrive can use Files On-Demand ("Free up space") to drop the local
copy, at the cost of quota; local-only bulk belongs outside the synced tree
entirely.

## 4 · Verification

- Progress numbers: `runs/phase4_stage55-flagship-cell_20260704_005801.log`
  (eval lines at 20k/25k/30k) and
  `runs/stage55_seed7_before35000_pitstop_20260704.md`.
- Sizes: measured 2026-07-05 with `Get-ChildItem | Measure-Object Length`;
  git blob sizes via `git rev-list --objects --all | git cat-file
  --batch-check`.
- Nothing was deleted, moved, or exported in this pass; training inputs are
  untouched.

## 5 · Addendum (2026-07-06) · seed 7 complete, checkpoints relocated

Seed 7 finished the full 50,000 steps (exit 0): final sampled val NLL
`0.556410`, `0.802730` bits/char, peak CUDA `4,007.55 MiB`, summary in
`runs/stage55_flagship_200m_b50000_seed7.md`. The 20k-30k val-NLL drift
watch item is CLOSED as sampled-eval noise: the curve bounced (35k `0.5715`,
40k `0.5852`, 45k `0.5496`, 50k `0.5513`, final `0.5564`) inside a `~0.03`
band and ended below the 20k reading; no late-LR divergence, no `stage55b`
needed.

Storage facts changed mid-run. At step 35,000 (02:33) the OneDrive-backed
checkpoint save failed on the final `.pt.tmp` to `.pt` rename (WinError 5,
preserved in `runs/stage55_flagship_200m_b50000_seed7_onedrive_checkpoint_
error_20260706.*`), killing the run; recovery moved `--checkpoint-dir` to
`C:\Users\senso\AppData\Local\Temp\cassandra_runs\stage55_flagship_checkpoints`.
Consequences for the plan above:

1. Steps 35k, 40k, 45k, 50k, and the final file live in `%TEMP%`, and
   Windows Storage Sense is ENABLED with temp cleaning ON, so that location
   is purgeable. The two final artifacts (`..._step050000.pt` and the
   no-suffix final, `1.51 GB` each) were copied on 2026-07-06 to the durable
   `C:\cassandra_runs\stage55_flagship_checkpoints` with byte counts
   verified. The ONNX export must load from this secured copy.
2. The OneDrive `runs\stage55_flagship_checkpoints` directory (steps 5k-30k
   plus a stranded `1.51 GB` `step035000.pt.tmp`) is now fully superseded:
   the whole directory, `10.8 GB`, is Tier B deletable after the ONNX is
   verified; the `.tmp` is deletable any time.
3. The seed-11 20k replica (started 14:32) writes checkpoints to the same
   `%TEMP%` directory. Codex later copied seed 11 and seed 19 final and
   no-suffix checkpoints to
   `C:\cassandra_runs\stage55_flagship_checkpoints` with byte-count
   verification. Seed 19 also trained in `%TEMP%` before the addendum was read;
   the launcher default now points to `C:\cassandra_runs` for future runs.
4. The MUSAHIT nightly Ollama task (daily 02:00, GPU-hungry) was suppressed
   for 2026-07-07 only, via a one-shot `SKIP_NEXT_RUN.flag` consumed by its
   own launcher script, to protect the replica window.

## 6 · Related docs

- `docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md`
- `docs/phase4-intake.md` · `docs/phase4-codex-goal-prompt.md`
- `docs/nsight-dl-designer-workflow.md`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 53-55)
- `experiments/tiny_language_lab/runs/stage55_d3a_sizing_resume_summary.md`
