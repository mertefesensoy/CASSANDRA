---
name: storage-cleanup
description: Execute the tiered checkpoint and artifact cleanup safely. Use when disk or OneDrive pressure appears, or after a phase closes and its ONNX export is verified.
disable-model-invocation: true
---

# storage-cleanup · reclaim space without losing evidence

When to use: run artifacts are crowding the disk or OneDrive quota, AND
the phase's protective conditions are met (final checkpoints secured
outside OneDrive and `%TEMP%`, ONNX exported and parity-verified).

## Exact steps, in order

1. Read the current cleanup plan first
   (`docs/phase4-flagship-midrun-report.md` Section 3 plus its addendum,
   or the successor doc). Never delete from memory of a plan; plans
   change as incidents happen.
2. Verify the protective gates: (a) the final checkpoints exist in
   `C:\cassandra_runs\...` with byte counts matching their sources;
   (b) the ONNX artifact exists under
   `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/` and its
   parity check is recorded (max logit diff `8.6e-6` class, top-1
   agreement `1.0`); (c) no training process is running
   (`nvidia-smi` = 0 MiB, `Get-Process python*` empty).
3. Measure before deleting: list each target directory with sizes, and
   present the keep-versus-delete table to the user for confirmation.
4. Delete in tier order, verifying between tiers: superseded resume-proof
   checkpoints first, then superseded intermediate checkpoints, then
   re-derivable corpus downloads. Never delete jsonl, md, or log
   evidence; it is the durable record and costs almost nothing.
5. Git hygiene in the same pass: confirm no large corpus file is tracked
   (`git status`; `git rm --cached` if one is), and never commit a file
   over 50 MB.
6. Record what was deleted and what was kept, with sizes, in the session
   log or the plan's addendum.

## Full example of a good final output (real table: `docs/phase4-flagship-midrun-report.md` Section 3, measured 2026-07-05)

```markdown
| Item | Size | Verdict |
| --- | ---: | --- |
| `runs/stage55_resume_checkpoints` | 9.27 GB | Delete. The resume proof is recorded (PASS) in RESULTS.md and the jsonl/md summaries. Nothing references these files. Safe even before training resumes. |
| `runs/stage55_flagship_checkpoints` | 9.27 GB now, ~17 GB at seed-7 completion | Prune after ONNX is verified. Keep the final checkpoint only (optionally plus step 25000 as a half-run reference). During the run keep the newest two. |
| `corpus/tinystories_raw` | 524 MB | Delete or archive after Phase 4. Re-downloadable; derivation is scripted; the `.meta.json` records provenance. |
| `corpus/tinystories_char_seed.txt` + shards | 950 MB | KEEP. The launcher requires both (seed file provides val split and vocab, shards feed training). |
| `runs/stage52_prior_cache` | 338 MB | Keep until the publication fork (ADR 0013 D5) is decided; rebuildable in minutes if desperate. |
| `runs/stage51_checkpoints` | 291 MB | Review PASSED 2026-07-06, no longer blocked. Low priority: keep through Phase 4 closeout as the 25M reference model, then optional delete. |
| All jsonl / md / log evidence | < 10 MB | Keep forever. This is the durable record. |
```

## Mistakes to avoid (each one actually happened here)

- **OneDrive is not a safe write target for big files.** A checkpoint
  rename failed mid-run with WinError 5 and killed 8,426 seconds of
  training. Heavy artifacts live outside the synced tree.
- **`%TEMP%` is not storage.** Windows Storage Sense was enabled with
  temp cleaning ON while the only copies of the flagship's final weights
  sat in `%TEMP%\cassandra_runs`. Secure copies FIRST, delete second.
- **Delete only after the export is verified.** The rule is
  ONNX-before-delete: parity-checked export, then pruning. Never the
  other way around.
- **The launcher requires files you might think are redundant.** The
  monolithic seed corpus AND the shard directory are both load-bearing
  (vocab plus val split come from the seed file); deleting one breaks
  every future run.
- **Git history already carries about 1.07 GB of corpus blobs** that
  exceed GitHub's limits; cleanup includes untracking, and history
  surgery happens only with a bundle backup and explicit user sign-off.
- **Keep every checkpoint referenced by a pending decision.** The
  Stage 51 checkpoints were held until the human review passed; a
  pending gate freezes its artifacts.
