---
name: run-pitstop
description: Safely pause, verify, document, and resume a GPU training run. Use when the user asks to pause or stop a running training job, or to resume one that was paused or crashed.
---

# run-pitstop · pause and resume a training run without losing evidence

When to use: any time a long training run must stop (meeting, sleep, GPU
needed elsewhere, crash recovery) or continue after a stop.

## Exact steps, in order

### Pausing

1. Identify the run: find the newest `phase*_*visible_pid.txt` and
   launcher log in `experiments/tiny_language_lab/runs/`. Note the
   launcher PID and the python training PID.
2. Check the newest `[checkpoint]` line in the run's `.log` file. The run
   may ONLY be stopped after a checkpoint you are willing to resume from
   exists. If a checkpoint write is imminent (steps come every ~1.9 s,
   checkpoints every 5000 steps), prefer waiting for it.
3. Stop the python training PID first, then the launcher PID
   (`Stop-Process -Id <pid>`).
4. Verify the stop: `nvidia-smi` must report `0 MiB` used and no running
   `cassandra_compare.py` process. `Get-Process python*` must be empty.
5. Sanity-load the durable checkpoint (torch.load, check `step`,
   optimizer state present, generator state present) or at minimum
   record its size and mtime.
6. Write a pit-stop note at
   `experiments/tiny_language_lab/runs/<run>_pitstop_<yyyymmdd>.md`
   following the example below: stop time, stopped PIDs, nvidia-smi
   verification, the durable checkpoint path, checkpoint facts, and the
   EXACT resume command.

### Resuming

7. Confirm the GPU is free (`nvidia-smi` = 0 MiB) and no conflicting
   scheduled job is imminent (the 02:00 nightly Ollama task is a known
   collider; see mistakes below).
8. Use the resume command from the newest pit-stop note verbatim. It
   must pass `-ResumeFrom <durable checkpoint>` and the SAME budget,
   seed, and checkpoint dir as the original run.
9. Watch the first `[eval]`/`[train]` lines: the resumed run should
   report `resume_loaded=True` at the checkpoint's step and continue.

## Full example of a good final output (real file: `runs/stage55_seed7_before35000_pitstop_20260704.md`)

```markdown
# Stage 55 Seed 7 Before-35000 Pit Stop

Date: 2026-07-04

Status: paused immediately by user request before reaching the step-35000
checkpoint.

Training was force-stopped without waiting for the next checkpoint. The visible
launcher process and Python child were stopped:

- launcher PID `37832`
- Python PID `37708`

After stopping, `nvidia-smi` reported `0 MiB` GPU memory used and no active
`cassandra_compare.py`, `run_phase4_visible.ps1`, or `stage55_flagship`
training process remained.

## Latest Durable Checkpoint

Use this checkpoint for the next continuation:

    experiments/tiny_language_lab/runs/stage55_flagship_checkpoints/stage55_flagship_200m_b50000_seed7_random_full_seed7_step030000.pt

Checkpoint sanity load:

- `step=30000`
- `formation_forward_passes=60000`
- optimizer state present
- generator state present
- file size: `1,619,477,793` bytes
- recorded eval rows: `5000`, `15000`, `20000`, `25000`, `30000`
- step-30000 eval: train NLL `0.621661`, val NLL `0.578902`,
  bits/char `0.835180`

The run log reached step `32000/50000`, but step 32000 was not checkpointed.
Those post-checkpoint steps are intentionally discarded for a safe resume.

## Resume Command

    powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 `
      -Mode stage55-flagship-cell `
      -Budget 50000 `
      -Seed 7 `
      -ResumeFrom .\experiments\tiny_language_lab\runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step030000.pt
```

## Mistakes to avoid (each one actually happened here)

- **Never point `--checkpoint-dir` inside OneDrive.** The seed-7 run
  CRASHED at step 35,000 when OneDrive locked the `.pt.tmp` rename
  (WinError 5). Checkpoints go to `C:\cassandra_runs\...` only.
- **Never leave the only copy of a checkpoint in `%TEMP%`.** Windows
  Storage Sense purges it; the flagship's final weights sat there until
  byte-verified copies were made. Secure to `C:\cassandra_runs` first.
- **Do not resume from the log's last step.** Steps after the last
  checkpoint are discarded by design (the seed-7 log reached 32,000 but
  the resume anchor was 30,000). Resuming "from the log" corrupts the
  evidence chain.
- **Check the 02:00 collision window.** A nightly task (`MUSAHIT_Nightly`)
  loads an Ollama model onto the same 8 GB GPU at 02:00. One night can be
  skipped by creating `SKIP_NEXT_RUN.flag` next to that project's
  `run_nightly.ps1`; the task itself is admin-locked.
- **Do not sum wall-clock across resume legs naively.** `elapsed` resets
  per invocation; the closeout must say "seconds are per final
  invocation" (the Stage 55 closeout does exactly this).
- **Preserve interrupted evidence.** A crashed run's partial jsonl/md gets
  a suffixed name (for example `..._interrupted_partial.jsonl`), never
  deleted (Stage 53 convention).
