# INSTRUCTIONS.md · How Mert works (read me before touching anything)

For any model working this repo cold. CLAUDE.md explains the codebase;
this file explains the human, the workflow, and the bar for "done".

## 1 · Stack, tools, clients, formats

- **Machine:** Windows 11 laptop, RTX 4070 8 GB, PowerShell 7 primary
  (Git Bash available). The repo sits under a OneDrive-synced Desktop
  with a non-ASCII path (`Masaüstü`): treat OneDrive as a hazard, never
  write checkpoints or big artifacts into it.
- **Canonical locations:** training checkpoints go to
  `C:\cassandra_runs\...` (never OneDrive, never `%TEMP%`); ONNX exports
  to `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/`; run
  artifacts to `experiments/tiny_language_lab/runs/` (gitignored; the
  durable record is `RESULTS.md`).
- **Python:** torch `2.12.1+cu126` (CUDA works), matplotlib, gradio,
  onnxruntime. GPU jobs launch through the visible-terminal protocol
  (ADR 0012) via `run_phase*_visible.ps1`.
- **Three AI roles, strict lanes:** Claude writes hypotheses, ADRs,
  roadmaps, and reports. Codex writes code, runs matrices, and writes
  RESULTS.md stage entries. Gemini writes prior-art notes under
  `research/`. Handoffs travel as repo files
  (`docs/phaseN-codex-goal-prompt.md`, `research/theme_*/NN_*.md`),
  sometimes pasted manually. Stay in your lane.
- **Decision formats:** hypotheses at `docs/hypotheses/NNN-slug.md` with
  a mandatory pass-or-fail line; ADRs at `docs/decisions/NNNN-slug.md`
  with a mandatory reopen-or-reverse condition; every stage lands in
  `RESULTS.md` with command shape, seeds (`7 11 19`), parameter counts,
  val NLL and bits/char, wall clock, and an interpretation that states
  what it does NOT prove.
- **Evaluation conventions (ADR 0014):** closeout claims use the chunked
  deterministic eval; `--eval-mode sampled --eval-batches 16` is in-run
  monitoring only; zero-shot text8 is the standing external anchor.
- **Mert's messages** are fast and typo-prone ("müşhait" meant the
  MUSAHIT scheduled task; "ONMX" meant ONNX). Resolve by evidence in the
  files, not by guessing; when a fork is genuinely his, ask short
  structured questions. He answers tersely and expects autonomy between
  questions.

## 2 · Tone and writing rules

- **No em dashes, no en dashes, anywhere in repo docs.** Use the middle
  dot `·` or restructure. A PostToolUse hook flags violations.
- Numbers are backticked, cited to a run file, and never invented: a
  number is measured or it is "to be measured by Codex".
- Claims are scoped tightly; every strong result ends by naming what it
  does not prove. Negative and failed results are kept as evidence,
  never deleted (supersede with suffixed filenames).
- Absolute dates only ("before 2026-07-20", never "next week").
- Sample of the house style (real text, Stage 53):

  > This result does not say the order-4 prior is useless. It says the
  > current full-body Muon recipe does not get the prior for free: the
  > additive frozen base helps early, then becomes a late-training
  > handicap relative to random initialization. [...] This stage remains
  > a local NLL result, not a coherence or sample-quality claim.

## 3 · The weekly tasks and the skill that covers each

| Weekly task | Skill file (`.claude/skills/...`) |
| --- | --- |
| Pause, verify, document, resume GPU runs | `run-pitstop/SKILL.md` |
| Write the RESULTS.md entry for a finished run | `new-stage-entry/SKILL.md` |
| Fold Codex or Gemini output into the roadmap | `read-back/SKILL.md` |
| Reclaim disk safely after a phase closes | `storage-cleanup/SKILL.md` (user-only) |
| Commit the week's work without hazards | `weekly-commit/SKILL.md` (user-only) |
| Record session outcomes in persistent memory | `memory-update/SKILL.md` |
| Rebuild report figures from raw artifacts | `regen-figures/SKILL.md` |
| Scaffold a new hypothesis or ADR | `new-hypothesis/SKILL.md`, `new-adr/SKILL.md` |

Each skill contains the exact steps, a real gold-standard output, and
the mistakes that actually happened here. Follow them literally.

## 4 · What a good day's output looks like, concretely

The 2026-07-07 session is the reference day. It produced, in order: a
read-back of the finished runs (verdicts named, numbers spot-checked
against jsonl); a verification pass (ONNX parity `8.6e-6`, an 8M-char
re-eval confirming every recorded number); one new measurement (zero-shot
text8, `2.8817` bits/char, full test split); a report with five figures
regenerated from raw artifacts; one ADR with reopen conditions; one
hypothesis with a calibrated pass-or-fail line (threshold moved from
`1.60` to `1.70` against the published record BEFORE running); memory
updated so the next session starts oriented.

The bar, distilled: **every claim traces to an artifact or a citation;
every decision carries the condition that would reverse it; every
artifact worth keeping is in its canonical location; nothing pending is
described as done; and the user was asked only the questions that were
genuinely his to answer.** A day that trains nothing but leaves the
record verified, falsifiable, and tidy is a good day. A day that produces
numbers nobody can regenerate is a bad day regardless of how good they
look.

## 5 · Standing hazards (learned the hard way)

OneDrive kills checkpoint renames mid-run; Storage Sense purges `%TEMP%`;
`Tee-Object` logs are UTF-16; `runs/` is gitignored so gitignore-aware
search tools skip it; git history carries oversized corpus blobs and the
repo must not be pushed before the planned history surgery; a nightly
02:00 task (MUSAHIT) loads Ollama onto the GPU and collides with
overnight training (one-shot skip: `SKIP_NEXT_RUN.flag` beside its
`run_nightly.ps1`).
